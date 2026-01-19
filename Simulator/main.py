from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable
import heapq
import itertools
import random
from agent import DecisionAgent, LongestQueueAgent
from observation_manager import ObservationManager
from job import Job


# =====================================================
# Process time models
# =====================================================

def const_time(x: float) -> Callable:
    return lambda e, rng: x

def lognormal_time(mu: float, sigma: float) -> Callable:
    return lambda e, rng: rng.lognormvariate(mu, sigma)




# =====================================================
# Engine
# =====================================================

@dataclass(order=True)
class Event:
    time: float
    priority: int
    seq: int
    etype: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)


class Engine:
    def __init__(self, seed=42):
        self.now = 0.0
        self.queue: List[Event] = []
        self.seq = itertools.count()
        self.handlers = {}
        self.rng = random.Random(seed)
        self.on_advance_time = None

    def register(self, etype, handler):
        self.handlers[etype] = handler

    def schedule(self, time, etype, payload, priority=50):
        heapq.heappush(
            self.queue,
            Event(time, priority, next(self.seq), etype, payload)
        )

    def run(self, until):
        while self.queue:
            ev = heapq.heappop(self.queue)
            if ev.time > until:
                break

            dt = ev.time - self.now
            if dt > 0 and self.on_advance_time:
                self.on_advance_time(dt)

            self.now = ev.time
            self.handlers[ev.etype](ev.payload)


# =====================================================
# Resource
# =====================================================

@dataclass
class ResourcePool:
    capacity: int
    resource_id: str = ""
    observer: ObservationManager | None = None
    busy: int = 0

    def acquire(self):
        self.busy += 1
        if self.observer:
            self.observer.add(self.resource_id)

    def release(self):
        self.busy -= 1
        if self.busy == 0 and self.observer:
            # 👇 從 busy → available
            self.observer.remove(self.resource_id)

    def available(self):
        return self.capacity - self.busy


# =====================================================
# Entity
# =====================================================

@dataclass
class Entity:
    eid: int
    created: float
    held_resources: Dict[str, ResourcePool] = field(default_factory=dict)
    dispatch_target: str | None = None
    site: str
    


# =====================================================
# Station (base)
# =====================================================

@dataclass
class Station:
    name: str
    engine: Engine
    resources: List[ResourcePool]
    process_time: Callable[[Entity, random.Random], float]

    queue: List[Entity] = field(default_factory=list)
    routes: List[Tuple[float, str, float]] = field(default_factory=list)

    busy_time: float = 0.0

    def arrive(self, payload):
        entity = payload.entity
        self.queue.append(entity)
        self.try_start()

    def try_start(self):
        while self.queue and self.check_resources():
            e = self.queue.pop(0)
            self.resource.acquire()

            duration = self.process_time(e, self.engine.rng)

            self.engine.schedule(
                self.engine.now + duration,
                f"END_{self.name}",
                {"eid": e.eid},
                priority=10
            )

    def end(self, e: Entity):
        self.release_resources()

        # probabilistic routing
        if self.routes:
            r = self.engine.rng.random()
            acc = 0.0
            for prob, event, delay in self.routes:
                acc += prob
                if r <= acc:
                    self.engine.schedule(
                        self.engine.now + delay,
                        event,
                        {"eid": e.eid},
                        priority=20
                    )
                    break

        self.try_start()

    
    def on_advance_time(self, dt: float):
        total_busy = sum(pool.busy for pool in self.resources)
        self.busy_time += total_busy * dt

    def utilization(self, sim_time: float) -> float:
        total_capacity = sum(pool.capacity for pool in self.resources)
        return self.busy_time / (total_capacity * sim_time)
    
    def check_resources(self) -> bool:
        return all(res.available() > 0 for res in self.resources)
    
    def acquire_resources(self):
        for res in self.resources:
            res.acquire()

    def release_resources(self):
        for res in self.resources:
            res.release()



# =====================================================
# LoadStation (A)
# =====================================================

@dataclass
class LoadStation(Station):
    excavator_pools: Dict[str, "ResourcePool"]

    def try_start(self):
        i = 0
        while self.queue:
            entity = self.queue[0]
            site = entity.site
            pool = self.excavator_pools[site]

            if pool is None:
                raise RuntimeError(
                    f"No excavator pool defined for site '{site}'"
                )

            if pool.available() > 0:
                pool.acquire()

                # remove item 
                self.queue.pop(i)

                duration = self.process_time(entity, self.engine.rng)

                self.engine.schedule(
                    self.engine.now + duration,
                    f"END_{self.name}",
                    {"eid": entity.eid},
                    priority=10,
                )
            else:
                i += 1

    def end(self, entity):
        pool = self.excavator_pools[entity.site]
        pool.release()

        super().end(entity)

# =====================================================
# UnloadStation (C)
# =====================================================

@dataclass
class UnloadStation(Station):
    def end(self, e: Entity):
        self.release_resources()

        for pool in e.held_resources.values():
            pool.release()
        e.held_resources.clear()

        self.try_start()


# =====================================================
# World
# =====================================================

class World:
    def __init__(self, engine: Engine, agent: DecisionAgent):
        self.engine = engine
        self.agent = agent

        self.observation_manager = ObservationManager(on_trigger=self.request_dispatch)

        self.pending_tasks = [
            Entity(eid=1, created=0.0),
            Entity(eid=2, created=0.0),
            Entity(eid=3, created=0.0),
            Entity(eid=4, created=0.0),
            Entity(eid=5, created=0.0),
        ]
        self.active_tasks = {}
        self.complete_tasks = []

        self.machines = ResourcePool(2)
        self.trucks = ResourcePool(2, "truck_pool", self.observation_manager )
        self.drivers = ResourcePool(2, "driver_pool", self.observation_manager)
        self.scales = ResourcePool(1)
        self.unloaders = ResourcePool(1)
        self.inspectors = ResourcePool(1)

        self.entities: Dict[int, Entity] = {}

        self.A = LoadStation(
            name="A",
            engine=engine,
            resources=[self.machines, self.trucks, self.drivers],
            trucks=self.trucks,
            drivers=self.drivers,
            process_time=lognormal_time(0.0, 0.25),
            routes=[
                (0.9, "ARRIVE_B", 2.0),
                (0.1, "ARRIVE_D", 3.0),
            ],
        )

        self.B = Station(
            name="B",
            engine=engine,
            resources=[self.scales],
            process_time=const_time(0.5),
            routes=[(1.0, "ARRIVE_C", 2.0)],
        )

        self.D = Station(
            name="D",
            engine=engine,
            resources=[self.inspectors],
            process_time=const_time(1.5),
            routes=[(1.0, "ARRIVE_C", 1.0)],
        )

        self.C = UnloadStation(
            name="C",
            engine=engine,
            resources=[self.unloaders],
            process_time=const_time(1.0),
        )

        self.stations = [self.A, self.B, self.D, self.C]

        engine.register("ARRIVE_A", lambda p: self.A.arrive(self.entities[p["eid"]]))
        engine.register("END_A", lambda p: self.A.end(self.entities[p["eid"]]))

        engine.register("ARRIVE_B", lambda p: self.B.arrive(self.entities[p["eid"]]))
        engine.register("END_B", lambda p: self.B.end(self.entities[p["eid"]]))

        engine.register("ARRIVE_D", lambda p: self.D.arrive(self.entities[p["eid"]]))
        engine.register("END_D", lambda p: self.D.end(self.entities[p["eid"]]))

        engine.register("ARRIVE_C", lambda p: self.C.arrive(self.entities[p["eid"]]))
        engine.register("END_C", lambda p: self.C.end(self.entities[p["eid"]]))

        engine.on_advance_time = self.on_advance_time


    def on_advance_time(self, dt: float):
        for st in self.stations:
            st.on_advance_time(dt)

    def report_utilization(self, sim_time: float):
        return {st.name: st.utilization(sim_time) for st in self.stations}
    
    def request_dispatch(self):
        observation = {
            "time": self.engine.now,
            "available_trucks": self.trucks.available(),
            "pending_tasks": [t.eid for t in self.pending_tasks],
            "active_tasks": list(self.active_tasks.keys()),
        }

        action = self.agent.decide(observation)
        if action:
            self.apply_action(action)

    def get_task(self, task_id) -> Entity:
        for i, task in enumerate(self.pending_tasks):
            if task.eid == task_id:
                return self.pending_tasks.pop(i)
        
    def apply_action(self, action):
        task_id = action["task_id"]
        target_station = action["from_location"]

        task = self.get_task(task_id)
        task.dispatch_target = action["dispatch_target"]

        # 現在才正式讓任務進系統
        self.active_tasks[task.eid] = task

        self.engine.schedule(
            self.engine.now,
            f"ARRIVE_{target_station}",
            {"entity": task}
        )

    def start(self):
        self.request_dispatch()





if __name__ == "__main__":
    eng = Engine(seed=7)

    agent = LongestQueueAgent()
    world = World(eng, agent)
    world.start()

    horizon = 50
    eng.run(until=horizon)

    print("Utilization:")
    for k, v in world.report_utilization(horizon).items():
        print(f"  Station {k}: {v:.2%}")
