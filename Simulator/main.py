from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable
import heapq
import itertools
import random
from agent import DecisionAgent, RandomDispatchAgent
from observation_manager import ObservationManager
from job import Job


# =====================================================
# Process time models
# =====================================================

def const_time(x: float) -> Callable:
    return lambda e, rng: x

def lognormal_time(mu: float, sigma: float) -> Callable:
    return lambda e, rng: rng.lognormvariate(mu, sigma)


depots = {
    "Depot_A": {"x": 0, "y": 0},
    "Depot_B": {"x": 10, "y": 0},
    "Depot_C": {"x": 5, "y": 8},
}
disaster_sites = {
    "Site_1": {"x": 3, "y": 12},
    "Site_2": {"x": 15, "y": 5},
    "Site_3": {"x": 7, "y": 40},
}
# fuel_station = {"x": 6, "y": 3}

def transport_time(truck_loc: str, task_site: str) -> float:


    x1, y1 = depots[truck_loc]["x"], depots[truck_loc]["y"]
    x2, y2 = disaster_sites[task_site]["x"], disaster_sites[task_site]["y"]

    distance = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
    speed = 5.0  # km/h or whatever unit

    duration = distance / speed
    return duration


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

            # print(ev.payload["eid"])
            print(ev.etype, ev)
            self.now = ev.time
            self.handlers[ev.etype](ev.payload)


# =====================================================
# Resource
# =====================================================
@dataclass
class ResourcePool:
    capacity: int
    resource_id: str = ""

    items: Dict[str, Any] | None = None   # truck_id -> meta
    in_use: Dict[str, Any] = field(default_factory=dict)

    observer: ObservationManager | None = None
    busy: int = 0

    # ==========================
    # Acquire
    # ==========================
    def acquire(self, rid: str | None = None) -> str | None:
        if self.busy >= self.capacity:
            raise RuntimeError(f"{self.resource_id} over capacity")

        # 🔽 individual resource mode
        if self.items is not None:
            # 指定 resource
            if rid is not None:
                if rid not in self.items:
                    raise KeyError(f"{rid} not in {self.resource_id}")
                if rid in self.in_use:
                    raise RuntimeError(f"{rid} already in use")

                self.in_use[rid] = self.items[rid]
                self.busy += 1
                if self.observer:
                    self.observer.add(self.resource_id)
                return rid

            # 未指定 → 自動挑一個 idle
            for rid2, meta in self.items.items():
                if rid2 not in self.in_use:
                    self.in_use[rid2] = meta
                    self.busy += 1
                    if self.observer:
                        self.observer.add(self.resource_id)
                    return rid2

            raise RuntimeError(f"No available resource in {self.resource_id}")

        # 🔽 pool-only mode（driver / excavator）
        self.busy += 1
        if self.observer:
            self.observer.add(self.resource_id)
        return None

    def release(self, rid: str | None = None):
        self.busy -= 1

        if rid is not None:
            self.in_use.pop(rid, None)

        if self.available and self.observer:
            self.observer.remove(self.resource_id)

    def available(self) -> int:
        return self.capacity - self.busy


# =====================================================
# Entity
# =====================================================

@dataclass
class Entity:
    eid: int
    created: float
    site: str
    truck_id: str = None
    resource_loc: str = None
    held_resources: Dict[str, ResourcePool] = field(default_factory=dict)
    # dispatch_target: str | None = None



# =====================================================
# Station (base)
# =====================================================

@dataclass
class Station:
    name: str
    engine: Engine
    process_time: Callable[[Entity, random.Random], float]
    resources: Dict[str, ResourcePool] | None = None

    queue: List[Entity] = field(default_factory=list)
    routes: List[Tuple[float, str, float]] = field(default_factory=list)


    busy_time: float = 0.0

    def arrive(self, entity):
        self.queue.append(entity)
        self.try_start()

    def try_start(self):
        while self.queue and self.check_resources():
            e = self.queue.pop(0)
            self.acquire_resources

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
        if not self.resources:
            return

        total_busy = sum(pool.busy for pool in self.resources.values())
        self.busy_time += total_busy * dt


    def utilization(self, sim_time: float) -> float:
        if not self.resources:
            return 0.0

        total_capacity = sum(pool.capacity for pool in self.resources.values())
        return self.busy_time / (total_capacity * sim_time) if total_capacity > 0 else 0.0


    def check_resources(self) -> bool:
        if not self.resources:
            return True

        return all(pool.available() > 0 for pool in self.resources.values())


    def acquire_resources(self):
        if not self.resources:
            return

        for pool in self.resources.values():
            pool.acquire()


    def release_resources(self):
        if not self.resources:
            return

        for pool in self.resources.values():
            pool.release()

# =====================================================
# TransportStation
# =====================================================
class TransportStation(Station):

    def try_start(self):
        while self.queue and self.check_resources():
            e = self.queue.pop(0)
            # print(e.truck_id)

            # because of parallel so acquire truck when make action
            # self.resources["trucks"].acquire(e.truck_id)

            e.held_resources["truck"] = self.resources["trucks"]


            truck_loc = e.resource_loc
            task_site = e.site
            duration = self.process_time(truck_loc, task_site)

            self.engine.schedule(
                self.engine.now + duration,
                "END_Transportation",
                {"eid": e.eid},
                priority=10
            )

# =====================================================
# LoadStation
# =====================================================

@dataclass
class LoadStation(Station):
    excavator_pools: Dict[str, ResourcePool] = None

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
# UnloadStation
# =====================================================

@dataclass
class DumpStation(Station):
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

        # ==================================================
        # Resources (global, shared)
        # ==================================================
        self.trucks = ResourcePool(
            capacity=6,
            resource_id="truck_pool",
            observer=self.observation_manager,
            items={
                "Truck_1": {"location": "Depot_A"},
                "Truck_2": {"location": "Depot_A"},
                "Truck_3": {"location": "Depot_A"},
                "Truck_4": {"location": "Depot_B"},
                "Truck_5": {"location": "Depot_B"},
                "Truck_6": {"location": "Depot_C"},
            }
        )

        self.drivers = ResourcePool(
            capacity=6,
            resource_id="driver_pool",
            observer=self.observation_manager
        )

        self.excavators = {
            "Site_1": ResourcePool(2),
            "Site_2": ResourcePool(1),
            "Site_3": ResourcePool(3),
        }

         # ==================================================

        # ==================================================
        # Task state
        # ==================================================
        self.pending_tasks = [
            Entity(eid=1, created=0.0, site="Site_1"),
            Entity(eid=2, created=0.0, site="Site_1"),
            Entity(eid=3, created=0.0, site="Site_1"),
            Entity(eid=4, created=0.0, site="Site_1"),
            Entity(eid=5, created=0.0, site="Site_1"),
            Entity(eid=6, created=0.0, site="Site_2"),
            # Entity(eid=7, created=0.0, site="Site_2"),
            # Entity(eid=8, created=0.0, site="Site_2"),
            # Entity(eid=9, created=0.0, site="Site_3"),
            # Entity(eid=10, created=0.0, site="Site_3"),
        ]

        self.active_tasks = {}
        self.complete_tasks = []


        # ==================================================
        # Stations (physical operations)
        # ==================================================
        self.Transportation = TransportStation(
            name="Transportation",
            engine=engine,
            resources={"trucks": self.trucks},
            process_time=transport_time,  # <-- dynamic
            routes=[
                (0.9, "ARRIVE_Load", 0.0),
                (0.1, "ARRIVE_Fuel", 0.0),
            ],
        )

        self.Load = LoadStation(
            name="Load",
            engine=engine,
            resources={},
            excavator_pools=self.excavators,
            process_time=const_time(1.0),
            routes=[(1.0, "ARRIVE_Haul", 0.0)],
        )

        self.Haul = Station(
            name="Haul",
            engine=engine,
            resources={},
            process_time=const_time(3.0),
            routes=[(1.0, "ARRIVE_Dump", 0.0)],
        )

        self.Dump = DumpStation(
            name="Dump",
            engine=engine,
            resources={},
            process_time=const_time(1.0),
            routes=[(1.0, "ARRIVE_Transportation", 0.0)],
        )

        self.Fuel = Station(
            name="Fuel",
            engine=engine,
            resources={},
            process_time=const_time(2.0),
            routes=[(1.0, "ARRIVE_Transportation", 0.0)],
        )

        self.stations = [
            self.Transportation,
            self.Load,
            self.Haul,
            self.Dump,
            self.Fuel,
        ]

        # ==================================================
        # Event registration (same pattern as before)
        # ==================================================
        engine.register("ARRIVE_Transportation",
                        lambda p: self.Transportation.arrive(self.active_tasks[p["eid"]]))
        engine.register("END_Transportation",
                        lambda p: self.Transportation.end(self.active_tasks[p["eid"]]))

        engine.register("ARRIVE_Load",
                        lambda p: self.Load.arrive(self.active_tasks[p["eid"]]))
        engine.register("END_Load",
                        lambda p: self.Load.end(self.active_tasks[p["eid"]]))

        engine.register("ARRIVE_Haul",
                        lambda p: self.Haul.arrive(self.active_tasks[p["eid"]]))
        engine.register("END_Haul",
                        lambda p: self.Haul.end(self.active_tasks[p["eid"]]))

        engine.register("ARRIVE_Dump",
                        lambda p: self.Dump.arrive(self.active_tasks[p["eid"]]))
        engine.register("END_Dump",
                        lambda p: self._handle_end_dump(self.active_tasks[p["eid"]]))

        engine.register("ARRIVE_Fuel",
                        lambda p: self.Fuel.arrive(self.active_tasks[p["eid"]]))
        engine.register("END_Fuel",
                        lambda p: self.Fuel.end(self.active_tasks[p["eid"]]))

        engine.on_advance_time = self.on_advance_time

    def _handle_end_dump(self, e):
        print("handle_end_dump:", e)


        # todo modify it in the future
        if e.truck_id is not None:
            self.trucks.items[e.truck_id]["location"] = e.resource_loc

        # 1. 讓 Station 做該做的事
        self.Dump.end(e)

        # 2. World 管理任務生命週期
        self.active_tasks.pop(e.eid)
        self.complete_tasks.append(e)

    def on_advance_time(self, dt: float):
        for st in self.stations:
            st.on_advance_time(dt)

    def report_utilization(self, sim_time: float):
        return {st.name: st.utilization(sim_time) for st in self.stations}

    def request_dispatch(self):
        # status
        observation = {
            "time": self.engine.now,
            "truck_locations": {
                rid: meta["location"]
                for rid, meta in self.trucks.items.items()
                if rid not in self.trucks.in_use
            },
            "pending_tasks": [t.eid for t in self.pending_tasks],
        }
        # reward function R(S)

        print("observation", observation)

        # action
        action = self.agent.decide(observation)
        if action:
            self.apply_action(action)

    def get_task(self, task_id) -> Entity:
        for i, task in enumerate(self.pending_tasks):
            if task.eid == task_id:
                return self.pending_tasks.pop(i)

    def apply_action(self, action):
        truck_id = action["truck_id"]
        self.trucks.acquire(truck_id)

        task_id = action["task_id"]

        task = self.get_task(task_id)
        # task.dispatch_target = action["dispatch_target"]

        task.truck_id = truck_id
        current_location = self.trucks.items[truck_id]["location"]
        task.resource_loc = current_location


        # 現在才正式讓任務進系統
        self.active_tasks[task.eid] = task

        self.engine.schedule(
            self.engine.now,
            "ARRIVE_Transportation",
            {"eid": task.eid, "truck_loc": current_location, "entity": task}
        )

    def start(self):
        self.request_dispatch()





if __name__ == "__main__":
    eng = Engine(seed=7)

    agent = RandomDispatchAgent()
    world = World(eng, agent)
    world.start()

    horizon = 50
    eng.run(until=horizon)

    print("=====================")
    print(len(world.complete_tasks))
    
    # print("Utilization:")
    # for k, v in world.report_utilization(horizon).items():
    #     print(f"  Station {k}: {v:.2%}")
