from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import heapq
import itertools
import random
import math


# =========================
# Policies (pluggable)
# =========================

class DispatchPolicy(Protocol):
    def select(self, queue: List["Entity"], *, world: "World", station: "Station") -> "Entity":
        ...


class FIFO(DispatchPolicy):
    def select(self, queue: List["Entity"], *, world: "World", station: "Station") -> "Entity":
        return queue[0]


class SPT(DispatchPolicy):
    """
    Shortest Processing Time first: choose entity with smallest expected processing time at this station.
    (Uses world.estimate_process_time())
    """
    def select(self, queue: List["Entity"], *, world: "World", station: "Station") -> "Entity":
        best_i = 0
        best_t = float("inf")
        for i, e in enumerate(queue):
            t = world.estimate_process_time(e, station.step_id)
            if t < best_t:
                best_t = t
                best_i = i
        return queue[best_i]


# =========================
# Model / World
# =========================

@dataclass
class Entity:
    eid: int
    route: List[str]              # list of step_ids
    created_time: float
    step_index: int = 0
    started_times: Dict[str, float] = field(default_factory=dict)
    finished_times: Dict[str, float] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.step_index >= len(self.route)

    @property
    def current_step(self) -> Optional[str]:
        if self.done:
            return None
        return self.route[self.step_index]


@dataclass
class ResourcePool:
    capacity: int
    busy: int = 0

    def available(self) -> int:
        return self.capacity - self.busy

    def acquire(self) -> bool:
        if self.busy < self.capacity:
            self.busy += 1
            return True
        return False

    def release(self) -> None:
        if self.busy <= 0:
            raise RuntimeError("ResourcePool.release() called when busy == 0")
        self.busy -= 1


@dataclass
class Station:
    step_id: str
    resources: ResourcePool
    dispatch_policy: DispatchPolicy
    queue: List[Entity] = field(default_factory=list)


class World:
    """
    A simple flow-shop style world:
    - Entities follow a route of steps
    - Each step maps to a Station with a queue + resource pool
    - On ARRIVAL or MOVE_IN, entity is queued and we try_start()
    - On END_PROCESS, release resource, move entity forward, try_start() again
    """
    # Event priorities (smaller = earlier when time ties)
    PRIO_RESOURCE_AVAILABLE = 10
    PRIO_END_PROCESS = 20
    PRIO_ARRIVAL_MOVEIN = 30

    def __init__(self, engine: Engine):
        self.engine = engine

        # system components
        self.stations: Dict[str, Station] = {}
        self.entities: Dict[int, Entity] = {}

        # process-time models: step_id -> callable(entity, rng) -> duration
        self.process_time: Dict[str, Callable[[Entity, random.Random], float]] = {}

        # metrics
        self.completed: List[Entity] = []
        self.area_wip: float = 0.0
        self.area_queue: Dict[str, float] = {}
        self.area_busy: Dict[str, float] = {}
        self.last_wip: int = 0

        # register event handlers
        engine.register("ARRIVAL", self._on_arrival)
        engine.register("END_PROCESS", self._on_end_process)

        # time-weighted metrics hook
        engine.on_advance_time = self._on_advance_time

    # -------- setup --------

    def add_station(self, step_id: str, capacity: int, policy: DispatchPolicy) -> None:
        self.stations[step_id] = Station(
            step_id=step_id,
            resources=ResourcePool(capacity=capacity),
            dispatch_policy=policy,
        )
        self.area_queue.setdefault(step_id, 0.0)
        self.area_busy.setdefault(step_id, 0.0)

    def set_process_time(self, step_id: str, sampler: Callable[[Entity, random.Random], float]) -> None:
        self.process_time[step_id] = sampler

    def create_entity(self, eid: int, route: List[str], at_time: float) -> Entity:
        e = Entity(eid=eid, route=route, created_time=at_time)
        self.entities[eid] = e
        return e

    # -------- process time helpers --------

    def sample_process_time(self, entity: Entity, step_id: str) -> float:
        if step_id not in self.process_time:
            raise KeyError(f"No process_time defined for step '{step_id}'")
        dur = self.process_time[step_id](entity, self.engine.rng)
        if dur < 0:
            raise ValueError("Process time must be >= 0")
        return dur

    def estimate_process_time(self, entity: Entity, step_id: str) -> float:
        # A simple estimate: call sampler with a deterministic RNG clone or use mean.
        # Here: we just use one sample from a separate RNG seeded by (eid, step_id) for stability.
        seed = hash((entity.eid, step_id, 12345)) & 0xFFFFFFFF
        tmp_rng = random.Random(seed)
        dur = self.process_time[step_id](entity, tmp_rng)
        return max(dur, 0.0)

    # -------- metrics --------

    def current_wip(self) -> int:
        # WIP = entities not yet completed
        return len(self.entities) - len(self.completed)

    def _on_advance_time(self, dt: float) -> None:
        # time-weighted WIP
        wip = self.current_wip()
        self.area_wip += wip * dt

        # time-weighted per-station queue length & utilization
        for sid, st in self.stations.items():
            self.area_queue[sid] += len(st.queue) * dt
            self.area_busy[sid] += st.resources.busy * dt

    # -------- event handlers --------

    def _on_arrival(self, payload: Dict[str, Any]) -> None:
        eid = payload["eid"]
        step_id = payload["step_id"]
        st = self.stations[step_id]
        e = self.entities[eid]

        # enqueue
        st.queue.append(e)
        # attempt to start immediately
        self.try_start(step_id)

    def _on_end_process(self, payload: Dict[str, Any]) -> None:
        eid = payload["eid"]
        step_id = payload["step_id"]

        st = self.stations[step_id]
        e = self.entities[eid]

        # mark finish time and release resource
        e.finished_times[step_id] = self.engine.now
        st.resources.release()

        # advance entity route
        e.step_index += 1

        # try start another job at this station (resource just freed)
        self.try_start(step_id)

        # move entity to next step if exists, else complete
        if e.done:
            self.completed.append(e)
        else:
            next_step = e.current_step
            # immediate move-in (transport time could be added here)
            self.engine.schedule(
                time=self.engine.now,
                etype="ARRIVAL",
                payload={"eid": eid, "step_id": next_step},
                priority=self.PRIO_ARRIVAL_MOVEIN,
            )

    # -------- core station logic --------

    def try_start(self, step_id: str) -> None:
        st = self.stations[step_id]

        while st.queue and st.resources.available() > 0:
            # pick entity by dispatch policy
            chosen = st.dispatch_policy.select(st.queue, world=self, station=st)

            # remove chosen from queue (careful: chosen may not be at index 0)
            idx = st.queue.index(chosen)
            e = st.queue.pop(idx)

            ok = st.resources.acquire()
            if not ok:
                # should not happen because we checked available()
                st.queue.insert(0, e)
                return

            # mark start time
            e.started_times[step_id] = self.engine.now

            # schedule end of processing
            dur = self.sample_process_time(e, step_id)
            self.engine.schedule(
                time=self.engine.now + dur,
                etype="END_PROCESS",
                payload={"eid": e.eid, "step_id": step_id},
                priority=self.PRIO_END_PROCESS,
            )

    # -------- reporting --------

    def report(self, horizon_time: float) -> Dict[str, Any]:
        total_time = max(horizon_time, 1e-9)

        # cycle time stats
        cts = []
        for e in self.completed:
            ct = (max(e.finished_times.values()) - e.created_time) if e.finished_times else 0.0
            cts.append(ct)

        cts_sorted = sorted(cts)
        def percentile(p: float) -> float:
            if not cts_sorted:
                return 0.0
            k = (len(cts_sorted) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return cts_sorted[int(k)]
            return cts_sorted[f] * (c - k) + cts_sorted[c] * (k - f)

        util = {sid: self.area_busy[sid] / (self.stations[sid].resources.capacity * total_time)
                for sid in self.stations}

        avg_q = {sid: self.area_queue[sid] / total_time for sid in self.stations}
        avg_wip = self.area_wip / total_time
        throughput = len(self.completed) / total_time

        return {
            "completed": len(self.completed),
            "throughput_per_time": throughput,
            "avg_wip": avg_wip,
            "avg_queue_len": avg_q,
            "utilization": util,
            "ct_mean": (sum(cts) / len(cts)) if cts else 0.0,
            "ct_p50": percentile(0.50),
            "ct_p90": percentile(0.90),
        }
