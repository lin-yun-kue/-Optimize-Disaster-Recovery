"""
constrobe_engine.py

ConStrobe-backed Simulation Engine for Disaster Response.
Precomputes the disaster timeline from the ScenarioConfig, generates a static
JSTRX graph, and uses Python IPC Callbacks to evaluate dynamic Policies and track metrics.
"""

from dataclasses import dataclass
import math
from typing import Callable, List, Dict, Tuple, Any, Optional

from Generator.ProcessManager import ProcessManager
from Generator.ResultsParser import ResultsParser
from Generator.JSTRXGenerator import (
    JSTRXGenerator,
    QueueNode,
    CombiNode,
    ActivityNode,
    ActivityCallbackData,
    AddToQueueAction,
    AssignAction,
    Get,
    RemoveFromQueueAction,
)
from Generator.expressions import Literal, Var

from SimPyTest.scenario_types import ScenarioConfig
from SimPyTest.metrics_tracker import SimulationMetrics
from SimPyTest.evaluation import SimulationSummary
from SimPyTest.simulation import ResourceType
from SimPyTest.calendar import SimulationCalendar, Season
from SimPyTest.engine import SimulationRNG
from SimPyTest.gis_utils import sample_clatsop_local_utm, meters_to_miles, travel_minutes_from_distance

# ============================================================================
# MARK: Proxy Objects for Policy Compatibility
# ============================================================================


class ProxyDisaster:
    """Mock Disaster state to track rosters and dirt outside of ConStrobe."""

    def __init__(
        self, d_id: int, d_type: str, initial_size: float, loc: Tuple[float, float], engine, spawn_time: float
    ):
        self.id = d_id
        self.disaster_type = d_type
        self.initial_size = initial_size
        self.current_size = initial_size
        self.location = loc
        self.engine = engine
        self.spawn_time = spawn_time

        self.active = False
        self.required_resources = (ResourceType.EXCAVATOR, ResourceType.TRUCK)
        self.roster = {ResourceType.TRUCK: set(), ResourceType.EXCAVATOR: set()}
        self.created_time = 0.0

    def percent_remaining(self) -> float:
        return self.current_size / self.initial_size if self.initial_size > 0 else 0.0

    def get_scale(self) -> float:
        return self.current_size / 5000.0  # Dummy scaling max

    @property
    def __class__(self):
        class DuckClass:
            __name__ = "Landslide" if self.disaster_type == "landslide" else "WildfireDebris"

        return DuckClass


@dataclass
class Policy:
    name: str
    func: Callable[[ResourceType, list[ProxyDisaster]], ProxyDisaster | None]


# ============================================================================
# MARK: ConStrobe Engine
# ============================================================================


class ConStrobeSimulationEngine:
    """
    Simulation engine wrapper that drives ConStrobe via the Python JSTRXGenerator.
    """

    MAX_SIM_TIME: float = 600_000.0

    def __init__(self, *, policy: Policy, scenario_config: ScenarioConfig, seed: int = 0, track_metrics: bool = True):
        self.policy = policy
        self.scenario_config = scenario_config
        self.seed = seed
        self.rng = SimulationRNG(seed)
        self.track_metrics = track_metrics

        self.current_sim_time: float = 0.0
        self.metrics = SimulationMetrics()

        self.total_spent: float = 0.0
        self.time_with_disasters: float = 0.0
        self.total_drive_time: float = 0.0
        self.total_resource_hours: float = 0.0
        self.decisions_made: int = 0

        self.precomputed_disasters: List[ProxyDisaster] = []

        # ID Counters
        self._next_resource_id = 1

        self.graph = JSTRXGenerator()
        self.manager = ProcessManager()

        # non gis locations
        self.idle_location = (0.0, 0.0)  # central depot for idle distance calc
        self.dump_location = (100.0, 100.0)  # central dump site

    # ------------------------------------------------------------------------
    # State Helpers
    # ------------------------------------------------------------------------
    def get_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        euclidean_meters = math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])
        return meters_to_miles(euclidean_meters)

    def _get_travel_time(self, loc1: Tuple[float, float], loc2: Tuple[float, float], r_type: ResourceType) -> float:
        dist = self.get_distance(loc1, loc2)
        return travel_minutes_from_distance(dist, r_type.specs["speed"])

    def _sync_time(self, data: ActivityCallbackData):
        """Update python clock from ConStrobe callbacks."""
        self.current_sim_time = data["sim_time"]

    # ------------------------------------------------------------------------
    # Precomputation Phase
    # ------------------------------------------------------------------------
    def _precompute_timeline(self):
        """Re-implements calendar.py logic to statically generate disasters."""
        cal = SimulationCalendar(self.scenario_config.calendar_start_date, self.scenario_config.calendar_duration_years)
        max_sim_minutes = cal.duration_years * 525600
        sim_year_span = math.floor(cal.current_date + cal.duration_years) + 1
        profiles = self.scenario_config.seasonal_spawn

        event_times = []
        for year in range(sim_year_span):
            for season in Season:
                for d_type, profile in profiles.items():
                    count_range = profile.event_count_range_by_season.get(season.name.lower(), (0, 0))
                    count = self.rng.randint(count_range[0], count_range[1])
                    for _ in range(count):
                        base_time = (self.rng.uniform(0.0, 0.25) + season.value) % 1
                        if base_time < cal.current_date:
                            base_time += 1
                        ev_time = (base_time - cal.current_date) * 525600

                        size_range = profile.size_range_by_season.get(season.name.lower(), (1, 1))
                        ev_size = self.rng.randint(size_range[0], size_range[1])

                        event_times.append((ev_time, d_type, ev_size))

        event_times.sort(key=lambda x: x[0])

        d_id = 1
        for ev_time, d_type, d_size in event_times:
            if ev_time > max_sim_minutes:
                continue

            loc = sample_clatsop_local_utm(self.rng)
            proxy = ProxyDisaster(d_id, d_type, float(d_size), loc, self, ev_time)
            self.precomputed_disasters.append(proxy)
            d_id += 1

    # ------------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------------
    def build_network(self):
        self._precompute_timeline()
        truck_count = self.scenario_config.resolve_resource_count(self.rng, ResourceType.TRUCK)
        excav_count = self.scenario_config.resolve_resource_count(self.rng, ResourceType.EXCAVATOR)

        with self.graph:
            # Idle Pools
            idle_trucks = QueueNode(name="IdleTrucks", initialContent=truck_count)
            idle_excavs = QueueNode(name="IdleExcavs", initialContent=excav_count)

            self.graph.moveCursor(400, 100)

            # Single Dump Site Logic
            dump_q = QueueNode(name="DumpQueue", initialContent=0)
            dump_slots = QueueNode(name="DumpSlots", initialContent=3)
            dump_combi = CombiNode(name="DumpCombi", duration=ResourceType.TRUCK.specs["dump_time"])
            dump_q.linkTo(dump_combi)
            dump_slots.linkTo(dump_combi)
            dump_combi.linkTo(dump_slots)
            # self.graph.layout("vertical", dump_q, dump_combi, dump_slots)

            disp_t_available_semaphore = QueueNode(name="DispT_Available", initialContent=0)
            disp_e_available_semaphore = QueueNode(name="DispE_Available", initialContent=0)

            # Disasters Subgraphs
            for i, d in enumerate(self.precomputed_disasters):
                self._build_disaster_subgraph(d, idle_trucks, idle_excavs, dump_q, dump_combi, i)
            # self._build_disaster_subgraph(self.precomputed_disasters[0], idle_trucks, idle_excavs, dump_q, dump_combi)

            self._build_dispatcher(idle_trucks, idle_excavs)

    def _build_disaster_subgraph(
        self,
        d: ProxyDisaster,
        idle_trucks: QueueNode,
        idle_excavs: QueueNode,
        dump_q: QueueNode,
        dump_combi: CombiNode,
        i=0,
    ):
        """Constructs the nodes and logic gates for a single precomputed disaster."""
        self.graph.moveCursor(100, 250 * i + 200)

        disp_t_available_semaphore = self.graph.find_node("DispT_Available", QueueNode)
        disp_e_available_semaphore = self.graph.find_node("DispE_Available", QueueNode)

        # Site Queues
        d_trucks = QueueNode(name=f"D{d.id}_Trucks", initialContent=0)
        d_excavs = QueueNode(name=f"D{d.id}_Excavs", initialContent=0)
        d_dirt = QueueNode(name=f"D{d.id}_Dirt", initialContent=0)

        # Precalculate load chunks (1 Token = 1 Truck Load)
        loads_required = math.ceil(d.initial_size / ResourceType.TRUCK.specs["capacity"])

        self.graph.moveCursor(400, 250 * i + 200)

        # # Spawner Trigger
        spawn_trigger = QueueNode(name=f"D{d.id}_SpwnT", initialContent=1)
        spawn_delay = CombiNode(name=f"D{d.id}_SpwnD", duration=max(0, d.spawn_time))
        spawn_trigger.linkTo(spawn_delay)

        spawn_delay.linkTo(disp_t_available_semaphore)
        spawn_delay.linkTo(disp_e_available_semaphore)

        def make_on_spawn(pd: ProxyDisaster):
            def callback(data: ActivityCallbackData):
                self._sync_time(data)
                pd.active = True
                pd.created_time = data["sim_time"]
                if self.track_metrics:
                    self.metrics.record_disaster_created(pd.id, pd.disaster_type, data["sim_time"])

            return callback

        spawn_delay.onEnd(make_on_spawn(d))
        spawn_delay.onEnd(AddToQueueAction(d_dirt, amount=loads_required))

        # spawn_delay.onEnd(AddToQueueAction(disp_t_available_semaphore, amount=1))
        # spawn_delay.onEnd(AddToQueueAction(disp_e_available_semaphore, amount=1))

        self.graph.moveCursor(100, 250 * i + 260)

        # Work / Load Logic
        load = CombiNode(name=f"D{d.id}_Load", duration=ResourceType.TRUCK.specs["load_time"])
        d_trucks.linkTo(load)
        d_excavs.linkTo(load)
        d_dirt.linkTo(load, drawAmount=1)  # 1 Token = 1 Truck Capacity

        def make_on_load_end(pd: ProxyDisaster):
            def callback(data: ActivityCallbackData):
                self._sync_time(data)
                # Deduct dirt
                pd.current_size = max(0.0, pd.current_size - ResourceType.TRUCK.specs["capacity"])

                # Accrue Costs & Metrics
                load_hrs = ResourceType.TRUCK.specs["load_time"] / 60.0
                t_cost = load_hrs * ResourceType.TRUCK.specs["hourly_operating_cost"]
                e_cost = load_hrs * ResourceType.EXCAVATOR.specs["hourly_operating_cost"]

                self.total_spent += t_cost + e_cost
                self.total_resource_hours += load_hrs * 2

                if self.track_metrics:
                    self.metrics.record_resource_metrics(1, "TRUCK", load_hrs, t_cost)
                    self.metrics.record_resource_metrics(2, "EXCAVATOR", load_hrs, e_cost)

                # Resolution Check
                if pd.current_size <= 0 and pd.active:
                    pd.active = False
                    if self.track_metrics:
                        self.metrics.record_disaster_resolved(pd.id, data["sim_time"], 0.0)

            return callback

        load.onEnd(make_on_load_end(d))
        self.graph.onIf(
            Var("DispT_Available.CurCount").__le__(0.0), RemoveFromQueueAction(disp_t_available_semaphore, 1)
        )

        # Excavator loops back to site
        load.linkTo(d_excavs)

        # Truck travels to Dump Site
        t_to_dump = self._get_travel_time(d.location, self.dump_location, ResourceType.TRUCK)
        drive_to_dump = ActivityNode(name=f"D{d.id}_ToDump", duration=t_to_dump)
        load.linkTo(drive_to_dump)
        drive_to_dump.linkTo(dump_q)

        def make_on_dump_end(pd: ProxyDisaster):
            def callback(data: ActivityCallbackData):
                self._sync_time(data)
                # Remove from roster as it has left the disaster loop
                if self.track_metrics:
                    dump_hrs = ResourceType.TRUCK.specs["dump_time"] / 60.0
                    cost = dump_hrs * ResourceType.TRUCK.specs["hourly_operating_cost"]
                    self.total_spent += cost
                    self.total_resource_hours += dump_hrs
                    self.metrics.record_resource_metrics(1, "TRUCK", dump_hrs, cost)

                # Cleanup Roster
                if pd.roster[ResourceType.TRUCK]:
                    pd.roster[ResourceType.TRUCK].pop()

            return callback

        dump_combi.onEnd(make_on_dump_end(d))

        # Travel from Dump to Idle
        t_dump_to_idle = self._get_travel_time(self.dump_location, self.idle_location, ResourceType.TRUCK)
        drive_to_idle = ActivityNode(name=f"D{d.id}_ToIdl", duration=t_dump_to_idle)
        dump_combi.linkTo(drive_to_idle)
        drive_to_idle.linkTo(idle_trucks)

        # Cleanup Routine: Send Excavators home when resolved
        cleanup_e = CombiNode(name=f"D{d.id}_ClnE", duration=0)
        d_excavs.linkTo(cleanup_e, drawCondition=Get(lambda: 1.0 if not d.active else 0.0).eq(1.0))

        t_e_to_idle = self._get_travel_time(d.location, self.idle_location, ResourceType.EXCAVATOR)
        drive_e_to_idle = ActivityNode(name=f"D{d.id}_EToIdl", duration=t_e_to_idle)
        cleanup_e.linkTo(drive_e_to_idle)
        drive_e_to_idle.linkTo(idle_excavs)

        def make_on_cln_e(pd: ProxyDisaster):
            def callback(data: ActivityCallbackData):
                if pd.roster[ResourceType.EXCAVATOR]:
                    pd.roster[ResourceType.EXCAVATOR].pop()

            return callback

        cleanup_e.onEnd(make_on_cln_e(d))

        # Cleanup Routine: Send stuck Trucks home when resolved
        cleanup_t = CombiNode(name=f"D{d.id}_ClnT", duration=0)
        d_trucks.linkTo(cleanup_t, drawCondition=Get(lambda: 1.0 if not d.active else 0.0).eq(1.0))

        t_t_to_idle = self._get_travel_time(d.location, self.idle_location, ResourceType.TRUCK)
        drive_t_to_idle = ActivityNode(name=f"D{d.id}_TToIdl", duration=t_t_to_idle)
        cleanup_t.linkTo(drive_t_to_idle)
        drive_t_to_idle.linkTo(idle_trucks)

        cleanup_t.onEnd(make_on_cln_e(d))  # Reusing roster pop logic

    def _build_dispatcher(self, idle_trucks: QueueNode, idle_excavs: QueueNode):
        """Builds the dynamic policy routing hubs."""
        self.graph.moveCursor(900, 100)

        disp_t_available_semaphore = self.graph.find_node("DispT_Available", QueueNode)
        disp_t = CombiNode(name="DispT", duration=0)
        disp_t_available_semaphore.linkTo(disp_t)
        disp_t.linkTo(disp_t_available_semaphore)
        idle_trucks.linkTo(disp_t, drawAmount=1)

        self.graph.add_savevalue("T_Target", Literal(0))
        disp_t.onEnd(AssignAction("T_Target", Get(lambda: self._evaluate_policy(ResourceType.TRUCK))))

        disp_e_available_semaphore = self.graph.find_node("DispE_Available", QueueNode)
        disp_e = CombiNode(name="DispE", duration=0)
        disp_e_available_semaphore.linkTo(disp_e)
        disp_e.linkTo(disp_e_available_semaphore)
        idle_excavs.linkTo(disp_e, drawAmount=1)

        self.graph.add_savevalue("E_Target", Literal(0))
        disp_e.onEnd(AssignAction("E_Target", Get(lambda: self._evaluate_policy(ResourceType.EXCAVATOR))))

        # Route logic for each disaster
        for d in self.precomputed_disasters:
            # For Truck
            t_travel = self._get_travel_time(self.idle_location, d.location, ResourceType.TRUCK)
            drv_tq = QueueNode(name=f"DrTQ_{d.id}", initialContent=0)
            drv_t = CombiNode(name=f"DrT_{d.id}", duration=t_travel)
            self.graph.onIf(Var("T_Target").eq(d.id), AddToQueueAction(f"DrTQ_{d.id}"))
            self.graph.onIf(Var("T_Target").eq(d.id), AssignAction("T_Target", Literal(0.0)))
            disTQ = self.graph.find_node(f"D{d.id}_Trucks")
            drv_tq.linkTo(drv_t)
            drv_t.linkTo(disTQ)

            # For Excavator
            e_travel = self._get_travel_time(self.idle_location, d.location, ResourceType.EXCAVATOR)
            drv_eq = QueueNode(name=f"DrEQ_{d.id}", initialContent=0)
            drv_e = CombiNode(name=f"DrE_{d.id}", duration=e_travel)
            self.graph.onIf(Var("E_Target").eq(d.id), AddToQueueAction(f"DrEQ_{d.id}"))
            self.graph.onIf(Var("E_Target").eq(d.id), AssignAction("E_Target", Literal(0.0)))
            disEQ = self.graph.find_node(f"D{d.id}_Excavs")
            drv_eq.linkTo(drv_e)
            drv_e.linkTo(disEQ)

            # # Record First Response metric upon arrival at Site
            # def make_arrival_t(pd: ProxyDisaster):
            #     def callback(data: ActivityCallbackData):
            #         self._sync_time(data)
            #         mock_res = ProxyResource(self._next_resource_id, ResourceType.TRUCK, pd.location, self)
            #         pd.roster[ResourceType.TRUCK].add(mock_res)
            #         self._next_resource_id += 1
            #         if len(pd.roster[ResourceType.TRUCK]) + len(pd.roster[ResourceType.EXCAVATOR]) == 1:
            #             self.metrics.record_first_response(pd.id, data["sim_time"])

            #     return callback

            # drv_t.onEnd(make_arrival_t(d))

            # def make_arrival_e(pd: ProxyDisaster):
            #     def callback(data: ActivityCallbackData):
            #         self._sync_time(data)
            #         mock_res = ProxyResource(self._next_resource_id, ResourceType.EXCAVATOR, pd.location, self)
            #         pd.roster[ResourceType.EXCAVATOR].add(mock_res)
            #         self._next_resource_id += 1
            #         if len(pd.roster[ResourceType.TRUCK]) + len(pd.roster[ResourceType.EXCAVATOR]) == 1:
            #             self.metrics.record_first_response(pd.id, data["sim_time"])

            #     return callback

            # drv_e.onEnd(make_arrival_e(d))

    # ------------------------------------------------------------------------
    # Python Policy Evaluator
    # ------------------------------------------------------------------------
    def _needs_resources(self) -> float:
        return 1.0 if any(d.active and d.percent_remaining() > 0 for d in self.precomputed_disasters) else 0.0

    def _evaluate_policy(self, r_type: ResourceType) -> float:
        """Called by ConStrobe AssignAction via IPC."""
        actionable = [d for d in self.precomputed_disasters if d.active and d.percent_remaining() > 0]
        if not actionable:
            return 0.0  # Send to Park queue

        # Call the actual Python Policy!
        # chosen = self.policy.func(r_type, actionable)
        chosen = actionable[0]

        if chosen is None:
            print(0.0)
            return 0.0

        self.decisions_made += 1
        return float(chosen.id)

    # ------------------------------------------------------------------------
    # Execution & Metric Retrieval
    # ------------------------------------------------------------------------
    def summary(self) -> SimulationSummary:
        """Reconstructs the SimulationSummary from the tracked metrics."""
        summary_dict = self.metrics.get_summary()
        return SimulationSummary(
            terminal_outcome="SUCCESS",
            time_with_disasters=self.time_with_disasters,
            total_drive_time=self.total_drive_time,
            total_operating_cost=self.total_spent,
            total_fuel_cost=0.0,  # not calculable cus of resource properties
            total_spent=self.total_spent,
            total_resource_hours=self.total_resource_hours,
            disasters_created=int(summary_dict.get("total_disasters_created", 0)),
            disasters_resolved=int(summary_dict.get("total_disasters_resolved", 0)),
            resolution_rate=float(summary_dict.get("resolution_rate", 0.0)),
            avg_response_time=float(summary_dict.get("avg_response_time", 0.0)),
            avg_resolution_time=float(summary_dict.get("avg_resolution_time", 0.0)),
            total_weighted_closure_hours=float(summary_dict.get("total_weighted_closure_hours", 0.0)),
        )

    def run(self, max_decisions: Optional[int] = None) -> bool:
        """Compiles the network, boots ConStrobe, and executes the simulation."""
        self.build_network()
        jstrx_path = self.graph.write_jstrx()

        def handle_results(msg: str):
            results = ResultsParser.parse(msg)
            self.current_sim_time = results.sim_time

            # Calculate total time with active disasters by examining created/resolved times
            total_time = 0.0
            for metric in self.metrics.disaster_metrics.values():
                start = metric["start_time"]
                end = metric["end_time"] if metric["end_time"] is not None else results.sim_time
                total_time += end - start
            self.time_with_disasters = total_time

        self.manager.register_callback("MESSAGE", self.graph._post_callback)
        self.manager.register_callback("GET", self.graph._get_callback)
        self.manager.register_callback("RESULTS", handle_results)

        self.manager.load_jstrx(jstrx_path)
        self.manager.reset_model()
        self.manager.set_animate(True)

        self.manager.run_model(blocking=True)

        self.manager.close()
        self.manager.cleanup()

        return True
