from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from os import cpu_count
from time import perf_counter
from typing import TYPE_CHECKING

import simpy
from simpy.core import EmptySchedule

from SimPyTest.scenario_types import ScenarioConfig
from SimPyTest.engine import ReplayDecision, SimPySimulationEngine
from SimPyTest.simulation import Disaster, Resource

if TYPE_CHECKING:
    from SimPyTest.policies import Policy


@dataclass(slots=True)
class TournamentConfig:
    depth: int = 1
    max_decisions: int | None = None


@dataclass(slots=True)
class PolicyResult:
    terminal_outcome: str | None
    time_with_disasters: float
    policy: str | None
    move_id: int | None
    sim_time: float
    decisions_made: int
    disasters_remaining: int
    branch_decision: int | None


@dataclass(slots=True)
class TournamentProfileStats:
    tournament_calls: int = 0
    unanimous_checks: int = 0
    unanimous_time_s: float = 0.0
    materialize_calls: int = 0
    materialize_time_s: float = 0.0
    worker_calls: int = 0
    worker_cache_hits: int = 0
    worker_cache_misses: int = 0
    worker_time_s: float = 0.0
    worker_run_time_s: float = 0.0
    tree_calls: int = 0
    tree_time_s: float = 0.0
    live_leaf_calls: int = 0
    live_leaf_time_s: float = 0.0
    cache_store_entries: int = 0
    cache_prunes: int = 0
    cache_pruned_keys: int = 0


TOURNAMENT_DEPTH: TournamentConfig = TournamentConfig()

ROLLOUT_CACHE: dict[tuple[str, int, int | None, tuple[ReplayDecision, ...]], PolicyResult] = {}
PROFILE_STATS: TournamentProfileStats = TournamentProfileStats()
_rollout_executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None


def _decision_replay_entry(resource: Resource, move_id: int) -> ReplayDecision:
    return (resource.id, resource.resource_type.name, move_id)


def set_tournament_depth(depth: int):
    TOURNAMENT_DEPTH.depth = depth


def set_tournament_max_decisions(max_decisions: int | None):
    TOURNAMENT_DEPTH.max_decisions = max_decisions


def reset_tournament_profile_stats():
    for key, value in asdict(TournamentProfileStats()).items():
        setattr(PROFILE_STATS, key, value)


def get_tournament_profile_stats() -> dict[str, float | int]:
    return asdict(PROFILE_STATS)


def clear_tournament_cache():
    ROLLOUT_CACHE.clear()


def _use_rollout_cache() -> bool:
    return TOURNAMENT_DEPTH.depth > 1


def _get_rollout_executor() -> ProcessPoolExecutor | ThreadPoolExecutor:
    global _rollout_executor
    if _rollout_executor is None:
        available_cpu_count = cpu_count() or 1
        max_workers = max(1, available_cpu_count - 1)
        try:
            _rollout_executor = ProcessPoolExecutor(max_workers=max_workers)
        except PermissionError:
            _rollout_executor = ThreadPoolExecutor(max_workers=max_workers)
    return _rollout_executor


def _shutdown_rollout_executor() -> None:
    global _rollout_executor
    if _rollout_executor is not None:
        _rollout_executor.shutdown()
        _rollout_executor = None


atexit.register(_shutdown_rollout_executor)


def prune_tournament_cache(actual_history: list[ReplayDecision]):
    """Wipes parallel timelines"""
    if not actual_history:
        ROLLOUT_CACHE.clear()
        return

    actual_tuple = tuple(actual_history)
    actual_len = len(actual_tuple)

    keys_to_delete: list[tuple[str, int, int | None, tuple[ReplayDecision, ...]]] = []
    for key in ROLLOUT_CACHE.keys():
        _, _, _, cached_hist = key
        if len(cached_hist) < actual_len or cached_hist[:actual_len] != actual_tuple:
            keys_to_delete.append(key)

    for k in keys_to_delete:
        del ROLLOUT_CACHE[k]
    PROFILE_STATS.cache_prunes += 1
    PROFILE_STATS.cache_pruned_keys += len(keys_to_delete)


def _lookup_rollout_cache(policy_name: str, seed: int, max_decisions: int | None, history: list[ReplayDecision]) -> PolicyResult | None:
    return ROLLOUT_CACHE.get((policy_name, seed, max_decisions, tuple(history)))


def _store_rollout_cache_entries(
    *,
    policy_name: str,
    seed: int,
    max_decisions: int | None,
    history: list[ReplayDecision],
    result: PolicyResult,
) -> None:
    current_hist = tuple(history)
    ROLLOUT_CACHE[(policy_name, seed, max_decisions, current_hist)] = replace(result)
    PROFILE_STATS.cache_store_entries += 1


def _score_policy_result(result: PolicyResult) -> float:
    if result.terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS:
        return float(result.time_with_disasters)

    failure_penalty = float(SimPySimulationEngine.MAX_SIM_TIME * 10)
    remaining_penalty = float(result.disasters_remaining) * float(SimPySimulationEngine.MAX_SIM_TIME)
    return failure_penalty + remaining_penalty + float(result.time_with_disasters)


def _evaluate_policy_worker(
    policy: Policy,
    seed: int,
    history: list[ReplayDecision],
    max_decisions: int | None,
    scenario_config: ScenarioConfig,
) -> PolicyResult:
    PROFILE_STATS.worker_calls += 1
    worker_t0 = perf_counter()
    if _use_rollout_cache():
        cached = _lookup_rollout_cache(policy.name, seed, max_decisions, history)
        if cached is not None:
            PROFILE_STATS.worker_cache_hits += 1
            PROFILE_STATS.worker_time_s += perf_counter() - worker_t0
            return cached
        PROFILE_STATS.worker_cache_misses += 1

    # 2. If not, simulate the timeline to completion
    sim_fork = SimPySimulationEngine(policy=policy, seed=seed, scenario_config=scenario_config, track_metrics=False)
    sim_fork.initialize_world()
    sim_fork.replay_buffer = deque(history)
    run_t0 = perf_counter()
    sim_fork.run(max_decisions=max_decisions)
    PROFILE_STATS.worker_run_time_s += perf_counter() - run_t0

    result = PolicyResult(
        terminal_outcome=sim_fork.last_terminal_outcome,
        time_with_disasters=float(sim_fork.time_with_disasters),
        policy=policy.name,
        move_id=sim_fork.branch_decision,
        sim_time=sim_fork.env.now,
        decisions_made=sim_fork.decisions_made,
        disasters_remaining=len(sim_fork.disaster_store.items),
        branch_decision=sim_fork.branch_decision,
    )

    if _use_rollout_cache():
        _store_rollout_cache_entries(
            policy_name=policy.name,
            seed=seed,
            max_decisions=max_decisions,
            history=history,
            result=result,
        )
    PROFILE_STATS.worker_time_s += perf_counter() - worker_t0
    return result


def _run_policy_worker_task(
    policy_name: str,
    seed: int,
    history: list[ReplayDecision],
    max_decisions: int | None,
    scenario_config: ScenarioConfig,
) -> tuple[PolicyResult, float]:
    from .policies import POLICIES

    policy = next(policy for policy in POLICIES if policy.name == policy_name)
    sim_fork = SimPySimulationEngine(policy=policy, seed=seed, scenario_config=scenario_config, track_metrics=False)
    sim_fork.initialize_world()
    sim_fork.replay_buffer = deque(history)
    run_t0 = perf_counter()
    sim_fork.run(max_decisions=max_decisions)
    run_time = perf_counter() - run_t0

    result = PolicyResult(
        terminal_outcome=sim_fork.last_terminal_outcome,
        time_with_disasters=float(sim_fork.time_with_disasters),
        policy=policy.name,
        move_id=sim_fork.branch_decision,
        sim_time=sim_fork.env.now,
        decisions_made=sim_fork.decisions_made,
        disasters_remaining=len(sim_fork.disaster_store.items),
        branch_decision=sim_fork.branch_decision,
    )
    return result, run_time


def _build_terminal_result(
    engine: SimPySimulationEngine,
    *,
    move_id: int | None = None,
    policy_name: str | None = None,
) -> PolicyResult:
    return PolicyResult(
        terminal_outcome=engine.last_terminal_outcome,
        time_with_disasters=float(engine.time_with_disasters),
        policy=policy_name,
        move_id=move_id,
        # partial_history=tuple(history),
        # decision_signature_hash=None,
        sim_time=engine.env.now,
        decisions_made=engine.decisions_made,
        disasters_remaining=len(engine.disaster_store.items),
        branch_decision=move_id,
    )


def _materialize_decision_state(seed: int, history: list[ReplayDecision], scenario_config: ScenarioConfig) -> SimPySimulationEngine:
    from .policies import Policy

    PROFILE_STATS.materialize_calls += 1
    t0 = perf_counter()
    sim_fork = SimPySimulationEngine(
        policy=Policy("_tournament_noop", lambda resource, disasters, env: None),
        seed=seed,
        scenario_config=scenario_config,
        track_metrics=False,
    )
    sim_fork.initialize_world()
    sim_fork.replay_buffer = deque(history)
    sim_fork.stop_before_policy_decision = True
    sim_fork.disasters_process = sim_fork.env.process(sim_fork.add_disasters())
    sim_fork._main_loop_process = sim_fork.env.process(sim_fork.loop())
    sim_fork._initialize_runtime_tracking()
    while sim_fork.pending_decision_resource is None and sim_fork.last_terminal_outcome is None:
        try:
            sim_fork.advance_to_next_event()
        except EmptySchedule:
            sim_fork.last_terminal_outcome = sim_fork.infer_terminal_outcome(schedule_exhausted=True)
            break
    sim_fork.stop_before_policy_decision = False
    PROFILE_STATS.materialize_time_s += perf_counter() - t0
    return sim_fork


def _enumerate_policy_moves(
    resource: Resource,
    disasters: list[Disaster],
    env: simpy.Environment,
    all_policies: list[Policy],
) -> dict[int, str]:
    moves: dict[int, str] = {}
    for policy in all_policies:
        choice = policy.func(resource, disasters, env)
        if choice is None:
            continue
        moves.setdefault(choice.id, policy.name)
    return moves


def _find_unanimous_policy_move(
    resource: Resource,
    disasters: list[Disaster],
    env: simpy.Environment,
    all_policies: list[Policy],
) -> tuple[int, list[str]] | None:
    selected_move: int | None = None
    agreeing_policies: list[str] = []

    for policy in all_policies:
        choice = policy.func(resource, disasters, env)
        if choice is None:
            return None
        if selected_move is None:
            selected_move = choice.id
        elif choice.id != selected_move:
            return None
        agreeing_policies.append(policy.name)

    if selected_move is None:
        return None
    return selected_move, agreeing_policies


def _find_disaster_by_id(disasters: list[Disaster], move_id: int | None) -> Disaster | None:
    if move_id is None:
        return None
    for disaster in disasters:
        if disaster.id == move_id:
            return disaster
    return None


# def _decision_signature_hash(
#     resource: Resource,
#     disasters: list[Disaster],
# ) -> int:
#     return hash(
#         DecisionSignature(
#             resource_id=resource.id,
#             location_x=round(float(resource.location[0]), 2),
#             location_y=round(float(resource.location[1]), 2),
#             disasters=tuple(
#                 sorted(
#                     DisasterSignature(
#                         percent_remaining=round(float(disaster.percent_remaining()), 2),
#                         truck_count=len(disaster.roster.get(ResourceType.TRUCK, [])),
#                         excavator_count=len(disaster.roster.get(ResourceType.EXCAVATOR, [])),
#                     )
#                     for disaster in disasters
#                 )
#             ),
#         )
#     )


def _evaluate_policy_tree_impl(
    seed: int,
    history: list[ReplayDecision],
    depth: int,
    scenario_config: ScenarioConfig,
    all_policies: list[Policy],
    ancestry_has_more: tuple[bool, ...],
    # current_signature_hash: int | None = None,
) -> PolicyResult:
    PROFILE_STATS.tree_calls += 1
    tree_t0 = perf_counter()
    if depth <= 1:
        print("l", end="", flush=True)
        decision_state = _materialize_decision_state(seed, history, scenario_config)
        if decision_state.pending_decision_resource is None:
            result = _build_terminal_result(decision_state)
            PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
            return result

        moves = _enumerate_policy_moves(
            decision_state.pending_decision_resource,
            list(decision_state.disaster_store.items),
            decision_state.env,
            all_policies,
        )
        if not moves:
            result = _build_terminal_result(decision_state)
            PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
            return result

        best_final_result = None
        best_final_score = float("inf")
        move_items = list(moves.items())

        for move_id, policy_name in move_items:
            extended_history = [
                *history,
                _decision_replay_entry(decision_state.pending_decision_resource, move_id),
            ]
            best_move_result = None
            best_move_score = float("inf")

            for policy in all_policies:
                final_result = _evaluate_policy_worker(
                    policy=policy,
                    seed=seed,
                    history=extended_history,
                    max_decisions=TOURNAMENT_DEPTH.max_decisions,
                    scenario_config=scenario_config,
                )
                score = _score_policy_result(final_result)
                if score < best_move_score:
                    best_move_score = score
                    best_move_result = final_result

            if best_move_result is None:
                continue

            best_move_result.move_id = move_id
            best_move_result.branch_decision = move_id
            best_move_result.policy = policy_name

            if best_move_score < best_final_score:
                best_final_score = best_move_score
                best_final_result = best_move_result

        if best_final_result is None:
            raise RuntimeError("Tournament leaf produced no policy results.")
        PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
        return best_final_result

    decision_state = _materialize_decision_state(seed, history, scenario_config)
    if decision_state.pending_decision_resource is None:
        terminal_result = _build_terminal_result(decision_state)
        PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
        return terminal_result

    moves = _enumerate_policy_moves(
        decision_state.pending_decision_resource,
        list(decision_state.disaster_store.items),
        decision_state.env,
        all_policies,
    )
    if not moves:
        terminal_result = _build_terminal_result(decision_state)
        PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
        return terminal_result

    # parent_sig_hash = _decision_signature_hash(decision_state.pending_decision_resource, list(decision_state.disaster_store.items))

    best_final_result = None
    best_final_score = float("inf")
    move_items = list(moves.items())
    total_moves = len(move_items)
    for index, (move_id, policy_name) in enumerate(move_items):
        extended_history = [
            *history,
            _decision_replay_entry(decision_state.pending_decision_resource, move_id),
        ]
        final_result = _evaluate_policy_tree_impl(
            seed=seed,
            history=extended_history,
            depth=depth - 1,
            scenario_config=scenario_config,
            all_policies=all_policies,
            ancestry_has_more=(*ancestry_has_more, total_moves - index > 1),
            # current_signature_hash=parent_sig_hash,
        )
        final_result = final_result
        final_result.move_id = move_id
        final_result.branch_decision = move_id
        final_result.policy = policy_name
        # final_result.partial_history = tuple(extended_history)
        score = _score_policy_result(final_result)
        if score < best_final_score:
            best_final_score = score
            best_final_result = final_result

    if best_final_result is None:
        result = _build_terminal_result(decision_state)
        PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
        return result
    PROFILE_STATS.tree_time_s += perf_counter() - tree_t0
    return best_final_result


time_policy_chosen: dict[str, int] = {}


def _evaluate_live_decision_leaf(
    resource: Resource,
    disasters: list[Disaster],
    env: simpy.Environment,
    *,
    seed: int,
    history: list[ReplayDecision],
    scenario_config: ScenarioConfig,
    all_policies: list[Policy],
) -> PolicyResult:
    PROFILE_STATS.live_leaf_calls += 1
    leaf_t0 = perf_counter()
    moves = _enumerate_policy_moves(resource, disasters, env, all_policies)
    if not moves:
        result = PolicyResult(
            terminal_outcome=SimPySimulationEngine.TERMINAL_FAIL_INVALID_STATE,
            time_with_disasters=float(resource.engine.time_with_disasters),
            policy=None,
            move_id=None,
            sim_time=env.now,
            decisions_made=resource.engine.decisions_made,
            disasters_remaining=len(resource.engine.disaster_store.items),
            branch_decision=None,
        )
        PROFILE_STATS.live_leaf_time_s += perf_counter() - leaf_t0
        return result

    best_final_result = None
    best_final_score = float("inf")
    policy_names = [policy.name for policy in all_policies]
    executor = _get_rollout_executor()
    move_results: dict[int, tuple[str, list[PolicyResult]]] = {
        move_id: (policy_name, []) for move_id, policy_name in moves.items()
    }
    future_to_move_id: dict[Future[tuple[PolicyResult, float]], int] = {}

    for move_id in moves:
        extended_history = [*history, _decision_replay_entry(resource, move_id)]
        for policy_name in policy_names:
            PROFILE_STATS.worker_calls += 1
            future = executor.submit(
                _run_policy_worker_task,
                policy_name,
                seed,
                extended_history,
                TOURNAMENT_DEPTH.max_decisions,
                scenario_config,
            )
            future_to_move_id[future] = move_id

    for future in as_completed(future_to_move_id):
        move_id = future_to_move_id[future]
        final_result, worker_run_time = future.result()
        PROFILE_STATS.worker_time_s += worker_run_time
        PROFILE_STATS.worker_run_time_s += worker_run_time
        _, results = move_results[move_id]
        results.append(final_result)

    for move_id, (policy_name, results) in move_results.items():
        best_move_result = None
        best_move_score = float("inf")

        for final_result in results:
            score = _score_policy_result(final_result)
            if score < best_move_score:
                best_move_score = score
                best_move_result = final_result

        if best_move_result is None:
            continue

        best_move_result.move_id = move_id
        best_move_result.branch_decision = move_id
        best_move_result.policy = policy_name
        if best_move_score < best_final_score:
            best_final_score = best_move_score
            best_final_result = best_move_result

    if best_final_result is None:
        raise RuntimeError("Tournament leaf produced no policy results.")
    PROFILE_STATS.live_leaf_time_s += perf_counter() - leaf_t0
    return best_final_result


def run_tournament_policy(
    resource: Resource,
    disasters: list[Disaster],
    env: simpy.Environment,
    all_policies: list[Policy],
) -> Disaster | None:
    PROFILE_STATS.tournament_calls += 1
    depth = TOURNAMENT_DEPTH.depth

    master_engine = resource.engine
    current_history = list(master_engine.decision_trace)
    if _use_rollout_cache():
        prune_tournament_cache(current_history)
    master_seed = master_engine.seed

    fork_scenario_config: ScenarioConfig = master_engine.scenario_config

    if _use_rollout_cache() and len(current_history) == 0:
        clear_tournament_cache()

    PROFILE_STATS.unanimous_checks += 1
    unanimous_t0 = perf_counter()
    unanimous_move = _find_unanimous_policy_move(resource, disasters, env, all_policies)
    PROFILE_STATS.unanimous_time_s += perf_counter() - unanimous_t0
    if unanimous_move is not None:
        print("u", end="", flush=True)
        move_id, _ = unanimous_move
        master_engine.tournament_decisions.append((env.now, "unanimous"))
        return _find_disaster_by_id(disasters, move_id)
    print("T", end="", flush=True)
    if depth <= 1:
        result = _evaluate_live_decision_leaf(
            resource,
            disasters,
            env,
            seed=master_seed,
            history=current_history,
            scenario_config=fork_scenario_config,
            all_policies=all_policies,
        )
    else:
        result = _evaluate_policy_tree_impl(
            master_seed,
            current_history,
            depth,
            fork_scenario_config,
            all_policies,
            (),
            # _decision_signature_hash(resource, list(resource.engine.disaster_store.items)),
        )

    if result.policy is None:
        raise RuntimeError("Tournament produced no policy results.")
    if result.move_id is None:
        raise RuntimeError(f"Tournament produced no valid move (terminal={result.terminal_outcome}, policy={result.policy})")

    master_engine.tournament_decisions.append((env.now, result.policy))

    times = time_policy_chosen.get(result.policy, 0)
    time_policy_chosen[result.policy] = times + 1

    print(result.policy, end="", flush=True)
    print(times, end="", flush=True)
    print(". ", end="", flush=True)
    return _find_disaster_by_id(disasters, result.move_id)
