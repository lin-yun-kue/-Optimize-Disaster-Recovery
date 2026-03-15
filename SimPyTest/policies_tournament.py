from __future__ import annotations

import multiprocessing
import pickle
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import simpy

from SimPyTest.scenario_types import ScenarioConfig
from .evaluation import build_simulation_summary
from .engine import SimPySimulationEngine
from .simulation import Disaster, Resource

if TYPE_CHECKING:
    from .policies import Policy


@dataclass
class TournamentConfig:
    depth: int = 1


@dataclass(frozen=True)
class _StaticPolicy:
    name: str
    func: Any


@dataclass
class TournamentTrace:
    lines: list[str]
    leaf_rollouts: int
    materializations: int
    cache_hits: int


TOURNAMENT_DEPTH: TournamentConfig = TournamentConfig()
DISABLE_TOURNAMENT_MULTIPROCESSING = True
ROLLOUT_CACHE: dict[tuple[str, int, tuple[int, ...], int | None], dict[str, Any]] = {}


def set_tournament_depth(depth: int):
    TOURNAMENT_DEPTH.depth = depth


def clear_tournament_cache():
    ROLLOUT_CACHE.clear()


def _score_policy_result(result: dict[str, Any]) -> float:
    if result["success"]:
        return float(result["time"])
    return 100000.0 + float(result["disasters_remaining"]) * 100.0 + float(result["time"])


def _evaluate_policy_worker(
    policy: Policy,
    seed: int,
    history: list[int],
    max_decisions: int | None,
    scenario_config: ScenarioConfig,
) -> dict[str, Any]:
    cache_key = (policy.name, seed, tuple(history), max_decisions)
    cached = ROLLOUT_CACHE.get(cache_key)
    if cached is not None:
        result = dict(cached)
        result["cache_hit"] = True
        return result

    sim_fork = SimPySimulationEngine(policy=policy, seed=seed, scenario_config=scenario_config)
    sim_fork.initialize_world()
    sim_fork.replay_buffer = list(history)
    sim_fork.run(max_decisions=max_decisions)
    summary = build_simulation_summary(sim_fork)

    result = {
        "success": len(sim_fork.disaster_store.items) == 0,
        "time": float(summary.non_idle_time),
        "policy": policy.name,
        "move_id": sim_fork.branch_decision,
        "partial_history": list(history) + list(sim_fork.decision_log),
        "sim_time": sim_fork.env.now,
        "decisions_made": sim_fork.decisions_made,
        "disasters_remaining": len(sim_fork.disaster_store.items),
        "branch_decision": sim_fork.branch_decision,
        "cache_hit": False,
    }
    cached_result = dict(result)
    del cached_result["cache_hit"]
    ROLLOUT_CACHE[cache_key] = cached_result
    return result


def _build_terminal_result(
    history: list[int],
    engine: SimPySimulationEngine,
    *,
    move_id: int | None = None,
    policy_name: str | None = None,
) -> dict[str, Any]:
    return {
        "success": len(engine.disaster_store.items) == 0,
        "time": float(engine.non_idle_time),
        "policy": policy_name,
        "move_id": move_id,
        "partial_history": list(history),
        "sim_time": engine.env.now,
        "decisions_made": engine.decisions_made,
        "disasters_remaining": len(engine.disaster_store.items),
        "branch_decision": move_id,
    }


def _materialize_decision_state(seed: int, history: list[int], scenario_config: ScenarioConfig) -> SimPySimulationEngine:
    sim_fork = SimPySimulationEngine(policy=_NOOP_POLICY, seed=seed, scenario_config=scenario_config)
    sim_fork.initialize_world()
    sim_fork.replay_buffer = list(history)
    sim_fork.stop_before_policy_decision = True
    sim_fork.run(max_decisions=0)
    sim_fork.stop_before_policy_decision = False
    return sim_fork


def _format_history(history: list[int]) -> str:
    if not history:
        return "replay_len=0 tail=[]"
    tail = ", ".join(str(move_id) for move_id in history[-5:])
    return f"replay_len={len(history)} tail=[{tail}]"


def _format_tree_prefix(ancestry_has_more: tuple[bool, ...], is_last: bool) -> str:
    if not ancestry_has_more:
        return ""
    branches = ["|   " if has_more else "    " for has_more in ancestry_has_more[:-1]]
    branches.append("`-- " if is_last else "|-- ")
    return "".join(branches)


def _append_trace_line(
    trace: TournamentTrace,
    ancestry_has_more: tuple[bool, ...],
    is_last: bool,
    message: str,
) -> None:
    trace.lines.append(f"{_format_tree_prefix(ancestry_has_more, is_last)}{message}")


def _format_result_metrics(result: dict[str, Any]) -> str:
    outcome = "ok" if result["success"] else "fail"
    cache = " hit" if result.get("cache_hit") else ""
    return (
        f"score={_score_policy_result(result):.2f} time={float(result['time']):.2f} "
        f"sim_t={float(result['sim_time']):.2f} remaining={result['disasters_remaining']} "
        f"decisions={result['decisions_made']} {outcome}{cache}"
    )


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


def _evaluate_policy_tree_impl(
    seed: int,
    history: list[int],
    depth: int,
    scenario_config: ScenarioConfig,
    all_policies: list[Policy],
    trace: TournamentTrace | None,
    ancestry_has_more: tuple[bool, ...],
) -> dict[str, Any]:
    if depth <= 1:
        best_final_result = None
        best_final_score = float("inf")
        total_policies = len(all_policies)
        for index, policy in enumerate(all_policies):
            final_result = _evaluate_policy_worker(policy, seed, history, None, scenario_config)
            score = _score_policy_result(final_result)
            if trace is not None:
                trace.leaf_rollouts += 1
                if final_result.get("cache_hit"):
                    trace.cache_hits += 1
                _append_trace_line(
                    trace,
                    (*ancestry_has_more, total_policies - index > 1),
                    index == total_policies - 1,
                    (f"rollout policy={policy.name} history={_format_history(history)} " f"{_format_result_metrics(final_result)}"),
                )
            if score < best_final_score:
                best_final_score = score
                best_final_result = final_result
        if best_final_result is None:
            raise RuntimeError("Tournament leaf produced no policy results.")
        return best_final_result

    decision_state = _materialize_decision_state(seed, history, scenario_config)
    if trace is not None:
        trace.materializations += 1
    if decision_state.pending_decision_resource is None:
        terminal_result = _build_terminal_result(history, decision_state)
        if trace is not None:
            _append_trace_line(
                trace,
                ancestry_has_more,
                True,
                (f"terminal history={_format_history(history)} sim_t={float(decision_state.env.now):.2f} " f"remaining={terminal_result['disasters_remaining']}"),
            )
        return terminal_result

    moves = _enumerate_policy_moves(
        decision_state.pending_decision_resource,
        list(decision_state.disaster_store.items),
        decision_state.env,
        all_policies,
    )
    if trace is not None:
        move_summary = ", ".join(f"{move_id}:{policy_name}" for move_id, policy_name in moves.items()) or "none"
        _append_trace_line(
            trace,
            ancestry_has_more,
            False,
            (f"decision depth={depth} sim_t={float(decision_state.env.now):.2f} " f"history={_format_history(history)} options=[{move_summary}]"),
        )
    if not moves:
        terminal_result = _build_terminal_result(history, decision_state)
        if trace is not None:
            _append_trace_line(
                trace,
                ancestry_has_more,
                True,
                (f"dead-end history={_format_history(history)} sim_t={float(decision_state.env.now):.2f} " f"remaining={terminal_result['disasters_remaining']}"),
            )
        return terminal_result

    best_final_result = None
    best_final_score = float("inf")
    move_items = list(moves.items())
    total_moves = len(move_items)
    for index, (move_id, policy_name) in enumerate(move_items):
        extended_history = [*history, move_id]
        final_result = _evaluate_policy_tree_impl(
            seed,
            extended_history,
            depth - 1,
            scenario_config,
            all_policies,
            trace,
            (*ancestry_has_more, total_moves - index > 1),
        )
        final_result = dict(final_result)
        final_result["move_id"] = move_id
        final_result["branch_decision"] = move_id
        final_result["policy"] = policy_name
        final_result["partial_history"] = extended_history
        score = _score_policy_result(final_result)
        if trace is not None:
            _append_trace_line(
                trace,
                (*ancestry_has_more, total_moves - index > 1),
                index == total_moves - 1,
                (f"branch move={move_id} via={policy_name} history={_format_history(extended_history)} " f"{_format_result_metrics(final_result)}"),
            )
        if score < best_final_score:
            best_final_score = score
            best_final_result = final_result

    if best_final_result is None:
        return _build_terminal_result(history, decision_state)
    return best_final_result


def _evaluate_policy_tree(
    seed: int,
    history: list[int],
    depth: int,
    scenario_config: ScenarioConfig,
    all_policies: list[Policy],
) -> dict[str, Any]:
    return _evaluate_policy_tree_impl(seed, history, depth, scenario_config, all_policies, None, ())


def run_tournament_policy(
    resource: Resource,
    disasters: list[Disaster],
    env: simpy.Environment,
    all_policies: list[Policy],
) -> Disaster | None:
    depth = TOURNAMENT_DEPTH.depth

    master_engine = resource.engine
    current_history = list(master_engine.decision_log)
    master_seed = master_engine.seed

    fork_scenario_config: ScenarioConfig = master_engine.scenario_config
    use_multiprocessing = not DISABLE_TOURNAMENT_MULTIPROCESSING
    try:
        pickle.dumps(master_engine.scenario_config)
    except Exception:
        # Scenario config contains non-picklable GIS internals; keep local mode.
        use_multiprocessing = False

    t0: float = time.perf_counter()
    trace = TournamentTrace(lines=[], leaf_rollouts=0, materializations=0, cache_hits=0)

    if len(current_history) == 0:
        clear_tournament_cache()

    unanimous_move = _find_unanimous_policy_move(resource, disasters, env, all_policies)
    if unanimous_move is not None:
        move_id, agreeing_policies = unanimous_move
        master_engine.tournament_decisions.append((env.now, "unanimous"))
        progress_marker = "1" if len(disasters) <= 1 else "T"
        print(progress_marker, end="", flush=True)
        # print(f"[tournament] decision sim_t={float(env.now):.2f} depth={depth} history={_format_history(current_history)}")
        # print(f"[tournament] unanimous move={move_id} policies={','.join(agreeing_policies)} " f"rollouts_skipped={len(all_policies)}")
        target = next((d for d in disasters if d.id == move_id), None)
        return target

    if depth <= 1:
        tasks = [(p, master_seed, current_history, None, fork_scenario_config) for p in all_policies]
        if use_multiprocessing:
            with multiprocessing.Pool(processes=len(all_policies)) as pool:
                results = pool.starmap(_evaluate_policy_worker, tasks)
        else:
            results = [_evaluate_policy_worker(*task) for task in tasks]
        for index, result in enumerate(results):
            trace.leaf_rollouts += 1
            if result.get("cache_hit"):
                trace.cache_hits += 1
            _append_trace_line(
                trace,
                (len(results) - index > 1,),
                index == len(results) - 1,
                (f"rollout policy={result['policy']} history={_format_history(current_history)} " f"{_format_result_metrics(result)}"),
            )
    else:
        results = [
            _evaluate_policy_tree_impl(
                master_seed,
                current_history,
                depth,
                fork_scenario_config,
                all_policies,
                trace,
                (),
            )
        ]

    best_result = None
    best_score = float("inf")
    for result in results:
        score = _score_policy_result(result)
        if score < best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise RuntimeError("Tournament produced no policy results.")

    master_engine.tournament_decisions.append((env.now, best_result["policy"]))

    print("t", end="", flush=True)

    # elapsed = time.perf_counter() - t0
    # successes = sum(1 for r in results if r["success"])
    # print(f"[tournament] decision sim_t={float(env.now):.2f} depth={depth} history={_format_history(current_history)}")
    # for line in trace.lines:
    #     print(f"[tournament] {line}")
    # print(
    #     f"[tournament] selected policy={best_result['policy']} move={best_result['move_id']} "
    #     f"score={best_score:.2f} successes={successes}/{len(results)} wall={elapsed:.2f}s "
    #     f"leaf_rollouts={trace.leaf_rollouts} materializations={trace.materializations} "
    #     f"cache_hits={trace.cache_hits} cache_size={len(ROLLOUT_CACHE)}"
    # )

    targets = [d for d in disasters if d.id == best_result["move_id"]]
    if not targets:
        fallback = next((p for p in all_policies if p.name == best_result["policy"] and not p.name.startswith("tournament")), None)
        if fallback is not None:
            return fallback.func(resource, disasters, env)
        return None
    return targets[0]


_NOOP_POLICY = _StaticPolicy("_tournament_noop", lambda resource, disasters, env: None)
