from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import simpy

from SimPyTest.scenario_types import ScenarioConfig
from .engine import SimPySimulationEngine
from .simulation import Disaster, Resource

if TYPE_CHECKING:
    from .policies import Policy


@dataclass(slots=True)
class TournamentConfig:
    depth: int = 1


@dataclass(slots=True)
class TournamentTrace:
    lines: list[str]
    leaf_rollouts: int
    materializations: int
    cache_hits: int


@dataclass(slots=True)
class PolicyResult:
    terminal_outcome: str | None
    time_with_disasters: float
    policy: str | None
    move_id: int | None
    partial_history: tuple[int, ...]
    sim_time: float
    decisions_made: int
    disasters_remaining: int
    branch_decision: int | None


TOURNAMENT_DEPTH: TournamentConfig = TournamentConfig()


@dataclass(slots=True)
class RolloutCacheNode:
    result: PolicyResult | None = None
    children: dict[int, "RolloutCacheNode"] = field(default_factory=dict)


ROLLOUT_CACHE: dict[tuple[str, int, int | None], PolicyResult | RolloutCacheNode] = {}


def set_tournament_depth(depth: int):
    TOURNAMENT_DEPTH.depth = depth


def clear_tournament_cache():
    ROLLOUT_CACHE.clear()


def _clone_policy_result(result: PolicyResult) -> PolicyResult:
    return replace(result)


def _lookup_rollout_cache(policy_name: str, seed: int, history: list[int], max_decisions: int | None) -> PolicyResult | None:
    cached = ROLLOUT_CACHE.get((policy_name, seed, max_decisions))
    if cached is None:
        return None
    if max_decisions is not None:
        if isinstance(cached, RolloutCacheNode):
            return None
        return _clone_policy_result(cached)

    node = cached
    if not isinstance(node, RolloutCacheNode):
        return None

    for move_id in history:
        node = node.children.get(move_id)
        if node is None:
            return None

    if node.result is None:
        return None
    return _clone_policy_result(node.result)


def _score_policy_result(result: PolicyResult) -> float:
    if result.terminal_outcome == SimPySimulationEngine.TERMINAL_SUCCESS:
        return float(result.time_with_disasters)

    failure_penalty = float(SimPySimulationEngine.MAX_SIM_TIME * 10)
    remaining_penalty = float(result.disasters_remaining) * float(SimPySimulationEngine.MAX_SIM_TIME)
    return failure_penalty + remaining_penalty + float(result.time_with_disasters)


def _evaluate_policy_worker(
    policy: Policy,
    seed: int,
    history: list[int],
    max_decisions: int | None,
    scenario_config: ScenarioConfig,
) -> PolicyResult:
    cached = _lookup_rollout_cache(policy.name, seed, history, max_decisions)
    if cached is not None:
        return cached

    sim_fork = SimPySimulationEngine(policy=policy, seed=seed, scenario_config=scenario_config, track_metrics=False)
    sim_fork.initialize_world()
    sim_fork.replay_buffer = deque(history)
    sim_fork.run(max_decisions=max_decisions)

    result = _build_terminal_result(list(history) + list(sim_fork.decision_log), sim_fork, move_id=sim_fork.branch_decision, policy_name=policy.name)
    _store_rollout_cache_entries(
        policy_name=policy.name,
        seed=seed,
        history=history,
        decision_log=sim_fork.decision_log,
        result=result,
        max_decisions=max_decisions,
        total_decisions=sim_fork.decisions_made,
    )
    return result


def _build_terminal_result(
    history: list[int],
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
        partial_history=tuple(history),
        sim_time=engine.env.now,
        decisions_made=engine.decisions_made,
        disasters_remaining=len(engine.disaster_store.items),
        branch_decision=move_id,
    )


def _store_rollout_cache_entries(
    *,
    policy_name: str,
    seed: int,
    history: list[int],
    decision_log: list[int],
    result: PolicyResult,
    max_decisions: int | None,
    total_decisions: int,
) -> None:
    if max_decisions is not None:
        ROLLOUT_CACHE[(policy_name, seed, max_decisions)] = _clone_policy_result(result)
        return

    root = ROLLOUT_CACHE.get((policy_name, seed, None))
    if isinstance(root, RolloutCacheNode):
        node = root
    else:
        node = RolloutCacheNode()
        ROLLOUT_CACHE[(policy_name, seed, None)] = node

    for move_id in history:
        child = node.children.get(move_id)
        if child is None:
            child = RolloutCacheNode()
            node.children[move_id] = child
        node = child

    decision_count = len(decision_log)
    for prefix_len in range(decision_count + 1):
        next_move = decision_log[prefix_len] if prefix_len < decision_count else None
        node.result = PolicyResult(
            terminal_outcome=result.terminal_outcome,
            time_with_disasters=result.time_with_disasters,
            policy=policy_name,
            move_id=next_move,
            partial_history=(),
            sim_time=result.sim_time,
            decisions_made=max(0, total_decisions - prefix_len),
            disasters_remaining=result.disasters_remaining,
            branch_decision=next_move,
        )
        if prefix_len == decision_count:
            break

        move_id = decision_log[prefix_len]
        child = node.children.get(move_id)
        if child is None:
            child = RolloutCacheNode()
            node.children[move_id] = child
        node = child


def _materialize_decision_state(seed: int, history: list[int], scenario_config: ScenarioConfig) -> SimPySimulationEngine:
    from .policies import Policy

    sim_fork = SimPySimulationEngine(
        policy=Policy("_tournament_noop", lambda resource, disasters, env: None),
        seed=seed,
        scenario_config=scenario_config,
        track_metrics=False,
    )
    sim_fork.initialize_world()
    sim_fork.replay_buffer = deque(history)
    sim_fork.stop_before_policy_decision = True
    sim_fork.run(max_decisions=0)
    sim_fork.stop_before_policy_decision = False
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


def _evaluate_policy_tree_impl(
    seed: int,
    history: list[int],
    depth: int,
    scenario_config: ScenarioConfig,
    all_policies: list[Policy],
    ancestry_has_more: tuple[bool, ...],
) -> PolicyResult:
    if depth <= 1:
        best_final_result = None
        best_final_score = float("inf")

        for index, policy in enumerate(all_policies):
            final_result = _evaluate_policy_worker(policy, seed, history, None, scenario_config)
            score = _score_policy_result(final_result)
            if score < best_final_score:
                best_final_score = score
                best_final_result = final_result

        if best_final_result is None:
            raise RuntimeError("Tournament leaf produced no policy results.")
        return best_final_result

    decision_state = _materialize_decision_state(seed, history, scenario_config)
    if decision_state.pending_decision_resource is None:
        terminal_result = _build_terminal_result(history, decision_state)
        return terminal_result

    moves = _enumerate_policy_moves(
        decision_state.pending_decision_resource,
        list(decision_state.disaster_store.items),
        decision_state.env,
        all_policies,
    )
    if not moves:
        terminal_result = _build_terminal_result(history, decision_state)
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
            (*ancestry_has_more, total_moves - index > 1),
        )
        final_result = final_result
        final_result.move_id = move_id
        final_result.branch_decision = move_id
        final_result.policy = policy_name
        final_result.partial_history = tuple(extended_history)
        score = _score_policy_result(final_result)
        if score < best_final_score:
            best_final_score = score
            best_final_result = final_result

    if best_final_result is None:
        return _build_terminal_result(history, decision_state)
    return best_final_result


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

    if len(current_history) == 0:
        clear_tournament_cache()

    unanimous_move = _find_unanimous_policy_move(resource, disasters, env, all_policies)
    if unanimous_move is not None:
        move_id, _ = unanimous_move
        master_engine.tournament_decisions.append((env.now, "unanimous"))
        return _find_disaster_by_id(disasters, move_id)
    result = _evaluate_policy_tree_impl(
        master_seed,
        current_history,
        depth,
        fork_scenario_config,
        all_policies,
        (),
    )

    if result.policy is None:
        raise RuntimeError("Tournament produced no policy results.")

    master_engine.tournament_decisions.append((env.now, result.policy))
    return _find_disaster_by_id(disasters, result.move_id)
