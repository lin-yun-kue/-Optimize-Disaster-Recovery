"""
Canonical benchmark runner for heuristic, PPO, and dispatch-ML evaluation.
"""

from __future__ import annotations

import json
import statistics
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from SimPyTest.benchmark_catalog import (
    BENCHMARK_SUITES,
    OBJECTIVE_VERSION,
    SCENARIO_CATALOG_VERSION,
    SCENARIO_SPECS,
    create_scenario_config as catalog_create_scenario_config,
)
from SimPyTest.engine import SimPySimulationEngine
from SimPyTest.evaluation import ResearchMetricBundle, compute_research_metric_bundle, compute_training_objective_score
from SimPyTest.policies import POLICIES, TOURNAMENT_POLICY, Policy
from SimPyTest.policies_tournament import (
    clear_tournament_cache,
    get_tournament_profile_stats,
    reset_tournament_profile_stats,
    set_tournament_depth,
    set_tournament_max_decisions,
)
from SimPyTest.scenario_types import ScenarioConfig

if TYPE_CHECKING:
    from SimPyTest.gis_utils import GISConfig
    from mlp.ml_dispatch import TrainedDispatchPolicy
    from ppo.ppo_dispatch import EpisodeResult as PPOEpisodeResult


ROAD_SHAPEFILE = "maps/tl_2024_41007_roads/tl_2024_41007_roads.shp"
DEFAULT_REFERENCE_POLICY = "balanced_ratio"


class EpisodeRecord(TypedDict):
    seed: int
    success: bool
    terminal_outcome: str | None
    training_objective_score: float
    research_metrics: ResearchMetricBundle
    time_with_disasters: float
    wall_time_s: float
    decisions: int


class PolicyResult(TypedDict):
    records: list[EpisodeRecord]


class PPOModelSpec(TypedDict):
    label: str
    path: str
    metadata: dict[str, Any]


class DispatchModelSpec(TypedDict):
    label: str
    path: str
    metadata: dict[str, Any]


class PairedComparison(TypedDict):
    reference: str
    challenger: str
    shared_seed_count: int
    challenger_only_successes: int
    reference_only_successes: int
    terminal_outcomes: dict[str, int]
    mean_training_objective_delta: float | None
    mean_research_metric_delta: dict[str, float | None]


class ScenarioReport(TypedDict):
    scenario: str
    gis_enabled: bool
    reference_policy: str | None
    approaches: dict[str, dict[str, Any]]
    paired_vs_reference: dict[str, PairedComparison]


class SuiteReport(TypedDict):
    suite_name: str
    seeds: list[int]
    gis_enabled: bool
    reference_policy: str | None
    scenarios: list[ScenarioReport]
    macro_summary: dict[str, dict[str, Any]]


def create_gis_config() -> GISConfig | None:
    from SimPyTest.gis_utils import GISConfig, build_road_graph, load_and_prune_roads

    try:
        roads_gdf = load_and_prune_roads(ROAD_SHAPEFILE, enabled_types=["I", "U", "S"])
        road_graph = build_road_graph(roads_gdf)
    except Exception as exc:
        print(f"Warning: Could not load GIS data: {exc}")
        return None
    return GISConfig(roads_gdf=roads_gdf, road_graph=road_graph)


def create_scenario_config(preset: str, gis_config: GISConfig | None) -> ScenarioConfig:
    return catalog_create_scenario_config(preset, gis_config)


def _new_policy_result() -> PolicyResult:
    return {"records": []}


def default_ppo_label(model_path: Path, metadata: dict[str, Any]) -> str:
    scenario_name = metadata.get("scenario_name") or metadata.get("final_stage_scenario_name")
    if scenario_name:
        return f"ppo_{scenario_name}"
    return f"ppo_{model_path.stem}"


def load_ppo_model_spec(model_path: str | None) -> PPOModelSpec | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PPO model not found: {path}")
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"label": default_ppo_label(path, metadata), "path": str(path.resolve()), "metadata": metadata}


def default_dispatch_label(model_path: Path, metadata: dict[str, Any]) -> str:
    scenario_name = metadata.get("scenario_name") or metadata.get("difficulty")
    if scenario_name:
        return f"dispatch_ml_{scenario_name}"
    if model_path.stem == "dispatch_model":
        return f"dispatch_ml_{model_path.parent.name}"
    return f"dispatch_ml_{model_path.stem}"


def load_dispatch_model_spec(model_path: str | None) -> DispatchModelSpec | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Dispatch model not found: {path}")
    metadata_path = path.parent / "dispatch_model_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"label": default_dispatch_label(path, metadata), "path": str(path.resolve()), "metadata": metadata}


def run_policy_episode(policy: Policy, seed: int, scenario_config: ScenarioConfig, live_plot: bool) -> EpisodeRecord:
    t0 = time.perf_counter()
    if policy.name == TOURNAMENT_POLICY.name:
        reset_tournament_profile_stats()
    engine = SimPySimulationEngine(policy=policy, seed=seed, scenario_config=scenario_config)
    if live_plot:
        from SimPyTest.visualization import EngineVisualizer

        engine.visualizer = EngineVisualizer(engine)
    engine.initialize_world()
    success = engine.run()
    summary = engine.summary()
    if policy.name == TOURNAMENT_POLICY.name:
        stats = get_tournament_profile_stats()
        hit_rate = stats["worker_cache_hits"] / stats["worker_calls"] if stats["worker_calls"] else 0.0
        avg_worker_ms = (stats["worker_time_s"] / stats["worker_calls"] * 1000.0) if stats["worker_calls"] else 0.0
        avg_materialize_ms = (stats["materialize_time_s"] / stats["materialize_calls"] * 1000.0) if stats["materialize_calls"] else 0.0
        print(
            "      [tournament-profile] "
            f"calls={stats['tournament_calls']} "
            f"worker_calls={stats['worker_calls']} "
            f"cache_hit_rate={hit_rate:.1%} "
            f"materialize_calls={stats['materialize_calls']}"
        )
        print(
            "      [tournament-profile] "
            f"worker_time_s={stats['worker_time_s']:.2f} "
            f"worker_run_time_s={stats['worker_run_time_s']:.2f} "
            f"materialize_time_s={stats['materialize_time_s']:.2f} "
            f"tree_time_s={stats['tree_time_s']:.2f} "
            f"live_leaf_time_s={stats['live_leaf_time_s']:.2f}"
        )
        print(
            "      [tournament-profile] "
            f"avg_worker_ms={avg_worker_ms:.2f} "
            f"avg_materialize_ms={avg_materialize_ms:.2f} "
            f"cache_entries_written={stats['cache_store_entries']} "
            f"cache_pruned_keys={stats['cache_pruned_keys']}"
        )
    return {
        "seed": seed,
        "success": success,
        "terminal_outcome": engine.last_terminal_outcome,
        "training_objective_score": compute_training_objective_score(summary),
        "research_metrics": compute_research_metric_bundle(summary),
        "time_with_disasters": summary.time_with_disasters,
        "wall_time_s": time.perf_counter() - t0,
        "decisions": engine.decisions_made,
    }


def run_ppo_episode(model: object, seed: int, scenario_name: str, scenario_config: ScenarioConfig) -> EpisodeRecord:
    from ppo.ppo_dispatch import run_policy_episode as run_ppo_policy_episode

    episode: PPOEpisodeResult = run_ppo_policy_episode(model, seed, True, "ppo_benchmark", scenario_name, scenario_config)
    return {
        "seed": seed,
        "success": bool(episode["success"]),
        "terminal_outcome": episode["terminal_outcome"],
        "training_objective_score": episode["objective_score"],
        "research_metrics": episode["kpis"],
        "time_with_disasters": episode["time_with_disasters"],
        "wall_time_s": episode["wall_time_s"],
        "decisions": episode["decisions"],
    }


def run_dispatch_episode(
    model: "TrainedDispatchPolicy",
    seed: int,
    scenario_config: ScenarioConfig,
    scenario_name: str,
) -> EpisodeRecord:
    from SimPyTest.gym import DisasterResponseGym

    t0 = time.perf_counter()
    dispatch_env = DisasterResponseGym(
        max_visible_disasters=model.max_visible_disasters,
        sorting_strategy="nearest",
        scenario_config=scenario_config,
        controller_name="dispatch_ml_benchmark",
        scenario_name=scenario_name,
    )
    observation, info = dispatch_env.reset(seed=seed)
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = dispatch_env.step(action)
        _ = reward

    summary = info["summary"]
    return {
        "seed": seed,
        "success": bool(info["is_success"]),
        "terminal_outcome": info["terminal_outcome"],
        "training_objective_score": float(info["objective_score"]),
        "research_metrics": compute_research_metric_bundle(summary),
        "time_with_disasters": summary.time_with_disasters,
        "wall_time_s": time.perf_counter() - t0,
        "decisions": -1,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def _record_time_with_disasters(record: EpisodeRecord) -> float:
    return float(record.get("time_with_disasters", record["time_with_disasters"]))


def _is_qualified_approach(summary: dict[str, Any]) -> bool:
    terminal_outcomes = summary["terminal_outcomes"]
    return float(summary["success_rate"]) == 100.0 and set(terminal_outcomes.keys()) == {"SUCCESS"}


def _approach_summary(records: list[EpisodeRecord]) -> dict[str, Any]:
    success_records = [record for record in records if record["success"]]
    terminal_counts = Counter((record["terminal_outcome"] or "UNKNOWN") for record in records)
    research_metric_mean: dict[str, float | None] = {}
    if records:
        for key in sorted(records[0]["research_metrics"].keys()):
            research_metric_mean[key] = _mean([record["research_metrics"][key] for record in records])
    return {
        "runs": len(records),
        "successes": len(success_records),
        "failures": len(records) - len(success_records),
        "success_rate": (len(success_records) / len(records) * 100.0) if records else 0.0,
        "avg_training_objective_score": _mean([record["training_objective_score"] for record in records]),
        "avg_time_with_disasters": _mean([_record_time_with_disasters(record) for record in success_records]),
        "avg_wall_time_s": _mean([record["wall_time_s"] for record in records]),
        "terminal_outcomes": dict(sorted(terminal_counts.items())),
        "qualified": False,
        "research_metric_mean": research_metric_mean,
        "per_seed": records,
    }


def summarize_results(results: dict[str, PolicyResult]) -> dict[str, dict[str, Any]]:
    summaries = {name: _approach_summary(data["records"]) for name, data in results.items()}
    for data in summaries.values():
        data["qualified"] = _is_qualified_approach(data)
    return summaries


def paired_seed_comparison(reference_name: str, challenger_name: str, records_by_name: dict[str, list[EpisodeRecord]]) -> PairedComparison:
    reference_records = {record["seed"]: record for record in records_by_name.get(reference_name, [])}
    challenger_records = {record["seed"]: record for record in records_by_name.get(challenger_name, [])}
    shared_seeds = sorted(set(reference_records) & set(challenger_records))

    challenger_only_successes = 0
    reference_only_successes = 0
    terminal_outcomes: Counter[str] = Counter()
    training_objective_deltas: list[float] = []
    research_metric_deltas: defaultdict[str, list[float]] = defaultdict(list)

    for seed in shared_seeds:
        reference = reference_records[seed]
        challenger = challenger_records[seed]
        training_objective_deltas.append(challenger["training_objective_score"] - reference["training_objective_score"])

        if challenger["success"] and not reference["success"]:
            challenger_only_successes += 1
        if reference["success"] and not challenger["success"]:
            reference_only_successes += 1

        terminal_key = f"{reference['terminal_outcome'] or 'UNKNOWN'}->{challenger['terminal_outcome'] or 'UNKNOWN'}"
        terminal_outcomes[terminal_key] += 1
        for key in sorted(reference["research_metrics"].keys()):
            research_metric_deltas[key].append(challenger["research_metrics"][key] - reference["research_metrics"][key])

    mean_research_metric_delta = {key: _mean(values) for key, values in sorted(research_metric_deltas.items())}
    return {
        "reference": reference_name,
        "challenger": challenger_name,
        "shared_seed_count": len(shared_seeds),
        "challenger_only_successes": challenger_only_successes,
        "reference_only_successes": reference_only_successes,
        "terminal_outcomes": dict(sorted(terminal_outcomes.items())),
        "mean_training_objective_delta": _mean(training_objective_deltas),
        "mean_research_metric_delta": mean_research_metric_delta,
    }


def choose_reference_policy(summaries: dict[str, dict[str, Any]]) -> str | None:
    qualified_names = [name for name, data in summaries.items() if data["qualified"]]
    if not qualified_names:
        return None
    if DEFAULT_REFERENCE_POLICY in qualified_names:
        return DEFAULT_REFERENCE_POLICY
    return qualified_names[0]


def run_benchmark_single(
    gis_enabled: bool,
    seed_values: list[int],
    preset: str,
    policies: list[str] | None,
    ex_policies: list[str] | None,
    live_plot: bool,
    ppo_model_path: str | None,
    mlp_model_path: str | None,
    tournament_depths: list[int],
    tournament_max_decisions: list[int] | list[None],
) -> dict[str, PolicyResult]:
    if gis_enabled:
        gis_config = create_gis_config()
        if gis_config is None:
            raise RuntimeError("Could not load GIS config for GIS benchmark.")
    else:
        gis_config = None

    scenario_config = create_scenario_config(preset, gis_config)
    results: dict[str, PolicyResult] = defaultdict(_new_policy_result)

    run_policies = [*POLICIES, TOURNAMENT_POLICY] if policies is None else [policy for policy in [*POLICIES, TOURNAMENT_POLICY] if policy.name in policies]
    if ex_policies is not None:
        run_policies = [policy for policy in run_policies if policy.name not in ex_policies]

    ppo_spec = load_ppo_model_spec(ppo_model_path)
    dispatch_spec = load_dispatch_model_spec(mlp_model_path)
    if ppo_spec is not None or dispatch_spec is not None:
        from ppo.ppo_dispatch import load_model, select_device

        device = select_device()
        ppo_model = load_model(ppo_spec["path"], device) if ppo_spec is not None else None
    else:
        device = None
        ppo_model = None
    if dispatch_spec is not None and device is not None:
        from mlp.ml_dispatch import TrainedDispatchPolicy

        dispatch_model = TrainedDispatchPolicy.load(dispatch_spec["path"], device=device)
    else:
        dispatch_model = None

    set_tournament_depth(tournament_depths[0])
    set_tournament_max_decisions(tournament_max_decisions[0])

    print(f"\nScenario: {preset} | GIS={'yes' if gis_enabled else 'no'} | Seeds={seed_values}")
    for seed in seed_values:
        print(f"  Seed {seed}")
        for policy in run_policies:
            if policy.name == TOURNAMENT_POLICY.name and len(tournament_depths) >= 1 and len(tournament_max_decisions) >= 1:
                for depth in tournament_depths:
                    for max_decisions in tournament_max_decisions:
                        clear_tournament_cache()
                        set_tournament_depth(depth)
                        label = f"{TOURNAMENT_POLICY.name}_d{depth}_m{max_decisions}"
                        record = run_policy_episode(TOURNAMENT_POLICY, seed, scenario_config, live_plot)
                        results[label]["records"].append(record)
                        terminal = record["terminal_outcome"] or "UNKNOWN"
                        print(
                            f"    {label:<18} terminal={terminal:<18} train_obj={record['training_objective_score']:.2f} time_with_disasters={record['time_with_disasters']:.2f} wall_time_s={record['wall_time_s']:.2f} decisions={record['decisions']:.2f}"
                        )
                set_tournament_depth(tournament_depths[0])
                set_tournament_max_decisions(tournament_max_decisions[0])
                continue

            if policy.name != TOURNAMENT_POLICY.name:
                record = run_policy_episode(policy, seed, scenario_config, live_plot)
                results[policy.name]["records"].append(record)
                terminal = record["terminal_outcome"] or "UNKNOWN"
                print(
                    f"    {policy.name:<18} terminal={terminal:<18} train_obj={record['training_objective_score']:.2f} time_with_disasters={record['time_with_disasters']:.2f} wall_time_s={record['wall_time_s']:.2f} decisions={record['decisions']:.2f}"
                )

        if ppo_spec is not None and ppo_model is not None:
            record = run_ppo_episode(ppo_model, seed, preset, scenario_config)
            results[ppo_spec["label"]]["records"].append(record)
            terminal = record["terminal_outcome"] or "UNKNOWN"
            print(
                f"    {ppo_spec['label']:<18} terminal={terminal:<18} train_obj={record['training_objective_score']:.2f} time_with_disasters={record['time_with_disasters']:.2f} wall_time_s={record['wall_time_s']:.2f} decisions={record['decisions']:.2f}"
            )

        if dispatch_spec is not None and dispatch_model is not None:
            record = run_dispatch_episode(dispatch_model, seed, scenario_config, preset)
            results[dispatch_spec["label"]]["records"].append(record)
            terminal = record["terminal_outcome"] or "UNKNOWN"
            print(
                f"    {dispatch_spec['label']:<18} terminal={terminal:<18} train_obj={record['training_objective_score']:.2f} time_with_disasters={record['time_with_disasters']:.2f} wall_time_s={record['wall_time_s']:.2f}"
            )

    return dict(results)


def build_scenario_report(scenario_name: str, gis_enabled: bool, results: dict[str, PolicyResult]) -> ScenarioReport:
    summary = summarize_results(results)
    reference_policy = choose_reference_policy(summary)
    records_by_name = {name: data["per_seed"] for name, data in summary.items()}
    paired = {
        name: paired_seed_comparison(reference_policy, name, records_by_name) for name, data in summary.items() if reference_policy is not None and name != reference_policy and data["qualified"]
    }
    return {
        "scenario": scenario_name,
        "gis_enabled": gis_enabled,
        "reference_policy": reference_policy,
        "approaches": summary,
        "paired_vs_reference": paired,
    }


def _suite_macro_summary(scenarios: list[ScenarioReport]) -> dict[str, dict[str, Any]]:
    approach_names = sorted({name for scenario in scenarios for name in scenario["approaches"]})
    macro: dict[str, dict[str, Any]] = {}
    for name in approach_names:
        success_rates: list[float] = []
        training_objectives: list[float] = []
        time_with_disasters_values: list[float] = []
        qualified_scenarios = 0
        scenario_count = 0
        for scenario in scenarios:
            data = scenario["approaches"].get(name)
            if data is None:
                continue
            scenario_count += 1
            success_rates.append(float(data["success_rate"]))
            if data["avg_training_objective_score"] is not None:
                training_objectives.append(float(data["avg_training_objective_score"]))
            if data["avg_time_with_disasters"] is not None:
                time_with_disasters_values.append(float(data["avg_time_with_disasters"]))
            if data["qualified"]:
                qualified_scenarios += 1
        macro[name] = {
            "scenario_count": scenario_count,
            "qualified_scenarios": qualified_scenarios,
            "macro_success_rate": _mean(success_rates),
            "macro_avg_training_objective_score": _mean(training_objectives),
            "macro_avg_time_with_disasters": _mean(time_with_disasters_values),
        }
    return macro


def print_scenario_table(report: ScenarioReport) -> None:
    print(f"\nRESULTS: {report['scenario']} | GIS={'yes' if report['gis_enabled'] else 'no'}")
    print("=" * 182)
    print(
        f"{'APPROACH':<24} | {'QUAL':<5} | {'SUCCESS %':<10} | {'AVG TIME W/ DISASTERS':<22} | "
        f"{'AVG RESPONSE':<12} | {'AVG CLOSURE':<12} | {'AVG SPENT':<12} | {'TRAIN OBJ':<12} | {'AVG WALL':<10} | {'TERMINALS':<24}"
    )
    print("-" * 182)
    rows: list[tuple[str, dict[str, Any]]] = list(report["approaches"].items())
    rows.sort(
        key=lambda item: (
            0 if item[1]["qualified"] else 1,
            -(float(item[1]["success_rate"])),
            float(item[1]["avg_time_with_disasters"]) if item[1]["avg_time_with_disasters"] is not None else float("inf"),
            float(item[1]["research_metric_mean"]["avg_response_time_min"]) if item[1]["research_metric_mean"]["avg_response_time_min"] is not None else float("inf"),
            float(item[1]["research_metric_mean"]["total_spent"]) if item[1]["research_metric_mean"]["total_spent"] is not None else float("inf"),
        )
    )
    for name, data in rows:
        avg_time_with_disasters = "N/A" if data["avg_time_with_disasters"] is None else f"{data['avg_time_with_disasters']:.2f}"
        avg_response = "N/A" if data["research_metric_mean"]["avg_response_time_min"] is None else f"{data['research_metric_mean']['avg_response_time_min']:.2f}"
        avg_closure = "N/A" if data["research_metric_mean"]["total_weighted_closure_hours"] is None else f"{data['research_metric_mean']['total_weighted_closure_hours']:.2f}"
        avg_spent = "N/A" if data["research_metric_mean"]["total_spent"] is None else f"{data['research_metric_mean']['total_spent']:.2f}"
        train_obj = "N/A" if data["avg_training_objective_score"] is None else f"{data['avg_training_objective_score']:.2f}"
        avg_wall = "N/A" if data["avg_wall_time_s"] is None else f"{data['avg_wall_time_s']:.2f}s"
        terminals = ", ".join(f"{key}:{value}" for key, value in data["terminal_outcomes"].items())
        qualified = "yes" if data["qualified"] else "no"
        print(
            f"{name:<24} | {qualified:<5} | {data['success_rate']:<9.1f}% | {avg_time_with_disasters:<22} | "
            f"{avg_response:<12} | {avg_closure:<12} | {avg_spent:<12} | {train_obj:<12} | {avg_wall:<10} | {terminals:<24}"
        )


def print_paired_comparisons(report: ScenarioReport) -> None:
    if report["reference_policy"] is None:
        print("\nPaired benchmark comparison skipped: no approach achieved 100% SUCCESS terminal outcomes on this panel.")
        return
    if not report["paired_vs_reference"]:
        return
    print(f"\nPaired vs reference: {report['reference_policy']}")
    print("-" * 126)
    print(f"{'APPROACH':<24} | {'C-ONLY':<6} | {'R-ONLY':<6} | {'DELTA TIME':<12} | {'DELTA RESP':<12} | {'DELTA SPENT':<12} | {'DELTA CLOSE':<12}")
    print("-" * 126)
    for name, comparison in sorted(report["paired_vs_reference"].items()):
        metric_delta = comparison["mean_research_metric_delta"]
        delta_time = "N/A" if metric_delta["time_with_disasters"] is None else f"{metric_delta['time_with_disasters']:.2f}"
        delta_resp = "N/A" if metric_delta["avg_response_time_min"] is None else f"{metric_delta['avg_response_time_min']:.2f}"
        delta_spent = "N/A" if metric_delta["total_spent"] is None else f"{metric_delta['total_spent']:.2f}"
        delta_close = "N/A" if metric_delta["total_weighted_closure_hours"] is None else f"{metric_delta['total_weighted_closure_hours']:.2f}"
        print(
            f"{name:<24} | {comparison['challenger_only_successes']:<6} | " f"{comparison['reference_only_successes']:<6} | {delta_time:<12} | {delta_resp:<12} | {delta_spent:<12} | {delta_close:<12}"
        )


def print_suite_summary(report: SuiteReport) -> None:
    print(f"\nSUITE SUMMARY: {report['suite_name']}")
    print("=" * 114)
    print(f"{'APPROACH':<24} | {'SCENARIOS':<9} | {'QUALIFIED':<9} | {'MACRO SUCCESS %':<16} | {'MACRO TRAIN OBJ':<16} | {'MACRO TIME W/ DISASTERS':<24}")
    print("-" * 114)
    for name, data in sorted(report["macro_summary"].items()):
        success = "N/A" if data["macro_success_rate"] is None else f"{data['macro_success_rate']:.1f}%"
        objective = "N/A" if data["macro_avg_training_objective_score"] is None else f"{data['macro_avg_training_objective_score']:.2f}"
        time_with_disasters = "N/A" if data["macro_avg_time_with_disasters"] is None else f"{data['macro_avg_time_with_disasters']:.2f}"
        print(f"{name:<24} | {data['scenario_count']:<9} | {data['qualified_scenarios']:<9} | {success:<16} | {objective:<14} | {time_with_disasters:<24}")


def run_suite(
    suite_name: str,
    policies: list[str] | None,
    ex_policies: list[str] | None,
    live_plot: bool,
    ppo_model_path: str | None,
    mlp_model_path: str | None,
    tournament_depths: list[int],
    tournament_max_decisions: list[int] | list[None],
) -> SuiteReport:
    suite = BENCHMARK_SUITES[suite_name]
    seeds = list(suite.seeds)
    scenario_reports: list[ScenarioReport] = []

    for scenario_name in suite.scenario_names:
        results = run_benchmark_single(
            gis_enabled=suite.gis_enabled,
            seed_values=seeds,
            preset=scenario_name,
            policies=policies,
            ex_policies=ex_policies,
            live_plot=live_plot,
            ppo_model_path=ppo_model_path,
            mlp_model_path=mlp_model_path,
            tournament_depths=tournament_depths,
            tournament_max_decisions=tournament_max_decisions,
        )
        report = build_scenario_report(scenario_name, suite.gis_enabled, results)
        print_scenario_table(report)
        print_paired_comparisons(report)
        scenario_reports.append(report)

    return {
        "suite_name": suite_name,
        "seeds": seeds,
        "gis_enabled": suite.gis_enabled,
        "reference_policy": choose_reference_policy({name: data for scenario in scenario_reports for name, data in scenario["approaches"].items()}),
        "scenarios": scenario_reports,
        "macro_summary": _suite_macro_summary(scenario_reports),
    }


def write_results_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Wrote benchmark results to {path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Benchmark policies on shared seeds")
    parser.add_argument("--seeds", type=str, default="5", help="Either a seed count (e.g. 5) or comma-separated seeds (e.g. 1,3,5)")
    parser.add_argument("--preset", type=str, default="medium-winter", choices=sorted(SCENARIO_SPECS.keys()), help="Single scenario preset to benchmark")
    parser.add_argument("--compare-gis", action="store_true", help="Single-scenario mode only: run both GIS and non-GIS versions")
    parser.add_argument("--policies", type=str, default=None, help="Comma-separated list of policies to test")
    parser.add_argument("--ex-policies", type=str, default=None, help="Comma-separated list of policies to exclude from testing")
    parser.add_argument("--tournament-depth", type=str, default="1", help="Single depth or comma-separated list of tournament depths")
    parser.add_argument("--tournament-max-decisions", type=str, default="None", help="Single max decisions or comma-separated list of tournament max decisions")
    parser.add_argument("--suite", type=str, default=None, choices=sorted(BENCHMARK_SUITES.keys()), help="Run a named benchmark suite")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write machine-readable benchmark results")
    parser.add_argument("--live-plot", action="store_true", help="Show live plots")
    parser.add_argument("--ppo-model-path", type=str, default=None, help="Optional PPO checkpoint to include")
    parser.add_argument("--mlp-model-path", type=str, default=None, help="Optional MLP dispatch checkpoint to include")
    return parser


def _parse_seed_values(raw_seeds: str) -> list[int]:
    seed_text = raw_seeds.strip()
    if "," in seed_text:
        return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    if seed_text.startswith("-"):
        return [int(seed_text)]
    return list(range(int(seed_text)))


def _parse_depth_values(raw_depth: str) -> list[int] | None:
    if raw_depth == "None":
        return None
    depth_text = raw_depth.strip()
    if "," in depth_text:
        return [int(depth.strip()) for depth in depth_text.split(",") if depth.strip()]
    return [int(depth_text)]


def _single_mode_payload(
    preset: str,
    seeds: list[int],
    compare_gis: bool,
    policies: list[str] | None,
    ex_policies: list[str] | None,
    live_plot: bool,
    ppo_model_path: str | None,
    mlp_model_path: str | None,
    tournament_depths: list[int],
    tournament_max_decisions: list[int] | list[None],
) -> dict[str, Any]:
    no_gis_results = run_benchmark_single(False, seeds, preset, policies, ex_policies, live_plot, ppo_model_path, mlp_model_path, tournament_depths, tournament_max_decisions)
    no_gis_report = build_scenario_report(preset, False, no_gis_results)
    print_scenario_table(no_gis_report)
    print_paired_comparisons(no_gis_report)

    gis_report = None
    if compare_gis:
        gis_results = run_benchmark_single(True, seeds, preset, policies, ex_policies, live_plot, ppo_model_path, mlp_model_path, tournament_depths, tournament_max_decisions)
        gis_report = build_scenario_report(preset, True, gis_results)
        print_scenario_table(gis_report)
        print_paired_comparisons(gis_report)

    return {
        "mode": "single",
        "preset": preset,
        "seeds": seeds,
        "compare_gis": compare_gis,
        "scenario_catalog_version": SCENARIO_CATALOG_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "results_no_gis": no_gis_report,
        "results_gis": gis_report,
    }


if __name__ == "__main__":
    args = build_parser().parse_args()
    seeds = _parse_seed_values(str(args.seeds))
    tournament_depths = _parse_depth_values(str(args.tournament_depth)) or [1]
    tournament_max_decisions: list[int] | list[None] = _parse_depth_values(str(args.tournament_max_decisions)) or [None]
    policies = args.policies.split(",") if args.policies else None
    ex_policies = args.ex_policies.split(",") if args.ex_policies else None

    if args.suite is not None:
        suite_report = run_suite(
            suite_name=args.suite,
            policies=policies,
            ex_policies=ex_policies,
            live_plot=args.live_plot,
            ppo_model_path=args.ppo_model_path,
            mlp_model_path=args.mlp_model_path,
            tournament_depths=tournament_depths,
            tournament_max_decisions=tournament_max_decisions,
        )
        print_suite_summary(suite_report)
        payload = {
            "mode": "suite",
            "suite": args.suite,
            "scenario_catalog_version": SCENARIO_CATALOG_VERSION,
            "objective_version": OBJECTIVE_VERSION,
            "suite_report": suite_report,
        }
    else:
        payload = _single_mode_payload(
            preset=args.preset,
            seeds=seeds,
            compare_gis=bool(args.compare_gis),
            policies=policies,
            ex_policies=ex_policies,
            live_plot=bool(args.live_plot),
            ppo_model_path=args.ppo_model_path,
            mlp_model_path=args.mlp_model_path,
            tournament_depths=tournament_depths,
            tournament_max_decisions=tournament_max_decisions,
        )

    if args.output_json:
        write_results_json(args.output_json, payload)
