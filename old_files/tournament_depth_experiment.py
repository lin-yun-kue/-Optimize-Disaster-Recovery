from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from SimPyTest.engine import ScenarioConfig, SimPySimulationEngine
from SimPyTest.policies import (
    DISABLE_TOURNAMENT_MULTIPROCESSING,
    POLICIES,
    TOURNAMENT_POLICY_WHITELIST,
    clear_tournament_cache,
    set_tournament_candidate_policies,
    set_tournament_debug,
    set_tournament_depth,
)


@dataclass
class ScenarioSpec:
    name: str
    description: str
    seeds: list[int]
    tournament_depths: list[int]
    factory: Callable[[], ScenarioConfig]


def enabled_baseline_policies():
    return [p for p in POLICIES if not p.name.startswith("tournament")]


def tournament_policy():
    return next(p for p in POLICIES if p.name == "tournament_recursive")


def serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [serialize_value(v) for v in value]
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    return f"<{type(value).__name__}>"


def summarize_config(cfg: ScenarioConfig) -> dict[str, Any]:
    if not is_dataclass(cfg):
        return {"repr": repr(cfg)}
    summary: dict[str, Any] = {}
    for f in fields(cfg):
        summary[f.name] = serialize_value(getattr(cfg, f.name))
    return summary


def run_engine_once(policy_name: str, seed: int, cfg: ScenarioConfig) -> dict[str, Any]:
    policy = next(p for p in POLICIES if p.name == policy_name)
    t0 = time.perf_counter()
    engine = SimPySimulationEngine(policy=policy, seed=seed, live_plot=False, scenario_config=cfg)
    engine.initialize_world()
    success = engine.run()
    summary = engine.get_summary()
    wall_s = time.perf_counter() - t0
    return {
        "policy": policy_name,
        "seed": seed,
        "success": bool(success),
        "sim_time": float(summary["non_idle_time"]),
        "wall_time_s": float(wall_s),
        "env_time": float(engine.env.now),
        "disasters_remaining": int(len(engine.disaster_store.items)),
        "tournament_decisions": int(len(engine.tournament_decisions)),
    }


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint for a single independent simulation run."""
    depth = task.get("depth")
    if depth is not None:
        set_tournament_depth(int(depth))
        clear_tournament_cache()

    run = run_engine_once(str(task["policy_name"]), int(task["seed"]), task["cfg"])
    if depth is not None:
        run["depth"] = int(depth)

    return {"task": task, "run": run}


def get_max_workers(task_count: int) -> int:
    env_val = os.getenv("TDE_MAX_WORKERS")
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, cpu_count - 1)
    if env_val is None:
        return max(1, min(default_workers, task_count))
    try:
        requested = int(env_val)
    except ValueError:
        print(f"Invalid TDE_MAX_WORKERS={env_val!r}; using default={default_workers}", flush=True)
        return max(1, min(default_workers, task_count))
    return max(1, min(requested, task_count))


def mean_or_none(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def stdev_or_none(vals: list[float]) -> float | None:
    return statistics.stdev(vals) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    success_runs = [r for r in runs if r["success"]]
    fail_runs = [r for r in runs if not r["success"]]
    sim_vals = [r["sim_time"] for r in success_runs]
    wall_vals = [r["wall_time_s"] for r in success_runs]
    return {
        "runs": len(runs),
        "successes": len(success_runs),
        "failures": len(fail_runs),
        "success_rate": (len(success_runs) / len(runs) * 100.0) if runs else 0.0,
        "avg_sim_time": mean_or_none(sim_vals),
        "avg_wall_time_s": mean_or_none(wall_vals),
        "sim_stdev": stdev_or_none(sim_vals),
        "sim_min": min(sim_vals) if sim_vals else None,
        "sim_max": max(sim_vals) if sim_vals else None,
        "avg_remaining_on_fail": mean_or_none([r["disasters_remaining"] for r in fail_runs]),
    }


def rank_summary(summary: dict[str, Any]) -> tuple[float, float, float]:
    """Sort by completion first, then sim time, then wall time (all ascending)."""
    success_rate = float(summary.get("success_rate", 0.0))
    avg_sim = summary.get("avg_sim_time")
    avg_wall = summary.get("avg_wall_time_s")
    return (-success_rate, float("inf") if avg_sim is None else float(avg_sim), float("inf") if avg_wall is None else float(avg_wall))


def paired_seed_comparison(lhs_runs: list[dict[str, Any]], rhs_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare two strategies on shared seeds.

    Deltas are lhs - rhs (negative sim delta means lhs is faster).
    """
    lhs_by_seed = {int(r["seed"]): r for r in lhs_runs}
    rhs_by_seed = {int(r["seed"]): r for r in rhs_runs}
    shared_seeds = sorted(set(lhs_by_seed) & set(rhs_by_seed))

    both_success_sim_deltas: list[float] = []
    both_success_wall_deltas: list[float] = []
    lhs_better = 0
    rhs_better = 0
    ties = 0
    lhs_only_success = 0
    rhs_only_success = 0
    both_fail = 0

    for seed in shared_seeds:
        lhs = lhs_by_seed[seed]
        rhs = rhs_by_seed[seed]
        lhs_ok = bool(lhs["success"])
        rhs_ok = bool(rhs["success"])
        if lhs_ok and rhs_ok:
            sim_delta = float(lhs["sim_time"]) - float(rhs["sim_time"])
            wall_delta = float(lhs["wall_time_s"]) - float(rhs["wall_time_s"])
            both_success_sim_deltas.append(sim_delta)
            both_success_wall_deltas.append(wall_delta)
            if abs(sim_delta) < 1e-9:
                ties += 1
            elif sim_delta < 0:
                lhs_better += 1
            else:
                rhs_better += 1
        elif lhs_ok and not rhs_ok:
            lhs_only_success += 1
        elif rhs_ok and not lhs_ok:
            rhs_only_success += 1
        else:
            both_fail += 1

    return {
        "shared_seed_count": len(shared_seeds),
        "shared_seeds": shared_seeds,
        "both_success_count": len(both_success_sim_deltas),
        "lhs_only_success_count": lhs_only_success,
        "rhs_only_success_count": rhs_only_success,
        "both_fail_count": both_fail,
        "lhs_faster_on_both_success": lhs_better,
        "rhs_faster_on_both_success": rhs_better,
        "ties_on_both_success": ties,
        "avg_sim_delta_lhs_minus_rhs": mean_or_none(both_success_sim_deltas),
        "avg_wall_delta_s_lhs_minus_rhs": mean_or_none(both_success_wall_deltas),
    }


def build_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="clatsop_winter_ops_light",
            description="Realistic winter operations (seasonal + weather + dispatch priors), lighter load.",
            seeds=[i for i in range(12)],
            tournament_depths=[i for i in range(1, 4)],
            factory=lambda: ScenarioConfig(
                num_trucks=(12, 18),
                num_excavators=(6, 9),
                num_snowplows=(2, 4),
                num_assessment_vehicles=(1, 2),
                num_landslides=(8, 12),
                landslide_size_range=(200, 900),
                landslide_distance_range=(900, 2200),
                calendar_start_date=datetime(2024, 1, 1),
                calendar_duration_years=1,
                use_seasonal_disasters=True,
                use_weather_modifier=True,
                use_dispatch_delay_priors=True,
                seasonal_spawn_interval_minutes_range=(60.0, 720.0),
                track_costs=True,
                annual_budget=8_500_000,
                time_variance=0.1,
            ),
        ),
        ScenarioSpec(
            name="clatsop_summer_ops_light",
            description="Realistic summer operations (wildfire debris/flood mix), lighter load.",
            seeds=[i for i in range(12)],
            tournament_depths=[i for i in range(1, 4)],
            factory=lambda: ScenarioConfig(
                num_trucks=(12, 16),
                num_excavators=(6, 8),
                num_snowplows=(0, 1),
                num_assessment_vehicles=(1, 2),
                num_landslides=(8, 12),
                landslide_size_range=(150, 800),
                landslide_distance_range=(900, 2200),
                calendar_start_date=datetime(2024, 7, 1),
                calendar_duration_years=1,
                use_seasonal_disasters=True,
                use_weather_modifier=True,
                use_dispatch_delay_priors=True,
                seasonal_spawn_interval_minutes_range=(90.0, 1440.0),
                track_costs=True,
                annual_budget=8_000_000,
                time_variance=0.1,
            ),
        ),
        ScenarioSpec(
            name="clatsop_storm_stress",
            description="Storm-season stress test with tighter budget and high event pressure.",
            seeds=[i for i in range(10)],
            tournament_depths=[i for i in range(1, 4)],
            factory=lambda: ScenarioConfig(
                num_trucks=(10, 14),
                num_excavators=(5, 8),
                num_snowplows=(2, 4),
                num_assessment_vehicles=(1, 2),
                num_landslides=(12, 18),
                landslide_size_range=(250, 2000),
                landslide_distance_range=(900, 2600),
                calendar_start_date=datetime(2024, 11, 1),
                calendar_duration_years=1,
                use_seasonal_disasters=True,
                use_weather_modifier=True,
                use_dispatch_delay_priors=True,
                seasonal_spawn_interval_minutes_range=(45.0, 540.0),
                track_costs=True,
                annual_budget=4_000_000,
                time_variance=0.15,
            ),
        ),
    ]


def fmt_num(v: float | None, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}{suffix}"


def generate_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Tournament Depth Experiment Results")
    lines.append("")
    lines.append(f"- Generated: `{report['generated_at']}`")
    lines.append(f"- Tournament policy under test: `{report['tournament_policy_name']}`")
    lines.append(f"- Enabled baseline policies: `{', '.join(report['enabled_baselines'])}`")
    lines.append("- Tournament candidate policies: " f"`{', '.join(report['tournament_candidate_policies'])}`")
    lines.append(f"- Tournament multiprocessing disabled: `{report['tournament_config']['disable_multiprocessing']}`")
    lines.append(f"- Tournament candidate whitelist: `{report['tournament_config']['candidate_whitelist']}`")
    lines.append(f"- Tournament depths tested: `{report['depths_tested']}`")
    lines.append(f"- Total runs: `{report['meta']['total_runs']}`")
    lines.append(f"- Total wall time: `{report['meta']['total_wall_time_s']:.2f}s`")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Scenario | Best Baseline (success-first) | T d=1 | T d=2 | T d=3 | Best Tournament Depth |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for scenario_name, sdata in report["scenarios"].items():
        baseline_best = sdata["comparisons"]["best_baseline_by_rank"]
        t_by_depth = sdata["tournament_by_depth_summary"]
        cells = []
        for d in [1, 2, 3]:
            summ = t_by_depth.get(str(d))
            cells.append("N/A" if not summ else fmt_num(summ["avg_sim_time"]))
        best_td = sdata["comparisons"]["best_tournament_depth_by_rank"]
        best_td_label = f"d={best_td}" if best_td is not None else "N/A"
        baseline_label = "N/A"
        if baseline_best:
            baseline_label = f"{fmt_num(baseline_best['avg_sim_time'])} ({baseline_best['policy']}, " f"{fmt_num(baseline_best['success_rate'])}%)"
        lines.append(f"| `{scenario_name}` | " f"{baseline_label} | " f"{cells[0]} | {cells[1]} | {cells[2]} | {best_td_label} |")

    lines.append("")
    lines.append("## Per-Scenario Details")
    lines.append("")

    for scenario_name, sdata in report["scenarios"].items():
        lines.append(f"### `{scenario_name}`")
        lines.append("")
        lines.append(f"- {sdata['description']}")
        lines.append(f"- Seeds: `{sdata['seeds']}`")
        lines.append(f"- Seed count: `{len(sdata['seeds'])}`")
        lines.append(f"- Tournament depths: `{sdata['tournament_depths']}`")
        lines.append(f"- Tournament policy: `{report['tournament_policy_name']}`")
        lines.append("- Scenario config:")
        for key, value in sdata["scenario_config"].items():
            lines.append(f"  - `{key}`: `{value}`")
        lines.append("")

        lines.append("**Baselines (enabled policies)**")
        lines.append("")
        lines.append("| Policy | Success % | Avg Sim | Avg Wall (s) | Sim Stdev |")
        lines.append("|---|---:|---:|---:|---:|")
        baseline_rows = sorted(sdata["baseline_summary"].items(), key=lambda item: rank_summary(item[1]))
        for pname, psum in baseline_rows:
            lines.append(f"| `{pname}` | {fmt_num(psum['success_rate'])}% | {fmt_num(psum['avg_sim_time'])} | " f"{fmt_num(psum['avg_wall_time_s'])} | {fmt_num(psum['sim_stdev'])} |")

        lines.append("")
        lines.append("**Tournament Depth Sweep**")
        lines.append("")
        lines.append("| Depth | Success % | Avg Sim | Avg Wall (s) | Sim Stdev | Delta vs Best Baseline (Sim) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        best_baseline = sdata["comparisons"]["best_baseline_by_rank"]
        baseline_avg = best_baseline["avg_sim_time"] if best_baseline else None
        tournament_rows = sorted(
            ((int(depth), dsum) for depth, dsum in sdata["tournament_by_depth_summary"].items()),
            key=lambda item: item[0],
        )
        for depth_int, dsum in tournament_rows:
            delta = None
            if baseline_avg is not None and dsum["avg_sim_time"] is not None:
                delta = dsum["avg_sim_time"] - baseline_avg
            lines.append(
                f"| {depth_int} | {fmt_num(dsum['success_rate'])}% | {fmt_num(dsum['avg_sim_time'])} | " f"{fmt_num(dsum['avg_wall_time_s'])} | {fmt_num(dsum['sim_stdev'])} | {fmt_num(delta)} |"
            )

        lines.append("")
        lines.append("**Paired Seed Comparisons (Tournament vs Best Baseline)**")
        lines.append("")
        lines.append("| Depth | Shared Seeds | Both Success | T faster | Baseline faster | T-only success | Baseline-only success | Avg Sim Delta (T-B) | Avg Wall Delta s (T-B) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for depth_int, _ in tournament_rows:
            comp = sdata["comparisons"]["paired_vs_best_baseline_by_depth"].get(str(depth_int))
            if not comp:
                continue
            lines.append(
                f"| {depth_int} | {comp['shared_seed_count']} | {comp['both_success_count']} | "
                f"{comp['lhs_faster_on_both_success']} | {comp['rhs_faster_on_both_success']} | "
                f"{comp['lhs_only_success_count']} | {comp['rhs_only_success_count']} | "
                f"{fmt_num(comp['avg_sim_delta_lhs_minus_rhs'])} | {fmt_num(comp['avg_wall_delta_s_lhs_minus_rhs'])} |"
            )

        lines.append("")
        lines.append("**Paired Depth Comparisons (Tournament, seed-matched)**")
        lines.append("")
        lines.append("| Comparison | Shared Seeds | Both Success | Left faster | Right faster | Avg Sim Delta (L-R) | Avg Wall Delta s (L-R) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for label, comp in sdata["comparisons"]["paired_depth_comparisons"].items():
            lines.append(
                f"| `{label}` | {comp['shared_seed_count']} | {comp['both_success_count']} | "
                f"{comp['lhs_faster_on_both_success']} | {comp['rhs_faster_on_both_success']} | "
                f"{fmt_num(comp['avg_sim_delta_lhs_minus_rhs'])} | {fmt_num(comp['avg_wall_delta_s_lhs_minus_rhs'])} |"
            )

        depth_trend = sdata["comparisons"]["depth_trend"]
        if depth_trend:
            lines.append("")
            lines.append("Depth trend notes:")
            for note in depth_trend:
                lines.append(f"- {note}")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `Avg Sim` is the simulation's `non_idle_time` metric (not wall-clock runtime).")
    lines.append("- `Avg Wall` is measured with `time.perf_counter()` around each full run.")
    lines.append("- Tournament depth cost scales quickly with number of decision points; wall-time trends matter more than sim-time trends for usability.")
    return "\n".join(lines) + "\n"


def main():
    set_tournament_debug(False)
    set_tournament_candidate_policies(None)

    baselines = enabled_baseline_policies()
    tpolicy = tournament_policy()
    scenarios = build_scenarios()

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tournament_policy_name": tpolicy.name,
        "enabled_baselines": [p.name for p in baselines],
        "tournament_candidate_policies": [p.name for p in baselines],
        "tournament_config": {
            "disable_multiprocessing": bool(DISABLE_TOURNAMENT_MULTIPROCESSING),
            "candidate_whitelist": sorted(TOURNAMENT_POLICY_WHITELIST) if TOURNAMENT_POLICY_WHITELIST is not None else None,
        },
        "depths_tested": sorted({d for s in scenarios for d in s.tournament_depths}),
        "scenarios": {},
        "meta": {"total_runs": 0, "total_wall_time_s": 0.0},
    }

    exp_start = time.perf_counter()

    for spec in scenarios:
        print(f"\n=== Scenario: {spec.name} ===", flush=True)
        print(f"{spec.description}", flush=True)
        scenario_config_preview = summarize_config(spec.factory())
        baseline_runs: dict[str, list[dict[str, Any]]] = {p.name: [] for p in baselines}
        tournament_runs_by_depth: dict[str, list[dict[str, Any]]] = {str(d): [] for d in spec.tournament_depths}
        tasks: list[dict[str, Any]] = []

        # Baselines
        for p in baselines:
            for seed in spec.seeds:
                tasks.append(
                    {
                        "kind": "baseline",
                        "scenario": spec.name,
                        "policy_name": p.name,
                        "seed": seed,
                        "depth": None,
                        "cfg": spec.factory(),
                    }
                )

        # Tournament by depth
        for depth in spec.tournament_depths:
            for seed in spec.seeds:
                tasks.append(
                    {
                        "kind": "tournament",
                        "scenario": spec.name,
                        "policy_name": tpolicy.name,
                        "seed": seed,
                        "depth": depth,
                        "cfg": spec.factory(),
                    }
                )

        max_workers = get_max_workers(len(tasks))
        print(
            f"Running {len(tasks)} jobs in parallel with {max_workers} worker process(es) "
            f"(set TDE_MAX_WORKERS to override)",
            flush=True,
        )

        completed = 0
        if max_workers == 1:
            for task in tasks:
                result = run_task(task)
                task_info = result["task"]
                run = result["run"]
                completed += 1
                if task_info["kind"] == "baseline":
                    baseline_runs[str(task_info["policy_name"])].append(run)
                    print(
                        f"[done {completed}/{len(tasks)}] [baseline] scenario={spec.name} "
                        f"policy={task_info['policy_name']} seed={task_info['seed']} "
                        f"-> success={run['success']} sim={run['sim_time']:.2f} wall={run['wall_time_s']:.2f}s",
                        flush=True,
                    )
                else:
                    tournament_runs_by_depth[str(task_info["depth"])].append(run)
                    print(
                        f"[done {completed}/{len(tasks)}] [tournament] scenario={spec.name} "
                        f"depth={task_info['depth']} seed={task_info['seed']} "
                        f"-> success={run['success']} sim={run['sim_time']:.2f} wall={run['wall_time_s']:.2f}s "
                        f"t_decisions={run['tournament_decisions']}",
                        flush=True,
                    )
                report["meta"]["total_runs"] += 1
                report["meta"]["total_wall_time_s"] += run["wall_time_s"]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_task, task): task for task in tasks}
                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        print(
                            f"Task failed: kind={task['kind']} scenario={task['scenario']} "
                            f"policy={task['policy_name']} depth={task['depth']} seed={task['seed']}",
                            flush=True,
                        )
                        raise

                    task_info = result["task"]
                    run = result["run"]
                    completed += 1

                    if task_info["kind"] == "baseline":
                        baseline_runs[str(task_info["policy_name"])].append(run)
                        print(
                            f"[done {completed}/{len(tasks)}] [baseline] scenario={spec.name} "
                            f"policy={task_info['policy_name']} seed={task_info['seed']} "
                            f"-> success={run['success']} sim={run['sim_time']:.2f} wall={run['wall_time_s']:.2f}s",
                            flush=True,
                        )
                    else:
                        tournament_runs_by_depth[str(task_info["depth"])].append(run)
                        print(
                            f"[done {completed}/{len(tasks)}] [tournament] scenario={spec.name} "
                            f"depth={task_info['depth']} seed={task_info['seed']} "
                            f"-> success={run['success']} sim={run['sim_time']:.2f} wall={run['wall_time_s']:.2f}s "
                            f"t_decisions={run['tournament_decisions']}",
                            flush=True,
                        )

                    report["meta"]["total_runs"] += 1
                    report["meta"]["total_wall_time_s"] += run["wall_time_s"]

        for runs in baseline_runs.values():
            runs.sort(key=lambda r: int(r["seed"]))
        for runs in tournament_runs_by_depth.values():
            runs.sort(key=lambda r: int(r["seed"]))

        baseline_summary = {name: summarize_runs(runs) for name, runs in baseline_runs.items()}
        tournament_summary = {depth: summarize_runs(runs) for depth, runs in tournament_runs_by_depth.items()}

        # Comparisons
        baseline_candidates = [{"policy": name, **summ} for name, summ in baseline_summary.items()]
        best_baseline = min(baseline_candidates, key=rank_summary) if baseline_candidates else None

        tournament_candidates = [{"depth": int(depth), **summ} for depth, summ in tournament_summary.items()]
        best_tournament = min(tournament_candidates, key=rank_summary) if tournament_candidates else None

        paired_depth_comparisons: dict[str, Any] = {}
        sorted_depths = sorted(int(d) for d in tournament_summary.keys())
        for d_prev, d_next in zip(sorted_depths, sorted_depths[1:]):
            paired_depth_comparisons[f"d{d_prev}_vs_d{d_next}"] = paired_seed_comparison(
                tournament_runs_by_depth[str(d_prev)],
                tournament_runs_by_depth[str(d_next)],
            )

        paired_vs_best_baseline_by_depth: dict[str, Any] = {}
        if best_baseline is not None:
            best_baseline_runs = baseline_runs[best_baseline["policy"]]
            for depth in spec.tournament_depths:
                paired_vs_best_baseline_by_depth[str(depth)] = paired_seed_comparison(
                    tournament_runs_by_depth[str(depth)],
                    best_baseline_runs,
                )

        depth_trend: list[str] = []
        for d_prev, d_next in zip(sorted_depths, sorted_depths[1:]):
            prev = tournament_summary[str(d_prev)]
            nxt = tournament_summary[str(d_next)]
            if prev["avg_sim_time"] is None or nxt["avg_sim_time"] is None:
                depth_trend.append(f"d={d_prev} -> d={d_next}: insufficient successful runs")
                continue
            sim_delta = nxt["avg_sim_time"] - prev["avg_sim_time"]
            wall_delta = (nxt["avg_wall_time_s"] or 0.0) - (prev["avg_wall_time_s"] or 0.0)
            depth_trend.append(f"d={d_prev} -> d={d_next}: sim delta {sim_delta:+.2f}, wall delta {wall_delta:+.2f}s")

        report["scenarios"][spec.name] = {
            "description": spec.description,
            "seeds": spec.seeds,
            "tournament_depths": spec.tournament_depths,
            "scenario_config": scenario_config_preview,
            "baseline_runs": baseline_runs,
            "tournament_runs_by_depth": tournament_runs_by_depth,
            "baseline_summary": baseline_summary,
            "tournament_by_depth_summary": tournament_summary,
            "comparisons": {
                "best_baseline_by_rank": (
                    {
                        "policy": best_baseline["policy"],
                        "success_rate": best_baseline["success_rate"],
                        "avg_sim_time": best_baseline["avg_sim_time"],
                        "avg_wall_time_s": best_baseline["avg_wall_time_s"],
                    }
                    if best_baseline
                    else None
                ),
                "best_tournament_depth_by_rank": (best_tournament["depth"] if best_tournament else None),
                "paired_depth_comparisons": paired_depth_comparisons,
                "paired_vs_best_baseline_by_depth": paired_vs_best_baseline_by_depth,
                "depth_trend": depth_trend,
            },
        }

    report["meta"]["total_wall_time_s"] = time.perf_counter() - exp_start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("experiment_results") / "tournament_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"tournament_depth_experiment_{timestamp}.json"
    md_path = out_dir / f"tournament_depth_experiment_{timestamp}.md"
    latest_md = out_dir / "TOURNAMENT_DEPTH_EXPERIMENT_RESULTS.md"
    latest_json = out_dir / "TOURNAMENT_DEPTH_EXPERIMENT_RESULTS.json"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_text = generate_markdown(report)
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Experiment Complete ===", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)
    print(f"Latest Markdown: {latest_md}", flush=True)


if __name__ == "__main__":
    main()
