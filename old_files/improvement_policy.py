import simpy
from SimPyTest.engine import SimPySimulationEngine, ScenarioConfig
from SimPyTest.simulation import Resource, Disaster, ResourceType
from score_model import DeepSetsChooser, DISASTER_TYPES, type_to_idx, model_config
import torch.nn as nn
from collections import defaultdict
from SimPyTest.main import PolicyResult
import statistics
import pickle
import torch
from SimPyTest.policies import Policy, record_decision

from pathlib import Path
from typing import Optional

from dataclasses import dataclass
from typing import Any
import numpy as np
import copy
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
import json

# model_name = "best_model.pt"
model_name = "imp_policy1.pt"
output_filename = "p1.json"

output_data = False
num_seed = 5

def one_hot_disaster_type(disaster_type: str) -> list[float]:
    vec = np.zeros((len(DISASTER_TYPES),), dtype=np.float32)
    idx = type_to_idx.get(disaster_type, type_to_idx["other"])
    vec[idx] = 1.0
    return vec


def get_resource_features(resource: Resource) -> list[float]:
    resource_type = resource.resource_type.value

    if hasattr(resource, "prev_location"):
        px, py = resource.prev_location
    else:
        px, py = 0.0, 0.0

    return [resource_type, float(px), float(py)]


def get_disaster_features(resource: Resource, disaster: Disaster) -> list[float]:
    disaster_type = getattr(disaster, "disaster_type", "other")
    type_vec = one_hot_disaster_type(disaster_type)

    location = getattr(disaster, "location", [0.0, 0.0])
    x, y = location

    remaining = disaster.percent_remaining()
    remain_size = disaster.dirt.level
    truck = len(disaster.roster[ResourceType.TRUCK])
    excavators = len(disaster.roster[ResourceType.EXCAVATOR])

    px, py = resource.prev_location
    dx = x - px
    dy = y - py
    dist = (dx ** 2 + dy ** 2) ** 0.5

    return np.concatenate([
        type_vec,
        np.array([remaining, remain_size, float(x), float(y), truck, excavators, dist], dtype=np.float32)
        ], axis=0)


@dataclass
class PolicyConfig:
    max_disasters: int = 10
    device: str = "cpu"


class ModelPolicy:
    def __init__(self, checkpoint_path: str, config: PolicyConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.model = DeepSetsChooser(
            r_dim= model_config["r_dim"],
            d_dim= model_config["d_dim"],
            h= model_config["hidden_width"],
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

    def _build_inputs(self, resource: Any, disasters: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Any]]:
        r = get_resource_features(resource)

        # 保留前 max_disasters 個候選；若你想更穩，可先用 heuristic 篩選
        candidates = disasters[: self.config.max_disasters]

        D = []
        kept_objects = []
        for d in candidates:
            d_feature = get_disaster_features(resource, d)
            D.append(d_feature)
            kept_objects.append(d)

        # padding
        while len(D) < self.config.max_disasters:
            D.append([0.0] * model_config["d_dim"])

        mask = [1.0] * len(kept_objects) + [0.0] * (self.config.max_disasters - len(kept_objects))

        r_t = torch.tensor([r], dtype=torch.float32, device=self.device)                  # [1, r_dim]
        
        D = np.array(D, dtype=np.float32)
        D_t = torch.tensor(D, device=self.device).unsqueeze(0)
        # D_t = torch.tensor([D], dtype=torch.float32, device=self.device)                  # [1, N, d_dim]
        mask_t = torch.tensor([mask], dtype=torch.float32, device=self.device)            # [1, N]

        return r_t, D_t, mask_t, kept_objects

    @torch.no_grad()
    def select(self, resource: Any, disasters: list[Any]) -> Optional[Any]:
        if len(disasters) == 0:
            return None

        r_t, D_t, mask_t, kept_objects = self._build_inputs(resource, disasters)

        logits = self.model(r_t, D_t, mask_t)[0]   # [N]
        pred_idx = int(torch.argmax(logits).item())
        
        if pred_idx >= len(kept_objects):
            return None

        return kept_objects[pred_idx]


_POLICY: ModelPolicy | None = None

def init_improvement_policy(checkpoint_path: str, device: str = "cpu") -> None:
    global _POLICY
    config = PolicyConfig(device=device)
    _POLICY = ModelPolicy(checkpoint_path=checkpoint_path, config=config)

class ImprovementPolicy:
    def __init__(self, d: Disaster):
        self.d = d
        self._decision_count = 0
        self.name = "Improvement Policy Expolre"
        

    
    def func(self, resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster | None:
        if self._decision_count == 0:
            self._decision_count += 1
            
            # print("Did: ", [d.id for d in disasters])
            # print("target: ", self.d.id)

            target = [d for d in disasters if d.id == self.d.id][0]


            if target is None:
                raise Exception("target not found")

            return target
        else:
            target = score_model_policy_func(resource, disasters, env)
            # print("Model Seclect Target:", target)
            return target
        

def evaluate_improvement_policy(d: Disaster, master_seed: int, current_history: list[int], config: ScenarioConfig):
    # init_improvement_policy("best_model.pt", device="cpu")
    
    improve_policy = ImprovementPolicy(d)

    sim_fork = SimPySimulationEngine(
        improve_policy, 
        master_seed, 
        False, 
        config
        )
    sim_fork.initialize_world()

    sim_fork.replay_buffer = list(current_history)
    success = sim_fork.run()

    time_taken = sim_fork.get_summary()["non_idle_time"]

    print("move_id: ", d.id, "time: ", time_taken)
    return {
        "success": success,
        "time": time_taken,
        "move_id": d.id,
        "policy": "improvement_policy"
    }


def score_model_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster | None:
    global _POLICY
    if _POLICY is None:
        init_improvement_policy("best_model.pt", device="cpu")
        # raise RuntimeError("Policy not initialized. Call init_improvement_policy() first.")

    selection = _POLICY.select(resource, disasters)
    return selection

def improvement_policy_func(resource: Resource, disasters: list[Disaster], env: simpy.Environment) -> Disaster | None:

    master_engine = resource.engine
    current_history = list(master_engine.decision_log)
    master_seed = master_engine.seed
    fork_scenario_config: ScenarioConfig = copy.deepcopy(master_engine.scenario_config)
    # fork_scenario_config: ScenarioConfig | None = None
    # try:
    #     fork_scenario_config = master_engine.scenario_config
    # except Exception:
    #     fork_scenario_config = None

    tasks = []
    for d in disasters:
        tasks.append((d, master_seed, current_history, fork_scenario_config))

    # with multiprocessing.Pool(processes=len(tasks)) as pool:
        # results = pool.starmap(evaluate_improvement_policy, tasks)

    results = []
    for t in tasks:
        r = evaluate_improvement_policy(*t)
        results.append(r)


    best_result = None
    min_time = float("inf")

    for result in results:
        if result["success"] and result["time"] < min_time:
            min_time = result["time"]
            best_result = result

    if best_result is None:
        raise Exception("No policy was able to complete the simulation.")
    
    target = [d for d in disasters if d.id == best_result["move_id"]][0]
    
    id_score = {}
    for r in results:
        if r["success"]:
            id_score[r["move_id"]] = r["time"]
    record_decision(resource, disasters, best_result, min_time, id_score)
    
    return target


if __name__ == "__main__":
    aggregated_results: defaultdict[str, PolicyResult] = defaultdict(lambda: {"success": [], "fail": 0})
    records: list[dict[str, Any]] = []

    init_improvement_policy(model_name, device="cpu")
    # policy = Policy("improvement_policy", improvement_policy_func)
    
    policy = Policy("score_model_policy", score_model_policy_func)

    for seed in range(num_seed):

        engine = SimPySimulationEngine(policy=policy, seed=1, live_plot=False)
        engine.initialize_world()
        success = engine.run()
    
        duration = engine.get_summary()["non_idle_time"]
    
        print("success: ", success, "seed: ", seed)
        if success:
            aggregated_results[policy.name]["success"].append(duration)
        else:
            aggregated_results[policy.name]["fail"] += 1

        records.extend(engine.records)

    print("\n" + "=" * 85)
    print(f"{'POLICY':<20} | {'SUCCESS %':<10} | {'AVG TIME':<10} | {'STDEV':<10} | {'MIN':<8} | {'MAX':<8}")
    print("-" * 85)

        # Calculate statistics
    final_stats: list[tuple[str, float, float, float, float, float]] = []
    for name, data in aggregated_results.items():
        success_times = data["success"]
        fail_count = data["fail"]
        total_runs = len(success_times) + fail_count

        success_rate = (len(success_times) / total_runs) * 100 if total_runs > 0 else 0

        if success_times:
            avg = statistics.mean(success_times)
            stdev = statistics.stdev(success_times) if len(success_times) > 1 else 0.0
            mn = min(success_times)
            mx = max(success_times)
        else:
            avg = float("inf")  # Sort failures to bottom
            stdev = 0.0
            mn = 0
            mx = 0

        final_stats.append((name, success_rate, avg, stdev, mn, mx))

     # Sort by Success Rate (Desc), then Avg Time (Asc)
    final_stats.sort(key=lambda x: (-x[1], x[2]))

    for name, rate, avg, stdev, mn, mx in final_stats:
        avg_str = f"{avg:.2f}" if avg != float("inf") else "N/A"
        stdev_str = f"{stdev:.2f}" if avg != float("inf") else "N/A"
        mn_str = f"{mn:.0f}" if avg != float("inf") else "N/A"
        mx_str = f"{mx:.0f}" if avg != float("inf") else "N/A"

        print(f"{name:<20} | {rate:<9.1f}% | {avg_str:<10} | {stdev_str:<10} | {mn_str:<8} | {mx_str:<8}")
    print("=" * 85 + "\n")

    if output_data:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
