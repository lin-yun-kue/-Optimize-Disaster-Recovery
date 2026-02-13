import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# =========================
# 1) read JSONL
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# =========================
# 2) Feature 
# =========================
def one_hot_resource_type(rt: int, num_types: int) -> List[float]:
    vec = [0.0] * num_types
    idx = rt - 1  # if resource_type is 0-based，idx = rt
    if 0 <= idx < num_types:
        vec[idx] = 1.0
    return vec


def build_pair_feature(record: Dict[str, Any], i: int, num_resource_types: int) -> np.ndarray:
    rt = int(record["resource_type"])
    rx, ry = record["prev_location"]
    dx, dy = record["locations"][i]

    remaining = float(record["remainging"][i])
    remain_size = float(record["remain_size"][i])
    dist = math.hypot(rx - dx, ry - dy)

    features = []
    features += one_hot_resource_type(rt, num_resource_types)
    features += [float(rx), float(ry)]
    features += [float(dx), float(dy)]
    features += [float(dist), float(remain_size), float(remaining)]
    return np.array(features, dtype=np.float32)


def build_pairwise_dataset(records: List[Dict[str, Any]], num_resource_types: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []

    for rec in records:
        target_id = int(rec["target_id"])
        ids = rec["id"]
        n = len(ids)

        for i in range(n):
            x = build_pair_feature(rec, i, num_resource_types)
            y = 1 if int(ids[i]) == target_id else 0
            X_list.append(x)
            y_list.append(y)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# =========================
# 3) Scoring Model
# =========================
class ScoringDispatchModel:
    def __init__(self, num_resource_types: int = 2):
        self.num_resource_types = num_resource_types
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

    def fit(self, train_records: List[Dict[str, Any]]):
        X, y = build_pairwise_dataset(train_records, self.num_resource_types)
        self.model.fit(X, y)
        return self

    def score_candidates(self, record: Dict[str, Any]) -> np.ndarray:
        X = np.vstack([
            build_pair_feature(record, i, self.num_resource_types)
            for i in range(len(record["id"]))
        ])
        probs = self.model.predict_proba(X)[:, 1]  # P(y=1) 當 score
        return probs

    def predict_target_id(self, record: Dict[str, Any]) -> int:
        probs = self.score_candidates(record)
        best_idx = int(np.argmax(probs))
        return int(record["id"][best_idx])


# =========================
# 4) Evaluate：Top-1 & Top-k accuracy
# =========================
def evaluate_topk(
    model: ScoringDispatchModel,
    test_records: List[Dict[str, Any]],
    ks: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    n = len(test_records)
    hits = {k: 0 for k in ks}

    for rec in test_records:
        probs = model.score_candidates(rec)
        order = np.argsort(-probs)  # 分數由高到低的 index
        ids = [int(x) for x in rec["id"]]
        target = int(rec["target_id"])

        for k in ks:
            top_ids = {ids[idx] for idx in order[: min(k, len(ids))]}
            if target in top_ids:
                hits[k] += 1

    # Top-1 accuracy 就是 k=1 的 Top-k accuracy
    return {f"top_{k}_accuracy": hits[k] / n for k in ks}


# =========================
# 5) MAIN：80/20 split data
# =========================
if __name__ == "__main__":
    dataset_path = "dataset.jsonl"
    records = load_jsonl(dataset_path)
    print(f"Loaded {len(records)} records")

    train_records, test_records = train_test_split(
        records,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    print(f"Train: {len(train_records)}  Test: {len(test_records)}")

    model = ScoringDispatchModel(num_resource_types=2).fit(train_records)
    print("Training completed.")

    metrics = evaluate_topk(model, test_records, ks=[1, 3, 5])
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    # rec = test_records[0]
    # pred = model.predict_target_id(rec)
    # print("\nExample:")
    # print("True target_id:", rec["target_id"])
    # print("Pred target_id:", pred)
