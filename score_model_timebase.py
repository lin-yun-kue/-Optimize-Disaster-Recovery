from typing import Dict, Any, List, Tuple
import math
import numpy as np

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def build_candidates_from_step(step: Dict[str, Any], disaster_type: str = "landslide"):
    """
    回傳 candidates: List[Dict]
    每個 candidate = {
        "id": int,
        "time_score": float,
        "distance": float,
        "disaster_feature": np.ndarray,   # 你可以改成 torch tensor
    }
    """
    prev_loc = step["prev_location"]               # [x, y]
    time_score_map = step.get("time_score", {})    # keys: "541" 這種字串

    # 取出指定災害類型的欄位
    info = step["disaster_info"][disaster_type]
    ids = info["id"]
    remainings = info["remainging"]
    remain_sizes = info["remain_size"]
    locs = info["locations"]
    trucks = info["truck"]
    excavators = info["excavators"]
    onehots = info["onehot_index"]

    candidates = []
    for i, did in enumerate(ids):
        key = str(did)  # time_score 的 key 是字串
        if key not in time_score_map:
            continue  # 只保留有出現在 time_score 的 disaster

        dist = euclidean(prev_loc, locs[i])

        # 這裡示範做一個基本的 disaster_feature（你可以依你模型需求調整）
        disaster_feat = np.array([
            onehots[i],
            remainings[i],
            remain_sizes[i],
            trucks[i],
            excavators[i],
            dist,                # 也可以把距離當作 disaster feature 的一部分
        ], dtype=np.float32)

        candidates.append({
            "id": did,
            "time_score": float(time_score_map[key]),
            "distance": float(dist),
            "disaster_feature": disaster_feat,
        })

    return candidates