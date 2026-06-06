import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

model_config = {
    "r_dim": 3,
    "d_dim": 9,
    "hidden_width": 256
}
dataset_path = "p1.json"
model_name = "imp_policy1.pt"

# =========================
# read JSONL
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records

DISASTER_TYPES = ["landslide", "other"]
type_to_idx = {t:i for i,t in enumerate(DISASTER_TYPES)}

def featurize_sample(sample, max_disasters=10, shuffle_candidates=False, seed=None):
    # ---- 1) global resource features r ----
    resource_type = float(sample["resource_type"])
    prev_x, prev_y = sample["prev_location"]
    r = np.array([resource_type, float(prev_x), float(prev_y)], dtype=np.float32)

    # ---- 2) flatten disasters to a list of (disaster_id, feature_vec) ----
    items = []
    for dtype, info in sample["disaster_info"].items():
        L = len(info["id"])  # number of disasters of this type
        for k in range(L):
            did = int(info["id"][k])
            x, y = info["locations"][k]
            remaining = float(info["remainging"][k])
            remain_size = float(info["remain_size"][k])
            truck = float(info["truck"][k])
            excavators = float(info["excavators"][k])

            dx = x - prev_x
            dy = y - prev_y
            dist = (dx ** 2 + dy ** 2) ** 0.5

            # type one-hot
            tvec = np.zeros((len(DISASTER_TYPES),), dtype=np.float32)
            tvec[type_to_idx.get(dtype, type_to_idx["other"])] = 1.0

            dvec = np.concatenate([
                tvec,
                np.array([remaining, remain_size, float(x), float(y), truck, excavators, dist], dtype=np.float32)
            ], axis=0)
            # dvec = np.array([remaining, remain_size, float(x), float(y), truck, excavators], dtype=np.float32)

            items.append((did, dvec))

        # ---- 3) optional shuffle to simulate random ordering ----
        if shuffle_candidates:
            rng = np.random.default_rng(seed)
            rng.shuffle(items)

        # ---- 4) pad/truncate to max_disasters ----
        d_dim = model_config["d_dim"]
        # d_dim = 6 # number of disaster feature
        D = np.zeros((max_disasters, d_dim), dtype=np.float32)
        mask = np.zeros((max_disasters,), dtype=np.float32)
        ids = np.full((max_disasters,), -1, dtype=np.int64)

        items = items[:max_disasters]
        for i, (did, dvec) in enumerate(items):
            D[i] = dvec
            mask[i] = 1.0
            ids[i] = did

        # ---- 5) label index y (0..max_disasters-1) ----
        target_id = int(sample["target_id"])
        idx = np.where(ids == target_id)[0]
        y = int(idx[0]) if len(idx) > 0 else None

        return r, D, mask, y, ids

def masked_mean(E, mask, eps=1e-8):
    # E: [B, N, H], mask: [B, N] in {0,1}
    m = mask.unsqueeze(-1)              # [B,N,1]
    summed = (E * m).sum(dim=1)         # [B,H]
    denom = m.sum(dim=1).clamp_min(eps) # [B,1]
    return summed / denom               # [B,H]

class DeepSetsChooser(nn.Module):
    def __init__(self, r_dim, d_dim, h=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(r_dim + d_dim, 128),
            nn.ReLU(),
            nn.Linear(128, h),
            nn.ReLU(),
        )
        self.g = nn.Sequential(
            nn.Linear(r_dim + d_dim + h, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, r, D, mask):
        # r: [B,r_dim], D:[B,N,d_dim], mask:[B,N]
        B, N, _ = D.shape
        r_expand = r.unsqueeze(1).expand(B, N, r.shape[-1])
        x = torch.cat([r_expand, D], dim=-1)   # [B,N,r+d]
        E = self.phi(x)                        # [B,N,h]
        z = masked_mean(E, mask)               # [B,h]
        z_expand = z.unsqueeze(1).expand(B, N, z.shape[-1])

        x2 = torch.cat([r_expand, D, z_expand], dim=-1)
        S = self.g(x2).squeeze(-1)             # [B,N] logits

        # avoid padding
        S = S.masked_fill(mask == 0, -1e9)
        return S

# def collate(batch):
#     rs, Ds, masks, ys = [], [], [], []
#     for (r, D, mask, y, ids) in batch:
#         if y is None:
#             continue  # target 不在 top-10 就跳過（或你改 max_disasters）
#         rs.append(r); Ds.append(D); masks.append(mask); ys.append(y)

#     r = torch.tensor(np.stack(rs), dtype=torch.float32)
#     D = torch.tensor(np.stack(Ds), dtype=torch.float32)
#     mask = torch.tensor(np.stack(masks), dtype=torch.float32)
#     y = torch.tensor(np.array(ys), dtype=torch.long)
#     return r, D, mask, y

class DisasterDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r, D, mask, y = self.data[idx]
        return (
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(D, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
    
def train(model, optimizer, criterion, train_loader, val_loader):
    best_val_acc = 0

    for epoch in range(100):

        # ====== TRAIN ======
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for r, D, mask, y in train_loader:
            logits = model(r, D, mask)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * r.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            total += r.size(0)

        train_acc = train_correct / total
        train_loss /= total


        # ====== VALIDATION ======
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for r, D, mask, y in val_loader:
                logits = model(r, D, mask)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += r.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print("-" * 40)

        # ====== SAVE BEST ======
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_name)

def permutation_stress_test(model, raw_samples, trials=10):
    model.eval()
    stable = 0

    for sample in raw_samples[:100]:  # 取 100 筆測試
        preds = []
        for t in range(trials):
            r, D, mask, y, ids = featurize_sample(
                sample,
                shuffle_candidates=True,
                seed=1000+t
            )
            if y is None:
                continue

            with torch.no_grad():
                r_t = torch.tensor(r[None,:], dtype=torch.float32)
                D_t = torch.tensor(D[None,:,:], dtype=torch.float32)
                m_t = torch.tensor(mask[None,:], dtype=torch.float32)
                logits = model(r_t, D_t, m_t)[0]
                idx = logits.argmax().item()
                preds.append(ids[idx])

        if len(set(preds)) == 1:
            stable += 1

    print("Permutation stable ratio:", stable / 100)

def evaluate(model, dataloader, device="cpu"):
    model.eval()

    total = 0
    correct = 0
    top3_correct = 0
    reciprocal_rank_sum = 0.0

    with torch.no_grad():
        for r, D, mask, y in dataloader:

            r = r.to(device)
            D = D.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(r, D, mask)   # [B, 10]

            # ---- Top-1 accuracy ----
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()

            # ---- Top-3 accuracy ----
            top3 = torch.topk(logits, k=3, dim=1).indices
            top3_correct += sum(
                y[i] in top3[i] for i in range(len(y))
            )

            # ---- MRR ----
            # rank of true label
            sorted_indices = torch.argsort(logits, dim=1, descending=True)
            for i in range(len(y)):
                rank = (sorted_indices[i] == y[i]).nonzero(as_tuple=True)[0].item() + 1
                reciprocal_rank_sum += 1.0 / rank

            total += r.size(0)

    metrics = {
        "acc": correct / total,
        "top3_acc": top3_correct / total,
        "MRR": reciprocal_rank_sum / total,
    }

    return metrics

# =========================
# MAIN：80/20 split data
# =========================
if __name__ == "__main__":
    raw_data = load_jsonl(dataset_path)
    print(f"Loaded {len(raw_data)} records")
    # print(raw_data[0])

    processed = []
    for sample in raw_data:
        r, D, mask, y, ids = featurize_sample(
            sample,
            max_disasters=10,
            shuffle_candidates=False
        )
        if y is not None:
            processed.append((r, D, mask, y))

    # print(processed[0][0])

    train_data, temp_data = train_test_split(
        processed,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,   
        random_state=42,
        shuffle=True,
    )

    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(val_data)}")
    print(f"Test:  {len(test_data)}")

    train_loader = DataLoader(
        DisasterDataset(train_data),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        DisasterDataset(val_data),
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        DisasterDataset(test_data),
        batch_size=64,
        shuffle=False
    )

    model = DeepSetsChooser(
        r_dim= model_config["r_dim"],
        d_dim= model_config["d_dim"],
        h= model_config["hidden_width"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, optimizer, criterion, train_loader, val_loader)

    model.load_state_dict(torch.load(model_name))
    permutation_stress_test(model, raw_data)

    test_metrics = evaluate(model, test_loader)
    

    print("===== Test Results =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
