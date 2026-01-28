from __future__ import annotations
from typing import Dict, Tuple
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_one_client(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    loader = make_loader(X, y, batch_size=batch_size, shuffle=True)
    losses = []
    t0 = time.time()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
    t1 = time.time()
    stats = {"loss": float(np.mean(losses)) if losses else 0.0, "time_sec": float(t1 - t0)}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, stats

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    preds = []
    trues = []
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(float(loss.detach().cpu()))
        p = torch.argmax(logits, dim=1)
        preds.append(p.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    y_true = np.concatenate(trues) if trues else np.array([])
    avg = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0)) if y_true.size else 0.0
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return {"loss": float(np.mean(losses)) if losses else 0.0, "f1": f1, "acc": acc}
