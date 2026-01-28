from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> EvalMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return EvalMetrics(acc, float(pr), float(rc), float(f1))

def update_variance(flat_updates: np.ndarray) -> float:
    # flat_updates: [n_clients, n_params]
    return float(np.mean(np.var(flat_updates, axis=0)))

def leakage_score_proxy(asr: float, upd_var: float) -> float:
    # simple proxy: higher ASR and higher variance imply higher leakage
    return float(0.6 * (asr / 100.0) + 0.4 * min(1.0, upd_var))

def attack_success_rate_proxy(upd_var: float) -> float:
    # heuristic proxy: lower variance -> lower ASR
    # map variance to [10, 30]
    return float(10.0 + 20.0 * min(1.0, upd_var))
