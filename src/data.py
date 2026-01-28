from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler

def load_csv_dataset(
    csv_path: str,
    label_col: str,
    time_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    df = pd.read_csv(csv_path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in CSV columns.")
    # Keep numeric features
    y = df[label_col].astype(str).values
    X = df.drop(columns=[label_col])
    # Cast to numeric where possible
    X = X.apply(pd.to_numeric, errors="coerce")
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    df_num = X.copy()
    df_num[label_col] = y_enc
    return df_num, le

def make_synthetic_dataset(
    n_samples: int = 300000,
    n_features: int = 32,
    n_classes: int = 2,
    seed: int = 42
) -> Tuple[pd.DataFrame, LabelEncoder]:
    rng = np.random.default_rng(seed)
    # Create mildly separable classes
    X = rng.normal(0, 1, size=(n_samples, n_features))
    w = rng.normal(0, 1, size=(n_features,))
    logits = X @ w + 0.25 * rng.normal(size=(n_samples,))
    if n_classes == 2:
        y = (logits > np.percentile(logits, 60)).astype(int)
    else:
        # multi-class via quantiles
        bins = np.quantile(logits, np.linspace(0, 1, n_classes+1)[1:-1])
        y = np.digitize(logits, bins)
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    le = LabelEncoder()
    le.fit([str(i) for i in range(n_classes)])
    return df, le

def split_scale(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
) -> DatasetBundle:
    y = df[label_col].values.astype(int)
    X = df.drop(columns=[label_col]).values.astype(np.float32)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    le = LabelEncoder()
    le.fit(y)  # y already encoded ints, but keep for completeness
    return DatasetBundle(X_train, y_train, X_val, y_val, X_test, y_test, le, scaler)

def to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if seq_len <= 1:
        return X[:, None, :], y
    n = X.shape[0]
    if n < seq_len:
        raise ValueError("Not enough samples to form one sequence window.")
    Xs = []
    ys = []
    # sliding windows
    for i in range(0, n - seq_len + 1, seq_len):  # step=seq_len for speed; change to 1 for dense windows
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.stack(Xs).astype(np.float32), np.array(ys).astype(int)

def partition_iid(y: np.ndarray, n_clients: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    splits = np.array_split(idx, n_clients)
    return [s for s in splits]

def partition_noniid_dirichlet(y: np.ndarray, n_clients: int, alpha: float, seed: int = 42, min_size: int = 1000) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = int(y.max() + 1)
    idx_by_class = [np.where(y == c)[0] for c in range(n_classes)]
    for arr in idx_by_class:
        rng.shuffle(arr)

    client_indices = [[] for _ in range(n_clients)]
    # repeat until each client has at least min_size
    while True:
        client_indices = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = idx_by_class[c]
            proportions = rng.dirichlet(alpha * np.ones(n_clients))
            # balance by current load
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cuts)
            for k in range(n_clients):
                client_indices[k].extend(splits[k].tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            break
        # re-sample with different seed jitter
        alpha = max(alpha * 0.95, 1e-3)

    return [np.array(ci, dtype=int) for ci in client_indices]

def make_client_datasets(
    bundle: DatasetBundle,
    seq_len: int,
    n_clients: int,
    scenario: str,
    alpha: float,
    seed: int
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    # Sequence conversion for train/val/test separately (keeps eval global)
    Xtr_seq, ytr_seq = to_sequences(bundle.X_train, bundle.y_train, seq_len)
    Xva_seq, yva_seq = to_sequences(bundle.X_val, bundle.y_val, seq_len)
    Xte_seq, yte_seq = to_sequences(bundle.X_test, bundle.y_test, seq_len)

    # Partition only training sequences across clients
    y_part = ytr_seq
    if scenario.lower() == "iid":
        parts = partition_iid(y_part, n_clients=n_clients, seed=seed)
    elif scenario.lower() in ["noniid", "non-iid", "dirichlet"]:
        parts = partition_noniid_dirichlet(y_part, n_clients=n_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError("scenario must be one of: iid, noniid")

    clients = {}
    for cid, idx in enumerate(parts):
        clients[cid] = {
            "train": (Xtr_seq[idx], ytr_seq[idx]),
            "val": (Xva_seq, yva_seq),   # global val for consistent tracking
            "test": (Xte_seq, yte_seq),  # global test
        }
    return clients
