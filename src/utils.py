from __future__ import annotations
import os, random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_pref: str = "cuda_if_available") -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda_if_available" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def sizeof_model_mb(model: torch.nn.Module) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return (n_params * 4) / (1024 ** 2)  # float32 approx

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
