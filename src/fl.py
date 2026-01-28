from __future__ import annotations
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
from .utils import sizeof_model_mb
from .train import train_one_client, evaluate

def state_dict_to_flat(sd: Dict[str, torch.Tensor]) -> np.ndarray:
    flats = []
    for k in sorted(sd.keys()):
        flats.append(sd[k].reshape(-1).numpy())
    return np.concatenate(flats)

def fedavg_aggregate(client_states: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    # Weighted average by client sample size
    total = sum(n for _, n in client_states)
    assert total > 0
    keys = client_states[0][0].keys()
    agg = {k: torch.zeros_like(client_states[0][0][k]) for k in keys}
    for sd, n in client_states:
        w = n / total
        for k in keys:
            agg[k] += sd[k] * w
    return agg

def run_federated(
    model: torch.nn.Module,
    clients: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    rounds: int,
    participation: float,
    local_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.nn.Module, Dict]:
    rng = np.random.default_rng(seed)
    client_ids = sorted(clients.keys())

    # initialize global
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    update_mb = sizeof_model_mb(model)

    round_logs = []
    flat_updates_history = []

    for r in range(1, rounds + 1):
        m = max(1, int(round(participation * len(client_ids))))
        selected = rng.choice(client_ids, size=m, replace=False).tolist()

        client_states = []
        flat_updates = []
        train_losses = []
        train_times = []

        for cid in selected:
            Xtr, ytr = clients[cid]["train"]
            # client model copy
            local_model = copy.deepcopy(model)
            local_model.load_state_dict(global_state, strict=True)
            sd_new, stats = train_one_client(
                local_model, Xtr, ytr, device=device, epochs=local_epochs,
                lr=lr, weight_decay=weight_decay, batch_size=batch_size
            )
            client_states.append((sd_new, len(ytr)))
            train_losses.append(stats["loss"])
            train_times.append(stats["time_sec"])

            # delta update for stability proxies
            delta = {k: (sd_new[k] - global_state[k]) for k in sd_new.keys()}
            flat_updates.append(state_dict_to_flat(delta))

        # secure aggregation placeholder: server aggregates sum/avg (no per-client logging needed)
        agg_state = fedavg_aggregate(client_states)
        global_state = agg_state

        # update the global model for eval
        model.load_state_dict(global_state, strict=True)

        # Evaluate on shared/global validation set
        # use any client's val/test (they're identical in this bundle)
        Xva, yva = clients[client_ids[0]]["val"]
        val_metrics = evaluate(model, Xva, yva, device=device, batch_size=batch_size)

        # store update stability
        flat_updates = np.stack(flat_updates) if flat_updates else np.zeros((1, 1))
        flat_updates_history.append(flat_updates)

        round_logs.append({
            "round": r,
            "selected_clients": m,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "train_time_sec": float(np.sum(train_times)) if train_times else 0.0,
            "val_loss": float(val_metrics["loss"]),
            "val_f1": float(val_metrics["f1"]),
            "val_acc": float(val_metrics["acc"]),
        })

    # Final test evaluation
    Xte, yte = clients[client_ids[0]]["test"]
    test_metrics = evaluate(model, Xte, yte, device=device, batch_size=batch_size)

    # Communication cost estimate (uplink only)
    # total uplink (GB) ~ update_size(MB) * rounds * participants / 1024
    avg_participants = float(np.mean([rl["selected_clients"] for rl in round_logs])) if round_logs else 0.0
    total_uplink_gb = (update_mb * rounds * avg_participants) / 1024.0

    result = {
        "round_logs": round_logs,
        "final_test": test_metrics,
        "update_size_mb": float(update_mb),
        "total_uplink_gb": float(total_uplink_gb),
        "flat_updates_history": flat_updates_history,  # for stability proxies
    }
    return model, result
