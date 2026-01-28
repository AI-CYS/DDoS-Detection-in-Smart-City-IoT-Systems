from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from src.utils import set_seed, get_device, ensure_dir
from src.data import make_synthetic_dataset, split_scale, make_client_datasets
from src.model import TransformerClassifier
from src.fl import run_federated
from src.plots import plot_participation_sweep

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, default="iid", choices=["iid", "noniid"])
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--epochs-local", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs/participation_sweep")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("cuda_if_available")

    df, _ = make_synthetic_dataset(n_samples=140000, n_features=32, n_classes=2, seed=args.seed)
    bundle = split_scale(df, label_col="label", seed=args.seed)
    clients = make_client_datasets(bundle, seq_len=args.seq_len, n_clients=args.clients,
                                   scenario=args.scenario, alpha=args.alpha, seed=args.seed)

    Xtr0, ytr0 = clients[0]["train"]
    model = TransformerClassifier(n_features=Xtr0.shape[-1], n_classes=int(ytr0.max()+1), seq_len=args.seq_len)

    ensure_dir(args.outdir)
    results = {}
    for p_rate in [0.2, 0.4, 0.6, 0.8]:
        m = TransformerClassifier(n_features=Xtr0.shape[-1], n_classes=int(ytr0.max()+1), seq_len=args.seq_len)
        _, res = run_federated(
            model=m, clients=clients, rounds=args.rounds, participation=p_rate,
            local_epochs=args.epochs_local, lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, device=device, seed=args.seed
        )
        results[p_rate] = float(res["final_test"]["f1"])
        print("Participation", p_rate, "F1", results[p_rate])

    with open(os.path.join(args.outdir, "participation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_participation_sweep(results, os.path.join(args.outdir, "participation_vs_f1.png"))

if __name__ == "__main__":
    main()
