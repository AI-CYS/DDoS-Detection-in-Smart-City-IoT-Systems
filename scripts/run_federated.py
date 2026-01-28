from __future__ import annotations
import argparse, json, os, csv
import numpy as np
import torch

from src.utils import set_seed, get_device, ensure_dir
from src.data import load_csv_dataset, make_synthetic_dataset, split_scale, make_client_datasets
from src.model import TransformerClassifier
from src.fl import run_federated, state_dict_to_flat
from src.metrics import update_variance, attack_success_rate_proxy, leakage_score_proxy
from src.plots import plot_convergence, plot_f1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--time-col", type=str, default="")
    p.add_argument("--drop-cols", type=str, default="")
    p.add_argument("--scenario", type=str, default="iid", choices=["iid", "noniid"])
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--participation", type=float, default=0.6)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--epochs-local", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs/federated")
    p.add_argument("--plots", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("cuda_if_available")

    if args.csv:
        drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else None
        df, _ = load_csv_dataset(args.csv, label_col=args.label_col, time_col=args.time_col or None, drop_cols=drop_cols, seed=args.seed)
    else:
        df, _ = make_synthetic_dataset(n_samples=160000, n_features=32, n_classes=2, seed=args.seed)

    bundle = split_scale(df, label_col=args.label_col, seed=args.seed)
    clients = make_client_datasets(bundle, seq_len=args.seq_len, n_clients=args.clients,
                                   scenario=args.scenario, alpha=args.alpha, seed=args.seed)

    # infer shapes
    Xtr0, ytr0 = clients[0]["train"]
    n_features = Xtr0.shape[-1]
    n_classes = int(ytr0.max() + 1)

    model = TransformerClassifier(
        n_features=n_features, n_classes=n_classes, seq_len=args.seq_len,
        d_model=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.1
    )

    model, result = run_federated(
        model=model,
        clients=clients,
        rounds=args.rounds,
        participation=args.participation,
        local_epochs=args.epochs_local,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    # Stability/privacy proxies (based on per-round client deltas)
    # compute mean update variance over last 10 rounds (or all if fewer)
    hist = result["flat_updates_history"]
    tail = hist[-min(len(hist), 10):]
    vars_ = []
    for upd in tail:
        vars_.append(update_variance(upd))
    upd_var = float(np.mean(vars_)) if vars_ else 0.0
    asr = attack_success_rate_proxy(upd_var)
    leak = leakage_score_proxy(asr, upd_var)

    ensure_dir(args.outdir)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))

    # Save per-round logs
    csv_path = os.path.join(args.outdir, "round_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["round_logs"][0].keys()))
        w.writeheader()
        w.writerows(result["round_logs"])

    summary = {
        "scenario": args.scenario,
        "alpha": args.alpha if args.scenario == "noniid" else None,
        "clients": args.clients,
        "participation": args.participation,
        "rounds": args.rounds,
        "epochs_local": args.epochs_local,
        "seq_len": args.seq_len,
        "final_test": result["final_test"],
        "update_size_mb": result["update_size_mb"],
        "total_uplink_gb": result["total_uplink_gb"],
        "update_variance_proxy": upd_var,
        "attack_success_rate_proxy": asr,
        "leakage_score_proxy": leak,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.plots:
        plot_convergence(result["round_logs"], os.path.join(args.outdir, "plots", "convergence_val_loss.png"),
                         ykey="val_loss", title="Federated Convergence (Val Loss)")
        plot_f1(result["round_logs"], os.path.join(args.outdir, "plots", "val_f1.png"),
                title="Federated Validation F1")

    print("Federated summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
