from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from src.utils import set_seed, get_device, ensure_dir
from src.data import load_csv_dataset, make_synthetic_dataset, split_scale, to_sequences
from src.model import TransformerClassifier
from src.train import train_one_client, evaluate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--time-col", type=str, default="")
    p.add_argument("--drop-cols", type=str, default="")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="outputs/centralized")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device("cuda_if_available")

    if args.csv:
        drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else None
        df, _ = load_csv_dataset(args.csv, label_col=args.label_col, time_col=args.time_col or None, drop_cols=drop_cols, seed=args.seed)
    else:
        df, _ = make_synthetic_dataset(n_samples=120000, n_features=32, n_classes=2, seed=args.seed)

    bundle = split_scale(df, label_col=args.label_col, seed=args.seed)

    Xtr, ytr = to_sequences(bundle.X_train, bundle.y_train, args.seq_len)
    Xva, yva = to_sequences(bundle.X_val, bundle.y_val, args.seq_len)
    Xte, yte = to_sequences(bundle.X_test, bundle.y_test, args.seq_len)

    n_features = Xtr.shape[-1]
    n_classes = int(max(ytr.max(), yva.max(), yte.max()) + 1)

    model = TransformerClassifier(
        n_features=n_features, n_classes=n_classes, seq_len=args.seq_len,
        d_model=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.1
    )

    sd, train_stats = train_one_client(
        model, Xtr, ytr, device=device, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size
    )
    model.load_state_dict(sd, strict=True)

    val = evaluate(model, Xva, yva, device=device, batch_size=args.batch_size)
    test = evaluate(model, Xte, yte, device=device, batch_size=args.batch_size)

    ensure_dir(args.outdir)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    summary = {"train": train_stats, "val": val, "test": test, "seq_len": args.seq_len}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Centralized summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
