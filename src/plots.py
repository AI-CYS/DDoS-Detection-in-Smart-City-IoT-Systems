from __future__ import annotations
from typing import List, Dict
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(round_logs: List[Dict], outpath: str, ykey: str = "val_loss", title: str = "Convergence") -> None:
    xs = [r["round"] for r in round_logs]
    ys = [r[ykey] for r in round_logs]
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Communication Rounds")
    plt.ylabel(ykey.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_f1(round_logs: List[Dict], outpath: str, title: str = "Validation F1") -> None:
    xs = [r["round"] for r in round_logs]
    ys = [r["val_f1"] for r in round_logs]
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Communication Rounds")
    plt.ylabel("F1-score")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_participation_sweep(results: Dict[float, float], outpath: str, title: str = "Participation vs F1") -> None:
    ps = sorted(results.keys())
    fs = [results[p] for p in ps]
    plt.figure()
    plt.plot(ps, fs, marker="o")
    plt.xlabel("Participation Rate")
    plt.ylabel("Final Test F1")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
