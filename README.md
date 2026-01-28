# Federated Transformer Networks for DDoS Detection (Python Reference Implementation)

This repository provides a **research-oriented** Python implementation of a **federated Transformer** for DDoS detection in smart-city IoT settings.  
It supports:

- Centralized training (upper-bound reference)
- Federated learning with **FedAvg**
- Client partitioning under **IID** and **non-IID (Dirichlet label-skew)** distributions
- Participation rate control (partial participation)
- Communication cost estimates, convergence logging, and simple privacy/stability proxies
- Plot generation for results-style figures

> ✅ The code runs end-to-end with **synthetic data** by default.  
> ✅ You can also use your own CSV dataset (e.g., CICIoT2023 / ToN-IoT exports) by providing paths and column names.

---

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2) Quickstart (Synthetic Data)

### Centralized (upper-bound reference)
```bash
python scripts/run_centralized.py --outdir outputs/centralized_demo
```

### Federated (IID)
```bash
python scripts/run_federated.py --scenario iid --outdir outputs/fed_iid_demo
```

### Federated (non-IID Dirichlet label-skew)
```bash
python scripts/run_federated.py --scenario noniid --alpha 0.3 --outdir outputs/fed_noniid_demo
```

---

## 3) Using Your Own Dataset (CSV)

Your CSV should contain **numeric features** and a **label** column.

Example:
```bash
python scripts/run_federated.py \
  --csv /path/to/your_dataset.csv \
  --label-col label \
  --scenario noniid \
  --alpha 0.3 \
  --clients 10 \
  --rounds 50 \
  --participation 0.6 \
  --seq-len 32 \
  --outdir outputs/custom_run
```

### Notes
- If you have a timestamp column, you can pass `--time-col` to preserve order before sequence windowing.
- If your dataset is already sequence-form, you can disable windowing by setting `--seq-len 1`.

---

## 4) Outputs

Each run writes into `--outdir`:
- `metrics.json` (summary)
- `round_metrics.csv` (per-round logs)
- `model.pt` (final global model)
- plots in `plots/` (when enabled)

---

## 5) Repository Layout

- `src/data.py` — loading, preprocessing, sequence windowing, IID/non-IID partitioning
- `src/model.py` — Transformer classifier + positional encoding
- `src/train.py` — training + evaluation utilities
- `src/fl.py` — FedAvg loop + participation + aggregation
- `src/metrics.py` — F1/acc + stability/communication/privacy proxies
- `src/plots.py` — Matplotlib plots for paper-style figures
- `scripts/*.py` — runnable entry points

---

## 6) Reproducibility

All scripts accept `--seed`.  
Set `--runs N` for repeated runs to estimate variance.

---

## Disclaimer

This is a **reference research implementation** designed to mirror the experimental flow in the manuscript.  
Exact performance numbers depend on dataset feature engineering and label definitions.
