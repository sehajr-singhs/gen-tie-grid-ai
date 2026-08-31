"""Print every number the paper needs, from the committed result JSONs.

Run after screening_metrics.json and control_metrics.json exist.
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")

print("=" * 70)
print("SCREENING")
print("=" * 70)
rows = json.load(open(os.path.join(RESULTS, "screening_metrics.json")))


def agg(r, key, field):
    v = np.array([m[key][field] for m in r])
    return v.mean(), v.std()


def fmt(v, nd=1):
    return f"{v[0]:.{nd}f}+-{v[1]:.{nd}f}"


print(f"n seeds: {len(rows)}")
for key in ["ho", "ood", "topo"]:
    print(f"GNN {key}: ml_mae={fmt(agg(rows, key, 'ml_mae'))} "
          f"ml_rmse={fmt(agg(rows, key, 'ml_rmse'))} "
          f"acc={agg(rows, key, 'acc')[0]:.3f} "
          f"v_mae={fmt(agg(rows, key, 'v_mae'), 3)}")
for key in ["mlp_ho", "mlp_ood", "mlp_topo"]:
    print(f"MLP {key}: ml_rmse={fmt(agg(rows, key, 'ml_rmse'))} "
          f"acc={agg(rows, key, 'acc')[0]:.3f}")
for key in ["lin_ho", "lin_ood", "lin_topo"]:
    print(f"LIN {key}: ml_rmse={fmt(agg(rows, key, 'ml_rmse'))}")

# pooled margins
lt = np.concatenate([np.load(os.path.join(RESULTS, f"margin_preds_seed{s}.npz"))["lam_true"]
                     for s in [r["seed"] for r in rows]])
lp = np.concatenate([np.load(os.path.join(RESULTS, f"margin_preds_seed{s}.npz"))["lam_pred"]
                     for s in [r["seed"] for r in rows]])
kt = np.concatenate([np.load(os.path.join(RESULTS, f"margin_preds_seed{s}.npz"))["kap_true"]
                     for s in [r["seed"] for r in rows]])
kp = np.concatenate([np.load(os.path.join(RESULTS, f"margin_preds_seed{s}.npz"))["kap_pred"]
                     for s in [r["seed"] for r in rows]])
print(f"margins pooled n={len(lt)}: rho_V={spearmanr(lt, lp).statistic:.3f} "
      f"rho_C={spearmanr(kt, kp).statistic:.3f} "
      f"rmse_V={np.sqrt(np.mean((lt-lp)**2)):.3f} rmse_C={np.sqrt(np.mean((kt-kp)**2)):.3f}")
infer = np.array([r["infer_ms"] for r in rows])
print(f"infer_ms={infer.mean():.2f}+-{infer.std():.2f} pf_ms={rows[0]['pf_ms']:.0f}")

print()
print("=" * 70)
print("CONTROL")
print("=" * 70)
c = json.load(open(os.path.join(RESULTS, "control_metrics.json")))
pol = c["policy"]
base = c["baselines"]
n = len(pol)
print(f"n seeds: {n}")
for k in ["ho_J", "ood_J", "ho_cong", "ood_cong", "ho_v", "ood_v",
          "ho_loss", "ood_loss", "ho_curt", "ood_curt"]:
    v = np.array([r[k] for r in pol])
    print(f"{k}: {v.mean():.4f}+-{v.std():.4f}")
print("baselines (held-out / OOD):")
for k in ["base", "droop", "fixed_pf", "oracle"]:
    b = base[k]
    print(f"  {k}: ho J={b['ho']['J']:.4f} cong={b['ho']['cong']:.3f} vd={b['ho']['vd']:.4f} "
          f"| ood J={b['ood']['J']:.4f} cong={b['ood']['cong']:.3f} vd={b['ood']['vd']:.4f}")
