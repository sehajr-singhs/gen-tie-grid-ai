"""Regenerate every figure in the paper from committed result files.

Body figures:
  fig1_screening.pdf  - screening scatter + margin scatter + congestion ROC
  fig2_margin3d.pdf   - loadability margin over (plant output, load regime)
  fig3_control.pdf    - closed-loop control outcomes vs baselines + oracle
Appendix figures:
  fig4_ood_speed.pdf  - OOD degradation bars + screening speed
  fig5_failures.pdf   - screening failure cases and control failure cases
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Physical-Intelligence-style palette: one accent + neutrals
ACCENT = "#1a5fb4"
ACCENT2 = "#c01c28"
GRAY = "#5c6470"
LGRAY = "#c8cdd4"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_preds():
    p = np.load(os.path.join(RESULTS, "screening_preds_seed0.npz"))
    m = np.load(os.path.join(DATA, "screening_margins.npz"))
    return p, m


def fig1_screening():
    p, _ = load_preds()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    ax = axes[0]
    t, y = p["ho_ml_true"], p["ho_ml_pred"]
    ax.scatter(t, y, s=4, c=ACCENT, alpha=0.35, rasterized=True)
    lim = [min(t.min(), y.min()) - 5, max(t.max(), y.max()) + 5]
    ax.plot(lim, lim, color=GRAY, lw=0.8, ls="--")
    ax.axhline(100, color=GRAY, lw=0.6, ls=":")
    ax.axvline(100, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel("true max line loading (%)")
    ax.set_ylabel("GNN prediction (%)")
    ax.set_title("(a) congestion")

    ax = axes[1]
    t, y = p["m_lambda"], p["m_pred_lambda"]
    ax.scatter(t, y, s=4, c=ACCENT, alpha=0.4, rasterized=True)
    lim = [min(t.min(), y.min()), max(t.max(), y.max())]
    ax.plot(lim, lim, color=GRAY, lw=0.8, ls="--")
    ax.axvspan(0, 0.2, color=LGRAY, alpha=0.4)
    ax.set_xlabel("true loadability margin $\\lambda$")
    ax.set_ylabel("GNN prediction")
    ax.set_title("(b) voltage-stability headroom")

    ax = axes[2]
    # congestion screening: label = any line above 100% loading
    ct = p["ho_ml_true"] > 100.0
    s = p["ho_ml_pred"]
    pos = s[ct]
    neg = s[~ct]
    if len(pos) and len(neg):
        thresh = np.sort(np.unique(np.concatenate([pos, neg])))
        tpr = np.array([(pos >= th).mean() for th in thresh])
        fpr = np.array([(neg >= th).mean() for th in thresh])
        ax.plot(fpr, tpr, color=ACCENT, lw=1.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("(c) congestion screening ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(FIG, "fig1_screening.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig2_margin3d():
    """Loadability margin as a 3-D surface over (plant output, load regime)."""
    m = np.load(os.path.join(DATA, "screening_margins.npz"))
    tr = np.load(os.path.join(DATA, "screening_train.npz"))
    sid_map = {int(s): i for i, s in enumerate(tr["sid"])}
    idx = np.array([sid_map[int(s)] for s in m["sid"]])
    P = tr["plant_p"][idx]
    G = tr["regime"][idx]
    lam = m["lambda_V"]

    fig = plt.figure(figsize=(3.4, 2.9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P, G, lam, c=lam, cmap="YlGnBu_r", s=9,
                    alpha=0.85, edgecolors="none")
    ax.set_xlabel("plant output (MW)")
    ax.set_ylabel("load regime")
    ax.set_zlabel("margin $\\lambda$")
    ax.set_zlim(0, max(lam.max(), 0.1))
    ax.view_init(elev=24, azim=-132)
    ax.tick_params(pad=-2)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08, label="$\\lambda$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_margin3d.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig3_control():
    c = np.load(os.path.join(DATA, "control_data.npz"))
    met = json.load(open(os.path.join(RESULTS, "control_metrics.json")))
    pol = met["policy"][0]
    base = met["baselines"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    # (a) learned Q vs oracle Q (held-out, seed 0)
    ax = axes[0]
    p = np.load(os.path.join(RESULTS, "control_preds_seed0.npz"))
    mask = p["names"] == "ho"
    ho = np.where(mask)[0]
    qmax_ho = p["qmax"][ho]
    q_star_n = p["q_star"][ho] / qmax_ho
    q_ml_n = p["q_ml"][ho] / qmax_ho  # Q applied in the closed loop, normalized
    ax.scatter(q_star_n, q_ml_n, s=4, c=ACCENT, alpha=0.4, rasterized=True)
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("oracle reactive setpoint $Q^\\star/Q_{\\max}$")
    ax.set_ylabel("learned policy $Q/Q_{\\max}$")
    ax.set_title("(a) reactive dispatch match")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # (b) closed-loop objective components per policy
    ax = axes[1]
    labels = ["no control", "volt-var", "fixed PF", "learned", "oracle"]
    keys = ["base", "droop", "fixed_pf", "learned", "oracle"]
    cong = [base[k]["ho"]["cong"] for k in ["base", "droop", "fixed_pf"]] + \
        [pol["ho_cong"], base["oracle"]["ho"]["cong"]]
    vd = [base[k]["ho"]["vd"] for k in ["base", "droop", "fixed_pf"]] + \
        [pol["ho_v"], base["oracle"]["ho"]["vd"]]
    x = np.arange(len(labels))
    w = 0.28
    b1 = ax.bar(x - w, cong, w, color=ACCENT, label="congestion rate")
    b2 = ax.bar(x, vd, w, color=ACCENT2, label="voltage deviation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=6.5)
    ax.set_ylabel("held-out scenarios")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(b) system-level effect in the loop")

    # (c) curtailment vs screening signal
    ax = axes[2]
    z = p["zeta"][ho]
    r = 1.0 - p["r_ml"][ho]
    ax.scatter(z, r, s=4, c=ACCENT, alpha=0.4, rasterized=True)
    ax.set_xlabel("screening signal $\\zeta$ (predicted max loading, %)")
    ax.set_ylabel("curtailment $1-r$")
    ax.set_title("(c) curtailment driven by the signal")
    ax.set_ylim(-0.03, 1.03)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(FIG, "fig3_control.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig4_ood_speed():
    met = json.load(open(os.path.join(RESULTS, "screening_metrics.json")))
    seeds = met
    def col(key):
        v = np.array([m[key]["ml_rmse"] for m in seeds])
        return v.mean(), v.std()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))

    ax = axes[0]
    labels = ["in-dist", "OOD load", "N-1 topo"]
    gnn = [col("ho"), col("ood"), col("topo")]
    mlp = [col("mlp_ho"), col("mlp_ood"), col("mlp_topo")]
    lin = [col("lin_ho"), col("lin_ood"), col("lin_topo")]
    x = np.arange(3)
    w = 0.26
    for xi, (g, c, lab) in enumerate([(gnn, ACCENT, "GNN"),
                                      (mlp, "#e5a50a", "MLP"),
                                      (lin, LGRAY, "linear")]):
        off = (xi - 1) * w
        ax.bar(x + off, [v[0] for v in g], w, yerr=[v[1] for v in g],
               color=c, label=lab, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE, max line loading (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("(a) out-of-distribution degradation (5 seeds)")

    ax = axes[1]
    pf_ms = seeds[0]["pf_ms"]
    inf_ms = seeds[0]["infer_ms"]
    bars = ax.bar([0, 1], [pf_ms, inf_ms], 0.5,
                  color=[GRAY, ACCENT])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"AC power flow\n({pf_ms:.0f} ms)", f"GNN screening\n({inf_ms:.2f} ms)"])
    ax.set_ylabel("wall-clock per scenario (ms)")
    ax.set_yscale("log")
    ax.set_title("(b) screening speed")
    ax.annotate(f"${pf_ms/inf_ms:.0f}\\times$", xy=(0.5, (pf_ms * inf_ms) ** 0.5),
                ha="center", fontsize=8, color=GRAY)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(FIG, "fig4_ood_speed.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig5_failures():
    p, _ = load_preds()
    met = json.load(open(os.path.join(RESULTS, "screening_metrics.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))

    ax = axes[0]
    err = np.abs(p["ho_ml_pred"] - p["ho_ml_true"])
    order = np.argsort(err)[-12:]
    ax.barh(np.arange(12), err[order], color=ACCENT)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels([f"#{i}" for i in order], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("absolute error, max loading (%)")
    ax.set_title("(a) worst held-out screening errors")

    ax = axes[1]
    c = np.load(os.path.join(DATA, "control_data.npz"))
    cp = np.load(os.path.join(RESULTS, "control_preds_seed0.npz"))
    ho = np.where(cp["names"] == "ho")[0]
    # J components for learned vs droop
    tl = c["base_tl"][cp["ev"][ho]]
    def j_of(ml, v, loss, tl):
        if np.isnan(ml):
            return np.inf
        return max(0.0, ml - 100.0) / 100.0 + max(0.0, abs(v - 1.0) - 0.012) / 0.1 + 0.5 * loss / tl
    j_ml = np.array([j_of(cp["ml_ml"][ho][k], cp["v_ml"][ho][k], cp["loss_ml"][ho][k], tl[k])
                     for k in range(len(ho))])
    j_dr = np.array([j_of(c["droop_ml"][cp["ev"][ho][k]], c["droop_v"][cp["ev"][ho][k]],
                          c["droop_loss"][cp["ev"][ho][k]], tl[k]) for k in range(len(ho))])
    worse = j_ml > j_dr + 1e-9
    ax.scatter(j_dr, j_ml, s=5, c=np.where(worse, ACCENT2, ACCENT),
               alpha=0.5, rasterized=True)
    lim = [0, max(np.nanmax(j_dr), np.nanmax(j_ml)) * 1.05]
    ax.plot(lim, lim, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("volt-var objective $J$")
    ax.set_ylabel("learned objective $J$")
    ax.set_title(f"(b) learned vs droop, {worse.mean()*100:.1f}\\% worse")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(FIG, "fig5_failures.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_screening()
    fig2_margin3d()
    fig3_control()
    fig4_ood_speed()
    fig5_failures()
    print("figures written to", FIG)
