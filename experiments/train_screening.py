"""Train and evaluate the GNN fast-screening model.

Targets (all from full AC power flow):
  - max_loading: max line loading % (congestion)
  - min_vm: minimum bus voltage pu
  - converged: binary AC-PF feasibility
  - lambda_V: loadability margin (on the margin subset)
  - kappa_C: injection congestion headroom (on the margin subset)

Models: GIN-style GNN (node + edge features), MLP on mean-pooled node
features, linear regression on the same features. 5 seeds. Evaluated on
held-out train scenarios, OOD load regimes (1.45, 1.60), and an N-1
topology. Reports per-target MAE/RMSE, congestion-classification accuracy,
margin Spearman rank correlation, and screening speed vs pandapower PF.

Resume-friendly: per-seed results are saved to results/screening_metrics.json
after each seed; seeds already present are skipped.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool

sys.path.insert(0, os.path.dirname(__file__))
from features import fit_stats, normalize_node, normalize_edge, append_feats

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

NORM = {}


def set_norm(nm, ns, em, es):
    NORM["nm"], NORM["ns"], NORM["em"], NORM["es"] = nm, ns, em, es


SEEDS = [0, 1, 2, 3]
HID = 96
EPOCHS = 70
LR = 3e-3
BATCH = 256


class GNN(nn.Module):
    def __init__(self, n_node_feat, n_edge_feat, hidden=HID, n_out=3):
        super().__init__()
        self.edge_emb = nn.Sequential(
            nn.Linear(n_edge_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        layers = []
        for _ in range(3):
            layers.append(GINEConv(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden)), eps=0.0))
        self.convs = nn.ModuleList(layers)
        self.node_enc = nn.Linear(n_node_feat, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.7)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        x, ei, ef = data.x, data.edge_index, data.edge_attr
        x = self.node_enc(x)
        e = self.edge_emb(ef)
        for conv in self.convs:
            x = conv(x, ei, e) + x
        return self.head(global_add_pool(x, data.batch))


class MLP(nn.Module):
    def __init__(self, n_in, hidden=HID, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def load_graph():
    g = np.load(os.path.join(DATA, "graph_static.npz"))
    return torch.tensor(g["edge_index"]), torch.tensor(g["edge_feat"], dtype=torch.float)


def load_topo_graph():
    """N-1 graph for the OOD topology split.

    The topo split was generated on the base network with the most-loaded
    line removed (the data generator picks it deterministically as line 21,
    buses 15-18, in the intact base solve). The screening GNN must be fed
    this modified graph to exercise structural generalization; the MLP and
    linear baselines receive node features only, which carry no topology
    information at all.
    """
    ei, ef = load_graph()
    ei = ei.numpy()
    ef = ef.numpy()
    keep = [i for i in range(ei.shape[1])
            if set((int(ei[0, i]), int(ei[1, i]))) != {15, 18}]
    assert len(keep) == ei.shape[1] - 1, "expected exactly one edge 15-18"
    return torch.tensor(ei[:, keep]), torch.tensor(ef[keep])


def load_split(name):
    return np.load(os.path.join(DATA, f"screening_{name}.npz"))


def make_dataset(d, edge_index, edge_feat, feats=None):
    """feats: optional [plant_p, regime] appended per scenario."""
    X = np.nan_to_num(d["node_feat"].astype(np.float32), nan=0.0)  # (n, 40, 7)
    if feats is not None:
        X = append_feats(X, feats)  # (n, 40, 9)
    X = normalize_node(X, NORM["nm"], NORM["ns"])
    n = X.shape[0]
    ef = normalize_edge(edge_feat.numpy(), NORM["em"], NORM["es"])
    data_list = []
    for i in range(n):
        data_list.append(Data(x=torch.tensor(X[i]), edge_index=edge_index,
                              edge_attr=torch.tensor(ef)))
    return data_list, X


def feats_of(d):
    return np.stack([d["regime"] / 1.3, d["plant_p"] / 900.0], axis=1)


def flat_feats(d, feats, nm, ns):
    X = np.nan_to_num(d["node_feat"].astype(np.float32), nan=0.0)
    X = append_feats(X, feats)
    return normalize_node(X, nm, ns).reshape(len(d["sid"]), -1)


def train_model(model, data_list, Y, epochs=EPOCHS, batch=BATCH):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(data_list)
    Yt = torch.tensor(Y, dtype=torch.float)
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            b = Batch.from_data_list([data_list[j] for j in idx])
            yb = Yt[idx]
            pred = model(b)
            loss = F.mse_loss(pred[:, :2], yb[:, :2]) + \
                F.binary_cross_entropy_with_logits(pred[:, 2], yb[:, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
    return model


def predict(model, data_list, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(data_list), batch):
            b = Batch.from_data_list(data_list[i:i + batch])
            outs.append(model(b).numpy())
    return np.concatenate(outs, axis=0)


def col_mae(a, b):
    """Per-column MAE, NaN-safe."""
    out = []
    for j in range(a.shape[1]):
        nanm = ~np.isnan(a[:, j])
        out.append(float(np.mean(np.abs(a[nanm, j] - b[nanm, j]))) if nanm.sum() else np.nan)
    return out


def col_rmse(a, b):
    out = []
    for j in range(a.shape[1]):
        nanm = ~np.isnan(a[:, j])
        out.append(float(np.sqrt(np.mean((a[nanm, j] - b[nanm, j]) ** 2))) if nanm.sum() else np.nan)
    return out


def cong_acc(y_ml, p_ml):
    """Accuracy of the >100% loading threshold decision."""
    yb = (y_ml > 100.0).astype(int)
    pb = (p_ml > 100.0).astype(int)
    return float((yb == pb).mean())


def train_model_mlp(model, X, Y, epochs=EPOCHS, batch=256):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    n = X.shape[0]
    Yt = torch.tensor(Y, dtype=torch.float)
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = model(X[idx])
            loss = F.mse_loss(pred[:, :2], Yt[idx][:, :2]) + \
                F.binary_cross_entropy_with_logits(pred[:, 2], Yt[idx][:, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()


def predict_mlp(model, X, batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            outs.append(model(X[i:i + batch]).numpy())
    return np.concatenate(outs, axis=0)


def main():
    edge_index, edge_feat = load_graph()
    topo_ei, topo_ef = load_topo_graph()
    train = load_split("train")
    ood = load_split("ood_load")
    topo = load_split("ood_topo")
    margins = np.load(os.path.join(DATA, "screening_margins.npz"))

    # align margins with train rows
    sid_map = {int(s): i for i, s in enumerate(train["sid"])}
    m_idx = np.array([sid_map[int(s)] for s in margins["sid"]])
    lambda_V = margins["lambda_V"]
    kappa_C = margins["kappa_C"]

    # y targets: [max_loading, min_vm, converged]
    def Y(d):
        ml = np.nan_to_num(d["max_loading"], nan=250.0)
        mv = np.nan_to_num(d["min_vm"], nan=0.5)
        cv = d["converged"].astype(np.float32)
        return np.stack([ml, mv, cv], axis=1)

    Ytr, Yood, Ytopo = Y(train), Y(ood), Y(topo)
    # NaN masks for reporting only (targets are NaN for divergent scenarios)
    ml_nan = np.isnan(train["max_loading"]).astype(int)

    # ---- timing: pandapower PF per scenario (wall clock) ----
    import pandapower as pp
    import pandapower.networks as pn
    net = pn.case39()
    t0 = time.perf_counter()
    for _ in range(20):
        pp.runpp(net, numba=False)
    t1 = time.perf_counter()
    pf_ms = (t1 - t0) / 20 * 1000

    feats_tr = feats_of(train).astype(np.float32)
    feats_ood = feats_of(ood).astype(np.float32)
    feats_topo = feats_of(topo).astype(np.float32)
    nm, ns, em, es = fit_stats(train["node_feat"], feats_tr, edge_feat.numpy())
    set_norm(nm, ns, em, es)

    Xm_tr_all = flat_feats(train, feats_tr, nm, ns)
    Xm_ood = flat_feats(ood, feats_ood, nm, ns)
    Xm_topo = flat_feats(topo, feats_topo, nm, ns)

    # resume: load rows already computed
    json_path = os.path.join(OUT, "screening_metrics.json")
    rows = []
    if os.path.exists(json_path):
        rows = json.load(open(json_path))
        print(f"[resume] {len(rows)} seeds already done", flush=True)
    done_seeds = {r["seed"] for r in rows}

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"[skip] seed {seed} (already done)", flush=True)
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        tr_list, _ = make_dataset(train, edge_index, edge_feat, feats_tr)
        od_list, _ = make_dataset(ood, edge_index, edge_feat, feats_ood)
        tp_list, _ = make_dataset(topo, topo_ei, topo_ef, feats_topo)
        # hold out 20% of train for in-distribution eval
        n = len(tr_list)
        perm = np.random.permutation(n)
        n_ho = n // 5
        ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]
        tr_list_ho = [tr_list[j] for j in tr_idx]
        ho_list = [tr_list[j] for j in ho_idx]

        gnn = GNN(9, 6)
        train_model(gnn, tr_list_ho, Ytr[tr_idx])
        t0 = time.perf_counter()
        p_ho = predict(gnn, ho_list)
        p_ood = predict(gnn, od_list)
        p_topo = predict(gnn, tp_list)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) / (len(ho_list) + len(od_list) + len(tp_list)) * 1000

        mlp = MLP(40 * 9)
        train_model_mlp(mlp, torch.tensor(Xm_tr_all[tr_idx]), Ytr[tr_idx])
        p_mlp_ho = predict_mlp(mlp, torch.tensor(Xm_tr_all[ho_idx]))
        p_mlp_ood = predict_mlp(mlp, torch.tensor(Xm_ood))
        p_mlp_topo = predict_mlp(mlp, torch.tensor(Xm_topo))

        from sklearn.linear_model import LinearRegression
        lin = LinearRegression().fit(Xm_tr_all[tr_idx], Ytr[tr_idx])
        p_lin_ho = lin.predict(Xm_tr_all[ho_idx])
        p_lin_ood = lin.predict(Xm_ood)
        p_lin_topo = lin.predict(Xm_topo)

        def m(Yt, p):
            return dict(
                ml_mae=col_mae(Yt, p)[0], ml_rmse=col_rmse(Yt, p)[0],
                v_mae=col_mae(Yt, p)[1], v_rmse=col_rmse(Yt, p)[1],
                acc=cong_acc(Yt[:, 0], p[:, 0]))

        g_ho, g_ood, g_topo = m(Ytr[ho_idx], p_ho), m(Yood, p_ood), m(Ytopo, p_topo)
        mlp_ho, mlp_ood, mlp_topo = m(Ytr[ho_idx], p_mlp_ho), m(Yood, p_mlp_ood), m(Ytopo, p_mlp_topo)
        lin_ho, lin_ood, lin_topo = m(Ytr[ho_idx], p_lin_ho), m(Yood, p_lin_ood), m(Ytopo, p_lin_topo)

        # margins: held-out split of the margin subset (no leakage)
        from scipy.stats import spearmanr
        rng_m = np.random.RandomState(seed)
        perm_m = rng_m.permutation(len(m_idx))
        n_m_ho = max(1, len(m_idx) // 5)
        m_ho_pos = perm_m[:n_m_ho]
        m_tr_pos = perm_m[n_m_ho:]
        m_tr_list = [tr_list[j] for j in m_idx[m_tr_pos]]
        m_ho_list = [tr_list[j] for j in m_idx[m_ho_pos]]
        gnn_m = GNN(9, 6)
        Ym_tr = np.stack([lambda_V[m_tr_pos], kappa_C[m_tr_pos],
                          Ytr[m_idx[m_tr_pos]][:, 2]], axis=1)
        train_model(gnn_m, m_tr_list, Ym_tr, epochs=50)
        p_m = predict(gnn_m, m_ho_list)
        lam_ho = lambda_V[m_ho_pos]
        kap_ho = kappa_C[m_ho_pos]
        rho_V = float(spearmanr(lam_ho, p_m[:, 0]).statistic)
        rho_C = float(spearmanr(kap_ho, p_m[:, 1]).statistic)
        rmse_V = float(np.sqrt(np.mean((lam_ho - p_m[:, 0]) ** 2)))
        rmse_C = float(np.sqrt(np.mean((kap_ho - p_m[:, 1]) ** 2)))
        np.savez(os.path.join(OUT, f"margin_preds_seed{seed}.npz"),
                 lam_true=lam_ho, lam_pred=p_m[:, 0],
                 kap_true=kap_ho, kap_pred=p_m[:, 1])

        rows.append(dict(
            seed=seed,
            ho=g_ho, ood=g_ood, topo=g_topo,
            mlp_ho=mlp_ho, mlp_ood=mlp_ood, mlp_topo=mlp_topo,
            lin_ho=lin_ho, lin_ood=lin_ood, lin_topo=lin_topo,
            rho_V=rho_V, rho_C=rho_C, rmse_V=rmse_V, rmse_C=rmse_C,
            infer_ms=infer_ms, pf_ms=pf_ms,
            ho_div=float(ml_nan[ho_idx].mean()), ood_div=float(np.isnan(ood["max_loading"]).mean()),
        ))
        print(f"seed {seed}: ho_ml_mae={g_ho['ml_mae']:.2f}% ood={g_ood['ml_mae']:.2f}% "
              f"topo={g_topo['ml_mae']:.2f}% | mlp_ood={mlp_ood['ml_mae']:.2f}% "
              f"lin_ood={lin_ood['ml_mae']:.2f}% | rho_V={rho_V:.3f} rho_C={rho_C:.3f} "
              f"| acc ho={g_ho['acc']:.3f} topo={g_topo['acc']:.3f}", flush=True)

        # incremental checkpoint
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

        if seed == 0:
            np.savez(os.path.join(OUT, "screening_preds_seed0.npz"),
                     ho_ml_true=Ytr[ho_idx][:, 0], ho_ml_pred=p_ho[:, 0],
                     ho_v_true=Ytr[ho_idx][:, 1], ho_v_pred=p_ho[:, 1],
                     ood_ml_true=Yood[:, 0], ood_ml_pred=p_ood[:, 0],
                     topo_ml_true=Ytopo[:, 0], topo_ml_pred=p_topo[:, 0],
                     m_lambda=lam_ho, m_kappa=kap_ho,
                     m_pred_lambda=p_m[:, 0], m_pred_kappa=p_m[:, 1],
                     ho_conv_true=Ytr[ho_idx][:, 2], ho_conv_logit=p_ho[:, 2],
                     ood_conv_true=Yood[:, 2], ood_conv_logit=p_ood[:, 2],
                     topo_conv_true=Ytopo[:, 2], topo_conv_logit=p_topo[:, 2])

    print("saved screening_metrics.json", flush=True)


if __name__ == "__main__":
    main()
