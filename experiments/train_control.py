"""Learned local inverter controller: policy pi(V_pcc, P, zeta) -> (Q, curtail).

zeta = GNN-predicted max line loading (the screening module's broadcast
signal). Training targets are the dense-scan oracle (Q*, P*) from
control_data.npz. Evaluation closes the loop: policy applied to measured
PCC voltage, full AC power flow re-solved to a fixed point, compared
against IEEE 1547 volt-var droop, fixed 0.95 power factor, and the oracle,
on held-out and OOD regimes.
"""
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandapower as pp
import pandapower.networks as pn
from scipy.stats import spearmanr
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.dirname(__file__))
from features import fit_stats, normalize_node, normalize_edge, append_feats

NORM = {}


def set_norm(nm, ns, em, es):
    NORM["nm"], NORM["ns"], NORM["em"], NORM["es"] = nm, ns, em, es

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
CONN_BUS = 3
PLANT_MAX_MW = 900.0
PLANT_MVA = 1000.0
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 200
LR = 1e-3
HID = 64


def qmax(p):
    return float(np.sqrt(max(0.0, PLANT_MVA ** 2 - p ** 2)))


# ---------------- screening GNN (produces the zeta signal) ----------------
class GNN(nn.Module):
    def __init__(self, n_node_feat, n_edge_feat, hidden=128, n_out=3):
        super().__init__()
        from torch_geometric.nn import GINEConv, global_add_pool
        self._pool = global_add_pool
        self.edge_emb = nn.Sequential(
            nn.Linear(n_edge_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden)), eps=0.0)
            for _ in range(3)])
        self.node_enc = nn.Linear(n_node_feat, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_out))

    def forward(self, data):
        x = self.node_enc(data.x)
        e = self.edge_emb(data.edge_attr)
        for conv in self.convs:
            x = conv(x, data.edge_index, e) + x
        return self.head(self._pool(x, data.batch))


def _feats(regime, plant_p):
    return np.stack([regime / 1.3, plant_p / 900.0], axis=1).astype(np.float32)


def train_screening_gnn():
    """Train a screening GNN (max-loading head) on the screening dataset."""
    g = np.load(os.path.join(DATA, "graph_static.npz"))
    ei = torch.tensor(g["edge_index"])
    ef = torch.tensor(g["edge_feat"], dtype=torch.float)
    tr = np.load(os.path.join(DATA, "screening_train.npz"))
    feats = _feats(tr["regime"], tr["plant_p"])
    nm, ns, em, es = fit_stats(tr["node_feat"], feats, ef.numpy())
    set_norm(nm, ns, em, es)
    X = normalize_node(append_feats(tr["node_feat"], feats), nm, ns)
    n = X.shape[0]
    ef_n = normalize_edge(ef.numpy(), em, es)
    y = np.nan_to_num(tr["max_loading"], nan=250.0).astype(np.float32)
    data_list = [Data(x=torch.tensor(X[i]), edge_index=ei,
                      edge_attr=torch.tensor(ef_n)) for i in range(n)]
    model = GNN(9, 6)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    n_ho = n // 5
    perm = np.random.RandomState(0).permutation(n)
    ho, tr_idx = perm[:n_ho], perm[n_ho:]
    yt = torch.tensor(y)
    for ep in range(100):
        p2 = torch.randperm(len(tr_idx))
        model.train()
        for i in range(0, len(tr_idx), 256):
            idx = tr_idx[p2[i:i + 256]]
            b = Batch.from_data_list([data_list[j] for j in idx])
            opt.zero_grad()
            loss = F.mse_loss(model(b)[:, 0], yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model, ei, ef_n


def predict_zeta(model, ei, ef, node_feat, plant_p, regime):
    """GNN max-loading prediction for control scenarios."""
    X = normalize_node(append_feats(node_feat, _feats(regime, plant_p)),
                       NORM["nm"], NORM["ns"])
    n = X.shape[0]
    dl = [Data(x=torch.tensor(X[i]), edge_index=ei,
               edge_attr=torch.tensor(ef)) for i in range(n)]
    out = []
    with torch.no_grad():
        for i in range(0, n, 512):
            b = Batch.from_data_list(dl[i:i + 512])
            out.append(model(b)[:, 0].numpy())
    return np.concatenate(out)


# ---------------- inverter policy ----------------
class Policy(nn.Module):
    def __init__(self, hidden=HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))  # [q_rel (tanh), curtail (sigmoid)]

    def forward(self, x):
        h = self.net(x)
        return torch.tanh(h[:, 0]), torch.sigmoid(h[:, 1])


def train_policy(policy, X, yq, yr, epochs=EPOCHS):
    opt = torch.optim.Adam(policy.parameters(), lr=LR, weight_decay=1e-5)
    yq_t = torch.tensor(yq, dtype=torch.float)
    yr_t = torch.tensor(yr, dtype=torch.float)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        policy.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            q, r = policy(X[idx])
            loss = F.mse_loss(q, yq_t[idx]) + 0.5 * F.mse_loss(r, yr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


def rebuild_net(load_mults, gen_mults, plant_p):
    net = pn.case39()
    net.bus["max_vm_pu"] = 1.10
    net.bus["min_vm_pu"] = 0.90
    pb = pp.create_bus(net, name="plant", vn_kv=230.0, type="b")
    pp.create_line_from_parameters(
        net, pb, CONN_BUS, length_km=15.0, r_ohm_per_km=0.06,
        x_ohm_per_km=0.32, c_nf_per_km=12.0, max_i_ka=3.2,
        type="ol", name="gen-tie")
    # PQ injection: a gen with vm_pu set would hold voltage and ignore Q
    pp.create_sgen(net, pb, p_mw=plant_p, q_mvar=0.0, name="plant")
    net.load["p_mw"] *= load_mults
    net.load["q_mvar"] *= load_mults
    idx = net.gen.index
    net.gen.loc[idx, "p_mw"] *= gen_mults
    net.gen["vm_pu"] = 1.0
    return net


def closed_loop(policy, load_mults, gen_mults, plant_p, zeta, n_iter=3):
    """Apply pi(v_measured, P, zeta) -> (Q, r); re-solve AC PF to fixed point."""
    net = rebuild_net(load_mults, gen_mults, plant_p)
    pi = net.sgen.index[net.sgen.name == "plant"][0]
    v = 1.0
    q, r = 0.0, 1.0
    for _ in range(n_iter):
        with torch.no_grad():
            x = torch.tensor([[v, plant_p / PLANT_MAX_MW, zeta / 250.0]],
                             dtype=torch.float)
            qn, rn = policy(x)
            q = float(qn) * qmax(plant_p)
            r = float(rn)
        net.sgen.at[pi, "p_mw"] = plant_p * (0.2 + 0.8 * r)
        net.sgen.at[pi, "q_mvar"] = q
        try:
            pp.runpp(net, numba=False, max_iteration=20)
        except Exception:
            net.converged = False
        if not net.converged:
            return dict(converged=False, max_loading=np.nan, v_pcc=np.nan,
                        losses=np.nan, q=q, r=r, p=plant_p * (0.2 + 0.8 * r))
        v = float(net.res_bus.at[CONN_BUS, "vm_pu"])
    ml = float(net.res_line["loading_percent"].max())
    losses = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum())
    return dict(converged=True, max_loading=ml, v_pcc=v, losses=losses,
                q=q, r=r, p=plant_p * (0.2 + 0.8 * r))


def J(o, total_load):
    if not o["converged"]:
        return np.inf
    cong = max(0.0, o["max_loading"] - 100.0) / 100.0
    volt = max(0.0, abs(o["v_pcc"] - 1.0) - 0.012) / 0.1
    loss = o["losses"] / total_load
    return cong + volt + 0.5 * loss


def main():
    c = np.load(os.path.join(DATA, "control_data.npz"))
    n = len(c["sid"])
    is_ood = c["is_ood"].astype(bool)
    train_idx = np.where(~is_ood)[0]
    ood_idx = np.where(is_ood)[0]

    # zeta signal from the screening GNN (out-of-sample for control scenarios)
    model, ei, ef = train_screening_gnn()
    zeta = predict_zeta(model, ei, ef, c["node_feat"], c["plant_p"], c["regime"])
    zeta_true = c["base_ml"]

    # policy training inputs / targets
    v_in = c["base_v"].astype(np.float32)
    p_in = (c["plant_p"] / PLANT_MAX_MW).astype(np.float32)
    yq = (c["q_star"] / c["qmax"]).astype(np.float32)
    yr = (c["p_star"] / c["plant_p"]).astype(np.float32)
    tl = c["base_tl"]

    json_path = os.path.join(OUT, "control_metrics.json")
    rows = []
    if os.path.exists(json_path):
        rows = json.load(open(json_path))["policy"]
        print(f"[resume] {len(rows)} seeds already done", flush=True)
    done_seeds = {r["seed"] for r in rows}

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"[skip] seed {seed} (already done)", flush=True)
            continue
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_idx))
        n_ho = len(train_idx) // 5
        ho = train_idx[perm[:n_ho]]
        tr = train_idx[perm[n_ho:]]

        Xtr = np.stack([v_in[tr], p_in[tr], zeta[tr] / 250.0], axis=1)
        Xho = np.stack([v_in[ho], p_in[ho], zeta[ho] / 250.0], axis=1)
        Xood = np.stack([v_in[ood_idx], p_in[ood_idx], zeta[ood_idx] / 250.0], axis=1)

        policy = Policy()
        train_policy(policy, torch.tensor(Xtr), yq[tr], yr[tr])

        # closed-loop evaluation of the learned policy (parallel)
        ev = list(ho) + list(ood_idx)
        names = ["ho"] * len(ho) + ["ood"] * len(ood_idx)
        from multiprocessing import Pool
        args = [(policy, c["load_mults"][i], c["gen_mults"][i],
                 c["plant_p"][i], zeta[i]) for i in ev]
        with Pool(8) as pool:
            ml_out = pool.starmap(closed_loop, args, chunksize=16)
        ho_js, ood_js, ho_cong, ood_cong = [], [], [], []
        ho_v, ood_v, ho_loss, ood_loss, ho_curt, ood_curt = [], [], [], [], [], []
        for i, nm, o in zip(ev, names, ml_out):
            j = J(o, tl[i]) if o["converged"] else np.inf
            cong = 1.0 if (o["converged"] and o["max_loading"] > 100) else 0.0
            vd = abs(o["v_pcc"] - 1.0) if o["converged"] else 0.5
            loss = o["losses"] if o["converged"] else np.nan
            curt = 1.0 - o["r"]
            (ho_js if nm == "ho" else ood_js).append(j)
            (ho_cong if nm == "ho" else ood_cong).append(cong)
            (ho_v if nm == "ho" else ood_v).append(vd)
            (ho_loss if nm == "ho" else ood_loss).append(loss)
            (ho_curt if nm == "ho" else ood_curt).append(curt)
        rows.append(dict(seed=seed,
                         ho_J=np.nanmean(ho_js), ood_J=np.nanmean(ood_js),
                         ho_cong=np.mean(ho_cong), ood_cong=np.mean(ood_cong),
                         ho_v=np.nanmean(ho_v), ood_v=np.nanmean(ood_v),
                         ho_loss=np.nanmean(ho_loss), ood_loss=np.nanmean(ood_loss),
                         ho_curt=np.nanmean(ho_curt), ood_curt=np.nanmean(ood_curt)))
        print(f"seed {seed}: ho_J={rows[-1]['ho_J']:.4f} ood_J={rows[-1]['ood_J']:.4f} "
              f"ho_cong={rows[-1]['ho_cong']:.3f} ood_cong={rows[-1]['ood_cong']:.3f}",
              flush=True)
        with open(json_path, "w") as f:
            json.dump(dict(policy=rows), f, indent=2)

        if seed == 0:
            np.savez(os.path.join(OUT, "control_preds_seed0.npz"),
                     ev=np.array(ev, dtype=int),
                     names=np.array(names),
                     ml_ml=np.array([o["max_loading"] for o in ml_out]),
                     v_ml=np.array([o["v_pcc"] for o in ml_out]),
                     loss_ml=np.array([o["losses"] for o in ml_out]),
                     r_ml=np.array([o["r"] for o in ml_out]),
                     q_ml=np.array([o["q"] for o in ml_out]),
                     zeta=zeta[ev], zeta_true=zeta_true[ev],
                     plant_p=c["plant_p"][ev],
                     base_ml=c["base_ml"][ev],
                     droop_ml=c["droop_ml"][ev],
                     q_star=c["q_star"][ev],
                     qmax=c["qmax"][ev],
                     v_base=c["base_v"][ev])

    # seed-independent baselines on a fixed held-out split + OOD, reported
    # separately so they are comparable to the policy's per-split numbers
    ho_fixed = train_idx[np.random.RandomState(0).permutation(len(train_idx))[:len(train_idx) // 5]]
    droop_o = dict(ml=c["droop_ml"], v=c["droop_v"], loss=c["droop_loss"])
    fixed_o = dict(ml=c["fixed_ml"], v=c["fixed_v"], loss=c["fixed_loss"])
    orac_o = dict(ml=c["oracle_ml"], v=c["oracle_v"], loss=c["oracle_loss"])
    base_o = dict(ml=c["base_ml"], v=c["base_v"], loss=c["base_loss"])

    def evals(d, idx):
        js, cong, vd, loss = [], [], [], []
        for i in idx:
            conv = not np.isnan(d["ml"][i])
            js.append(J(dict(converged=conv, max_loading=d["ml"][i],
                             v_pcc=d["v"][i], losses=d["loss"][i]), tl[i]) if conv else np.inf)
            cong.append(1.0 if conv and d["ml"][i] > 100 else 0.0)
            vd.append(abs(d["v"][i] - 1.0) if conv else 0.5)
            loss.append(d["loss"][i] if conv else np.nan)
        return dict(J=np.nanmean(js), cong=np.mean(cong), vd=np.nanmean(vd),
                    loss=np.nanmean(loss))

    baseline_metrics = {
        k: dict(ho=evals(o, ho_fixed), ood=evals(o, ood_idx))
        for k, o in [("base", base_o), ("droop", droop_o),
                     ("fixed_pf", fixed_o), ("oracle", orac_o)]
    }
    with open(os.path.join(OUT, "control_metrics.json"), "w") as f:
        json.dump(dict(policy=rows, baselines=baseline_metrics), f, indent=2)
    print("saved control_metrics.json")
    print("baselines ho:", {k: round(v["ho"]["J"], 4) for k, v in baseline_metrics.items()})
    print("baselines ood:", {k: round(v["ood"]["J"], 4) for k, v in baseline_metrics.items()})


if __name__ == "__main__":
    main()
