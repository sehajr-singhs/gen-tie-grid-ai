"""Inverter-coordination dataset on IEEE 39-bus with the gen-tie plant as an
inverter-based resource.

Per scenario (randomized loads, redispatch, real-time plant output P):
  - base outcome: AC power flow at (P, Q=0): max loading, PCC voltage,
    losses.
  - oracle target: dense scan over Q (51 points within the plant's
    apparent-power capability) minimizing
        J = cong_pen + volt_pen + 0.5 * loss_frac
    where cong_pen = max(0, max_loading - 100)/100,
          volt_pen = max(0, |V_pcc - 1| - 0.02)/0.1,
          loss_frac = losses / total_load.
    If the base scenario is congested (max_loading > 100), a joint
    (P_curtail x Q) scan over 6 curtail levels x 25 Q points finds the
    jointly optimal (P*, Q*).
  - baseline outcomes (fixed-point iteration, full AC PF in the loop):
    IEEE 1547-style volt-var droop, and fixed 0.95 power factor.

All quantities come from pandapower AC power flow.
"""
import copy
import os
import time
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from multiprocessing import Pool

RNG_SEED = 20260831
CONN_BUS = 3
PLANT_MAX_MW = 900.0
PLANT_MVA = 1000.0
N_CONTROL = 600
N_OOD = 150
N_Q = 41
N_Q_CURT = 25
CURT_LEVELS = [0.90, 0.80, 0.65, 0.50, 0.35, 0.20]
DEADBAND = 0.012
DROOP_SLOPE = 1.0 / 0.06   # Qmax reached at |V-1| = 6%
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTDIR, exist_ok=True)

_TEMPLATE = {}


def base_net():
    if "base" in _TEMPLATE:
        return copy.deepcopy(_TEMPLATE["base"])
    net = pn.case39()
    net.bus["max_vm_pu"] = 1.10
    net.bus["min_vm_pu"] = 0.90
    pb = pp.create_bus(net, name="plant", vn_kv=230.0, type="b")
    pp.create_line_from_parameters(
        net, pb, CONN_BUS, length_km=15.0, r_ohm_per_km=0.06,
        x_ohm_per_km=0.32, c_nf_per_km=12.0, max_i_ka=3.2,
        type="ol", name="gen-tie",
    )
    # the inverter-based plant is a PQ injection (fixed P, Q), not a
    # voltage-controlled bus: a gen with vm_pu set would hold the bus
    # voltage and ignore the reactive schedule entirely
    pp.create_sgen(net, pb, p_mw=0.0, q_mvar=0.0, name="plant")
    solve(net)  # warm pandapower's internal cache; deepcopies inherit it
    _TEMPLATE["base"] = net
    return copy.deepcopy(net)


def qmax(p):
    return float(np.sqrt(max(0.0, PLANT_MVA ** 2 - p ** 2)))


def solve(net, max_iter=20):
    try:
        pp.runpp(net, numba=False, max_iteration=max_iter)
    except Exception:
        net.converged = False
    return net


def outcome(net, plant_p, plant_q):
    pi = net.sgen.index[net.sgen.name == "plant"][0]
    net.sgen.at[pi, "p_mw"] = plant_p
    net.sgen.at[pi, "q_mvar"] = plant_q
    solve(net)
    if not net.converged:
        return dict(converged=False, max_loading=np.nan, v_pcc=np.nan,
                    losses=np.nan, total_load=np.nan)
    ml = float(net.res_line["loading_percent"].max())
    v_pcc = float(net.res_bus.at[CONN_BUS, "vm_pu"])
    losses = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum())
    total_load = float(net.load["p_mw"].sum())
    return dict(converged=True, max_loading=ml, v_pcc=v_pcc,
                losses=losses, total_load=total_load)


def node_features(net, plant_p):
    """40 x 7 node features, same schema as the screening dataset."""
    nb = len(net.bus)
    X = np.zeros((nb, 7))
    for _, row in net.load.iterrows():
        X[int(row.bus), 0] += row.p_mw
        X[int(row.bus), 1] += row.q_mvar
    for _, row in net.ext_grid.iterrows():
        X[int(row.bus), 4] = 1.0
    for _, row in net.gen.iterrows():
        b = int(row.bus)
        X[b, 2] += row.p_mw
        q = row.get("q_mvar", 0.0)
        X[b, 3] += 0.0 if not np.isfinite(q) else float(q)
        X[b, 6] = 1.0
    for _, row in net.sgen.iterrows():
        b = int(row.bus)
        if row.name == "plant":
            X[b, 2] = plant_p
            X[b, 5] = 1.0
        else:
            X[b, 2] += row.p_mw
            X[b, 3] += row.q_mvar
        X[b, 6] = 1.0
    return X


def J_of(o):
    if not o["converged"]:
        return np.inf
    cong = max(0.0, o["max_loading"] - 100.0) / 100.0
    volt = max(0.0, abs(o["v_pcc"] - 1.0) - DEADBAND) / 0.1
    loss = o["losses"] / o["total_load"]
    return cong + volt + 0.5 * loss


def droop_q(v_pcc, p):
    """IEEE 1547-style volt-var curve at the PCC."""
    dev = 1.0 - v_pcc
    if abs(dev) <= DEADBAND:
        return 0.0
    q = np.clip(dev * DROOP_SLOPE * qmax(p), -qmax(p), qmax(p))
    return float(q)


def baseline_outcome(net, plant_p, qfun, iters=3):
    """Fixed-point iteration: Q = qfun(V_pcc), then re-solve."""
    pi = net.sgen.index[net.sgen.name == "plant"][0]
    q = 0.0
    o = None
    for _ in range(iters):
        net.sgen.at[pi, "p_mw"] = plant_p
        net.sgen.at[pi, "q_mvar"] = q
        solve(net)
        if not net.converged:
            return dict(converged=False, max_loading=np.nan, v_pcc=np.nan,
                        losses=np.nan, total_load=np.nan, q=np.nan)
        o = outcome(net, plant_p, q)
        q = qfun(o["v_pcc"], plant_p)
    o["q"] = q
    return o


def scenario_worker(args):
    sid, params = args
    regime, load_mults, gen_mults, plant_p = params
    net = base_net()
    net.load["p_mw"] *= load_mults
    net.load["q_mvar"] *= load_mults
    idx = net.gen.index
    net.gen.loc[idx, "p_mw"] *= gen_mults

    base = outcome(net, plant_p, 0.0)
    qm = qmax(plant_p)

    # dense Q scan at full P
    qs = np.linspace(-qm, qm, N_Q)
    best_q, best_j, best_o = 0.0, np.inf, None
    for q in qs:
        o = outcome(net, plant_p, float(q))
        j = J_of(o)
        if j < best_j:
            best_j, best_q, best_o = j, float(q), o

    # curtailment scan if congested at the optimum
    p_star = plant_p
    q_star = best_q
    j_star = best_j
    oracle = best_o
    if base["converged"] and base["max_loading"] > 100.0:
        for cl in CURT_LEVELS:
            p2 = plant_p * cl
            qm2 = qmax(p2)
            qs2 = np.linspace(-qm2, qm2, N_Q_CURT)
            for q in qs2:
                o = outcome(net, p2, float(q))
                j = J_of(o)
                if j < j_star:
                    j_star, p_star, q_star = j, p2, float(q)
                    oracle = o

    # baselines (fixed point, in the loop)
    droop = baseline_outcome(net, plant_p, droop_q)
    fixedpf = baseline_outcome(net, plant_p,
                               lambda v, p: float(-0.329 * p))

    X = node_features(net, plant_p)

    return dict(sid=sid, regime=regime, plant_p=plant_p,
                base=base, q_star=q_star, p_star=p_star, j_star=j_star,
                oracle=oracle, droop=droop, fixedpf=fixedpf, qmax=qm,
                node_feat=X, load_mults=load_mults, gen_mults=gen_mults)


PARTIAL = os.path.join(OUTDIR, "control_data_partial.npz")
FINAL = os.path.join(OUTDIR, "control_data.npz")


def save_rows(rows, path):
    rows = sorted(rows, key=lambda r: r["sid"])
    np.savez(path,
             sid=np.array([r["sid"] for r in rows]),
             regime=np.array([r["regime"] for r in rows]),
             plant_p=np.array([r["plant_p"] for r in rows]),
             is_ood=np.array([i >= N_CONTROL for i in range(len(rows))]),
             q_star=np.array([r["q_star"] for r in rows]),
             p_star=np.array([r["p_star"] for r in rows]),
             j_star=np.array([r["j_star"] for r in rows]),
             qmax=np.array([r["qmax"] for r in rows]),
             node_feat=np.stack([r["node_feat"] for r in rows]),
             load_mults=np.stack([r["load_mults"] for r in rows]),
             gen_mults=np.stack([r["gen_mults"] for r in rows]),
             oracle_ml=np.array([r["oracle"]["max_loading"] if r["oracle"] else np.nan for r in rows]),
             oracle_v=np.array([r["oracle"]["v_pcc"] if r["oracle"] else np.nan for r in rows]),
             oracle_loss=np.array([r["oracle"]["losses"] if r["oracle"] else np.nan for r in rows]),
             base_ml=np.array([r["base"]["max_loading"] for r in rows]),
             base_v=np.array([r["base"]["v_pcc"] for r in rows]),
             base_loss=np.array([r["base"]["losses"] for r in rows]),
             base_tl=np.array([r["base"]["total_load"] for r in rows]),
             droop_ml=np.array([r["droop"]["max_loading"] for r in rows]),
             droop_v=np.array([r["droop"]["v_pcc"] for r in rows]),
             droop_loss=np.array([r["droop"]["losses"] for r in rows]),
             droop_q=np.array([r["droop"]["q"] for r in rows]),
             fixed_ml=np.array([r["fixedpf"]["max_loading"] for r in rows]),
             fixed_v=np.array([r["fixedpf"]["v_pcc"] for r in rows]),
             fixed_loss=np.array([r["fixedpf"]["losses"] for r in rows]))
    return len(rows)


def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    params = []
    for i in range(N_CONTROL + N_OOD):
        regime = float([0.85, 1.0, 1.15, 1.30][i % 4]) if i < N_CONTROL \
            else float([1.45, 1.60][(i - N_CONTROL) % 2])
        g = rng.uniform(0.85, 1.15, 21) * regime
        gm = rng.uniform(0.9, 1.1, 9) * regime  # generation tracks load regime
        plant_p = float(rng.uniform(0.3, 1.0) * PLANT_MAX_MW)
        params.append((regime, g, gm, plant_p))

    # keep a full dict sid -> row; merge on resume from the partial checkpoint
    rows = {}
    if os.path.exists(PARTIAL):
        d = np.load(PARTIAL)
        # reconstruct scenario dicts from the partial arrays
        n = len(d["sid"])
        for i in range(n):
            sid = int(d["sid"][i])
            rows[sid] = dict(
                sid=sid, regime=float(d["regime"][i]), plant_p=float(d["plant_p"][i]),
                q_star=float(d["q_star"][i]), p_star=float(d["p_star"][i]),
                j_star=float(d["j_star"][i]), qmax=float(d["qmax"][i]),
                node_feat=d["node_feat"][i],
                load_mults=d["load_mults"][i], gen_mults=d["gen_mults"][i],
                base=dict(max_loading=float(d["base_ml"][i]), v_pcc=float(d["base_v"][i]),
                          losses=float(d["base_loss"][i]), total_load=float(d["base_tl"][i]),
                          converged=True),
                oracle=None if np.isnan(d["oracle_ml"][i]) else dict(
                    max_loading=float(d["oracle_ml"][i]), v_pcc=float(d["oracle_v"][i]),
                    losses=float(d["oracle_loss"][i]), total_load=float(d["base_tl"][i]),
                    converged=True),
                droop=dict(max_loading=float(d["droop_ml"][i]), v_pcc=float(d["droop_v"][i]),
                           losses=float(d["droop_loss"][i]), total_load=float(d["base_tl"][i]),
                           converged=True, q=float(d["droop_q"][i])),
                fixedpf=dict(max_loading=float(d["fixed_ml"][i]), v_pcc=float(d["fixed_v"][i]),
                             losses=float(d["fixed_loss"][i]), total_load=float(d["base_tl"][i]),
                             converged=True, q=np.nan),
            )
        print(f"[resume] {len(rows)} scenarios loaded from partial", flush=True)

    todo = [i for i in range(len(params)) if i not in rows]
    print(f"computing {len(todo)} remaining scenarios", flush=True)
    with Pool(8) as pool:
        for r in pool.imap_unordered(scenario_worker, [(i, params[i]) for i in todo],
                                     chunksize=3):
            rows[r["sid"]] = r
            if len(rows) % 50 == 0:
                save_rows(list(rows.values()), PARTIAL)
                print(f"  {len(rows)}/{len(params)} scenarios ({time.time()-t0:.0f}s)",
                      flush=True)

    save_rows(list(rows.values()), FINAL)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)
    n_cong = int((np.array([r["base"]["max_loading"] for r in rows.values()]) > 100).sum())
    n_curt = int((np.array([r["p_star"] for r in rows.values()]) <
                  np.array([r["plant_p"] for r in rows.values()])).sum())
    print(f"control: n={len(rows)} congested_at_base={n_cong} "
          f"curtailed_by_oracle={n_curt}")
    print(f"q* nonzero: {(np.abs(np.array([r['q_star'] for r in rows.values()])) > 1).sum()}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
