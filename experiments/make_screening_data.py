"""Generate the screening dataset for the gen-tie paper (robust version).

IEEE 39-bus (pandapower case39) + a dedicated gen-tie line connecting a new
plant bus (index 39) to bus 3 (classic bus 4, 230 kV load area).

Each worker process builds its own net ONCE and reuses it (value mutation,
no deepcopy in the hot loop). Per scenario (randomized loads, redispatch,
plant injection P):
  - AC power flow at kappa=1: converged?, max line loading %, min voltage
  - loadability margin lambda_V (bisection over a scaling of loads + plant P)
  - injection congestion headroom kappa_C (bisection over plant P alone)

Train regimes {0.85, 1.0, 1.15, 1.30}; OOD load regimes {1.45, 1.60};
OOD topology: same regimes, most-loaded line removed (N-1).

All numbers from pandapower AC power flow. Outputs are checkpointed per split.
"""
import os
import time
import numpy as np
import pandapower as pp
import pandapower.networks as pn

RNG_SEED = 20260830
CONN_BUS = 3
PLANT_MAX_MW = 900.0
TRAIN_REGIMES = [0.85, 1.0, 1.15, 1.30]
OOD_REGIMES = [1.45, 1.60]
N_TRAIN = 3000
N_OOD_LOAD = 500
N_OOD_TOPO = 500
N_MARGIN = 250
MARGIN_KMAX_V = 2.2
MARGIN_KMAX_C = 3.0
BISECT_ITERS = 8
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTDIR, exist_ok=True)

_W = {}  # per-worker state


def make_template(remove_line=None):
    net = pn.case39()
    net.bus["max_vm_pu"] = 1.10
    net.bus["min_vm_pu"] = 0.90
    pb = pp.create_bus(net, name="plant", vn_kv=230.0, type="b")
    pp.create_line_from_parameters(
        net, pb, CONN_BUS, length_km=15.0, r_ohm_per_km=0.06,
        x_ohm_per_km=0.32, c_nf_per_km=12.0, max_i_ka=3.2,
        type="ol", name="gen-tie")
    pp.create_gen(net, pb, p_mw=0.0, q_mvar=0.0, vm_pu=1.0,
                  name="plant", slack=False, controllable=True)
    if remove_line is not None:
        net.line.drop(remove_line, inplace=True)
    net.gen["vm_pu"] = 1.0
    return net


def solve(net, max_iter=20):
    try:
        pp.runpp(net, numba=False, max_iteration=max_iter)
    except Exception:
        net.converged = False
    return net


def init_worker(remove_line, base_idx):
    _W.clear()
    _W["net"] = make_template(remove_line)
    _W["plant_idx"] = _W["net"].gen.index[_W["net"].gen.name == "plant"][0]
    _W["gen_idx"] = _W["net"].gen.index[_W["net"].gen.name != "plant"]
    _W["load_p"] = _W["net"].load["p_mw"].copy().values
    _W["load_q"] = _W["net"].load["q_mvar"].copy().values
    _W["gen_p"] = _W["net"].gen.loc[_W["gen_idx"], "p_mw"].copy().values
    _W["base_idx"] = base_idx
    solve(_W["net"])  # warm internal cache


def node_features(net, plant_p, plant_q=0.0):
    nb = len(net.bus)
    X = np.zeros((nb, 7))
    for _, row in net.load.iterrows():
        X[int(row.bus), 0] += row.p_mw
        X[int(row.bus), 1] += row.q_mvar
    for _, row in net.ext_grid.iterrows():
        X[int(row.bus), 4] = 1.0
    for name, row in net.gen.iterrows():
        b = int(row.bus)
        if name == "plant":
            X[b, 2] = plant_p
            X[b, 3] = plant_q
            X[b, 5] = 1.0
        else:
            X[b, 2] += row.p_mw
            X[b, 3] += row.q_mvar
        X[b, 6] = 1.0
    return X


def scenario_worker(args):
    sid, params = args
    regime, lm, gm, plant_p = params
    net = _W["net"]
    net.load["p_mw"] = _W["load_p"] * lm
    net.load["q_mvar"] = _W["load_q"] * lm
    net.gen.loc[_W["gen_idx"], "p_mw"] = _W["gen_p"] * gm
    net.gen.at[_W["plant_idx"], "p_mw"] = plant_p
    net.gen.at[_W["plant_idx"], "q_mvar"] = 0.0
    solve(net)
    conv = bool(net.converged)
    if conv:
        ml = float(net.res_line["loading_percent"].max())
        mv = float(net.res_bus["vm_pu"].min())
        loss = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum())
    else:
        ml = mv = loss = np.nan
    X = node_features(net, plant_p)
    return dict(sid=sid, regime=regime, plant_p=plant_p, converged=conv,
                max_loading=ml, min_vm=mv, losses=loss, node_feat=X,
                load_mults=lm, gen_mults=gm)


def margin_worker(args):
    sid, params = args
    regime, lm, gm, plant_p = params
    net = _W["net"]
    lp = _W["load_p"]; lq = _W["load_q"]; gp = _W["gen_p"]
    pi = _W["plant_idx"]; gidx = _W["gen_idx"]

    def set_loads(k):
        net.load["p_mw"] = lp * lm * k
        net.load["q_mvar"] = lq * lm * k

    def bisect(direction, lo, hi, iters, ok):
        set_loads(1.0)
        net.gen.loc[gidx, "p_mw"] = gp * gm
        net.gen.at[pi, "q_mvar"] = 0.0
        best = lo
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            if direction == "V":
                set_loads(mid)
                net.gen.at[pi, "p_mw"] = plant_p * mid
            else:  # "C": plant P alone, loads fixed
                set_loads(1.0)
                net.gen.at[pi, "p_mw"] = plant_p * mid
            solve(net)
            if ok(net):
                best = mid; lo = mid
            else:
                hi = mid
        return best

    ok_conv = lambda n: n.converged
    ok_cong = lambda n: n.converged and \
        float(n.res_line["loading_percent"].max()) <= 100.0

    lambda_V = bisect("V", 1.0, MARGIN_KMAX_V, BISECT_ITERS, ok_conv) - 1.0
    kappa_C = bisect("C", 1.0, MARGIN_KMAX_C, BISECT_ITERS, ok_cong)
    return dict(sid=sid, lambda_V=lambda_V, kappa_C=kappa_C)


def make_params(n, regimes, rng):
    params = []
    for i in range(n):
        regime = float(regimes[i % len(regimes)])
        lm = rng.uniform(0.85, 1.15, 21) * regime
        # generation tracks the load regime plus +/-10%% redispatch, so the
        # slack only balances residuals (realistic dispatch scaling)
        gm = rng.uniform(0.9, 1.1, 9) * regime
        plant_p = float(rng.uniform(0.0, PLANT_MAX_MW))
        params.append((regime, lm, gm, plant_p))
    return params


def main():
    from multiprocessing import Pool
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    # static graph from a reference net
    ref = make_template()
    nb = len(ref.bus); nl = len(ref.line); nt = len(ref.trafo)
    edge_index = np.zeros((2, nl + nt), dtype=np.int64)
    edge_feat = np.zeros((nl + nt, 6))
    for i, (_, row) in enumerate(ref.line.iterrows()):
        edge_index[0, i] = row.from_bus; edge_index[1, i] = row.to_bus
        vn = float(ref.bus.at[row.from_bus, "vn_kv"])
        edge_feat[i] = [row.r_ohm_per_km, row.x_ohm_per_km, row.c_nf_per_km,
                        row.max_i_ka * vn * np.sqrt(3) / 1000, 1.0, 0.0]
    for j, (_, row) in enumerate(ref.trafo.iterrows()):
        i = nl + j
        edge_index[0, i] = row.hv_bus; edge_index[1, i] = row.lv_bus
        edge_feat[i] = [row.vk_percent, row.vkr_percent, 0.0, row.sn_mva,
                        row.tap_pos, 1.0]
    np.savez(os.path.join(OUTDIR, "graph_static.npz"),
             edge_index=edge_index, edge_feat=edge_feat)

    # most-loaded line in the intact base case (for N-1 OOD)
    solve(ref)
    worst = int(ref.res_line["loading_percent"].idxmax())
    print(f"worst line for N-1: {worst} "
          f"({ref.res_line.loc[worst, 'loading_percent']:.1f}%)", flush=True)

    train_params = make_params(N_TRAIN, TRAIN_REGIMES, rng)
    ood_params = make_params(N_OOD_LOAD, OOD_REGIMES, rng)
    topo_params = make_params(N_OOD_TOPO, OOD_REGIMES, rng)

    def run_pool(func, params, remove_line, chunksize=32):
        with Pool(8, initializer=init_worker, initargs=(remove_line, 0)) as pool:
            return pool.map(func, list(enumerate(params)), chunksize=chunksize)

    def pack(rows, name):
        rows = sorted(rows, key=lambda r: r["sid"])
        np.savez(os.path.join(OUTDIR, f"screening_{name}.npz"),
                 sid=np.array([r["sid"] for r in rows]),
                 regime=np.array([r["regime"] for r in rows]),
                 plant_p=np.array([r["plant_p"] for r in rows]),
                 converged=np.array([r["converged"] for r in rows]),
                 max_loading=np.array([r["max_loading"] for r in rows]),
                 min_vm=np.array([r["min_vm"] for r in rows]),
                 losses=np.array([r["losses"] for r in rows]),
                 node_feat=np.stack([r["node_feat"] for r in rows]),
                 load_mults=np.stack([r["load_mults"] for r in rows]),
                 gen_mults=np.stack([r["gen_mults"] for r in rows]))
        conv = rows and int(np.sum([r["converged"] for r in rows]))
        print(f"{name}: n={len(rows)} converged={conv} "
              f"congested={(np.nan_to_num(np.array([r['max_loading'] for r in rows]), nan=0) > 100).sum()} "
              f"({time.time()-t0:.0f}s)", flush=True)

    train_rows = run_pool(scenario_worker, train_params, None)
    pack(train_rows, "train")
    ood_rows = run_pool(scenario_worker, ood_params, None)
    pack(ood_rows, "ood_load")
    topo_rows = run_pool(scenario_worker, topo_params, worst)
    pack(topo_rows, "ood_topo")

    rng2 = np.random.default_rng(RNG_SEED + 1)
    sub = rng2.choice(N_TRAIN, size=N_MARGIN, replace=False)
    margin_args = [(i, train_params[i]) for i in sorted(sub)]
    with Pool(8, initializer=init_worker, initargs=(None, 0)) as pool:
        margin_rows = pool.map(margin_worker, margin_args, chunksize=8)
    m = {r["sid"]: r for r in margin_rows}
    order = sorted(m.keys())
    d = dict(sid=np.array(order),
             lambda_V=np.array([m[s]["lambda_V"] for s in order]),
             kappa_C=np.array([m[s]["kappa_C"] for s in order]))
    np.savez(os.path.join(OUTDIR, "screening_margins.npz"), **d)
    print(f"margins: n={len(order)} lambda_V in [{d['lambda_V'].min():.3f},{d['lambda_V'].max():.3f}] "
          f"kappa_C in [{d['kappa_C'].min():.3f},{d['kappa_C'].max():.3f}] "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()