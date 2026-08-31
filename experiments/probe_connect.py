"""Probe candidate gen-tie connection buses: where does plant injection
produce the strongest congestion / stability signal on IEEE 39-bus?"""
import copy
import numpy as np
import pandapower as pp
import pandapower.networks as pn

rng = np.random.default_rng(7)

PLANT_BASE = 300.0  # MW base injection
PLANT_MAX = 900.0   # MW max injection


def build(conn_bus, load_mult, plant_p, plant_q=0.0):
    net = pn.case39()
    net.bus["max_vm_pu"] = 1.10
    net.bus["min_vm_pu"] = 0.90
    # plant bus + gen-tie line at the connection bus voltage level
    vn = float(net.bus.at[conn_bus, "vn_kv"])
    pb = pp.create_bus(net, name="plant", vn_kv=vn, type="b")
    pp.create_line_from_parameters(
        net, pb, conn_bus, length_km=15.0, r_ohm_per_km=0.06,
        x_ohm_per_km=0.32, c_nf_per_km=12.0, max_i_ka=3.2,
        type="ol", name="gen-tie",
    )
    pp.create_gen(net, pb, p_mw=plant_p, q_mvar=plant_q, vm_pu=1.0,
                  name="plant", slack=False, controllable=True)
    # scale loads
    net.load["p_mw"] *= load_mult
    net.load["q_mvar"] *= load_mult
    # silence vm limits warning by clamping gen vm_pu to bus limits
    net.gen["vm_pu"] = 1.0
    return net


def max_loading(net):
    return float(net.res_line["loading_percent"].max()) if net.converged else np.nan


def run(net):
    try:
        pp.runpp(net, numba=False)
    except Exception:
        net.converged = False
    return net


def loadability_margin(net, kmax=3.0):
    """Scale loads and plant P together until PF diverges."""
    lo, hi = 1.0, kmax
    best = 1.0
    # widen until divergence or cap
    n = copy.deepcopy(net)
    plant_idx = n.gen.index[n.gen.name == "plant"][0]
    p0 = n.gen.at[plant_idx, "p_mw"]
    load_p0 = n.load["p_mw"].copy()
    load_q0 = n.load["q_mvar"].copy()
    if not run(n).converged:
        return 1.0
    for k in np.linspace(1.0, kmax, 25):
        m = copy.deepcopy(n)
        m.load["p_mw"] = load_p0 * k
        m.load["q_mvar"] = load_q0 * k
        m.gen.at[plant_idx, "p_mw"] = p0 * k
        run(m)
        if m.converged:
            best = k
        else:
            break
    return best


for conn in [3, 7, 15, 30, 38]:  # 0-indexed candidates across both voltage levels
    rows = []
    for i in range(8):
        lm = float(rng.uniform(0.7, 1.35))
        pp_ = float(rng.uniform(0.0, PLANT_MAX))
        net = build(conn, lm, pp_)
        run(net)
        ml = max_loading(net)
        mg = loadability_margin(net, kmax=2.2)
        rows.append((lm, pp_, ml, mg))
    arr = np.array(rows)
    ml_range = arr[:, 2]
    mg_range = arr[:, 3]
    print(f"conn bus {conn:3d}: max_loading range [{ml_range.min():6.1f}, {ml_range.max():6.1f}]% "
          f"margin range [{mg_range.min():.3f}, {mg_range.max():.3f}] n_div={np.isnan(ml_range).sum()}")
