import time
import numpy as np
import sys
sys.path.insert(0, "experiments")
import make_screening_data as m

m._TEMPLATE.clear()
net = m.base_net()
base_load_p = net.load["p_mw"].copy()
base_load_q = net.load["q_mvar"].copy()
gen_idx = net.gen.index[net.gen.name != "plant"]
base_gen_p = net.gen.loc[gen_idx, "p_mw"].copy()
pi = net.gen.index[net.gen.name == "plant"][0]
rng = np.random.default_rng(1)


def solve_nb(net, max_iter=20):
    try:
        import pandapower as pp
        pp.runpp(net, numba=True, max_iteration=max_iter)
    except Exception:
        net.converged = False


t0 = time.time()
for i in range(20):
    net.load["p_mw"] = base_load_p * rng.uniform(0.85, 1.15, 21)
    net.load["q_mvar"] = base_load_q * rng.uniform(0.85, 1.15, 21)
    net.gen.loc[gen_idx, "p_mw"] = base_gen_p * rng.uniform(0.9, 1.1, 9)
    net.gen.at[pi, "p_mw"] = round(float(rng.uniform(0, 900)))
    net.gen.at[pi, "q_mvar"] = 0.0
    solve_nb(net, 20)
print(f"numba reuse path: {(time.time()-t0)/20*1000:.0f} ms/scenario")