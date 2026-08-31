import numpy as np
import sys
sys.path.insert(0, "experiments")
import make_screening_data as m

m.init_worker(None, 0)
rng = np.random.default_rng(0)
params = []
for i in range(6):
    lm = rng.uniform(0.85, 1.15, 21) * 1.0
    gm = rng.uniform(0.9, 1.1, 9)
    params.append((1.0, lm, gm, float(rng.uniform(0, 900))))

for sid, p in enumerate(params):
    r = m.scenario_worker((sid, p))
    print(f"scen {sid}: conv={r['converged']} ml={float(r['max_loading']):.1f}% "
          f"mv={float(r['min_vm']):.3f} feat_shape={r['node_feat'].shape}")
for sid, p in enumerate(params[:3]):
    r = m.margin_worker((sid, p))
    print(f"margin {sid}: lambda_V={r['lambda_V']:.3f} kappa_C={r['kappa_C']:.3f}")
print("SMOKE OK")