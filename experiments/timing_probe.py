import time
import numpy as np
import pandapower as pp
import pandapower.networks as pn

net = pn.case39()
# run once to trigger compilation
pp.runpp(net)
t0 = time.perf_counter()
for _ in range(100):
    pp.runpp(net)
t1 = time.perf_counter()
print(f"PF: {(t1-t0)/100*1000:.1f} ms/solve")

# OPF timing
try:
    t0 = time.perf_counter()
    for _ in range(20):
        pp.runopp(net, verbose=False)
    t1 = time.perf_counter()
    print(f"OPF: {(t1-t0)/20*1000:.1f} ms/solve")
except Exception as e:
    print("OPF failed:", e)
