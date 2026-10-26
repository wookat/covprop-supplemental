import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covprop_r3_dump import run_once_r3, load_any_r3
for ds in ["weather","FR","NP","ETTh1"]:
    data, source = load_any_r3(ds)
    o, pw, dg = run_once_r3(data, 0, source, mode="real")
    np.savez_compressed(f"ribbon_{ds}.npz", **dg["_ribbon"])
    print(ds, "Y:", dg["_ribbon"]["Y"].shape, flush=True)
