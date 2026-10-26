"""Dump per-window ribbons (Y/mag/rho/half + method intervals), all 7 streams.

Produces results/ribbon/ribbon_<ds>.npz used for the Sec. VII-D width-|eta|
correlation and the share-tercile diagnostics. Single seed (0); 1500 windows.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covprop_r3_dump import run_once_r3, load_any_r3, ALL_STREAMS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'ribbon')
os.makedirs(OUT, exist_ok=True)
for ds in ALL_STREAMS:
    data, source = load_any_r3(ds)
    o, pw, dg = run_once_r3(data, 0, source, mode='real')
    np.savez_compressed(os.path.join(OUT, f'ribbon_{ds}.npz'), **dg['_ribbon'])
    print(ds, 'half:', dg['_ribbon']['half'].shape, flush=True)
