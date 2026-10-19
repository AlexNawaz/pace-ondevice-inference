"""Shared loading and analysis of the PACE hardware sweeps (1B full grid, 3B and 8B cross grids)."""
import csv, collections, statistics as st, math, json, glob, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FILES = {"1B": os.path.join(D, "pace_hw_Llama-3.2-1B-Instruct-Q8_0_20261008_2303.csv"),
         "3B": os.path.join(D, "pace_hw_Llama-3.2-3B-Instruct-Q8_0_20261009_0021.csv"),
         "8B": os.path.join(D, "pace_hw_Meta-Llama-3.1-8B-Instruct-Q4_K_M_20261009_0150.csv")}
CORES = [600, 900, 1200, 1500, 1800, 2100, 2400, 0]
MEMS = [6001, 7001, 0]
TESTS = [("prefill", 128, 0), ("prefill", 1024, 0), ("decode", 0, 0), ("decode", 0, 1024)]
D_TTFT, D_TPOT = 300.0, 25.0

def load(model):
    return list(csv.DictReader(open(FILES[model], encoding="utf-8")))

def grid(rows):
    G = collections.defaultdict(list)
    for r in rows:
        if r["stage"] == "grid":
            G[(r["phase"], int(r["n_prompt"]), int(r["n_depth"]), int(r["core_lock_MHz"]), int(r["mem_lock_MHz"]))].append(r)
    return G

def val(G, key, f):
    rows = G.get(key)
    return st.mean(float(x[f]) for x in rows) if rows else None

def settings_of(G, t):
    return sorted({(k[3], k[4]) for k in G if k[:3] == t})

def slope(points):
    pts = [(math.log(1.0 / f), math.log(t)) for f, t in points if f and t]
    if len(pts) < 2: return None
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den if den > 1e-6 else None
