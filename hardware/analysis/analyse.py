"""Section 7: time and energy per token against the core clock (memory clock at its default),
relative to the default clock pair, for the 1B, 3B and 8B models.

Reads the measured CSVs in ../data and writes the values behind Table 9 and Fig. 13 to
../../results_hw/hw_clocks_data.json. Run:  python analyse.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hwlib import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results_hw") + os.sep
os.makedirs(OUT, exist_ok=True)
NAMES = {"1B": "Llama-3.2-1B, Q8_0", "3B": "Llama-3.2-3B, Q8_0", "8B": "Llama-3.1-8B, Q4_K_M"}
SER = [(("prefill", 1024, 0), "Prefill (1024-token prompt)"),
       (("decode", 0, 0), "Decode (from an empty context)")]

data = {}
for model in ("1B", "3B", "8B"):
    G = grid(load(model)); data[model] = {}
    for t, label in SER:
        T0 = val(G, t + (0, 0), "ms_per_token"); E0 = val(G, t + (0, 0), "energy_mJ_per_token")
        pts = []
        for c in CORES:
            pts.append({"lock": c, "core_MHz": round(val(G, t + (c, 0), "core_MHz")),
                        "time_rel": val(G, t + (c, 0), "ms_per_token") / T0,
                        "energy_rel": val(G, t + (c, 0), "energy_mJ_per_token") / E0,
                        "ms_per_token": val(G, t + (c, 0), "ms_per_token"),
                        "mJ_per_token": val(G, t + (c, 0), "energy_mJ_per_token")})
        data[model][t[0]] = pts
json.dump(data, open(OUT + "hw_clocks_data.json", "w"), indent=1)
print("written", OUT + "hw_clocks_data.json")
for model in data:
    for ph in data[model]:
        print(model, ph, [(p["lock"] or "def", p["core_MHz"], round(p["time_rel"], 3), round(p["energy_rel"], 3))
                          for p in data[model][ph]])
