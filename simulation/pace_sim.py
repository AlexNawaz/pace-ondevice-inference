
import os
import math
import json
import numpy as np
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Reproducibility & plotting style
# ----------------------------------------------------------------------------
SEED = 20260918  # the conference dates :)
rng = np.random.default_rng(SEED)

OUT = "results"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# Colour-blind-friendly palette (Okabe-Ito)
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442",
    "sky": "#56B4E9", "black": "#000000", "grey": "#999999",
}

# ----------------------------------------------------------------------------
# 1. Hardware model -- a heterogeneous mobile-class SoC
# ----------------------------------------------------------------------------
# Each processing element (PE) exposes two independent DVFS domains:
#   s_core in [smin_core, 1]  -> scales compute throughput and dynamic
#                                 compute energy (E_dyn_comp ~ s_core^2)
#   s_mem  in [smin_mem, 1]   -> scales effective memory bandwidth and
#                                 dynamic memory energy (E_dyn_mem ~ s_mem)
# A phase's latency follows a roofline overlap model:
#   L = max( W_comp / (C * s_core),  W_mem / (B * s_mem) )
# Energy = dyn-compute + dyn-memory + leakage*L.
#
# Values are representative of a contemporary smartphone SoC. The NPU is
# compute-rich but bandwidth-starved (a well-documented characteristic of
# mobile NPUs); the GPU has the highest sustained memory bandwidth.

@dataclass
class PE:
    name: str
    C: float          # peak compute, FLOP/s, at s_core = 1
    B: float          # peak memory bandwidth, bytes/s, at s_mem = 1
    e_comp: float     # dynamic compute energy, J/FLOP, at s_core = 1
    e_mem: float      # dynamic memory energy, J/byte, at s_mem = 1
    p_static: float   # leakage / static power, W
    smin_core: float
    smin_mem: float

    def run_phase(self, w_comp, w_mem, s_core, s_mem):
        """Return (latency_s, energy_J) for one phase chunk."""
        s_core = min(max(s_core, self.smin_core), 1.0)
        s_mem = min(max(s_mem, self.smin_mem), 1.0)
        t_comp = w_comp / (self.C * s_core)
        t_mem = w_mem / (self.B * s_mem)
        L = max(t_comp, t_mem)                       # compute/memory overlap
        e_dyn_comp = w_comp * self.e_comp * s_core ** 2
        e_dyn_mem = w_mem * self.e_mem * s_mem
        e_static = self.p_static * L
        return L, e_dyn_comp + e_dyn_mem + e_static

    # smallest s_core that keeps a compute-bound phase under budget D
    def min_score_for(self, w_comp, D):
        if D <= 0:
            return 1.0
        need = w_comp / (self.C * D)
        return min(max(need, self.smin_core), 1.0)

    # smallest s_mem that keeps a memory-bound phase under budget D
    def min_smem_for(self, w_mem, D):
        if D <= 0:
            return 1.0
        need = w_mem / (self.B * D)
        return min(max(need, self.smin_mem), 1.0)

    # is budget D feasible for a compute-bound phase even at s_core = 1?
    def comp_feasible(self, w_comp, D):
        return w_comp / (self.C * 1.0) <= D + 1e-12

    def mem_feasible(self, w_mem, D):
        return w_mem / (self.B * 1.0) <= D + 1e-12


def build_soc():
    return {
        "LITTLE": PE("LITTLE", C=0.06e12, B=18e9, e_comp=4.0e-12, e_mem=50e-12,
                     p_static=0.10, smin_core=0.30, smin_mem=0.40),
        "big":    PE("big",    C=0.40e12, B=34e9, e_comp=3.0e-12, e_mem=48e-12,
                     p_static=0.35, smin_core=0.30, smin_mem=0.40),
        "GPU":    PE("GPU",    C=1.80e12, B=60e9, e_comp=1.5e-12, e_mem=32e-12,
                     p_static=0.55, smin_core=0.40, smin_mem=0.40),
        "NPU":    PE("NPU",    C=4.50e12, B=30e9, e_comp=0.8e-12, e_mem=40e-12,
                     p_static=0.45, smin_core=0.50, smin_mem=0.40),
    }


# ----------------------------------------------------------------------------
# 2. Workload model -- two-phase generative inference with a growing KV cache
# ----------------------------------------------------------------------------
# Target model: a ~1.1B-parameter decoder-only Transformer in INT8
# (e.g. a Llama-3.2-1B-class model), 16 layers, hidden size 2048.

@dataclass
class ModelSpec:
    params: float = 1.1e9
    n_layers: int = 16
    d_model: int = 2048
    w_bytes: int = 1          # INT8 weights
    kv_bytes: int = 1         # INT8 KV cache
    def weight_bytes(self):
        return self.params * self.w_bytes
    def kv_per_token(self):   # K and V for every layer
        return 2 * self.n_layers * self.d_model * self.kv_bytes

MODEL = ModelSpec()

def prefill_work(model, L_in):
    """FLOPs and bytes to ingest an L_in-token prompt (compute-bound)."""
    w_comp = 2.0 * model.params * L_in \
        + 2.0 * model.n_layers * (L_in ** 2) * model.d_model     # attention
    w_mem = model.weight_bytes() + L_in * model.kv_per_token()    # weights read once
    return w_comp, w_mem

def decode_token_work(model, seq_len):
    """FLOPs and bytes for ONE decode step at current seq_len (memory-bound)."""
    w_comp = 2.0 * model.params \
        + 2.0 * model.n_layers * seq_len * model.d_model          # attention over cache
    w_mem = model.weight_bytes() + seq_len * model.kv_per_token() # weights + KV read
    return w_comp, w_mem

def sample_requests(n, rng):
    """Sample (L_in, L_out) pairs from log-normal distributions."""
    L_in = np.clip(rng.lognormal(mean=math.log(180), sigma=0.55, size=n), 16, 1024).astype(int)
    L_out = np.clip(rng.lognormal(mean=math.log(110), sigma=0.60, size=n), 8, 512).astype(int)
    return list(zip(L_in.tolist(), L_out.tolist()))


# ----------------------------------------------------------------------------
# 3. Scheduling policies
# ----------------------------------------------------------------------------
# Each policy consumes the SoC, a request (L_in, L_out_actual), the SLOs, and
# (for online policies) a *predicted* output length. It returns a dict with
# energy, ttft, per-token latencies, and miss statistics.

def simulate_request(policy, soc, req, D_ttft, D_tpot, pred_Lout=None):
    L_in, L_out = req
    return policy(soc, MODEL, L_in, L_out, D_ttft, D_tpot, pred_Lout)


# ---- helpers ---------------------------------------------------------------
def _decode_pass(pe, model, L_in, L_out, D_tpot, s_core, s_mem_fn, adapt=False):
    """Run L_out decode steps on PE; s_mem_fn(seq_len, w_mem) -> s_mem.
    If adapt, escalate s_mem to 1.0 the moment a token is projected to miss."""
    energy = 0.0
    lat = []
    boosted = False
    for t in range(L_out):
        seq = L_in + t
        wc, wm = decode_token_work(model, seq)
        s_mem = 1.0 if boosted else s_mem_fn(seq, wm)
        L, e = pe.run_phase(wc, wm, s_core, s_mem)
        if adapt and (not boosted) and L > D_tpot + 1e-12:
            # genuine under-provisioning: the growing KV cache pushed this
            # token over budget. Escalate the MEMORY clock (not the core clock).
            s_mem = 1.0
            L, e = pe.run_phase(wc, wm, s_core, s_mem)
            boosted = True
        energy += e
        lat.append(L)
    return energy, lat


def pol_perf_npu(soc, model, L_in, L_out, D_ttft, D_tpot, pred=None):
    """Performance baseline: run everything on the fastest PE (NPU) at max
    core+memory frequency. (race-to-finish)"""
    pe = soc["NPU"]
    wc, wm = prefill_work(model, L_in)
    ttft, e_pre = pe.run_phase(wc, wm, 1.0, 1.0)
    e_dec, lat = _decode_pass(pe, model, L_in, L_out, D_tpot, 1.0, lambda s, w: 1.0)
    return _pack(e_pre + e_dec, ttft, lat, D_ttft, D_tpot, "Perf-NPU")


def pol_eff_little(soc, model, L_in, L_out, D_ttft, D_tpot, pred=None):
    """Energy-naive baseline: most efficient PE (LITTLE) at minimum frequency."""
    pe = soc["LITTLE"]
    wc, wm = prefill_work(model, L_in)
    ttft, e_pre = pe.run_phase(wc, wm, pe.smin_core, pe.smin_mem)
    e_dec, lat = _decode_pass(pe, model, L_in, L_out, D_tpot,
                              pe.smin_core, lambda s, w: pe.smin_mem)
    return _pack(e_pre + e_dec, ttft, lat, D_ttft, D_tpot, "Eff-LITTLE")


def pol_edf_dvfs(soc, model, L_in, L_out, D_ttft, D_tpot, pred=None):
    """Classic deadline-driven *core* DVFS on the fastest PE (Pillai-Shin
    style): scale s_core to just meet the deadline, leave the memory clock at
    default (s_mem = 1). Single PE for both phases (no phase-aware mapping)."""
    pe = soc["NPU"]
    wc, wm = prefill_work(model, L_in)
    s_core = pe.min_score_for(wc, D_ttft)          # just-meet TTFT via core clock
    ttft, e_pre = pe.run_phase(wc, wm, s_core, 1.0)
    # decode: scale core clock down (memory-bound -> no latency effect) but
    # leave memory clock pinned high, as a core-only governor would.
    e_dec, lat = _decode_pass(pe, model, L_in, L_out, D_tpot,
                              pe.smin_core, lambda s, w: 1.0)
    return _pack(e_pre + e_dec, ttft, lat, D_ttft, D_tpot, "EDF-coreDVFS")


def pol_static_best(soc, model, L_in, L_out, D_ttft, D_tpot, pred=None):
    """Offline best *single* PE (+ per-phase frequency) for the whole request,
    minimising energy subject to the SLOs. Represents non-phase-aware
    heterogeneity awareness: it may pick the best PE, but the SAME PE must
    serve both phases."""
    wc_pre, wm_pre = prefill_work(model, L_in)
    best = None
    for pe in soc.values():
        # prefill at lowest core clock meeting TTFT (memory clock to min)
        s_core = pe.min_score_for(wc_pre, D_ttft)
        ttft, e_pre = pe.run_phase(wc_pre, wm_pre, s_core, pe.smin_mem)
        ttft_ok = ttft <= D_ttft + 1e-9
        # decode at the lowest *fixed* memory clock that meets TPOT for the
        # final (worst) token -- phase-aware DVFS, but locked to this one PE
        wc_d, wm_d = decode_token_work(model, L_in + L_out)
        s_mem = pe.min_smem_for(wm_d, D_tpot)
        e_dec, lat = _decode_pass(pe, model, L_in, L_out, D_tpot,
                                  pe.smin_core, lambda s, w: s_mem)
        tot = e_pre + e_dec
        miss = (not ttft_ok) + sum(1 for x in lat if x > D_tpot + 1e-9)
        key = (miss, tot)                          # prefer feasible, then min energy
        if best is None or key < best[0]:
            best = (key, tot, ttft, lat)
    return _pack(best[1], best[2], best[3], D_ttft, D_tpot, "Static-Best")


def pol_pace(soc, model, L_in, L_out, D_ttft, D_tpot, pred=None):
    """PACE (proposed). Phase-aware heterogeneous mapping + per-phase, per-
    domain DVFS + online memory-clock adaptation under output-length
    uncertainty.
       * Prefill -> most energy-efficient PE that can meet TTFT, at the lowest
         CORE clock that meets TTFT (memory clock to minimum).
       * Decode  -> most energy-efficient PE that can meet TPOT, at the lowest
         MEMORY clock that meets TPOT (core clock to minimum). If the predicted
         output length under-shoots and a token is about to miss as the KV
         cache grows, escalate the MEMORY clock (not the core clock)."""
    # ---- prefill PE/clock selection ----
    wc_pre, wm_pre = prefill_work(model, L_in)
    pre_choice = None
    for pe in soc.values():
        if not pe.comp_feasible(wc_pre, D_ttft):
            continue
        s_core = pe.min_score_for(wc_pre, D_ttft)
        _, e = pe.run_phase(wc_pre, wm_pre, s_core, pe.smin_mem)
        if pre_choice is None or e < pre_choice[0]:
            pre_choice = (e, pe, s_core)
    if pre_choice is None:                         # nobody meets TTFT -> fastest
        pe = soc["NPU"]; s_core = 1.0
        pre_choice = (pe.run_phase(wc_pre, wm_pre, 1.0, 1.0)[1], pe, 1.0)
    e_pre_energy, pe_pre, s_core_pre = pre_choice
    ttft, e_pre = pe_pre.run_phase(wc_pre, wm_pre, s_core_pre, pe_pre.smin_mem)

    # ---- decode PE selection (provision for the *predicted* final length) ----
    pred = pred if pred is not None else L_out
    seq_final_pred = L_in + pred
    wc_d, wm_d = decode_token_work(model, seq_final_pred)
    dec_choice = None
    for pe in soc.values():
        if not pe.mem_feasible(wm_d, D_tpot):
            continue
        s_mem = pe.min_smem_for(wm_d, D_tpot)
        _, e = pe.run_phase(wc_d, wm_d, pe.smin_core, s_mem)
        if dec_choice is None or e < dec_choice[0]:
            dec_choice = (e, pe)
    if dec_choice is None:                          # nobody meets TPOT -> highest BW
        pe_dec = max(soc.values(), key=lambda p: p.B)
        s_mem_prov = 1.0
    else:
        pe_dec = dec_choice[1]
        s_mem_prov = pe_dec.min_smem_for(wm_d, D_tpot)   # fixed op-point from prediction

    # ---- run decode at the provisioned op-point, with online adaptation -----
    # The memory clock is fixed at admission (no per-token reconfiguration). If
    # the true output runs longer than predicted, the growing KV cache pushes a
    # token toward its deadline and the adaptation escalates the MEMORY clock.
    e_dec, lat = _decode_pass(
        pe_dec, model, L_in, L_out, D_tpot, pe_dec.smin_core,
        lambda seq, w: s_mem_prov, adapt=True)

    out = _pack(e_pre + e_dec, ttft, lat, D_ttft, D_tpot, "PACE")
    out["pe_prefill"] = pe_pre.name
    out["pe_decode"] = pe_dec.name
    return out


def pol_oracle(soc, model, L_in, L_out, D_ttft, D_tpot, pred=None):
    """Clairvoyant lower bound: knows L_out exactly, free continuous choice of
    per-phase PE and frequency (fine search). Energy lower bound only."""
    wc_pre, wm_pre = prefill_work(model, L_in)
    grid = np.linspace(0.30, 1.0, 36)
    best_pre = math.inf; pre_ttft = None
    for pe in soc.values():
        for sc in grid:
            if sc < pe.smin_core:
                continue
            L, e = pe.run_phase(wc_pre, wm_pre, sc, pe.smin_mem)
            if L <= D_ttft + 1e-9 and e < best_pre:
                best_pre = e; pre_ttft = L
    if not math.isfinite(best_pre):
        pe = soc["NPU"]; best_pre = pe.run_phase(wc_pre, wm_pre, 1, 1)[1]; pre_ttft = pe.run_phase(wc_pre, wm_pre,1,1)[0]
    # decode: best per-token PE+s_mem at the *true* growing seq
    best_dec = 0.0; lat = []
    for t in range(L_out):
        seq = L_in + t
        wc, wm = decode_token_work(model, seq)
        bt = math.inf; bl = None
        for pe in soc.values():
            sm = pe.min_smem_for(wm, D_tpot)
            L, e = pe.run_phase(wc, wm, pe.smin_core, sm)
            if L <= D_tpot + 1e-9 and e < bt:
                bt = e; bl = L
        if not math.isfinite(bt):                  # infeasible -> highest BW @ max
            pe = max(soc.values(), key=lambda p: p.B)
            bl, bt = pe.run_phase(wc, wm, pe.smin_core, 1.0)
        best_dec += bt; lat.append(bl)
    return _pack(best_pre + best_dec, pre_ttft, lat, D_ttft, D_tpot, "Oracle")


def _pack(energy, ttft, lat, D_ttft, D_tpot, name):
    lat = np.asarray(lat, dtype=float)
    ttft_miss = 1.0 if ttft > D_ttft + 1e-9 else 0.0
    tpot_miss = float(np.mean(lat > D_tpot + 1e-9)) if lat.size else 0.0
    return {
        "policy": name, "energy": float(energy), "ttft": float(ttft),
        "lat": lat, "ttft_miss": ttft_miss, "tpot_miss": tpot_miss,
        "tok_s": float(len(lat) / max(np.sum(lat), 1e-9)) if lat.size else 0.0,
        "n_tokens": int(lat.size),
    }


POLICIES = {
    "Perf-NPU": pol_perf_npu,
    "Eff-LITTLE": pol_eff_little,
    "EDF-coreDVFS": pol_edf_dvfs,
    "Static-Best": pol_static_best,
    "PACE": pol_pace,
    "Oracle": pol_oracle,
}
ORDER = ["Perf-NPU", "Eff-LITTLE", "EDF-coreDVFS", "Static-Best", "PACE", "Oracle"]


# ----------------------------------------------------------------------------
# 4. Experiment driver
# ----------------------------------------------------------------------------
def run_batch(soc, requests, D_ttft, D_tpot, pred_err_cv=0.0, rng=None):
    """Run every policy over a batch of requests. pred_err_cv injects a
    lognormal multiplicative error on the predicted output length used by
    online policies (PACE)."""
    agg = {name: {"E": [], "ttft": [], "ttft_miss": [], "tpot_miss": [],
                  "tok_s": []} for name in POLICIES}
    pe_pre_count, pe_dec_count = {}, {}
    for (L_in, L_out) in requests:
        if pred_err_cv > 0 and rng is not None:
            pred = max(8, int(L_out * rng.lognormal(mean=0.0, sigma=pred_err_cv)))
        else:
            pred = L_out
        for name, fn in POLICIES.items():
            r = simulate_request(fn, soc, (L_in, L_out), D_ttft, D_tpot,
                                 pred_Lout=pred)
            agg[name]["E"].append(r["energy"])
            agg[name]["ttft"].append(r["ttft"])
            agg[name]["ttft_miss"].append(r["ttft_miss"])
            agg[name]["tpot_miss"].append(r["tpot_miss"])
            agg[name]["tok_s"].append(r["tok_s"])
            if name == "PACE":
                pe_pre_count[r["pe_prefill"]] = pe_pre_count.get(r["pe_prefill"], 0) + 1
                pe_dec_count[r["pe_decode"]] = pe_dec_count.get(r["pe_decode"], 0) + 1
    summary = {}
    for name in POLICIES:
        a = agg[name]
        summary[name] = {
            "E_mean": float(np.mean(a["E"])),
            "E_std": float(np.std(a["E"])),
            "ttft_mean_ms": float(np.mean(a["ttft"]) * 1e3),
            "ttft_miss": float(np.mean(a["ttft_miss"])),
            "tpot_miss": float(np.mean(a["tpot_miss"])),
            "tok_s_mean": float(np.mean(a["tok_s"])),
        }
    return summary, pe_pre_count, pe_dec_count


# ----------------------------------------------------------------------------
# 5. Run all experiments
# ----------------------------------------------------------------------------
def main():
    soc = build_soc()
    N = 2000
    requests = sample_requests(N, rng)

    D_TTFT = 0.300     # 300 ms time-to-first-token budget
    D_TPOT = 0.025     # 25 ms per-output-token budget (~40 tok/s floor)

    print("=" * 74)
    print(f"PACE evaluation  |  N={N} requests  |  D_ttft={D_TTFT*1e3:.0f} ms"
          f"  D_tpot={D_TPOT*1e3:.0f} ms")
    print("=" * 74)

    # ---- E1: headline comparison -------------------------------------------
    summary, pre_c, dec_c = run_batch(soc, requests, D_TTFT, D_TPOT)
    base = summary["Perf-NPU"]["E_mean"]
    print(f"\n{'Policy':<14}{'E (J)':>9}{'E norm':>9}{'TTFT ms':>10}"
          f"{'TTFT miss':>11}{'TPOT miss':>11}{'tok/s':>8}")
    for name in ORDER:
        s = summary[name]
        print(f"{name:<14}{s['E_mean']:>9.2f}{s['E_mean']/base:>9.2f}"
              f"{s['ttft_mean_ms']:>10.0f}{s['ttft_miss']*100:>10.1f}%"
              f"{s['tpot_miss']*100:>10.1f}%{s['tok_s_mean']:>8.1f}")
    print(f"\nPACE prefill PE usage: {pre_c}")
    print(f"PACE decode  PE usage: {dec_c}")

    pace = summary["PACE"]; perf = summary["Perf-NPU"]
    edf = summary["EDF-coreDVFS"]; orc = summary["Oracle"]
    save_pct_perf = 100 * (1 - pace["E_mean"] / perf["E_mean"])
    save_pct_edf = 100 * (1 - pace["E_mean"] / edf["E_mean"])
    gap_oracle = 100 * (pace["E_mean"] / orc["E_mean"] - 1)
    print(f"\nPACE energy saving vs Perf-NPU : {save_pct_perf:5.1f}%")
    print(f"PACE energy saving vs EDF-core : {save_pct_edf:5.1f}%")
    print(f"PACE gap above Oracle          : {gap_oracle:5.1f}%")

    _write_main_table(summary, base)
    _write_config_table(soc)

    # ---- E2: energy vs TPOT deadline (Pareto) ------------------------------
    tpots = np.array([0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050])
    pareto = {name: {"E": [], "miss": []} for name in ORDER}
    for dt in tpots:
        sm, _, _ = run_batch(soc, requests, D_TTFT, dt)
        for name in ORDER:
            pareto[name]["E"].append(sm[name]["E_mean"])
            # combined SLO violation = max(ttft_miss, tpot_miss)
            pareto[name]["miss"].append(max(sm[name]["ttft_miss"],
                                            sm[name]["tpot_miss"]))

    # ---- E3: per-token latency vs token index (single representative req) ---
    rep = (256, 256)
    per_token = {}
    for pe_name in ["NPU", "GPU", "big", "LITTLE"]:
        pe = soc[pe_name]
        lat = []
        for t in range(rep[1]):
            seq = rep[0] + t
            wc, wm = decode_token_work(MODEL, seq)
            L, _ = pe.run_phase(wc, wm, pe.smin_core, 1.0)   # max mem clock
            lat.append(L * 1e3)
        per_token[pe_name] = lat

    # ---- E4: sensitivity to output-length prediction error -----------------
    cvs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    sens = {"E": [], "tpot_miss": []}
    sens_noadapt = {"E": [], "tpot_miss": []}
    rng_local = np.random.default_rng(SEED + 7)
    for cv in cvs:
        sm, _, _ = run_batch(soc, requests[:800], D_TTFT, D_TPOT,
                             pred_err_cv=cv, rng=rng_local)
        sens["E"].append(sm["PACE"]["E_mean"])
        sens["tpot_miss"].append(sm["PACE"]["tpot_miss"])
        # PACE-without-adaptation: monkey-disable adapt by re-running a variant
        e_na, miss_na = _pace_no_adapt(soc, requests[:800], D_TTFT, D_TPOT, cv,
                                       np.random.default_rng(SEED + 7))
        sens_noadapt["E"].append(e_na)
        sens_noadapt["tpot_miss"].append(miss_na)

    # ---- E5: mixed-criticality (critical + best-effort streams) ------------
    mc = _mixed_criticality(soc, rng)

    # ---- figures -----------------------------------------------------------
    fig_energy(summary, base)
    fig_miss(summary)
    fig_pareto(tpots, pareto)
    fig_per_token(rep, per_token, D_TPOT)
    fig_sensitivity(cvs, sens, sens_noadapt, D_TPOT)
    fig_mixed(mc)

    # ---- text recap --------------------------------------------------------
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(f"N={N}, D_ttft={D_TTFT*1e3:.0f} ms, D_tpot={D_TPOT*1e3:.0f} ms\n\n")
        for name in ORDER:
            s = summary[name]
            f.write(f"{name:<14} E={s['E_mean']:.2f}J ({s['E_mean']/base:.2f}x) "
                    f"TTFT={s['ttft_mean_ms']:.0f}ms TTFTmiss={s['ttft_miss']*100:.1f}% "
                    f"TPOTmiss={s['tpot_miss']*100:.1f}% tok/s={s['tok_s_mean']:.1f}\n")
        f.write(f"\nPACE prefill PE usage: {pre_c}\n")
        f.write(f"PACE decode PE usage : {dec_c}\n")
        f.write(f"\nEnergy saving vs Perf-NPU : {save_pct_perf:.1f}%\n")
        f.write(f"Energy saving vs EDF-core : {save_pct_edf:.1f}%\n")
        f.write(f"Gap above Oracle          : {gap_oracle:.1f}%\n")
        f.write(f"\nMixed-criticality: {json.dumps(mc, indent=2)}\n")
    print(f"\nAll figures and tables written to ./{OUT}/")
    print("Headline numbers also in results/summary.txt")


def _pace_no_adapt(soc, requests, D_ttft, D_tpot, cv, rng):
    """PACE variant with online adaptation switched off (for ablation)."""
    Es, misses = [], []
    for (L_in, L_out) in requests:
        pred = max(8, int(L_out * rng.lognormal(mean=0.0, sigma=cv))) if cv > 0 else L_out
        # replicate pace prefill+decode-PE selection but no adapt
        wc_pre, wm_pre = prefill_work(MODEL, L_in)
        pre = None
        for pe in soc.values():
            if not pe.comp_feasible(wc_pre, D_ttft):
                continue
            sc = pe.min_score_for(wc_pre, D_ttft)
            _, e = pe.run_phase(wc_pre, wm_pre, sc, pe.smin_mem)
            if pre is None or e < pre[0]:
                pre = (e, pe, sc)
        if pre is None:
            pe = soc["NPU"]; pre = (pe.run_phase(wc_pre, wm_pre,1,1)[1], pe, 1.0)
        _, pe_pre, sc = pre
        ttft, e_pre = pe_pre.run_phase(wc_pre, wm_pre, sc, pe_pre.smin_mem)
        wc_d, wm_d = decode_token_work(MODEL, L_in + pred)
        dch = None
        for pe in soc.values():
            if not pe.mem_feasible(wm_d, D_tpot):
                continue
            sm = pe.min_smem_for(wm_d, D_tpot)
            _, e = pe.run_phase(wc_d, wm_d, pe.smin_core, sm)
            if dch is None or e < dch[0]:
                dch = (e, pe, sm)
        pe_dec = dch[1] if dch else max(soc.values(), key=lambda p: p.B)
        sm_fixed = pe_dec.min_smem_for(wm_d, D_tpot)   # provisioned for prediction
        e_dec, lat = 0.0, []
        for t in range(L_out):
            seq = L_in + t
            wc, wm = decode_token_work(MODEL, seq)
            L, e = pe_dec.run_phase(wc, wm, pe_dec.smin_core, sm_fixed)  # NO adapt
            e_dec += e; lat.append(L)
        lat = np.asarray(lat)
        Es.append(e_pre + e_dec)
        misses.append(float(np.mean(lat > D_tpot + 1e-9)) if lat.size else 0.0)
    return float(np.mean(Es)), float(np.mean(misses))


def _mixed_criticality(soc, rng):
    """A latency-critical stream (tight TPOT, must not miss) co-located with a
    best-effort stream (no deadline). Compare PACE-style criticality-aware
    placement vs a shared-NPU FCFS baseline. Reported: critical TPOT-miss and
    best-effort throughput (tok/s)."""
    crit = sample_requests(300, rng)
    D_ttft, D_tpot_c = 0.300, 0.020      # critical: 20 ms TPOT
    # PACE-aware: critical decode on GPU (high BW), best-effort on NPU
    crit_miss_pace, be_tok_pace = [], []
    crit_miss_base, be_tok_base = [], []
    for (L_in, L_out) in crit:
        # ---- PACE: critical on GPU ----
        rc = pol_pace(soc, MODEL, L_in, L_out, D_ttft, D_tpot_c)
        crit_miss_pace.append(rc["tpot_miss"])
        # best-effort token rate on the remaining PE (NPU) at min mem clock
        pe = soc["NPU"]
        wc, wm = decode_token_work(MODEL, L_in + L_out // 2)
        Lbe, _ = pe.run_phase(wc, wm, pe.smin_core, pe.smin_mem)
        be_tok_pace.append(1.0 / Lbe)
        # ---- baseline: both on NPU, FCFS (critical shares with best-effort) ----
        rb = pol_perf_npu(soc, MODEL, L_in, L_out, D_ttft, D_tpot_c)
        crit_miss_base.append(rb["tpot_miss"])
        be_tok_base.append(1.0 / Lbe * 0.5)   # contention halves best-effort
    return {
        "crit_tpot_miss_pace": float(np.mean(crit_miss_pace)) * 100,
        "crit_tpot_miss_base": float(np.mean(crit_miss_base)) * 100,
        "be_tok_s_pace": float(np.mean(be_tok_pace)),
        "be_tok_s_base": float(np.mean(be_tok_base)),
    }


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
def fig_energy(summary, base):
    names = ["Perf-NPU", "Eff-LITTLE", "EDF-coreDVFS", "Static-Best", "PACE", "Oracle"]
    vals = [summary[n]["E_mean"] / base for n in names]
    colors = [CB["grey"], CB["grey"], CB["sky"], CB["orange"], CB["green"], CB["blue"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color=CB["grey"], lw=0.8, ls=":")
    ax.set_ylabel("Energy per request\n(normalised to Perf-NPU)")
    ax.set_ylim(0, max(vals) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    _save(fig, "fig_energy")


def fig_miss(summary):
    names = ["Perf-NPU", "Eff-LITTLE", "EDF-coreDVFS", "Static-Best", "PACE"]
    ttft = [summary[n]["ttft_miss"] * 100 for n in names]
    tpot = [summary[n]["tpot_miss"] * 100 for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w/2, ttft, w, label="TTFT miss", color=CB["red"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, tpot, w, label="TPOT miss", color=CB["purple"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("SLO miss ratio (%)")
    ax.set_ylim(0, 122)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=1,
              loc="upper right", bbox_to_anchor=(0.995, 0.99))
    _save(fig, "fig_miss")


def fig_pareto(tpots, pareto):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    style = {"Perf-NPU": ("o", CB["grey"]), "EDF-coreDVFS": ("s", CB["sky"]),
             "Static-Best": ("^", CB["orange"]), "PACE": ("D", CB["green"]),
             "Oracle": ("*", CB["blue"])}
    for name, (mk, c) in style.items():
        ax1.plot(tpots * 1e3, pareto[name]["E"], marker=mk, color=c, label=name, lw=1.6, ms=6)
        ax2.plot(tpots * 1e3, np.array(pareto[name]["miss"]) * 100, marker=mk, color=c, label=name, lw=1.6, ms=6)
    ax1.set_xlabel("TPOT budget (ms)"); ax1.set_ylabel("Energy per request (J)")
    ax2.set_xlabel("TPOT budget (ms)"); ax2.set_ylabel("Worst-case SLO miss (%)")
    ax1.legend(frameon=False, fontsize=8.5)
    ax2.axhline(0, color=CB["grey"], lw=0.6)
    _save(fig, "fig_pareto")


def fig_per_token(rep, per_token, D_tpot):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    style = {"NPU": CB["red"], "GPU": CB["green"], "big": CB["orange"], "LITTLE": CB["grey"]}
    for name, c in style.items():
        ax.plot(range(rep[1]), per_token[name], color=c, label=name, lw=1.8)
    ax.axhline(D_tpot * 1e3, color=CB["black"], lw=1.2, ls="--", label="TPOT budget")
    ax.set_xlabel("Decode step (token index)")
    ax.set_ylabel("Per-token latency (ms)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    _save(fig, "fig_per_token")


def fig_sensitivity(cvs, sens, sens_na, D_tpot):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax1.plot(cvs, sens["E"], marker="D", color=CB["green"], label="PACE (adaptive)", lw=1.8)
    ax1.plot(cvs, sens_na["E"], marker="o", color=CB["orange"], label="PACE w/o adaptation", lw=1.8)
    ax1.set_xlabel("Output-length prediction error (CV)")
    ax1.set_ylabel("Energy per request (J)")
    ax1.legend(frameon=False, fontsize=9)
    ax2.plot(cvs, np.array(sens["tpot_miss"]) * 100, marker="D", color=CB["green"], label="PACE (adaptive)", lw=1.8)
    ax2.plot(cvs, np.array(sens_na["tpot_miss"]) * 100, marker="o", color=CB["orange"], label="PACE w/o adaptation", lw=1.8)
    ax2.set_xlabel("Output-length prediction error (CV)")
    ax2.set_ylabel("TPOT miss ratio (%)")
    ax2.legend(frameon=False, fontsize=9)
    _save(fig, "fig_sensitivity")


def fig_mixed(mc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax1.bar(["Shared-NPU\nFCFS", "PACE"],
            [mc["crit_tpot_miss_base"], mc["crit_tpot_miss_pace"]],
            color=[CB["grey"], CB["green"]], edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Critical-stream\nTPOT miss (%)")
    ax2.bar(["Shared-NPU\nFCFS", "PACE"],
            [mc["be_tok_s_base"], mc["be_tok_s_pace"]],
            color=[CB["grey"], CB["green"]], edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Best-effort\nthroughput (tok/s)")
    _save(fig, "fig_mixed")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. LaTeX tables
# ----------------------------------------------------------------------------
def _write_main_table(summary, base):
    rows = ["Perf-NPU", "Eff-LITTLE", "EDF-coreDVFS", "Static-Best", "PACE", "Oracle"]
    pretty = {"Perf-NPU": "Perf-NPU", "Eff-LITTLE": "Eff-LITTLE",
              "EDF-coreDVFS": "EDF core-DVFS~\\cite{pillai2001real}",
              "Static-Best": "Static-Best", "PACE": "\\textbf{PACE (ours)}",
              "Oracle": "Oracle (LB)"}
    lines = [
        "% Auto-generated by pace_sim.py -- paste into Table III",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Policy & $E$ (J) & $E$ norm. & TTFT (ms) & TTFT miss & TPOT miss \\\\",
        "\\midrule",
    ]
    for n in rows:
        s = summary[n]
        lines.append(
            f"{pretty[n]} & {s['E_mean']:.2f} & {s['E_mean']/base:.2f} & "
            f"{s['ttft_mean_ms']:.0f} & {s['ttft_miss']*100:.1f}\\% & "
            f"{s['tpot_miss']*100:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_config_table(soc):
    lines = [
        "% Auto-generated by pace_sim.py -- paste into Table I",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "PE & $C$ (TFLOP/s) & $B$ (GB/s) & $e_{comp}$ (pJ/FLOP) & "
        "$e_{mem}$ (pJ/B) & $P_{stat}$ (W) \\\\",
        "\\midrule",
    ]
    for pe in soc.values():
        lines.append(
            f"{pe.name} & {pe.C/1e12:.2f} & {pe.B/1e9:.0f} & "
            f"{pe.e_comp*1e12:.1f} & {pe.e_mem*1e12:.0f} & {pe.p_static:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "table_config.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()