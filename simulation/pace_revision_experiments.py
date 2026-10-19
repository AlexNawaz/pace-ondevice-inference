"""
"""
import math
import json
import os
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pace_sim as ps
from pace_sim import PE, ModelSpec, build_soc, prefill_work, decode_token_work, SEED, CB

OUT = "results_rev"
os.makedirs(OUT, exist_ok=True)
M11 = ModelSpec()
D_TTFT, D_TPOT = 0.300, 0.025
RES = {}


def requests_from(seed, n, lin=(180, 0.55, 16, 1024), lout=(110, 0.60, 8, 512)):
    """Log-normal (median, sigma, min, max) prompt and output lengths.
    With the defaults this is identical to pace_sim.sample_requests."""
    r = np.random.default_rng(seed)
    a = np.clip(r.lognormal(math.log(lin[0]), lin[1], n), lin[2], lin[3]).astype(int)
    b = np.clip(r.lognormal(math.log(lout[0]), lout[1], n), lout[2], lout[3]).astype(int)
    return list(zip(a.tolist(), b.tolist()))


# ----------------------------------------------------------------------------
# Engine with model knobs. With default knobs it reproduces pace_sim exactly
# (checked by exp_validate before anything else runs).
# ----------------------------------------------------------------------------
class Knobs:
    def __init__(self, beta=0.0, alpha=2.0, opp=None, ec=1.0, em=1.0, pst=1.0):
        self.beta = beta        # overlap loss: L = max(tc,tm) + beta*min(tc,tm)
        self.alpha = alpha      # dynamic compute energy ~ s_core**alpha
        self.opp = None if opp is None else np.asarray(sorted(opp), dtype=float)
        self.ec, self.em, self.pst = ec, em, pst   # coefficient multipliers


BASE = Knobs()


def _q(s, floor, K):
    s = min(max(s, floor), 1.0)
    if K.opp is not None:                     # round UP to a discrete operating point
        i = int(np.searchsorted(K.opp, s - 1e-12))
        s = float(K.opp[min(i, len(K.opp) - 1)])
    return s


def phase(pe, wc, wm, sc, sm, K=BASE):
    sc = _q(sc, pe.smin_core, K)
    sm = _q(sm, pe.smin_mem, K)
    tc = wc / (pe.C * sc)
    tm = wm / (pe.B * sm)
    L = max(tc, tm) + K.beta * min(tc, tm)
    E = (K.ec * pe.e_comp * wc * sc ** K.alpha + K.em * pe.e_mem * wm * sm
         + K.pst * pe.p_static * L)
    return L, E


def _bisect_min(f, lo, D, iters=60):
    """Smallest s in [lo, 1] with f(s) <= D, for f decreasing and f(1) <= D."""
    if f(lo) <= D + 1e-12:
        return lo
    a, b = lo, 1.0
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if f(mid) <= D + 1e-12:
            b = mid
        else:
            a = mid
    return b


def op_point(pe, wc, wm, D, lever, mode, K=BASE):
    """(feasible, s_core, s_mem) for one phase; lever 'c' (prefill) or 'm' (decode).
    'pub'  : published rule of pace_sim.py (closed form from the lever's own term).
    'aware': same single-lever rule on the full latency model; raises the other
             clock only if the lever alone cannot meet D.
    'joint': both clocks searched for minimum energy (reference)."""
    if mode == "pub":
        if lever == "c":
            ok = wc / pe.C <= D + 1e-12
            sc = min(max(wc / (pe.C * D), pe.smin_core), 1.0) if D > 0 else 1.0
            return ok, sc, pe.smin_mem
        ok = wm / pe.B <= D + 1e-12
        sm = min(max(wm / (pe.B * D), pe.smin_mem), 1.0) if D > 0 else 1.0
        return ok, pe.smin_core, sm
    if phase(pe, wc, wm, 1.0, 1.0, K)[0] > D + 1e-12:
        return False, 1.0, 1.0
    if lever == "c":
        f = lambda s: phase(pe, wc, wm, s, pe.smin_mem, K)[0]
        if f(1.0) <= D + 1e-12:
            pt = (True, _bisect_min(f, pe.smin_core, D), pe.smin_mem)
        else:
            g = lambda s: phase(pe, wc, wm, 1.0, s, K)[0]
            pt = (True, 1.0, _bisect_min(g, pe.smin_mem, D))
    else:
        f = lambda s: phase(pe, wc, wm, pe.smin_core, s, K)[0]
        if f(1.0) <= D + 1e-12:
            pt = (True, pe.smin_core, _bisect_min(f, pe.smin_mem, D))
        else:
            g = lambda s: phase(pe, wc, wm, s, 1.0, K)[0]
            pt = (True, _bisect_min(g, pe.smin_core, D), 1.0)
    if mode == "aware":
        return pt
    best = (phase(pe, wc, wm, pt[1], pt[2], K)[1], pt[1], pt[2])
    for sc in np.linspace(pe.smin_core, 1.0, 29):
        for sm in np.linspace(pe.smin_mem, 1.0, 25):
            L, E = phase(pe, wc, wm, sc, sm, K)
            if L <= D + 1e-12 and E < best[0]:
                best = (E, sc, sm)
    return True, best[1], best[2]


def _pack(E, ttft, lat, Dt, Dp, extra=None):
    lat = np.asarray(lat, dtype=float)
    r = {"E": float(E), "ttft": float(ttft), "lat": lat,
         "ttft_miss": 1.0 if ttft > Dt + 1e-9 else 0.0,
         "tpot_miss": float(np.mean(lat > Dp + 1e-9)) if lat.size else 0.0}
    if extra:
        r.update(extra)
    return r


def e_perf(soc, m, a, b, Dt, Dp, K=BASE, **kw):
    pe = soc["NPU"]
    wc, wm = prefill_work(m, a)
    ttft, E = phase(pe, wc, wm, 1.0, 1.0, K)
    lat = []
    for t in range(b):
        L, e = phase(pe, *decode_token_work(m, a + t), 1.0, 1.0, K)
        E += e
        lat.append(L)
    return _pack(E, ttft, lat, Dt, Dp)


def e_edf(soc, m, a, b, Dt, Dp, K=BASE, mode="pub", tau_tr=0.0, e_tr=0.0, **kw):
    pe = soc["NPU"]
    wc, wm = prefill_work(m, a)
    if mode == "pub":
        sc = min(max(wc / (pe.C * Dt), pe.smin_core), 1.0)
    else:
        f = lambda s: phase(pe, wc, wm, s, 1.0, K)[0]
        sc = _bisect_min(f, pe.smin_core, Dt) if f(1.0) <= Dt + 1e-12 else 1.0
    ttft, E = phase(pe, wc, wm, sc, 1.0, K)
    ttft += tau_tr
    E += e_tr
    lat = []
    for t in range(b):
        L, e = phase(pe, *decode_token_work(m, a + t), pe.smin_core, 1.0, K)
        if t == 0:
            L += tau_tr
            e += e_tr
        E += e
        lat.append(L)
    return _pack(E, ttft, lat, Dt, Dp)


def e_static(soc, m, a, b, Dt, Dp, K=BASE, mode="pub", tau_tr=0.0, e_tr=0.0, **kw):
    wc0, wm0 = prefill_work(m, a)
    wcd, wmd = decode_token_work(m, a + b)
    best = None
    for pe in soc.values():
        if mode == "pub":
            sc, smp = min(max(wc0 / (pe.C * Dt), pe.smin_core), 1.0), pe.smin_mem
            scd, smd = pe.smin_core, min(max(wmd / (pe.B * Dp), pe.smin_mem), 1.0)
        else:
            _, sc, smp = op_point(pe, wc0, wm0, Dt, "c", "aware", K)
            _, scd, smd = op_point(pe, wcd, wmd, Dp, "m", "aware", K)
        ttft, E = phase(pe, wc0, wm0, sc, smp, K)
        ttft += tau_tr
        E += e_tr
        lat = []
        for t in range(b):
            L, e = phase(pe, *decode_token_work(m, a + t), scd, smd, K)
            if t == 0:
                L += tau_tr
                e += e_tr
            E += e
            lat.append(L)
        miss = (ttft > Dt + 1e-9) + sum(1 for x in lat if x > Dp + 1e-9)
        if best is None or (miss, E) < best[0]:
            best = ((miss, E), E, ttft, lat, pe.name)
    return _pack(best[1], best[2], best[3], Dt, Dp, {"pe": best[4]})


def e_pace(soc, m, a, b, Dt, Dp, pred=None, K=BASE, mode="pub", adapt="predictive",
           handoff=None, tau_tr=0.0, e_tr=0.0, **kw):
    """PACE as in pace_sim.pol_pace, plus optional knobs.
    adapt   : 'predictive' (paper: the latency of token t is evaluated from Eq. (5)
              before t is dispatched), 'lagged' (breach observed after token t,
              escalation from t+1), or None.
    handoff : (tau_h [s], e_fix [J], overlapped_with_prefill [bool])."""
    wc0, wm0 = prefill_work(m, a)
    best = None
    for pe in soc.values():
        ok, sc, sm = op_point(pe, wc0, wm0, Dt, "c", mode, K)
        if not ok:
            continue
        e = phase(pe, wc0, wm0, sc, sm, K)[1]
        if best is None or e < best[0]:
            best = (e, pe, sc, sm)
    if best is None:
        pe = soc["NPU"]
        best = (None, pe, 1.0, pe.smin_mem if mode == "pub" else 1.0)
    _, pp, scp, smp = best
    ttft, E = phase(pp, wc0, wm0, scp, smp, K)
    ttft += tau_tr
    E += e_tr
    pred = b if pred is None else pred
    wcd, wmd = decode_token_work(m, a + pred)
    dbest = None
    for pe in soc.values():
        ok, sc, sm = op_point(pe, wcd, wmd, Dp, "m", mode, K)
        if not ok:
            continue
        e = phase(pe, wcd, wmd, sc, sm, K)[1]
        if dbest is None or e < dbest[0]:
            dbest = (e, pe, sc, sm)
    if dbest is None:
        pd = max(soc.values(), key=lambda p: p.B)
        scd, smd = pd.smin_core, 1.0
    else:
        _, pd, scd, smd = dbest
    E += e_tr
    switch = pd is not pp
    if handoff is not None and switch:
        E += handoff[1] + handoff[0] * (pp.p_static + pd.p_static)
    lat, boosted, esc, tr_next = [], False, None, False
    for t in range(b):
        wc, wm = decode_token_work(m, a + t)
        L, e = phase(pd, wc, wm, scd, 1.0 if boosted else smd, K)
        if adapt == "predictive" and not boosted and L > Dp + 1e-12:
            L, e = phase(pd, wc, wm, scd, 1.0, K)
            L += tau_tr
            e += e_tr
            boosted, esc = True, t
        elif adapt == "lagged":
            if tr_next:
                L += tau_tr
                e += e_tr
                tr_next = False
            if not boosted and L > Dp + 1e-12:
                boosted, esc, tr_next = True, t, True
        if t == 0:
            if not switch:
                L += tau_tr
            if handoff is not None and switch and not handoff[2]:
                L += handoff[0]
        E += e
        lat.append(L)
    return _pack(E, ttft, lat, Dt, Dp, {"pe_pre": pp.name, "pe_dec": pd.name, "esc": esc,
                                        "s_core_pre": scp, "s_mem_pre": smp,
                                        "s_core_dec": scd, "s_mem_dec": smd})


def e_pemap(soc, m, a, b, Dt, Dp, **kw):
    """Ablation of Sec. 6.3: PACE's phase-aware PE selection, every clock at maximum."""
    wc0, wm0 = prefill_work(m, a)
    best = None
    for pe in soc.values():
        if pe.comp_feasible(wc0, Dt):
            e = phase(pe, wc0, wm0, 1.0, 1.0)[1]
            if best is None or e < best[0]:
                best = (e, pe)
    pp = best[1] if best else soc["NPU"]
    ttft, E = phase(pp, wc0, wm0, 1.0, 1.0)
    wcd, wmd = decode_token_work(m, a + b)
    dbest = None
    for pe in soc.values():
        if pe.mem_feasible(wmd, Dp):
            e = phase(pe, wcd, wmd, 1.0, 1.0)[1]
            if dbest is None or e < dbest[0]:
                dbest = (e, pe)
    pd = dbest[1] if dbest else max(soc.values(), key=lambda p: p.B)
    lat = []
    for t in range(b):
        L, e = phase(pd, *decode_token_work(m, a + t), 1.0, 1.0)
        E += e
        lat.append(L)
    return _pack(E, ttft, lat, Dt, Dp)


def e_rule(soc, m, a, b, Dt, Dp, governor=False, **kw):
    """Rule-based split: prefill on the highest-FLOP PE, decode on the highest-
    bandwidth PE. governor=False: clocks at maximum. governor=True: a stock
    core-only deadline governor on each PE, memory clock left at maximum."""
    pp = max(soc.values(), key=lambda p: p.C)
    pd = max(soc.values(), key=lambda p: p.B)
    wc0, wm0 = prefill_work(m, a)
    sc = min(max(wc0 / (pp.C * Dt), pp.smin_core), 1.0) if governor else 1.0
    scd = pd.smin_core if governor else 1.0
    ttft, E = phase(pp, wc0, wm0, sc, 1.0)
    lat = []
    for t in range(b):
        L, e = phase(pd, *decode_token_work(m, a + t), scd, 1.0)
        E += e
        lat.append(L)
    return _pack(E, ttft, lat, Dt, Dp)


def offline_static_fit(soc, m, train, Dt, Dp):
    """One fixed (PE, clock) for prefill and one for decode, chosen offline on a
    training set: minimum mean energy subject to the unavoidable TTFT-miss rate
    (+0.5 pp) and a 0.1 % per-token miss cap."""
    a = np.array([x[0] for x in train], float)
    b = np.array([x[1] for x in train], int)
    wc0 = 2.0 * m.params * a + 2.0 * m.n_layers * a ** 2 * m.d_model
    wm0 = m.weight_bytes() + a * m.kv_per_token()
    floor_miss = np.mean(np.min(np.stack([np.maximum(wc0 / pe.C, wm0 / pe.B)
                                          for pe in soc.values()]), 0) > Dt + 1e-9)
    cap_t = floor_miss + 0.005
    pre = None
    for pe in soc.values():
        for sc in np.linspace(pe.smin_core, 1.0, 71):
            L = np.maximum(wc0 / (pe.C * sc), wm0 / (pe.B * pe.smin_mem))
            E = pe.e_comp * wc0 * sc ** 2 + pe.e_mem * wm0 * pe.smin_mem + pe.p_static * L
            if np.mean(L > Dt + 1e-9) <= cap_t and (pre is None or E.mean() < pre[0]):
                pre = (float(E.mean()), pe.name, float(sc))
    rid = np.repeat(np.arange(len(b)), b)
    seq = np.concatenate([np.arange(x) for x in b]) + np.repeat(a, b)
    wc = 2.0 * m.params + 2.0 * m.n_layers * seq * m.d_model
    wm = m.weight_bytes() + seq * m.kv_per_token()
    dec = None
    for pe in soc.values():
        for sm in np.linspace(pe.smin_mem, 1.0, 121):
            L = np.maximum(wc / (pe.C * pe.smin_core), wm / (pe.B * sm))
            E = pe.e_comp * wc * pe.smin_core ** 2 + pe.e_mem * wm * sm + pe.p_static * L
            miss = np.bincount(rid, weights=(L > Dp + 1e-9).astype(float), minlength=len(b)) / b
            if miss.mean() <= 0.001:
                Er = float(np.bincount(rid, weights=E, minlength=len(b)).mean())
                if dec is None or Er < dec[0]:
                    dec = (Er, pe.name, float(sm))
    return pre, dec, float(floor_miss)


def e_offline(soc, m, a, b, Dt, Dp, cfg=None, **kw):
    (_, pn, sc), (_, dn, sm) = cfg[0], cfg[1]
    pp, pd = soc[pn], soc[dn]
    wc0, wm0 = prefill_work(m, a)
    ttft, E = phase(pp, wc0, wm0, sc, pp.smin_mem)
    lat = []
    for t in range(b):
        L, e = phase(pd, *decode_token_work(m, a + t), pd.smin_core, sm)
        E += e
        lat.append(L)
    return _pack(E, ttft, lat, Dt, Dp)


def psw(fn, soc, m, Dt, Dp):
    """Wrap an unchanged pace_sim policy into the engine's record format."""
    def f(a, b, p=None):
        r = dict(fn(soc, m, a, b, Dt, Dp, p))
        r["E"] = r["energy"]
        return r
    return f


def run(fns, reqs, preds=None):
    out = {}
    for name, f in fns.items():
        recs = [f(a, b, None if preds is None else preds[i]) for i, (a, b) in enumerate(reqs)]
        s = {"E": float(np.mean([r["E"] for r in recs])),
             "ttft_miss": float(np.mean([r["ttft_miss"] for r in recs])),
             "tpot_miss": float(np.mean([r["tpot_miss"] for r in recs])),
             "recs": recs}
        s["worst"] = max(s["ttft_miss"], s["tpot_miss"])
        out[name] = s
    return out


def slim(S):
    return {k: {kk: vv for kk, vv in v.items() if kk != "recs"} for k, v in S.items()}


def save_pct(x, ref):
    return 100.0 * (1.0 - x / ref)


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def exp_validate():
    soc = build_soc()
    R = requests_from(SEED, 400)
    pairs = {"Perf-NPU": (ps.pol_perf_npu, e_perf), "EDF-coreDVFS": (ps.pol_edf_dvfs, e_edf),
             "Static-Best": (ps.pol_static_best, e_static), "PACE": (ps.pol_pace, e_pace)}
    noise = np.random.default_rng(3).lognormal(0.0, 0.4, len(R))
    worst = {}
    for name, (fo, fe) in pairs.items():
        d = 0.0
        for i, (a, b) in enumerate(R):
            p = max(8, int(b * noise[i]))
            ro = fo(soc, M11, a, b, D_TTFT, D_TPOT, p)
            re = fe(soc, M11, a, b, D_TTFT, D_TPOT, pred=p) if name == "PACE" else \
                fe(soc, M11, a, b, D_TTFT, D_TPOT)
            d = max(d, abs(ro["energy"] - re["E"]) / ro["energy"], abs(ro["ttft"] - re["ttft"]),
                    abs(ro["tpot_miss"] - re["tpot_miss"]))
        worst[name] = d
    RES["validation_max_abs_or_rel_diff"] = worst
    assert all(v < 1e-9 for v in worst.values()), worst
    print("validation vs pace_sim.py (max diff):", worst)


def exp_stats():
    soc = build_soc()
    R = requests_from(SEED, 2000)
    S = run({n: psw(ps.POLICIES[n], soc, M11, D_TTFT, D_TPOT) for n in ps.ORDER}, R)
    n = len(R)
    tab = {}
    for name in ps.ORDER:
        rec = S[name]["recs"]
        E = np.array([r["E"] for r in rec])
        T = np.array([r["ttft"] for r in rec]) * 1e3
        lat = np.concatenate([r["lat"] for r in rec]) * 1e3
        tm = np.array([r["tpot_miss"] for r in rec])
        fm = np.array([r["ttft_miss"] for r in rec])
        tab[name] = {"E": float(E.mean()), "E_ci": float(1.96 * E.std(ddof=1) / math.sqrt(n)),
                     "ttft_p50": float(np.percentile(T, 50)), "ttft_p99": float(np.percentile(T, 99)),
                     "ttft_p999": float(np.percentile(T, 99.9)),
                     "tok_p50": float(np.percentile(lat, 50)), "tok_p99": float(np.percentile(lat, 99)),
                     "tok_p999": float(np.percentile(lat, 99.9)),
                     "ttft_miss": float(fm.mean()), "ttft_miss_ci": wilson(float(fm.mean()), n),
                     "tpot_miss": float(tm.mean()),
                     "tpot_miss_ci": float(1.96 * tm.std(ddof=1) / math.sqrt(n))}
    rb = np.random.default_rng(12345)
    idx = rb.integers(0, n, size=(2000, n))
    Ep = np.array([r["E"] for r in S["PACE"]["recs"]])
    boot = {}
    for base in ["Perf-NPU", "EDF-coreDVFS", "Static-Best", "Oracle"]:
        Eb = np.array([r["E"] for r in S[base]["recs"]])
        if base == "Oracle":
            s, pt = 100 * (Ep[idx].mean(1) / Eb[idx].mean(1) - 1), 100 * (Ep.mean() / Eb.mean() - 1)
        else:
            s, pt = 100 * (1 - Ep[idx].mean(1) / Eb[idx].mean(1)), 100 * (1 - Ep.mean() / Eb.mean())
        boot[base] = (float(pt), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    seeds = [SEED] + list(range(1, 10))
    sv = {k: [] for k in ["Perf-NPU", "EDF-coreDVFS", "Static-Best", "Oracle"]}
    for sd in seeds:
        summ, _, _ = ps.run_batch(soc, requests_from(sd, 2000), D_TTFT, D_TPOT)
        pe = summ["PACE"]["E_mean"]
        for k in sv:
            ref = summ[k]["E_mean"]
            sv[k].append(100 * (pe / ref - 1) if k == "Oracle" else save_pct(pe, ref))
    seed_ci = {k: (float(np.mean(v)), float(2.262 * np.std(v, ddof=1) / math.sqrt(len(v))),
                   float(np.min(v)), float(np.max(v))) for k, v in sv.items()}
    RES["stats"] = {"table": tab, "bootstrap": boot, "seeds": seed_ci, "n_seeds": len(seeds)}
    RES["_chat_summary"] = slim(S)
    print("stats: bootstrap", boot)
    print("stats: seeds", seed_ci)


def exp_mixed(N=600):
    soc = build_soc()
    R = requests_from(SEED, N)
    out = {}
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        K = Knobs(beta=beta)
        fns = {
            "Perf-NPU": lambda a, b, p, K=K: e_perf(soc, M11, a, b, D_TTFT, D_TPOT, K=K),
            "EDF-coreDVFS": lambda a, b, p, K=K: e_edf(soc, M11, a, b, D_TTFT, D_TPOT, K=K, mode="aware"),
            "Static-Best": lambda a, b, p, K=K: e_static(soc, M11, a, b, D_TTFT, D_TPOT, K=K, mode="aware"),
            "PACE (published rule)": lambda a, b, p, K=K: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, K=K, mode="pub"),
            "PACE (model-aware)": lambda a, b, p, K=K: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, K=K, mode="aware"),
            "PACE (joint clocks)": lambda a, b, p, K=K: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, K=K, mode="joint"),
        }
        S = run(fns, R)
        base = S["Perf-NPU"]["E"]
        rec = S["PACE (model-aware)"]["recs"]
        second = np.mean([(r["s_mem_pre"] > soc[r["pe_pre"]].smin_mem + 1e-9) or
                          (r["s_core_dec"] > soc[r["pe_dec"]].smin_core + 1e-9) for r in rec])
        out[str(beta)] = {nm: {"E": S[nm]["E"], "E_norm": S[nm]["E"] / base,
                               "ttft_miss": S[nm]["ttft_miss"], "tpot_miss": S[nm]["tpot_miss"],
                               "worst": S[nm]["worst"]} for nm in fns}
        out[str(beta)]["second_lever_frac"] = float(second)
        print(f"mixed beta={beta}: PACE-aware E_norm={S['PACE (model-aware)']['E']/base:.3f} "
              f"worst={S['PACE (model-aware)']['worst']:.3f} | pub worst={S['PACE (published rule)']['worst']:.3f} "
              f"| Static worst={S['Static-Best']['worst']:.3f} | 2nd-lever={second:.3f}")
    RES["mixed"] = out


def soc_hetero(rc, rb):
    s = build_soc()
    g, n = s["GPU"], s["NPU"]
    s["GPU"] = PE("GPU", C=n.C / rc, B=g.B, e_comp=g.e_comp, e_mem=g.e_mem,
                  p_static=g.p_static, smin_core=g.smin_core, smin_mem=g.smin_mem)
    s["NPU"] = PE("NPU", C=n.C, B=g.B / rb, e_comp=n.e_comp, e_mem=n.e_mem,
                  p_static=n.p_static, smin_core=n.smin_core, smin_mem=n.smin_mem)
    return s


def exp_hetero(N=400):
    R = requests_from(SEED, N)
    rcs, rbs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    grid = {}
    for rc in rcs:
        for rb in rbs:
            soc = soc_hetero(rc, rb)
            S = run({nm: psw(ps.POLICIES[nm], soc, M11, D_TTFT, D_TPOT)
                     for nm in ["Perf-NPU", "EDF-coreDVFS", "Static-Best", "PACE"]}, R)
            split = float(np.mean([r["pe_prefill"] != r["pe_decode"] for r in S["PACE"]["recs"]]))
            grid[f"{rc}_{rb}"] = {
                "save_vs_static": save_pct(S["PACE"]["E"], S["Static-Best"]["E"]),
                "save_vs_edf": save_pct(S["PACE"]["E"], S["EDF-coreDVFS"]["E"]),
                "save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
                "static_worst": S["Static-Best"]["worst"], "static_ttft": S["Static-Best"]["ttft_miss"],
                "pace_worst": S["PACE"]["worst"], "split_frac": split}
    RES["hetero"] = {"rc": rcs, "rb": rbs, "grid": grid}
    print("hetero (2.5,2.0):", grid["2.5_2.0"], "| (1,1):", grid["1.0_1.0"])


def exp_handoff(N=1000):
    soc = build_soc()
    R = requests_from(SEED, N)
    ref = run({nm: psw(ps.POLICIES[nm], soc, M11, D_TTFT, D_TPOT)
               for nm in ["Perf-NPU", "EDF-coreDVFS", "Static-Best"]}, R)
    base = run({"PACE": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT)}, R)["PACE"]
    switch = float(np.mean([r["pe_pre"] != r["pe_dec"] for r in base["recs"]]))
    lat = {}
    for tau in [0.0, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050]:
        for ov in (False, True):
            S = run({"x": lambda a, b, p, tau=tau, ov=ov:
                     e_pace(soc, M11, a, b, D_TTFT, D_TPOT, handoff=(tau, 0.0, ov))}, R)["x"]
            lat[f"{tau}_{int(ov)}"] = {"E": S["E"], "tpot_miss": S["tpot_miss"], "ttft_miss": S["ttft_miss"]}
    be = {k: (ref[k]["E"] - base["E"]) / switch for k in ref}
    RES["handoff"] = {"pace_E": base["E"], "ref_E": {k: ref[k]["E"] for k in ref},
                      "ref_worst": {k: ref[k]["worst"] for k in ref}, "switch_frac": switch,
                      "breakeven_J": be, "breakeven_pct_of_pace": {k: 100 * v / base["E"] for k, v in be.items()},
                      "latency": lat}
    print("handoff break-even (J):", be, "| latency sweep:", lat)


def exp_dvfs(N=1000):
    soc = build_soc()
    R = requests_from(SEED, N)

    def fns_for(K, tau=0.0, etr=0.0):
        return {"Perf-NPU": lambda a, b, p: e_perf(soc, M11, a, b, D_TTFT, D_TPOT, K=K),
                "EDF-coreDVFS": lambda a, b, p: e_edf(soc, M11, a, b, D_TTFT, D_TPOT, K=K, tau_tr=tau, e_tr=etr),
                "Static-Best": lambda a, b, p: e_static(soc, M11, a, b, D_TTFT, D_TPOT, K=K, tau_tr=tau, e_tr=etr),
                "PACE": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, K=K, tau_tr=tau, e_tr=etr)}

    def summ(S):
        return {"save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
                "save_vs_edf": save_pct(S["PACE"]["E"], S["EDF-coreDVFS"]["E"]),
                "save_vs_static": save_pct(S["PACE"]["E"], S["Static-Best"]["E"]),
                "pace_E": S["PACE"]["E"], "pace_ttft": S["PACE"]["ttft_miss"], "pace_tpot": S["PACE"]["tpot_miss"],
                "static_ttft": S["Static-Best"]["ttft_miss"], "static_tpot": S["Static-Best"]["tpot_miss"]}
    disc = {}
    for lab, opp in [("continuous", None), ("16 levels", np.linspace(0.3, 1.0, 16)),
                     ("8 levels", np.linspace(0.3, 1.0, 8)), ("4 levels", np.linspace(0.3, 1.0, 4))]:
        disc[lab] = summ(run(fns_for(Knobs(opp=opp)), R))
        print("dvfs", lab, disc[lab])
    trans = {}
    K8 = Knobs(opp=np.linspace(0.3, 1.0, 8))
    for tau in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
        trans[str(tau)] = summ(run(fns_for(K8, tau, 1.0 * tau), R))   # 1 W during a transition
        print("dvfs transition", tau, trans[str(tau)])
    RES["dvfs"] = {"discrete": disc, "transition_8levels": trans}


def exp_energy_model(N=1000):
    soc = build_soc()
    R = requests_from(SEED, N)
    settings = [("alpha=1.5", Knobs(alpha=1.5)), ("alpha=2.0", Knobs()), ("alpha=2.5", Knobs(alpha=2.5)),
                ("e_c x0.5", Knobs(ec=0.5)), ("e_c x1.5", Knobs(ec=1.5)),
                ("e_m x0.5", Knobs(em=0.5)), ("e_m x1.5", Knobs(em=1.5)),
                ("P_stat x0.5", Knobs(pst=0.5)), ("P_stat x1.5", Knobs(pst=1.5))]
    out = {}
    for lab, K in settings:
        S = run({"Perf-NPU": lambda a, b, p, K=K: e_perf(soc, M11, a, b, D_TTFT, D_TPOT, K=K),
                 "EDF-coreDVFS": lambda a, b, p, K=K: e_edf(soc, M11, a, b, D_TTFT, D_TPOT, K=K),
                 "Static-Best": lambda a, b, p, K=K: e_static(soc, M11, a, b, D_TTFT, D_TPOT, K=K),
                 "PACE": lambda a, b, p, K=K: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, K=K)}, R)
        maps = {}
        for r in S["PACE"]["recs"]:
            maps[r["pe_pre"] + "/" + r["pe_dec"]] = maps.get(r["pe_pre"] + "/" + r["pe_dec"], 0) + 1
        out[lab] = {"save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
                    "save_vs_edf": save_pct(S["PACE"]["E"], S["EDF-coreDVFS"]["E"]),
                    "save_vs_static": save_pct(S["PACE"]["E"], S["Static-Best"]["E"]),
                    "pace_worst": S["PACE"]["worst"], "static_worst": S["Static-Best"]["worst"],
                    "mapping": maps}
        print("energy-model", lab, out[lab])
    opt = {}
    for alpha in (1.5, 2.0, 2.5):
        opt[str(alpha)] = {pe.name: (pe.p_static / (alpha * pe.e_comp * pe.C)) ** (1.0 / (alpha + 1.0))
                           for pe in soc.values()}
    opt["s_mem_star"] = {pe.name: math.sqrt(pe.p_static / (pe.e_mem * pe.B)) for pe in soc.values()}
    rec = run({"x": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT)}, R)["x"]["recs"]
    below = float(np.mean([(r["s_core_pre"] < opt["2.0"][r["pe_pre"]] - 1e-9) or
                           (r["s_mem_dec"] < opt["s_mem_star"][r["pe_dec"]] - 1e-9) for r in rec]))
    RES["energy_model"] = {"settings": out, "optimal_clock": opt, "frac_below_optimal_clock": below,
                           "min_chosen_s_core_pre": float(min(r["s_core_pre"] for r in rec)),
                           "min_chosen_s_mem_dec": float(min(r["s_mem_dec"] for r in rec))}
    print("energy-optimal clocks", opt, "| fraction of requests below:", below)


def exp_predictor(N=800):
    soc = build_soc()
    R = requests_from(SEED, N)
    perfect = run({"x": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT)}, R)["x"]
    fns = {"adapt": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, pred=p),
           "none": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, pred=p, adapt=None)}
    noise = np.random.default_rng(SEED + 11).lognormal(0.0, 0.3, N)
    bias = {}
    for bf in [0.6, 0.8, 1.0, 1.25, 1.5]:
        preds = [max(8, int(b * bf * noise[i])) for i, (a, b) in enumerate(R)]
        S = run(fns, R, preds)
        bias[str(bf)] = {"E_adapt": S["adapt"]["E"], "E_none": S["none"]["E"],
                         "tpot_adapt": S["adapt"]["tpot_miss"], "tpot_none": S["none"]["tpot_miss"],
                         "esc_frac": float(np.mean([r["esc"] is not None for r in S["adapt"]["recs"]]))}
        print("predictor bias", bf, bias[str(bf)])
    warm = [b for _, b in requests_from(SEED + 1, 64)]
    hist = {}
    for qv in (0.5, 0.75, 0.9):
        win = deque(warm, maxlen=64)
        preds = []
        for a, b in R:
            preds.append(max(8, int(np.quantile(list(win), qv))))
            win.append(b)
        S = run(fns, R, preds)
        tr = np.array([b for _, b in R])
        pr = np.array(preds)
        hist[str(qv)] = {"E_adapt": S["adapt"]["E"], "E_none": S["none"]["E"],
                         "tpot_adapt": S["adapt"]["tpot_miss"], "tpot_none": S["none"]["tpot_miss"],
                         "esc_frac": float(np.mean([r["esc"] is not None for r in S["adapt"]["recs"]])),
                         "under_frac": float(np.mean(pr < tr)),
                         "mape": float(np.mean(np.abs(pr - tr) / tr))}
        print("history predictor q", qv, hist[str(qv)])
    timing = {}
    for cv in (0.0, 0.2, 0.4, 0.6):
        nz = np.random.default_rng(SEED + 7).lognormal(0.0, cv, N) if cv > 0 else np.ones(N)
        preds = [max(8, int(b * nz[i])) for i, (a, b) in enumerate(R)]
        S = run({"predictive": fns["adapt"],
                 "predictive+0.5ms": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, pred=p,
                                                            tau_tr=0.0005, e_tr=0.0005),
                 "lagged+0.5ms": lambda a, b, p: e_pace(soc, M11, a, b, D_TTFT, D_TPOT, pred=p,
                                                        adapt="lagged", tau_tr=0.0005, e_tr=0.0005),
                 "none": fns["none"]}, R, preds)
        rec = S["predictive"]["recs"]
        pos = [r["esc"] / max(1, len(r["lat"])) for r in rec if r["esc"] is not None]
        timing[str(cv)] = {k: {"E": S[k]["E"], "tpot": S[k]["tpot_miss"]} for k in S}
        timing[str(cv)]["esc_frac"] = float(np.mean([r["esc"] is not None for r in rec]))
        timing[str(cv)]["esc_pos_median"] = float(np.median(pos)) if pos else None
        print("timing cv", cv, timing[str(cv)])
    RES["predictor"] = {"perfect_E": perfect["E"], "perfect_tpot": perfect["tpot_miss"],
                        "bias": bias, "history": hist, "timing": timing}


MODELS = [("1.1B", ModelSpec()), ("3B", ModelSpec(params=3e9, n_layers=28, d_model=3072)),
          ("7B", ModelSpec(params=7e9, n_layers=32, d_model=4096)),
          ("13B", ModelSpec(params=13e9, n_layers=40, d_model=5120))]


def exp_scale(N=600):
    soc = build_soc()
    R = requests_from(SEED, N)
    gpu = soc["GPU"]
    out = {}
    for lab, m in MODELS:
        k = m.params / 1.1e9
        Dt, Dp = D_TTFT * k, D_TPOT * k
        S = run({nm: psw(ps.POLICIES[nm], soc, m, Dt, Dp)
                 for nm in ["Perf-NPU", "EDF-coreDVFS", "Static-Best", "PACE", "Oracle"]}, R)
        wc, wm = decode_token_work(m, 256)
        out[lab] = {"D_ttft_ms": Dt * 1e3, "D_tpot_ms": Dp * 1e3,
                    "gap": 100 * (S["PACE"]["E"] / S["Oracle"]["E"] - 1),
                    "save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
                    "save_vs_edf": save_pct(S["PACE"]["E"], S["EDF-coreDVFS"]["E"]),
                    "save_vs_static": save_pct(S["PACE"]["E"], S["Static-Best"]["E"]),
                    "pace_ttft": S["PACE"]["ttft_miss"], "pace_tpot": S["PACE"]["tpot_miss"],
                    "oracle_tpot": S["Oracle"]["tpot_miss"],
                    "best_token_ms_at_256": gpu.run_phase(wc, wm, gpu.smin_core, 1.0)[0] * 1e3,
                    "kv_frac_at_290": m.kv_per_token() * 290 / m.weight_bytes()}
        print("scale", lab, out[lab])
    RES["scale"] = out


def exp_context():
    soc = build_soc()
    out = {}
    for mo, mx, n in [(110, 512, 1000), (500, 1024, 1000), (1500, 3000, 1000), (4000, 8000, 1000)]:
        R = requests_from(SEED, n, lout=(mo, 0.60, 8, mx))
        S = run({nm: psw(ps.POLICIES[nm], soc, M11, D_TTFT, D_TPOT)
                 for nm in ["Perf-NPU", "PACE", "Oracle"]}, R)
        out[str(mo)] = {"gap": 100 * (S["PACE"]["E"] / S["Oracle"]["E"] - 1),
                        "save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
                        "pace_tpot": S["PACE"]["tpot_miss"], "oracle_tpot": S["Oracle"]["tpot_miss"],
                        "mean_out": float(np.mean([b for _, b in R]))}
        print("context", mo, out[str(mo)])
    RES["context"] = out


def exp_7b4k(N=150):
    soc = build_soc()
    m = MODELS[2][1]
    k = m.params / 1.1e9
    Dt, Dp = D_TTFT * k, D_TPOT * k
    R = requests_from(SEED, N, lin=(180, 0.55, 16, 256), lout=(3500, 0.25, 1024, 3840))
    S = run({nm: psw(ps.POLICIES[nm], soc, m, Dt, Dp)
             for nm in ["Perf-NPU", "EDF-coreDVFS", "Static-Best", "PACE", "Oracle"]}, R)
    nz = np.random.default_rng(SEED + 7).lognormal(0.0, 0.3, N)
    preds = [max(8, int(b * nz[i])) for i, (a, b) in enumerate(R)]
    S2 = run({"adapt": lambda a, b, p: e_pace(soc, m, a, b, Dt, Dp, pred=p),
              "none": lambda a, b, p: e_pace(soc, m, a, b, Dt, Dp, pred=p, adapt=None)}, R, preds)
    out = {"D_ttft_ms": Dt * 1e3, "D_tpot_ms": Dp * 1e3,
           "mean_context": float(np.mean([a + b for a, b in R])),
           "gap": 100 * (S["PACE"]["E"] / S["Oracle"]["E"] - 1),
           "save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
           "save_vs_edf": save_pct(S["PACE"]["E"], S["EDF-coreDVFS"]["E"]),
           "save_vs_static": save_pct(S["PACE"]["E"], S["Static-Best"]["E"]),
           "pace_tpot": S["PACE"]["tpot_miss"], "oracle_tpot": S["Oracle"]["tpot_miss"],
           "cv03_adapt_tpot": S2["adapt"]["tpot_miss"], "cv03_none_tpot": S2["none"]["tpot_miss"],
           "cv03_adapt_E_over_perfect": S2["adapt"]["E"] / S["PACE"]["E"],
           "kv_frac_at_4096": m.kv_per_token() * 4096 / m.weight_bytes()}
    RES["7b4k"] = out
    print("7B-4K", out)


CLASSES = [("Short assistant", (60, 0.50, 8, 256), (30, 0.50, 4, 128), 0.300, 0.025, 800),
           ("Chat (default)", (180, 0.55, 16, 1024), (110, 0.60, 8, 512), 0.300, 0.025, 2000),
           ("Summarisation", (1500, 0.40, 512, 4096), (150, 0.50, 16, 512), 2.000, 0.025, 400),
           ("Long generation", (180, 0.55, 16, 1024), (1200, 0.50, 256, 4096), 0.300, 0.025, 150)]


def exp_classes():
    soc = build_soc()
    out = {}
    for lab, lin, lout, Dt, Dp, n in CLASSES:
        if lab == "Chat (default)":
            S = RES["_chat_summary"]
        else:
            S = slim(run({nm: psw(ps.POLICIES[nm], soc, M11, Dt, Dp) for nm in ps.ORDER},
                         requests_from(SEED, n, lin=lin, lout=lout)))
        out[lab] = {"lin": lin, "lout": lout, "D_ttft_ms": Dt * 1e3, "D_tpot_ms": Dp * 1e3, "N": n,
                    "save_vs_perf": save_pct(S["PACE"]["E"], S["Perf-NPU"]["E"]),
                    "save_vs_edf": save_pct(S["PACE"]["E"], S["EDF-coreDVFS"]["E"]),
                    "save_vs_static": save_pct(S["PACE"]["E"], S["Static-Best"]["E"]),
                    "gap": 100 * (S["PACE"]["E"] / S["Oracle"]["E"] - 1),
                    "pace_ttft": S["PACE"]["ttft_miss"], "pace_tpot": S["PACE"]["tpot_miss"],
                    "static_ttft": S["Static-Best"]["ttft_miss"], "static_tpot": S["Static-Best"]["tpot_miss"],
                    "edf_tpot": S["EDF-coreDVFS"]["tpot_miss"], "pace_E": S["PACE"]["E"]}
        print("class", lab, out[lab])
    RES["classes"] = out


def exp_baselines():
    soc = build_soc()
    R = requests_from(SEED, 2000)
    cfg = offline_static_fit(soc, M11, requests_from(SEED + 101, 2000), D_TTFT, D_TPOT)
    fns = {"Static-Best": psw(ps.pol_static_best, soc, M11, D_TTFT, D_TPOT),
           "Rule-split (max clocks)": lambda a, b, p: e_rule(soc, M11, a, b, D_TTFT, D_TPOT),
           "Rule-split + core governor": lambda a, b, p: e_rule(soc, M11, a, b, D_TTFT, D_TPOT, governor=True),
           "PE-map only": lambda a, b, p: e_pemap(soc, M11, a, b, D_TTFT, D_TPOT),
           "Offline-static": lambda a, b, p: e_offline(soc, M11, a, b, D_TTFT, D_TPOT, cfg=cfg),
           "PACE": psw(ps.pol_pace, soc, M11, D_TTFT, D_TPOT)}
    S = slim(run(fns, R))
    perf = RES["_chat_summary"]["Perf-NPU"]["E"]
    for k in S:
        S[k]["E_norm"] = S[k]["E"] / perf
        S[k]["pace_saving"] = save_pct(S["PACE"]["E"], S[k]["E"])
    RES["baselines"] = {"table": S, "offline_cfg": {"prefill": cfg[0], "decode": cfg[1],
                                                    "unavoidable_ttft_miss_train": cfg[2]}}
    print("baselines", {k: (round(v["E"], 3), round(v["pace_saving"], 1), v["ttft_miss"], v["tpot_miss"])
                        for k, v in S.items()}, "offline cfg", cfg)


def exp_contention():
    soc = build_soc()
    npu = soc["NPU"]
    crit = requests_from(SEED + 5, 300)
    Dc = 0.020
    shared_miss = float(np.mean([ps.pol_perf_npu(soc, M11, a, b, D_TTFT, Dc)["tpot_miss"] for a, b in crit]))
    out = {}
    for Bs in [math.inf, 120e9, 90e9, 75e9, 72e9, 70e9, 68e9, 66e9, 64e9]:
        mp, mr, bp, br, bsh, mg, bg = [], [], [], [], [], [], []
        for a, b in crit:
            wcb, wmb = decode_token_work(M11, a + b // 2)
            Lb, _ = npu.run_phase(wcb, wmb, npu.smin_core, npu.smin_mem)
            d_be = wmb / Lb                                   # best-effort demand, bytes/s
            bsh.append(0.5 / Lb)                              # shared NPU: 50/50 time slicing
            gpu = soc["GPU"]                                  # shared GPU (Static-Best placement): the two
            Lbg, _ = gpu.run_phase(wcb, wmb, 1.0, 1.0)        # streams alternate tokens on the same PE
            mg_cnt = 0
            for t in range(b):
                Lcg, _ = gpu.run_phase(*decode_token_work(M11, a + t), 1.0, 1.0)
                mg_cnt += (Lcg + Lbg) > Dc + 1e-9
            mg.append(mg_cnt / b)
            bg.append(1.0 / (Lcg + Lbg))
            r = ps.pol_pace(soc, M11, a, b, D_TTFT, Dc)
            pd = soc[r["pe_decode"]]
            s_prov = pd.min_smem_for(decode_token_work(M11, a + b)[1], Dc)
            boosted = False
            miss_p = miss_r = 0
            rate_p, rate_r = [], []
            for t in range(b):
                wc, wm = decode_token_work(M11, a + t)
                for _ in range(2):                            # second pass only after escalation
                    d_c = pd.B * (1.0 if boosted else s_prov)
                    share = 1.0 if not math.isfinite(Bs) else min(1.0, Bs / (d_c + d_be))
                    L = max(wc / (pd.C * pd.smin_core), wm / (d_c * share))
                    if boosted or L <= Dc + 1e-12:
                        break
                    boosted = True                            # PACE memory-clock escalation
                miss_p += L > Dc + 1e-9
                rate_p.append(d_be * share / wmb)
                d_res = pd.B * s_prov                         # bandwidth reserved for the critical stream
                avail = d_be if not math.isfinite(Bs) else max(0.0, Bs - d_res)
                Lr = max(wc / (pd.C * pd.smin_core), wm / min(d_res, Bs))
                miss_r += Lr > Dc + 1e-9
                rate_r.append(min(d_be, avail) / wmb)
            mp.append(miss_p / b)
            mr.append(miss_r / b)
            bp.append(np.mean(rate_p))
            br.append(np.mean(rate_r))
        key = "inf" if not math.isfinite(Bs) else str(int(Bs / 1e9))
        out[key] = {"pace_crit_miss": 100 * float(np.mean(mp)), "pace_be": float(np.mean(bp)),
                    "res_crit_miss": 100 * float(np.mean(mr)), "res_be": float(np.mean(br)),
                    "shared_crit_miss": 100 * shared_miss, "shared_be": float(np.mean(bsh)),
                    "gpu_crit_miss": 100 * float(np.mean(mg)), "gpu_be": float(np.mean(bg))}
        print("contention B_sys", key, out[key])
    RES["contention"] = out


def exp_batching():
    gpu = build_soc()["GPU"]
    m, seq = M11, 256
    rows = []
    for bsz in [1, 2, 4, 8, 16, 32]:
        wc1, _ = decode_token_work(m, seq)
        wc = bsz * wc1
        wm = m.weight_bytes() + bsz * seq * m.kv_per_token()
        L, E = gpu.run_phase(wc, wm, 1.0, 1.0)
        sc = gpu.min_score_for(wc, D_TPOT)
        sm = gpu.min_smem_for(wm, D_TPOT)
        Ld, Ed = gpu.run_phase(wc, wm, sc, sm)
        rows.append({"b": bsz, "intensity": wc / wm, "ridge": gpu.C / gpu.B,
                     "L_max_ms": L * 1e3, "tok_s_max": bsz / L,
                     "binding_at_max": "core" if wc / gpu.C > wm / gpu.B else "memory",
                     "feasible": max(wc / gpu.C, wm / gpu.B) <= D_TPOT + 1e-12,
                     "s_core": sc, "s_mem": sm, "E_per_tok_mJ": Ed / bsz * 1e3})
        print("batching", rows[-1])
    RES["batching"] = rows


# ----------------------------------------------------------------------------
# Figures (same style as pace_sim.py)
# ----------------------------------------------------------------------------
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def fig_mixed():
    d = RES["mixed"]
    betas = [float(b) for b in d]
    series = [("Static-Best", CB["orange"], "^"), ("EDF-coreDVFS", CB["sky"], "s"),
              ("PACE (published rule)", CB["grey"], "o"), ("PACE (model-aware)", CB["green"], "D"),
              ("PACE (joint clocks)", CB["blue"], "*")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.4))
    for nm, c, mk in series:
        a1.plot(betas, [d[str(b)][nm]["E_norm"] for b in betas], marker=mk, color=c, lw=1.6, label=nm)
        a2.plot(betas, [100 * d[str(b)][nm]["worst"] for b in betas], marker=mk, color=c, lw=1.6, label=nm)
    a1.set_xlabel(r"Overlap loss $\beta$")
    a1.set_ylabel("Energy per request\n(normalised to Perf-NPU)")
    a2.set_xlabel(r"Overlap loss $\beta$")
    a2.set_ylabel("Worst-case SLO miss (%)")
    a2.set_yscale("symlog", linthresh=1)
    a1.set_ylim(0.62, 1.0)
    a1.legend(fontsize=6.6, frameon=False, loc="center", ncol=2, bbox_to_anchor=(0.5, 0.62))
    _save(fig, "fig_rev_mixed")


def fig_hetero():
    h = RES["hetero"]
    rcs, rbs = h["rc"], h["rb"]
    A = np.array([[h["grid"][f"{rc}_{rb}"]["save_vs_static"] for rc in rcs] for rb in rbs])
    B = np.array([[100 * h["grid"][f"{rc}_{rb}"]["static_worst"] for rc in rcs] for rb in rbs])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, M, title, cmap, fmt in [(axs[0], A, "PACE saving vs Static-Best (%)", "Greens", "{:.1f}"),
                                    (axs[1], B, "Static-Best worst-case miss (%)", "Oranges", "{:.0f}")]:
        im = ax.imshow(M, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(rcs)))
        ax.set_xticklabels([f"{x:g}" for x in rcs])
        ax.set_yticks(range(len(rbs)))
        ax.set_yticklabels([f"{x:g}" for x in rbs])
        ax.set_xlabel(r"Compute ratio $C_{\mathrm{NPU}}/C_{\mathrm{GPU}}$")
        ax.set_ylabel(r"Bandwidth ratio $B_{\mathrm{GPU}}/B_{\mathrm{NPU}}$")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7.5,
                        color="black" if M[i, j] < 0.6 * M.max() else "white")
        ax.add_patch(plt.Rectangle((rcs.index(2.5) - 0.5, rbs.index(2.0) - 0.5), 1, 1,
                                   fill=False, ec="black", lw=1.8))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    _save(fig, "fig_rev_hetero")


def fig_handoff():
    h = RES["handoff"]
    ef = np.linspace(0.0, 1.0, 41)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.4))
    a1.plot(ef * 1e3, h["pace_E"] + ef * h["switch_frac"], color=CB["green"], lw=1.8, label="PACE + hand-off")
    a1.axhline(h["ref_E"]["Static-Best"], color=CB["orange"], ls="--", lw=1.4, label="Static-Best")
    a1.axhline(h["ref_E"]["EDF-coreDVFS"], color=CB["sky"], ls="--", lw=1.4, label="EDF-coreDVFS")
    a1.axvline(h["breakeven_J"]["Static-Best"] * 1e3, color=CB["grey"], ls=":", lw=1.0)
    a1.set_xlabel("Energy charged per hand-off (mJ)")
    a1.set_ylabel("Energy per request (J)")
    a1.legend(fontsize=7, frameon=False, loc="center right")
    taus = [0.0, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050]
    for ov, c, mk, lab in [(0, CB["red"], "o", "hand-off after prefill"), (1, CB["green"], "D", "overlapped with prefill")]:
        a2.plot([t * 1e3 for t in taus], [100 * h["latency"][f"{t}_{ov}"]["tpot_miss"] for t in taus],
                marker=mk, color=c, lw=1.6, label=lab)
    a2.set_xscale("symlog", linthresh=1)
    a2.set_xticks([0, 1, 2, 5, 10, 20, 50])
    a2.set_xticklabels(["0", "1", "2", "5", "10", "20", "50"])
    a2.set_xlabel("Hand-off latency (ms)")
    a2.set_ylabel("PACE TPOT miss (%)")
    a2.legend(fontsize=8, frameon=False)
    _save(fig, "fig_rev_handoff")


def fig_scale():
    s, c = RES["scale"], RES["context"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.4))
    labs = list(s)
    x = np.arange(len(labs))
    a1.bar(x, [s[l]["save_vs_perf"] for l in labs], 0.5, color=CB["green"], ec="black", lw=0.5,
           label="Saving vs Perf-NPU (%)")
    a1b = a1.twinx()
    a1b.plot(x, [s[l]["gap"] for l in labs], color=CB["blue"], marker="D", lw=1.6, label="Gap to Oracle (%)")
    a1.set_xticks(x)
    a1.set_xticklabels(labs)
    a1.set_xlabel("Model size (deadlines scaled with size)")
    a1.set_ylabel("Saving vs Perf-NPU (%)")
    a1b.set_ylabel("Gap to Oracle (%)", color=CB["blue"])
    a1b.set_ylim(0, max(0.5, 1.4 * max(s[l]["gap"] for l in labs)))
    a1b.grid(False)
    mos = list(c)
    xm = [c[k]["mean_out"] for k in mos]
    a2.plot(xm, [c[k]["gap"] for k in mos], color=CB["blue"], marker="D", lw=1.6, label="Gap to Oracle")
    a2.plot(xm, [100 * c[k]["pace_tpot"] for k in mos], color=CB["green"], marker="o", lw=1.4, ls="--", label="PACE TPOT miss")
    a2.plot(xm, [100 * c[k]["oracle_tpot"] for k in mos], color=CB["grey"], marker="s", lw=1.2, ls=":", label="Oracle TPOT miss")
    a2.set_xscale("log")
    a2.set_xlabel("Mean output length (tokens), 1.1 B model")
    a2.set_ylabel("Percent")
    a2.legend(fontsize=8, frameon=False)
    _save(fig, "fig_rev_scale")


def fig_predictor():
    p = RES["predictor"]
    bf = [float(k) for k in p["bias"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 6.2))
    a1.plot(bf, [100 * p["bias"][str(b)]["tpot_none"] for b in bf], color=CB["orange"], marker="o", lw=1.6, label="PACE w/o adaptation")
    a1.plot(bf, [100 * p["bias"][str(b)]["tpot_adapt"] for b in bf], color=CB["green"], marker="D", lw=1.6, label="PACE (adaptive)")
    a1.axvline(1.0, color=CB["grey"], ls=":", lw=0.8)
    a1.set_xlabel("Prediction bias (predicted / true)")
    a1.set_ylabel("TPOT miss (%)")
    a1.legend(fontsize=8, frameon=False)
    labs = [f"bias {b:g}" for b in bf] + ["history q=0.5", "history q=0.75", "history q=0.9"]
    vals = [100 * (p["bias"][str(b)]["E_adapt"] / p["perfect_E"] - 1) for b in bf] + \
           [100 * (p["history"][q]["E_adapt"] / p["perfect_E"] - 1) for q in ("0.5", "0.75", "0.9")]
    cols = [CB["green"]] * len(bf) + [CB["purple"]] * 3
    bars = a2.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.5)
    for bb, v in zip(bars, vals):
        a2.text(bb.get_x() + bb.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    a2.set_xticks(range(len(vals)))
    a2.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
    a2.set_ylabel("Energy premium of adaptive PACE\nover perfect prediction (%)")
    _save(fig, "fig_rev_predictor")


def fig_contention():
    d = RES["contention"]
    keys = list(d)                                  # inf, 120, 90, 75, 72, 70, 68, 66, 64
    xs = list(range(len(keys)))
    labs = ["inf" if k == "inf" else k for k in keys]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 5.4))
    for k_m, k_b, c, mk, lab in [("shared_crit_miss", "shared_be", CB["grey"], "s", "Shared-NPU FCFS"),
                                 ("gpu_crit_miss", "gpu_be", CB["sky"], "^", "Shared-GPU (time-sliced)"),
                                 ("pace_crit_miss", "pace_be", CB["orange"], "o", "PACE placement only"),
                                 ("res_crit_miss", "res_be", CB["green"], "D", "PACE + bandwidth reservation")]:
        a1.plot(xs, [d[k][k_m] for k in keys], color=c, marker=mk, lw=1.6, label=lab)
        a2.plot(xs, [d[k][k_b] for k in keys], color=c, marker=mk, lw=1.6, label=lab)
    for ax in (a1, a2):
        ax.set_xticks(xs)
        ax.set_xticklabels([r"$\infty$" if l == "inf" else l for l in labs])
        ax.set_xlabel("Shared DRAM bandwidth (GB/s)")
    a1.set_ylabel("Critical-stream TPOT miss (%)")
    a2.set_ylabel("Best-effort throughput (tok/s)")
    a1.legend(fontsize=7, frameon=False, loc="center left")
    _save(fig, "fig_rev_contention")


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


EXPERIMENTS = [exp_validate, exp_stats, exp_mixed, exp_hetero, exp_handoff, exp_dvfs,
               exp_energy_model, exp_predictor, exp_scale, exp_context, exp_7b4k,
               exp_classes, exp_baselines, exp_contention, exp_batching]
FIGURES = [fig_mixed, fig_hetero, fig_handoff, fig_scale, fig_predictor, fig_contention]
JSON_PATH = os.path.join(OUT, "revision_results.json")

if __name__ == "__main__":
    # python pace_revision_experiments.py            -> everything
    # python pace_revision_experiments.py stats mixed -> only those (results are merged)
    import sys
    t0 = time.time()
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            RES.update(json.load(f))
    wanted = sys.argv[1:]
    for fn in EXPERIMENTS:
        if wanted and fn.__name__[4:] not in wanted:
            continue
        t = time.time()
        fn()
        with open(JSON_PATH, "w") as f:
            json.dump(RES, f, indent=1, cls=_Enc)
        print(f"--- {fn.__name__} done in {time.time() - t:.1f}s", flush=True)
    if not wanted or "figures" in wanted:
        for fn in FIGURES:
            fn()
        print("figures written", flush=True)
    print(f"done in {time.time() - t0:.1f}s; results in ./{OUT}/")
