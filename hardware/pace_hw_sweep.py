#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pace_hw_sweep.py - measurement engine of the PACE hardware validation (NVIDIA GPU, Windows or Linux)

What it does
  It locks the GPU core clock and the GPU memory clock to each point of a grid. At every point it
  measures prefill and decode of a GGUF model with llama.cpp's llama-bench: time per token, power
  and energy. This checks the main claim of the paper on real hardware: prefill should follow the
  core clock and decode should follow the memory clock.
"""
import argparse
import collections
import csv
import json
import math
import os
import platform
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime

VERSION = "1.1"
FULL_MEMS = "6001,7001,8001,0"               # memory clock locks of the full sweep (MHz; 0 = driver default)

# Reasons the driver gives when it holds the clock below the request (bit mask from NVML).
# Bit 0x1 (GPU idle) and 0x2 (user-defined clocks, which is our own lock) are not limits.
REASONS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal",
           0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clock", 0x200: "board_limit",
           0x400: "reliability"}

FIELDS = ["run", "stage", "pass", "time", "gpu", "model", "core_lock_MHz", "mem_lock_MHz", "phase",
          "n_prompt", "n_gen", "n_depth", "reps", "reps_used", "rep_ms_mean", "rep_ms_median", "rep_ms_std",
          "ms_per_token", "tok_per_s", "power_W", "power_src", "power_counter_W", "power_instant_W",
          "power_avg_W", "idle_W", "energy_mJ_per_token", "energy_J_per_rep", "core_MHz", "mem_MHz",
          "util_gpu", "util_mem", "temp_C", "limit_W", "vram_MB", "throttle", "window_s", "note"]

ADMIN = ("The driver did not allow changing the GPU clocks.\n"
         "  Windows: close this window, open PowerShell with 'Run as administrator', go to this folder\n"
         "           again and repeat the command.\n"
         "  Linux:   run the command with sudo.")
RESET_HELP = "WARNING: could not reset the clocks. Please run:  nvidia-smi -rgc   and   nvidia-smi -rmc"

Test = collections.namedtuple("Test", "phase n_prompt n_gen depth")


class Stop(Exception):
    """A problem the user has to fix. Printed without a traceback."""


class BenchError(Exception):
    """llama-bench did not finish a test."""


# ----------------------------------------------------------------------------- small helpers
LOG = []                                     # everything printed, also saved into the zip file
LIVE = None                                  # optional file that receives every printed line at once
RESUME_HINT = None                           # main.py puts its own advice here
CLEANUP = []                                 # called when the console window is closed (Windows)
_KEEP = []                                   # keeps the Windows callback object alive


def say(msg=""):
    LOG.append(str(msg))
    if LIVE is not None:
        try:
            LIVE.write(str(msg) + "\n")
            LIVE.flush()
        except (OSError, ValueError):
            pass
    try:
        print(msg, flush=True)
    except (OSError, ValueError):            # the console is gone: keep working, the results are still saved
        pass


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rnd(x, n=3):
    return "" if x is None else round(x, n)


def rint(x):
    return "" if x is None else int(round(x))


def ints(text):
    return [int(v) for v in str(text).replace(";", ",").split(",") if v.strip() != ""]


def lab(mhz):
    return "auto" if not mhz else str(mhz)


def watts(x):
    return "%.1f W" % x if x is not None else "-"


def minutes(m):
    return "about %.0f minute%s" % (m, "" if round(m) == 1 else "s")


def valid_w(x, p_max):
    """A power reading in W, or None if it cannot be real. While a laptop GPU sleeps or wakes up,
    some drivers report values such as 590 W; anything above 1.5 times the power limit is dropped."""
    return x if x is not None and 0.5 < x <= p_max else None


def test_name(t):
    return "prefill %d" % t.n_prompt if t.phase == "prefill" else "decode d%d" % t.depth


def text_of(x):
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def here():
    return os.path.dirname(os.path.abspath(__file__))


def os_setup():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetErrorMode(ctypes.c_uint(0x8003))                  # no "missing DLL" pop-up from llama-bench
            k.SetThreadExecutionState(ctypes.c_uint(0x80000003))   # keep the laptop awake during the run
        except Exception:
            pass
        try:                                                       # closing the window must also free the clocks
            import ctypes
            from ctypes import wintypes

            def on_console_event(event):
                if event in (2, 5, 6):                             # window closed, log-off, shutdown
                    for fn in list(CLEANUP):
                        try:
                            fn()
                        except Exception:
                            pass
                return False                                       # Windows continues with its normal handling

            if not _KEEP:
                _KEEP.append(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(on_console_event))
                ctypes.windll.kernel32.SetConsoleCtrlHandler(_KEEP[0], True)
        except Exception:
            pass


# ----------------------------------------------------------------------------- GPU sensors (NVML)
class GPU:
    def __init__(self, N, index):
        self.N, self.index = N, index
        try:
            N.nvmlInit()
            self.h = N.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            raise Stop("Cannot reach the NVIDIA driver (%s). Is the NVIDIA driver installed, and is this "
                       "the right GPU index (--gpu)?" % exc)
        self.name = text_of(self._try(lambda: N.nvmlDeviceGetName(self.h)) or "GPU %d" % index)
        self.driver = text_of(self._try(N.nvmlSystemGetDriverVersion) or "?")
        self.count = int(self._try(N.nvmlDeviceGetCount) or 1)
        self._reasons = (getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None)
                         or getattr(N, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
        self.has_power = self._try(lambda: N.nvmlDeviceGetPowerUsage(self.h)) is not None
        self.has_energy = self._try(lambda: N.nvmlDeviceGetTotalEnergyConsumption(self.h)) is not None
        self.has_instant = self._instant() is not None
        self.has_util = self._try(lambda: N.nvmlDeviceGetUtilizationRates(self.h)) is not None
        sm = self._try(lambda: N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_SM))
        self._core_id = N.NVML_CLOCK_SM if sm is not None else N.NVML_CLOCK_GRAPHICS
        limits = [x for x in (self.limit(), self.default_limit()) if x]
        self.p_max = 1.5 * max(limits) if limits else 450.0       # higher power readings are not real

    @staticmethod
    def _try(fn):
        try:
            return fn()
        except Exception:
            return None

    def _instant(self):
        """Instantaneous power in W (NVML field 186); None when the GPU does not report it."""
        N = self.N
        try:
            v = N.nvmlDeviceGetFieldValues(self.h, [getattr(N, "NVML_FI_DEV_POWER_INSTANT", 186)])[0]
            if v.nvmlReturn != 0:
                return None
            u = v.value
            raw = (u.dVal, u.uiVal, u.ulVal, u.ullVal, u.sllVal, u.siVal, u.usVal)[int(v.valueType)]
            return float(raw) / 1000.0
        except Exception:
            return None

    def temperature(self):
        N = self.N
        try:
            return N.nvmlDeviceGetTemperature(self.h, N.NVML_TEMPERATURE_GPU)
        except Exception:
            return self._try(lambda: N.nvmlDeviceGetTemperatureV(self.h, N.NVML_TEMPERATURE_GPU))

    def clock(self, memory=False):
        N = self.N
        return self._try(lambda: N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_MEM if memory else self._core_id))

    def limit(self):
        v = self._try(lambda: self.N.nvmlDeviceGetEnforcedPowerLimit(self.h))
        return None if v is None else v / 1000.0

    def default_limit(self):
        v = self._try(lambda: self.N.nvmlDeviceGetPowerManagementDefaultLimit(self.h))
        return None if v is None else v / 1000.0

    def vram_mb(self):
        v = self._try(lambda: self.N.nvmlDeviceGetMemoryInfo(self.h).used)
        return None if v is None else v / 1048576.0

    def on_battery(self):
        N = self.N
        return self._try(lambda: N.nvmlDeviceGetPowerSource(self.h)) == getattr(N, "NVML_POWER_SOURCE_BATTERY", 1)

    def sample(self):
        """One reading: (t, average power W, instant power W, energy J, core MHz, memory MHz,
        temperature, GPU utilisation, memory utilisation, clock-limit reasons)."""
        N, h = self.N, self.h
        t = time.perf_counter()
        p = e = ug = um = None
        if self.has_power:
            try:
                p = N.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
        pi = self._instant() if self.has_instant else None
        if self.has_energy:
            try:
                e = N.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
            except Exception:
                pass
        if self.has_util:
            try:
                u = N.nvmlDeviceGetUtilizationRates(h)
                ug, um = int(u.gpu), int(u.memory)
            except Exception:
                pass
        try:
            rs = int(self._reasons(h)) if self._reasons else 0
        except Exception:
            rs = 0
        return (t, p, pi, e, self.clock(False), self.clock(True), self.temperature(), ug, um, rs)

    def idle_power(self, seconds=3.0):
        vals, t_end = [], time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            vals.append(valid_w(self.sample()[1], self.p_max))
            time.sleep(0.05)
        return med(vals)

    def close(self):
        self._try(self.N.nvmlShutdown)


# ----------------------------------------------------------------------------- clock locks
class Clocks:
    """Locks or frees the core clock and the memory clock. Uses NVML and falls back to nvidia-smi."""

    def __init__(self, gpu):
        self.gpu, self.use_smi, self.locked_once = gpu, False, False
        self.locked = {"core": 0, "mem": 0}          # what this script has locked (MHz); 0 = nothing

    def _nvml(self, domain, mhz):
        N, h = self.gpu.N, self.gpu.h
        if domain == "core":
            if mhz:
                N.nvmlDeviceSetGpuLockedClocks(h, mhz, mhz)
            else:
                N.nvmlDeviceResetGpuLockedClocks(h)
        elif mhz:
            N.nvmlDeviceSetMemoryLockedClocks(h, mhz, mhz)
        else:
            N.nvmlDeviceResetMemoryLockedClocks(h)

    def _smi(self, domain, mhz):
        flag = {"core": ("-rgc", "-lgc"), "mem": ("-rmc", "-lmc")}[domain][1 if mhz else 0]
        cmd = ["nvidia-smi"] + (["-i", str(self.gpu.index)] if self.gpu.count > 1 else []) + [flag]
        if mhz:
            cmd.append("%d,%d" % (mhz, mhz))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
        except FileNotFoundError:
            raise Stop("nvidia-smi was not found. It comes with the NVIDIA driver; please check the driver.")
        out = (r.stdout + "\n" + r.stderr).strip()
        low = out.lower()
        if "permission" in low:
            raise Stop(ADMIN)
        if r.returncode != 0 or "terminating early" in low or "not supported" in low or "invalid" in low:
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            raise ValueError("%s: %s" % (" ".join(cmd), lines[0] if lines else "error %s" % r.returncode))

    def _apply(self, domain, mhz):
        if not self.use_smi:
            for attempt in (1, 2):
                try:
                    self._nvml(domain, mhz)
                    return
                except Exception as exc:
                    code = getattr(exc, "value", None)
                    if code == 4:
                        raise Stop(ADMIN)
                    if code == 2:
                        raise ValueError("the driver does not accept a %s clock of %s MHz" % (
                            "memory" if domain == "mem" else domain, mhz))
                    if code in (3, 13) or isinstance(exc, AttributeError):
                        break                # not supported or not in this NVML: a second try cannot help
                    time.sleep(0.05)         # anything else may be a passing problem: try once more
            if mhz:
                self.use_smi = True          # locking through NVML is not available here: use nvidia-smi from now on
        self._smi(domain, mhz)

    def change(self, domain, mhz):
        """Lock one clock (domain 'core' or 'mem') to mhz; mhz = 0 gives it back to the driver."""
        mhz = int(mhz or 0)
        try:
            self._apply(domain, mhz)
        except ValueError:
            if mhz or self.locked[domain]:
                raise
            return                           # freeing a clock that is not locked: some drivers answer with an error
        self.locked[domain] = mhz
        self.locked_once = self.locked_once or bool(mhz)

    def set(self, core, mem):
        """core / mem in MHz; 0 means: leave this clock to the driver."""
        self.change("core", core)
        self.change("mem", mem)

    def reset(self):
        """Give both clocks back to the driver, through NVML and, if that fails, through nvidia-smi.
        Never raises. False if a lock of ours may still be active."""
        ok = True
        for domain in ("core", "mem"):
            freed = False
            for way in (self._nvml, self._smi):
                try:
                    way(domain, 0)
                    freed = True
                    break
                except BaseException:
                    pass
            if freed:
                self.locked[domain] = 0
            else:
                ok = ok and not self.locked[domain]  # an error for a clock we never locked does not matter
        return ok

    def finish(self):
        """Reset at the end of a run and tell the user what happened."""
        was_locked = self.locked_once
        if not self.reset():
            say(RESET_HELP)
        elif was_locked:
            say("Clocks returned to default.")


# ----------------------------------------------------------------------------- llama-bench
def parse_bench(out):
    i = out.find("[")
    if i >= 0:
        try:
            data = json.loads(out[i:])
            if data:
                return data[-1]
        except ValueError:
            pass
    e = {}                                   # fallback: read the numbers directly
    for k in ("n_prompt", "n_gen", "n_depth", "avg_ns", "stddev_ns", "build_number", "model_size", "model_n_params"):
        m = re.search(r'"%s":\s*(-?\d+)' % k, out)
        if m:
            e[k] = int(m.group(1))
    for k in ("avg_ts", "stddev_ts"):
        m = re.search(r'"%s":\s*(-?[0-9.eE+-]+)' % k, out)
        if m:
            e[k] = float(m.group(1))
    for k in ("backends", "gpu_info", "cpu_info", "model_type", "build_commit"):
        m = re.search(r'"%s":\s*"([^"]*)"' % k, out)
        if m:
            e[k] = m.group(1)
    m = re.search(r'"samples_ns":\s*\[([^\]]*)\]', out)
    if m:
        e["samples_ns"] = [int(v) for v in m.group(1).split(",") if v.strip()]
    if "avg_ns" not in e:
        raise BenchError("could not read the llama-bench result")
    return e


def bench_cmd(ctx, extra):
    """llama-bench is started inside the model folder and gets the bare file name. On Windows it
    reads its arguments in the local code page, so a folder name with non-English letters would
    otherwise not be found."""
    return [ctx.bench, "-m", ctx.model_name, "-ngl", str(ctx.ngl), "-o", "json"] + ctx.bench_args + extra


def run_bench(ctx, extra, timeout=600.0):
    """Run one llama-bench test and sample the GPU while it runs.
    Returns a dict: res (llama-bench result), samples, slow (power limit and GPU memory, read twice
    a second), t_start, t_exit and t_result (the moment the result was written, which is the end of
    the last repetition)."""
    cmd = bench_cmd(ctx, extra)
    errors, samples, slow = collections.deque(maxlen=40), [], []
    with tempfile.TemporaryFile() as fo:
        try:
            proc = subprocess.Popen(cmd, stdout=fo, stderr=subprocess.PIPE, cwd=ctx.model_dir)
        except OSError as exc:
            raise Stop("Could not start llama-bench (%s): %s" % (ctx.bench, exc))

        def reader():                                            # keeps the last messages for error reports
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line.strip():
                    errors.append(line)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        t_start = t_slow = now = time.perf_counter()
        t_sample, t_result = 0.0, None
        try:
            while proc.poll() is None:
                now = time.perf_counter()
                # llama-bench writes its result into the output file right after the last repetition.
                # That moment is the end of the measured work. (Its --progress lines cannot be used for
                # timing: on Windows they are held back in a buffer until the program ends.)
                if t_result is None and os.fstat(fo.fileno()).st_size > 64:    # more than the opening bracket
                    t_result = now
                if now - t_sample >= ctx.period:
                    samples.append(ctx.gpu.sample())
                    t_sample = now
                if now - t_slow >= 0.5:
                    slow.append((now, ctx.gpu.limit(), ctx.gpu.vram_mb()))
                    t_slow = now
                if now - t_start > timeout:
                    raise BenchError("no result after %.0f s (stopped)" % timeout)
                time.sleep(0.005)
            t_exit = time.perf_counter()
            if t_result is None and os.fstat(fo.fileno()).st_size > 64:
                t_result = (now + t_exit) / 2.0                  # written between the last look and the end
            samples.append(ctx.gpu.sample())
        finally:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
            th.join(timeout=3)
            try:
                proc.stderr.close()
            except Exception:
                pass
        fo.seek(0)
        out = fo.read().decode("utf-8", "replace")
    if proc.returncode != 0:
        if proc.returncode & 0xFFFFFFFF == 0xC0000135:
            raise Stop("llama-bench could not start because a DLL file is missing.\n"
                       "  If you set up llama.cpp by hand: unzip the 'cudart-llama-bin-win-cuda-...' file into the\n"
                       "  same folder as llama-bench.exe.\n"
                       "  Otherwise install the Microsoft Visual C++ runtime and try again:\n"
                       "      https://aka.ms/vs/17/release/vc_redist.x64.exe")
        tail = " | ".join(list(errors)[-3:]) or "no message"
        raise BenchError("llama-bench stopped with code %s: %s" % (proc.returncode, tail[:400]))
    return {"res": parse_bench(out), "samples": samples, "slow": slow, "messages": list(errors),
            "t_start": t_start, "t_exit": t_exit, "t_result": t_result}


def analyse(run, settle, p_max):
    """Find the repetitions that ran after the GPU had settled, and the power while they ran.

    The repetitions run back to back, and the last one ends when llama-bench writes its result, so
    the start and end of every repetition are known. Repetitions that start less than 'settle'
    seconds after the first one are not used: in that time a laptop GPU wakes up, raises its clocks,
    and its sensors still show older values."""
    res, samples = run["res"], run["samples"]
    reps_ns = res.get("samples_ns") or [res["avg_ns"]] * max(1, int(run.get("reps") or 1))
    notes = []
    if run.get("t_result") is not None:
        t_end = run["t_result"]
    else:                                                            # result not seen before the end: estimate
        t_end = run["t_exit"] - 0.15
        notes.append("window_estimated")
    starts, t = [], t_end
    for ns in reversed(reps_ns):
        t -= ns / 1e9
        starts.append(t)
    starts.reverse()
    use = [i for i, s in enumerate(starts) if s >= starts[0] + settle - 1e-6]
    if not use:                                                      # too short: only the last repetition
        use = [len(reps_ns) - 1]
        notes.append("short_run")
    t0, t1 = starts[use[0]], t_end
    dur = max(t1 - t0, 0.0)
    inside = [s for s in samples if t0 <= s[0] <= t1]

    p_counter = p_instant = p_avg = None
    lo = t0 + min(0.25, 0.1 * dur)
    if dur >= 1.0:                                                   # energy counter: delta E / delta t
        e = [(s[0], s[3]) for s in samples if s[3] is not None]
        # The counter may be updated less often than we read it, so only the readings at which
        # it moved are used as end points.
        e = [e[i] for i in range(1, len(e)) if e[i][1] != e[i - 1][1] and lo <= e[i][0] <= t1]
        if len(e) >= 2 and e[-1][0] - e[0][0] >= 0.5 * dur and e[-1][1] > e[0][1]:
            p_counter = (e[-1][1] - e[0][1]) / (e[-1][0] - e[0][0])
    v = [valid_w(s[2], p_max) for s in samples if lo <= s[0] <= t1]
    v = [x for x in v if x is not None]
    if len(v) >= 5:
        p_instant = mean(v)
    lag = min(1.0, 0.4 * dur)                                        # this reading is a 1-second average
    v = [valid_w(s[1], p_max) for s in samples if t0 + lag <= s[0] <= t1 + 0.05]
    v = [x for x in v if x is not None]
    if len(v) >= 5:
        p_avg = mean(v)
    if dur < 2.0:
        notes.append("short_window")
    if p_instant is not None:
        power, src = p_instant, "instant"
    elif p_avg is not None:
        power, src = p_avg, "average"
    elif p_counter is not None and p_counter <= p_max:
        power, src = p_counter, "counter"
    else:
        power, src = None, ""
        notes.append("no_power")
    flags = 0
    for s in inside:
        flags |= s[9] or 0
    core, mem = med([s[4] for s in inside]), med([s[5] for s in inside])

    def unsteady(k, m):
        xs = [s[k] for s in inside if s[k] is not None]
        return bool(xs and m) and sum(1 for x in xs if abs(x - m) > max(30, 0.03 * m)) > 0.2 * len(xs)

    if unsteady(4, core) or unsteady(5, mem):
        notes.append("clock_unsteady")
    ug = med([s[7] for s in inside])
    if ug is not None and ug < 50:
        notes.append("low_util")
    slow = [x for x in run["slow"] if t0 <= x[0] <= t1] or run["slow"][-1:]
    return {"t0": t0, "t1": t1, "dur": dur, "power": power, "src": src, "p_counter": p_counter,
            "p_instant": p_instant, "p_avg": p_avg, "core": core, "mem": mem,
            "temp": max([s[6] for s in inside if s[6] is not None], default=None),
            "ug": ug, "um": med([s[8] for s in inside]),
            "limit": med([x[1] for x in slow]), "vram": max([x[2] for x in slow if x[2] is not None], default=None),
            "throttle": "|".join(name for bit, name in REASONS.items() if flags & bit) or "none", "notes": notes,
            "reps_all": len(reps_ns), "win_ns": [reps_ns[i] for i in use]}


def bench_args_for(test, reps):
    a = ["-p", str(test.n_prompt), "-n", str(test.n_gen), "-r", str(reps)]
    if test.depth > 0:
        a += ["-d", str(test.depth)]
    return a


def reps_for(t_rep, target, low=3, high=4000):
    return int(max(low, min(high, math.ceil(target / max(t_rep, 1e-4)))))


# ----------------------------------------------------------------------------- one measurement
class Ctx:
    """Everything a measurement needs (paths, GPU, open files, counters)."""


def make_row(run_no, stage, pass_no, when, gpu, model, idle, test, core, mem, st):
    """One CSV line from the result of analyse()."""
    rep_ms = [x / 1e6 for x in st["win_ns"]]
    n_tok = test.n_prompt if test.phase == "prefill" else test.n_gen
    rep_mean = sum(rep_ms) / len(rep_ms)
    ms_tok = rep_mean / n_tok
    power = st["power"]
    notes = list(st["notes"])
    if core and st["core"] and abs(st["core"] - core) > max(30, 0.03 * core):
        notes.append("core_lock_not_held")
    if mem and st["mem"] and abs(st["mem"] - mem) > max(30, 0.03 * mem):
        notes.append("mem_lock_not_held")
    return {"run": run_no, "stage": stage, "pass": pass_no, "time": when,
            "gpu": gpu, "model": model, "core_lock_MHz": core, "mem_lock_MHz": mem, "phase": test.phase,
            "n_prompt": test.n_prompt, "n_gen": test.n_gen, "n_depth": test.depth, "reps": st["reps_all"],
            "reps_used": len(rep_ms), "rep_ms_mean": rnd(rep_mean, 4), "rep_ms_median": rnd(statistics.median(rep_ms), 4),
            "rep_ms_std": rnd(statistics.stdev(rep_ms) if len(rep_ms) > 1 else 0.0, 4),
            "ms_per_token": rnd(ms_tok, 5), "tok_per_s": rnd(1000.0 / ms_tok, 2),
            "power_W": rnd(power, 2), "power_src": st["src"], "power_counter_W": rnd(st["p_counter"], 2),
            "power_instant_W": rnd(st["p_instant"], 2), "power_avg_W": rnd(st["p_avg"], 2), "idle_W": rnd(idle, 2),
            "energy_mJ_per_token": rnd(power * ms_tok if power is not None else None, 4),
            "energy_J_per_rep": rnd(power * rep_mean / 1000.0 if power is not None else None, 5),
            "core_MHz": rint(st["core"]), "mem_MHz": rint(st["mem"]), "util_gpu": rint(st["ug"]),
            "util_mem": rint(st["um"]), "temp_C": rint(st["temp"]), "limit_W": rnd(st["limit"], 1),
            "vram_MB": rint(st["vram"]), "throttle": st["throttle"], "window_s": rnd(st["dur"], 2),
            "note": "|".join(notes)}


def measure(ctx, test, core, mem, stage, pass_no, guess, target=None):
    """Run one test at the current clock setting, write one CSV line and one raw record, return the line."""
    target = target or ctx.target
    reps = reps_for(guess, ctx.settle + target) + 1                  # +1: the repetition across the settle time
    run = st = None
    for attempt in range(3):
        run = run_bench(ctx, bench_args_for(test, reps), timeout=max(300.0, 60.0 * (target + ctx.settle)))
        run["reps"] = reps
        st = analyse(run, ctx.settle, ctx.gpu.p_max)
        if st["dur"] >= 0.6 * target or reps >= 4000 or attempt == 2:
            break
        all_ns = run["res"].get("samples_ns") or [run["res"]["avg_ns"]]
        t_rep = sum(all_ns) / len(all_ns) / 1e9                      # too short: repeat with more repetitions
        reps = reps_for(t_rep, 1.1 * (ctx.settle + target)) + 1
    res = run["res"]
    ctx.run += 1
    row = make_row(ctx.run, stage, pass_no, datetime.now().isoformat(timespec="seconds"), ctx.gpu.name, ctx.stem,
                   ctx.idle, test, core, mem, st)
    ctx.writer.writerow(row)
    ctx.csv_file.flush()
    t_start = run["t_start"]
    raw = dict(row)
    raw.update({"kind": "run", "samples_ns": res.get("samples_ns") or [res["avg_ns"]], "settle_s": ctx.settle,
                "window": [round(st["t0"] - t_start, 4), round(st["t1"] - t_start, 4)],
                "t_result": None if run["t_result"] is None else round(run["t_result"] - t_start, 4),
                "t_exit": round(run["t_exit"] - t_start, 4),
                "bench": {k: res.get(k) for k in ("avg_ns", "stddev_ns", "avg_ts", "stddev_ts", "n_depth", "build_number",
                                                  "build_commit", "backends", "gpu_info", "cpu_info", "model_type",
                                                  "model_size", "model_n_params", "n_batch", "n_ubatch", "n_threads",
                                                  "flash_attn", "type_k", "type_v")}})
    if ctx.trace:
        c = list(zip(*run["samples"])) if run["samples"] else [()] * 10

        def r2(col, n=2):
            return [None if x is None else round(x, n) for x in col]

        raw["trace"] = {"t": r2([t - t_start for t in c[0]], 4), "p_avg": r2(c[1]), "p_inst": r2(c[2]), "e_J": r2(c[3], 3),
                        "core": list(c[4]), "mem": list(c[5]), "temp": list(c[6]), "util_gpu": list(c[7]),
                        "util_mem": list(c[8]), "reasons": list(c[9])}
        raw["slow"] = [[round(x[0] - t_start, 3), x[1], None if x[2] is None else round(x[2], 1)] for x in run["slow"]]
    ctx.raw_file.write(json.dumps(raw) + "\n")
    ctx.raw_file.flush()
    return row


def show(row, prefix=""):
    e, p = row["energy_mJ_per_token"], row["power_W"]
    say("%s%-13s core %-5s mem %-5s| %8.4f ms/tok %8.1f tok/s | %5s W %8s mJ/tok | %s/%s MHz%s%s" % (
        prefix, test_name(Test(row["phase"], row["n_prompt"], row["n_gen"], row["n_depth"])),
        lab(row["core_lock_MHz"]), lab(row["mem_lock_MHz"]), row["ms_per_token"], row["tok_per_s"],
        "-" if p == "" else "%.1f" % p, "-" if e == "" else "%.3f" % e, row["core_MHz"], row["mem_MHz"],
        "" if row["throttle"] == "none" else "  limit: " + row["throttle"], "" if not row["note"] else "  [" + row["note"] + "]"))


# ----------------------------------------------------------------------------- plan
def plan(args):
    if args.quick:
        args.cores, args.mems = args.cores or "1200,0", args.mems or "6001,0"
        args.prompts, args.depths = args.prompts or "512", args.depths or "0,1024"
        args.char_prompts, args.char_depths = args.char_prompts or "128,1024", args.char_depths or "0"
        args.passes = args.passes or 1
        args.target = args.target or 3.0
    else:
        args.cores = args.cores or "600,900,1200,1500,1800,2100,2400,0"
        args.mems = args.mems or FULL_MEMS
        args.prompts, args.depths = args.prompts or "128,1024", args.depths or "0,1024"
        args.char_prompts = args.char_prompts or "32,64,128,256,512,1024,2048"
        args.char_depths = args.char_depths or "0,512,1024,2048,4096"
        args.passes = args.passes or 2
        args.target = args.target or 5.0
    cores, mems = ints(args.cores), ints(args.mems)
    ref_c = 0 if 0 in cores else max(cores)
    ref_m = 0 if 0 in mems else max(mems)
    if args.grid == "cross":
        settings = [(c, ref_m) for c in cores] + [(ref_c, m) for m in mems]
    else:
        settings = [(c, m) for m in mems for c in cores]
    settings = list(dict.fromkeys(settings))
    grid_tests = [Test("prefill", p, 0, 0) for p in ints(args.prompts)] + \
                 [Test("decode", 0, args.gen, d) for d in ints(args.depths)]
    char_tests = [] if args.no_char else \
        [Test("prefill", p, 0, 0) for p in ints(args.char_prompts)] + \
        [Test("decode", 0, args.gen, d) for d in ints(args.char_depths)]
    return settings, grid_tests, char_tests


def order_for(settings, pass_no, seed):
    """Default clocks first, the other settings in a random order that differs between passes."""
    rest = [s for s in settings if s != (0, 0)]
    random.Random(seed + pass_no).shuffle(rest)
    return ([(0, 0)] if (0, 0) in settings else []) + rest


def describe(args, settings, grid_tests, char_tests):
    n_all = len(settings) * len(grid_tests) * args.passes + len(char_tests) + 1
    say("pace_hw_sweep %s" % VERSION)
    say("Plan")
    say("  clock settings   : %d  (core lock %s MHz  x  memory lock %s MHz, grid '%s'; 0 = driver default)" % (
        len(settings), args.cores, args.mems, args.grid))
    say("  tests per setting: %s" % ", ".join(test_name(t) for t in grid_tests))
    say("  passes           : %d%s" % (args.passes, ", each in a different random order" if args.passes > 1 else ""))
    say("  at default clocks: %s" % (", ".join(test_name(t) for t in char_tests) or "(skipped)"))
    say("  measurements     : %d, each runs for about %.0f s -> %s for a 1B model" % (
        n_all, args.target + args.settle, minutes(n_all * (args.target + args.settle + 3.5) / 60.0)))
    return n_all


# ----------------------------------------------------------------------------- sweep
def find_bench(args):
    names = ["llama-bench.exe", "llama-bench"] if os.name == "nt" else ["llama-bench", "llama-bench.exe"]
    if args.bench:
        cands = [args.bench]
    else:
        folders = [os.getcwd(), here()] + ([os.path.dirname(os.path.abspath(args.model))] if args.model else [])
        cands = [os.path.join(f, n) for f in folders for n in names]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise Stop("llama-bench was not found. Put this script in the folder that contains llama-bench%s, or give "
               "its place with --bench." % (".exe" if os.name == "nt" else ""))


def load_nvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        raise Stop("The Python package 'nvidia-ml-py' is missing. Install it once with:\n"
                   "    python -m pip install nvidia-ml-py")


def open_ctx(args):
    ctx = Ctx()
    if not args.model or not os.path.isfile(args.model):
        raise Stop("Model file not found: %s" % args.model)
    ctx.model = os.path.abspath(args.model)
    ctx.model_dir, ctx.model_name = os.path.split(ctx.model)
    ctx.stem = os.path.splitext(ctx.model_name)[0]
    ctx.bench = find_bench(args)
    ctx.ngl, ctx.period, ctx.trace = args.ngl, args.period, not args.no_trace
    ctx.bench_args = shlex.split(args.bench_args, posix=(os.name != "nt")) if args.bench_args else []
    ctx.gpu = GPU(load_nvml(), args.gpu)
    ctx.clocks = Clocks(ctx.gpu)
    ctx.run, ctx.idle, ctx.target = 0, None, args.target or 5.0
    ctx.settle = max(0.0, args.settle)
    return ctx


def check_rights(ctx, core):
    """Stops with a clear message when the driver does not let us lock clocks."""
    ctx.clocks.reset()
    try:
        ctx.clocks.set(core, 0)
    except ValueError as exc:
        say("  note: the test lock was not accepted (%s)" % exc)
    finally:
        ctx.clocks.reset()


def preflight(ctx, args, first_core):
    """Checks before the sweep: charger, rights, llama-bench build, power limit under load.
    Returns the speed at default clocks, used to choose the number of repetitions."""
    g = ctx.gpu
    say("GPU      : %s (driver %s)" % (g.name, g.driver))
    say("Sensors  : power %s, instantaneous power %s, energy counter %s" % (
        "yes" if g.has_power else "NO", "yes" if g.has_instant else "no", "yes" if g.has_energy else "no"))
    if g.on_battery() and not args.force:
        raise Stop("The laptop runs on battery. Plug in the charger and run again (or add --force).")
    check_rights(ctx, first_core)
    say("Locks    : allowed (through %s)" % ("nvidia-smi" if ctx.clocks.use_smi else "NVML"))
    time.sleep(1.5)
    ctx.idle = g.idle_power()
    say("Idle     : %s, power limit now %s W (default %s W)" % (
        watts(ctx.idle) if ctx.idle is not None else "no valid reading (the GPU is asleep)", rint(g.limit()),
        rint(g.default_limit())))

    probe = Test("prefill", 512, 0, 0)
    try:
        first = run_bench(ctx, bench_args_for(probe, 3))
    except BenchError as exc:
        raise Stop("llama-bench could not run the model.\n  %s" % exc)
    pilot = first["res"]
    ctx.loaded = [m.strip()[:160] for m in first["messages"] if "load_backend" in m]
    backends, gpu_info = str(pilot.get("backends", "")), str(pilot.get("gpu_info", ""))
    if (backends.upper() == "CPU" or not gpu_info) and not args.force:
        raise Stop("llama-bench ran on the CPU, not on the GPU (backends: %s): its CUDA part did not load.\n"
                   "  Check that the NVIDIA driver works: the command nvidia-smi must show the GPU.\n"
                   "  If it does, delete the folder with llama.cpp (tools\\llama.cpp) and run main.py again, so that\n"
                   "  the files are unpacked again. If you set up llama.cpp by hand, use the 'bin-win-cuda' download\n"
                   "  and unzip the matching 'cudart-llama-bin-win-cuda' file into the same folder.\n"
                   "  What llama-bench reported:\n    %s" % (backends or "?", "\n    ".join(ctx.loaded) or "(nothing)"))
    say("Model    : %s, %.2f GB, %.2f B parameters | llama.cpp build %s, backend %s" % (
        pilot.get("model_type", "?"), (pilot.get("model_size") or 0) / 1e9, (pilot.get("model_n_params") or 0) / 1e9,
        pilot.get("build_number", "?"), backends or "?"))
    try:
        row = measure(ctx, probe, 0, 0, "preflight", 0, pilot["avg_ns"] / 1e9, target=max(3.0, min(ctx.target, 5.0)))
        show(row, "  ")
        p_s, p_a, p_c = row["power_instant_W"], row["power_avg_W"], row["power_counter_W"]
        say("Power    : GPU power sensor %s, its 1-second average %s, energy counter %s" % tuple(
            "-" if x == "" else "%.1f W" % x for x in (p_s, p_a, p_c)))
        ref = p_s if p_s != "" else p_a
        if p_c != "" and ref != "" and abs(p_c - ref) > max(5.0, 0.1 * ref):
            say("           The energy counter does not match the power sensor on this GPU. All results use the sensor.")
        else:
            say("           All results use the power sensor.")
        limit = row["limit_W"]
        if limit != "" and limit < args.min_power_limit and not args.force:
            raise Stop("The GPU power limit is only %.0f W while it works (default %s W), so the laptop is in a "
                       "power-saving mode.\n  Plug in the charger, choose the highest performance mode in the laptop's "
                       "control centre and run again.\n  To measure anyway, add --force." % (limit, rint(g.default_limit())))
        if "sw_power_cap" in row["throttle"]:
            say("  note: the driver reports a power cap during this test; high clock settings may be limited by it")
        pilot = run_bench(ctx, bench_args_for(Test("decode", 0, 32, 0), 2))["res"]
    except BenchError as exc:
        raise Stop("llama-bench could not run the model.\n  %s" % exc)
    f_def = row["core_MHz"] if row["core_MHz"] not in ("", 0) else None
    m_def = row["mem_MHz"] if row["mem_MHz"] not in ("", 0) else None
    return row["ms_per_token"] / 1000.0, pilot["avg_ns"] / 1e9 / 32.0, f_def, m_def


def check_mems(ctx, levels, t_dec):
    """Lock the memory clock to each level during a short decode run and read the clock it really
    runs at. Some laptops add a factory offset to the highest memory clock, so a lock can end up at
    the same clock as the default. Returns {level: MHz seen, or None}."""
    seen = {}
    n_gen = int(max(16, min(256, 0.8 / max(t_dec, 1e-4))))           # about 0.8 s per repetition
    for m in levels:
        try:
            ctx.clocks.set(0, m)
            run = run_bench(ctx, ["-p", "0", "-n", str(n_gen), "-r", "5"])
            run["reps"] = 5
            seen[m] = analyse(run, 2.0, ctx.gpu.p_max)["mem"]
        except (ValueError, BenchError) as exc:
            seen[m] = None
            say("  memory lock %s MHz: %s" % (m, exc))
        finally:
            ctx.clocks.reset()
    return seen


def usable_mems(seen, m_def, with_default):
    """The memory locks that give a clock of their own; the others are listed with the reason."""
    keep, got, dropped = [], [], []
    for m in sorted(seen):
        a = seen[m]
        if a is None:
            dropped.append((m, "was not accepted or could not be measured"))
        elif with_default and m_def and abs(a - m_def) <= max(30, 0.03 * m_def):
            dropped.append((m, "runs at %d MHz, the same as the default" % a))
        elif any(abs(a - b) <= max(30, 0.03 * b) for b in got):
            dropped.append((m, "runs at %d MHz, the same as a lower lock" % a))
        else:
            keep.append(m)
            got.append(a)
    return keep, dropped


def model_checksum(path):
    """SHA-256 of the model file if the setup stored it next to the file, else an empty string."""
    try:
        with open(path + ".sha256", encoding="utf-8") as fh:
            return fh.read().split()[0]
    except (OSError, IndexError):
        return ""


def guess_for(test, t_pf, t_dec, core, f_def, known):
    """Expected time of one repetition (s). Only used to choose how many repetitions to run.
    'known' holds the times already measured at default clocks."""
    g = known.get(test)
    if test.phase == "prefill":
        g = g or max(test.n_prompt * t_pf, t_dec)
        if core and f_def and core < f_def:
            g *= f_def / core
        return g
    return g or test.n_gen * t_dec


def read_done(path):
    done, last = set(), 0
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    done.add((r["stage"], int(r["pass"]), int(r["core_lock_MHz"]), int(r["mem_lock_MHz"]), r["phase"],
                              int(r["n_prompt"]), int(r["n_gen"]), int(r["n_depth"])))
                    last = max(last, int(r["run"]))
                except (KeyError, ValueError, TypeError):
                    pass
    return done, last


def old_header(path):
    """The column names of an existing CSV file, or None."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), None)
    except OSError:
        return None


def key_of(stage, pass_no, core, mem, t):
    return (stage, pass_no, core, mem, t.phase, t.n_prompt, t.n_gen, t.depth)


def sweep(args):
    settings, grid_tests, char_tests = plan(args)
    n_all = describe(args, settings, grid_tests, char_tests)
    if args.dry_run:
        return
    ctx = open_ctx(args)
    if args.resume:
        base = os.path.splitext(os.path.abspath(args.resume))[0]
    else:
        base = os.path.join(args.out_dir or here(), "pace_hw_%s%s_%s" % (
            "quick_" if args.quick else "", safe(ctx.stem), datetime.now().strftime("%Y%m%d_%H%M")))
    csv_path, raw_path, tr_path = base + ".csv", base + "_raw.jsonl", base + "_transition.csv"
    done, ctx.run = read_done(csv_path)
    new_file = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    state = {"failed": 0, "n": 0, "no_depth": False}
    fails, skipped, known = collections.Counter(), set(), {}
    t_begin = time.time()
    CLEANUP.append(ctx.clocks.reset)
    say("")
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as ctx.csv_file, \
                open(raw_path, "a", encoding="utf-8") as ctx.raw_file:
            fields = FIELDS if new_file else (old_header(csv_path) or FIELDS)   # a resumed file keeps its columns
            ctx.writer = csv.DictWriter(ctx.csv_file, fieldnames=fields, extrasaction="ignore")
            if new_file:
                ctx.writer.writeheader()
                ctx.csv_file.flush()
            opts = dict(vars(args))
            opts["model"] = ctx.model
            ctx.raw_file.write(json.dumps({
                "kind": "info", "version": VERSION, "time": datetime.now().isoformat(timespec="seconds"),
                "argv": sys.argv[1:], "options": opts, "gpu": ctx.gpu.name, "driver": ctx.gpu.driver,
                "has_power": ctx.gpu.has_power, "has_energy": ctx.gpu.has_energy, "has_instant": ctx.gpu.has_instant,
                "limit_W": ctx.gpu.limit(), "default_limit_W": ctx.gpu.default_limit(),
                "platform": platform.platform(), "python": platform.python_version(), "model": ctx.stem,
                "model_bytes": os.path.getsize(ctx.model), "model_sha256": model_checksum(ctx.model),
                "bench": os.path.basename(ctx.bench)}, default=str) + "\n")
            ctx.raw_file.flush()
            first_core = next((c for c, _ in settings if c), 1200)
            t_pf, t_dec, f_def, m_def = preflight(ctx, args, first_core)
            ctx.raw_file.write(json.dumps({"kind": "preflight", "idle_W": ctx.idle, "t_prefill_per_token_s": t_pf,
                                           "t_decode_per_token_s": t_dec, "core_MHz_default": f_def,
                                           "mem_MHz_default": m_def,
                                           "lock_via": "nvidia-smi" if ctx.clocks.use_smi else "NVML",
                                           "backends_loaded": getattr(ctx, "loaded", [])}) + "\n")
            plan_mems = sorted({m for _, m in settings if m})
            levels = sorted(set(plan_mems) | ({m for m in ints(FULL_MEMS) if m} if args.quick else set()))
            if levels and not args.no_mem_check:
                seen = check_mems(ctx, levels, t_dec)
                keep, dropped = usable_mems(seen, m_def, any(m == 0 for _, m in settings))
                say("Memory   : %s%s" % (", ".join("lock %d -> %s" % (m, "-" if seen[m] is None else "%d MHz" % seen[m])
                                                   for m in levels), "" if m_def is None else " (default %d MHz)" % m_def))
                ctx.raw_file.write(json.dumps({"kind": "mem_check", "seen": {str(m): seen[m] for m in levels},
                                               "default_MHz": m_def, "kept": keep,
                                               "dropped": [[m, why] for m, why in dropped]}) + "\n")
                gone = [(m, why) for m, why in dropped if m in plan_mems]
                if gone:
                    settings = [(c, m) for c, m in settings if m == 0 or m in keep]
                    n_all = len(settings) * len(grid_tests) * args.passes + len(char_tests) + 1
                    for m, why in gone:
                        say("           memory lock %d MHz is left out: it %s" % (m, why))
                    say("           -> %d clock settings, %d measurements, %s" % (
                        len(settings), n_all, minutes(n_all * (args.target + args.settle + 3.5) / 60.0)))
            ctx.raw_file.flush()
            say("Results  : %s" % csv_path)
            say("")
            say("Please leave the laptop alone while this runs. Ctrl+C stops it safely.")

            def one(test, core, mem, stage, pass_no):
                if key_of(stage, pass_no, core, mem, test) in done or test in skipped:
                    return
                if test.depth > 0 and state["no_depth"]:
                    return
                try:
                    row = measure(ctx, test, core, mem, stage, pass_no, guess_for(test, t_pf, t_dec, core, f_def, known))
                except BenchError as exc:
                    state["failed"] += 1
                    fails[test] += 1
                    say("  skipped %s at core %s, memory %s: %s" % (test_name(test), lab(core), lab(mem), exc))
                    if "invalid parameter" in str(exc) and test.depth > 0:
                        state["no_depth"] = True             # older llama-bench without the -d option
                        say("  note: this llama-bench has no -d option; decode is measured at depth 0 only")
                    elif fails[test] >= 2:
                        skipped.add(test)                    # e.g. the context does not fit this model or the GPU memory
                    if state["failed"] >= 4:
                        raise Stop("llama-bench failed four times in a row. Last message:\n  %s" % exc)
                    return
                state["failed"] = 0
                state["n"] += 1
                if not core and not mem:
                    known[test] = row["rep_ms_mean"] / 1000.0
                show(row, "  ")
                if state["n"] % 20 == 0:
                    left = (n_all - 1 - len(done) - state["n"]) * (time.time() - t_begin) / state["n"] / 60.0
                    if left >= 1.5:
                        say("  ... about %.0f minutes left" % left)

            if char_tests:
                say("")
                say("Default clocks (the driver chooses)")
                ctx.clocks.reset()
                for t in char_tests:
                    one(t, 0, 0, "char", 0)
            for p in range(1, args.passes + 1):
                order = order_for(settings, p, args.seed)
                say("")
                say("Clock grid, pass %d of %d (%d settings)" % (p, args.passes, len(order)))
                for core, mem in order:
                    if all(key_of("grid", p, core, mem, t) in done or t in skipped for t in grid_tests):
                        continue
                    try:
                        ctx.clocks.set(core, mem)
                    except ValueError as exc:
                        say("  setting core %s, memory %s skipped: %s" % (lab(core), lab(mem), exc))
                        ctx.clocks.reset()
                        continue
                    time.sleep(0.5)
                    for t in grid_tests:
                        one(t, core, mem, "grid", p)
            ctx.clocks.reset()
            if args.quick and not args.no_transition and not os.path.isfile(tr_path):
                say("")
                say("Clock-switch time (while the model is decoding)")
                try:
                    switch_report(switch_times(ctx, args, 8), tr_path)
                except (Stop, BenchError, ValueError, OSError) as exc:
                    say("  (not measured: %s)" % exc)
            time.sleep(2.0)
            end_idle = ctx.gpu.idle_power()
            ctx.raw_file.write(json.dumps({"kind": "end", "time": datetime.now().isoformat(timespec="seconds"),
                                           "idle_W_end": end_idle, "minutes": (time.time() - t_begin) / 60.0}) + "\n")
            say("")
            say("Finished in %.1f minutes. %s" % ((time.time() - t_begin) / 60.0, (
                "Idle power: no valid reading (the GPU was asleep)." if end_idle is None and ctx.idle is None else
                "Idle power at the end: %s (at the start: %s)." % (watts(end_idle), watts(ctx.idle)))))
    except KeyboardInterrupt:
        say("")
        if ctx.run == 0:
            say("Stopped by you before the first measurement. Nothing was saved.")
        else:
            say("Stopped by you. " + (RESUME_HINT or "To continue later:  python %s --resume \"%s\"" % (
                os.path.basename(__file__), csv_path)))
    finally:
        ctx.clocks.finish()
        del CLEANUP[:]
        ctx.gpu.close()
        if ctx.run == 0:                                     # nothing was measured: leave no empty files behind
            for p in (csv_path, raw_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
        else:
            pack(base, [csv_path, raw_path, tr_path])
    if ctx.run == 0:
        return
    try:
        summary([csv_path])
    except Exception as exc:                                 # the summary is a convenience; the data is already saved
        say("(the summary could not be printed: %s)" % exc)
    say("Please send this file:  %s.zip" % base)
    pack(base, [csv_path, raw_path, tr_path])                # again, so that the log in the zip is complete


def pack(base, paths):
    """Write the console log next to the results and put all result files into one zip file."""
    try:
        with open(base + "_log.txt", "a", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in LOG[pack.saved:]))
        pack.saved = len(LOG)
        with zipfile.ZipFile(base + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in paths + [base + "_log.txt"]:
                if os.path.isfile(p):
                    z.write(p, os.path.basename(p))
    except Exception as exc:
        say("(could not write the zip file: %s; please send the .csv and the _raw.jsonl instead)" % exc)


pack.saved = 0


def restore(args):
    """For --resume: take the options of the first run from the raw file, so the same plan is continued."""
    if not os.path.isfile(args.resume):
        raise Stop("File not found: %s" % args.resume)
    raw = os.path.splitext(os.path.abspath(args.resume))[0] + "_raw.jsonl"
    old = None
    try:
        with open(raw, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("kind") == "info":
                    old = rec.get("options", {})
                    break
    except Exception:
        old = None
    if not old:
        say("note: the options of the first run were not found; using the options given now")
        return
    for k in ("quick", "cores", "mems", "grid", "prompts", "depths", "gen", "char_prompts", "char_depths", "no_char",
              "passes", "target", "seed", "ngl", "gpu", "bench_args", "period", "no_trace", "no_transition", "settle",
              "no_mem_check"):
        if old.get(k) is not None:
            setattr(args, k, old[k])
    for k in ("model", "bench"):                             # a path given now wins over the stored one
        if not getattr(args, k) and old.get(k):
            setattr(args, k, old[k])


# ----------------------------------------------------------------------------- clock-switch time
def switch_times(ctx, args, reps):
    """Change each clock back and forth while the model is decoding and time every change:
    how long the driver call takes, and when the driver reports the new clock."""
    g = ctx.gpu
    cores = sorted(c for c in ints(args.trans_cores) if c)
    mems = sorted(m for m in ints(args.trans_mems) if m)
    cases = ([("core", cores[0], cores[-1])] if len(cores) >= 2 else []) + \
            ([("mem", mems[0], mems[-1])] if len(mems) >= 2 else [])
    rows = []
    idle = ctx.idle if ctx.idle is not None else g.idle_power(2.0)
    cmd = bench_cmd(ctx, ["-p", "0", "-n", "4096", "-r", "200"])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ctx.model_dir)
    try:
        t_end = time.perf_counter() + 90
        while time.perf_counter() < t_end:                    # wait until the model is loaded and decoding
            s = g.sample()
            if proc.poll() is not None:
                raise Stop("llama-bench stopped early. Run the --quick check first to see the reason.")
            p = valid_w(s[1], g.p_max)
            if (p is not None and idle is not None and p > idle + 5) or (s[7] is not None and s[7] > 50):
                break
            time.sleep(0.1)
        time.sleep(3.0)
        for domain, lo, hi in cases:
            is_mem = domain == "mem"
            for i in range(reps):
                target = lo if i % 2 == 0 else hi
                prev = g.clock(memory=is_mem)
                t0 = time.perf_counter()
                try:
                    ctx.clocks.change(domain, target)
                except ValueError as exc:
                    say("  %s clock %s MHz was not accepted: %s" % (domain, target, exc))
                    break
                call = time.perf_counter() - t0
                seen = got = None
                while time.perf_counter() - t0 < 3.0:         # poll without sleeping: the timer of Windows is coarse
                    cur = g.clock(memory=is_mem)
                    if cur is not None and prev is not None and (
                            abs(cur - target) <= max(20, 0.02 * target)
                            or (target != prev and (cur - prev) * (target - prev) > 0
                                and abs(cur - prev) >= 0.6 * abs(target - prev))):
                        seen, got = time.perf_counter() - t0, cur
                        break
                    time.sleep(0)
                rows.append({"domain": domain, "from_MHz": prev, "to_MHz": target, "got_MHz": "" if got is None else got,
                             "via": "nvidia-smi" if ctx.clocks.use_smi else "NVML", "call_ms": round(call * 1e3, 3),
                             "seen_ms": "" if seen is None else round(seen * 1e3, 3)})
                say("  %-4s %5s -> %5s MHz : call %8.2f ms, new clock seen after %s%s" % (
                    domain, prev, target, call * 1e3, "more than 3 s" if seen is None else "%.2f ms" % (seen * 1e3),
                    "" if got is None or abs(got - target) <= max(20, 0.02 * target) else " (runs at %s MHz)" % got))
                time.sleep(0.4)
            ctx.clocks.reset()
            time.sleep(1.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        ctx.clocks.reset()
    return rows


def switch_report(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for domain in ("core", "mem"):
        c = [r["call_ms"] for r in rows if r["domain"] == domain]
        s = [r["seen_ms"] for r in rows if r["domain"] == domain and r["seen_ms"] != ""]
        if c:
            say("  %-4s clock: the call takes %.2f ms (median; largest %.2f); the new clock is seen after %s ms "
                "(median; %d of %d changes seen)" % (domain, statistics.median(c), max(c),
                                                    "%.2f" % statistics.median(s) if s else "-", len(s), len(c)))


def transition(args):
    ctx = open_ctx(args)
    out = os.path.join(args.out_dir or here(), "pace_hw_transition_%s_%s.csv" % (
        safe(ctx.stem), datetime.now().strftime("%Y%m%d_%H%M")))
    say("pace_hw_sweep %s" % VERSION)
    say("GPU: %s (driver %s)" % (ctx.gpu.name, ctx.gpu.driver))
    say("Measuring how long a clock change takes while the model is decoding.")
    rows = []
    CLEANUP.append(ctx.clocks.reset)
    try:
        check_rights(ctx, next((c for c in ints(args.trans_cores) if c), 1200))
        rows = switch_times(ctx, args, args.trans_reps)
    except KeyboardInterrupt:
        say("Stopped by you.")
    finally:
        ctx.clocks.finish()
        del CLEANUP[:]
        ctx.gpu.close()
    if rows:
        say("")
        switch_report(rows, out)
        say("")
        say("Please send this file:  %s" % out)


# ----------------------------------------------------------------------------- results of an earlier run, again
def reanalyse(raw_path, settle):
    """Compute the CSV lines of an earlier run again from its raw samples, with the current rules.
    Writes <name>_reanalysed.csv next to the raw file and prints the tables."""
    if not os.path.isfile(raw_path):
        raise Stop("File not found: %s" % raw_path)
    recs = []
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    info = next((r for r in recs if r.get("kind") == "info"), {})
    limits = [x for x in (info.get("limit_W"), info.get("default_limit_W")) if x]
    p_max = 1.5 * max(limits) if limits else 450.0
    base = raw_path[:-len("_raw.jsonl")] if raw_path.endswith("_raw.jsonl") else os.path.splitext(raw_path)[0]
    out = base + "_reanalysed.csv"
    rows, missing = [], 0
    for r in recs:
        if r.get("kind") != "run":
            continue
        tr = r.get("trace")
        if not tr or not tr.get("t"):
            missing += 1
            continue
        samples = list(zip(tr["t"], tr["p_avg"], tr["p_inst"], tr["e_J"], tr["core"], tr["mem"], tr["temp"],
                           tr["util_gpu"], tr["util_mem"], tr["reasons"]))
        if "t_result" in r:
            t_result = r["t_result"]
        else:                                    # version 1.0 files: the window ended when the result was written
            t_result = None if "window_estimated" in (r.get("note") or "") else r["window"][1]
        slow = [tuple(x) for x in r.get("slow") or []] or [(-1.0, fnum(r.get("limit_W")), fnum(r.get("vram_MB")))]
        run = {"res": {"samples_ns": r.get("samples_ns"), "avg_ns": (r.get("bench") or {}).get("avg_ns")},
               "samples": samples, "slow": slow, "t_result": t_result, "t_exit": r.get("t_exit", tr["t"][-1]),
               "reps": r.get("reps")}
        st = analyse(run, settle, p_max)
        test = Test(r["phase"], int(r["n_prompt"]), int(r["n_gen"]), int(r["n_depth"]))
        rows.append(make_row(r["run"], r["stage"], r["pass"], r["time"], r["gpu"], r["model"], fnum(r.get("idle_W")),
                             test, int(r["core_lock_MHz"]), int(r["mem_lock_MHz"]), st))
    if not rows:
        raise Stop("There are no measurements with raw samples in %s" % raw_path)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    say("Computed again from the raw samples (settle time %.1f s): %d measurements%s" % (
        settle, len(rows), "" if not missing else "; %d without raw samples were left out" % missing))
    say("  %s" % out)
    for row in rows:
        show(row, "  ")
    summary([out])
    return out


# ----------------------------------------------------------------------------- summary tables
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def slope(points):
    """Least-squares slope of ln(time) against ln(1/clock): 1 = the time follows the clock, 0 = no effect."""
    pts = [(math.log(1.0 / f), math.log(t)) for f, t in points if f and t and f > 0 and t > 0]
    if len(pts) < 2:
        return None
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den < 1e-4:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def tkey(r):
    return (r["phase"], int(r["n_prompt"]), int(r["n_gen"]), int(r["n_depth"]))


def torder(k):
    return (k[0] != "prefill", k[1], k[3])                   # prefill first, then by size / depth


def summary(paths):
    rows = []
    for p in paths:
        if not os.path.isfile(p):
            raise Stop("File not found: %s" % p)
        with open(p, newline="", encoding="utf-8-sig") as fh:
            got = [r for r in csv.DictReader(fh) if r.get("stage") and r.get("model") and fnum(r.get("ms_per_token"))]
        if not got:
            say("(no sweep measurements in %s)" % os.path.basename(p))
        rows += got
    if not rows:
        return
    for model in sorted({r["model"] for r in rows}):
        R = [r for r in rows if r["model"] == model]
        say("")
        say("=" * 100)
        say("%s  |  %s" % (model, R[0].get("gpu", "")))
        say("=" * 100)

        def cell(sel, field):
            return mean([fnum(r[field]) for r in sel])

        char = [r for r in R if r["stage"] in ("char", "preflight")]
        if char:
            say("")
            say("Default clocks (the driver chooses)")
            say("  %-14s %10s %10s %9s %10s %9s %9s" % ("test", "ms/token", "tok/s", "power W", "mJ/token", "core MHz", "mem MHz"))
            for k in sorted({tkey(r) for r in char}, key=torder):
                sel = [r for r in char if tkey(r) == k]
                v = [cell(sel, f) for f in ("ms_per_token", "tok_per_s", "power_W", "energy_mJ_per_token", "core_MHz", "mem_MHz")]
                fmts = ("%.4f", "%.1f", "%.1f", "%.3f", "%.0f", "%.0f")
                say("  %-14s %10s %10s %9s %10s %9s %9s" % tuple(
                    [test_name(Test(*k))] + ["-" if x is None else f % x for f, x in zip(fmts, v)]))
        grid = [r for r in R if r["stage"] == "grid"]
        if not grid:
            continue
        cores = sorted({int(r["core_lock_MHz"]) for r in grid}, key=lambda c: (c == 0, c))
        mems = sorted({int(r["mem_lock_MHz"]) for r in grid}, key=lambda m: (m == 0, m))
        tests = sorted({tkey(r) for r in grid}, key=torder)
        ref_c = 0 if 0 in cores else max(cores)
        ref_m = 0 if 0 in mems else max(mems)

        def pick(k, c, m):
            return [r for r in grid if tkey(r) == k and int(r["core_lock_MHz"]) == c and int(r["mem_lock_MHz"]) == m]

        for title, field, fmt in (("Time per token (ms)", "ms_per_token", "%.4f"),
                                  ("Power (W)", "power_W", "%.1f"),
                                  ("Energy per token (mJ)", "energy_mJ_per_token", "%.3f")):
            say("")
            say("%s. Rows: core clock lock (MHz). Columns: memory clock lock (MHz). 'auto' = driver default." % title)
            for k in tests:
                say("  %-14s" % test_name(Test(*k)) + "".join("%11s" % lab(m) for m in mems))
                for c in cores:
                    vals = [cell(pick(k, c, m), field) for m in mems]
                    say("  %14s" % lab(c) + "".join("%11s" % ("-" if v is None else fmt % v) for v in vals))
        say("")
        say("Which clock does each phase follow?  (1 = the time follows this clock fully, 0 = no effect)")
        say("  %-14s %12s %14s" % ("test", "core clock", "memory clock"))
        for k in tests:
            pc = [(cell(pick(k, c, ref_m), "core_MHz"), cell(pick(k, c, ref_m), "ms_per_token")) for c in cores]
            pm = [(cell(pick(k, ref_c, m), "mem_MHz"), cell(pick(k, ref_c, m), "ms_per_token")) for m in mems]
            sc, sm = slope([p for p in pc if None not in p]), slope([p for p in pm if None not in p])
            say("  %-14s %12s %14s" % (test_name(Test(*k)), "-" if sc is None else "%.2f" % sc, "-" if sm is None else "%.2f" % sm))
        if len(cores) * len(mems) < 2 or not pick(tests[0], 0, 0):
            continue
        say("")
        say("Energy per token against default clocks: the best setting that costs at most 5% more time,")
        say("and the best setting when time does not matter")
        for k in tests:
            base_e, base_t = cell(pick(k, 0, 0), "energy_mJ_per_token"), cell(pick(k, 0, 0), "ms_per_token")
            if not base_e or not base_t:
                continue
            near = free = None
            for c in cores:
                for m in mems:
                    e, t = cell(pick(k, c, m), "energy_mJ_per_token"), cell(pick(k, c, m), "ms_per_token")
                    if e is None or t is None:
                        continue
                    if free is None or e < free[0]:
                        free = (e, t, c, m)
                    if t <= 1.05 * base_t and (near is None or e < near[0]):
                        near = (e, t, c, m)
            for tag, b in (("same speed", near), ("any speed", free)):
                if b is not None:
                    say("  %-14s %-10s : core %-5s mem %-5s  energy %+6.1f%%  time %+6.1f%%" % (
                        test_name(Test(*k)), tag, lab(b[2]), lab(b[3]),
                        100.0 * (b[0] / base_e - 1), 100.0 * (b[1] / base_t - 1)))
    say("")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path of the .gguf model")
    ap.add_argument("--bench", help="path of llama-bench (default: looked up next to this script)")
    ap.add_argument("--quick", action="store_true", help="small check that everything works (about 4 minutes)")
    ap.add_argument("--transition", action="store_true", help="measure how long a clock change takes")
    ap.add_argument("--summary", nargs="+", metavar="CSV", help="print the result tables of finished sweeps")
    ap.add_argument("--reanalyse", metavar="RAW", help="compute the results of an earlier run again from its _raw.jsonl")
    ap.add_argument("--resume", metavar="CSV", help="continue the sweep that wrote this CSV file")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, measure nothing")
    ap.add_argument("--cores", help="core clock locks in MHz, 0 = driver default (default: 600,...,2400,0)")
    ap.add_argument("--mems", help="memory clock locks in MHz, 0 = driver default (default: 6001,7001,8001,0)")
    ap.add_argument("--grid", choices=["full", "cross"], default="full",
                    help="full: every combination; cross: vary one clock at a time (shorter)")
    ap.add_argument("--prompts", help="prompt lengths for prefill on the grid (default: 128,1024)")
    ap.add_argument("--depths", help="context depths for decode on the grid (default: 0,1024)")
    ap.add_argument("--gen", type=int, default=256, help="generated tokens per decode repetition (default: 256)")
    ap.add_argument("--char-prompts", help="prompt lengths measured at default clocks")
    ap.add_argument("--char-depths", help="context depths measured at default clocks")
    ap.add_argument("--no-char", action="store_true", help="skip the measurements at default clocks")
    ap.add_argument("--passes", type=int, help="how many times the grid is measured (default: 2)")
    ap.add_argument("--target", type=float, help="seconds of running per measurement (default: 5)")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds at the start of every run that are not used, while the GPU wakes up (default: 2)")
    ap.add_argument("--no-mem-check", action="store_true",
                    help="do not check which clock each memory lock really gives before the sweep")
    ap.add_argument("--seed", type=int, default=20261006, help="seed of the random order")
    ap.add_argument("--ngl", type=int, default=99, help="layers on the GPU (default: 99 = all)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    ap.add_argument("--bench-args", default="", help="extra options passed to llama-bench, in quotes")
    ap.add_argument("--min-power-limit", type=float, default=60.0, help="stop if the GPU power limit is below this (W)")
    ap.add_argument("--force", action="store_true", help="measure even if a check fails")
    ap.add_argument("--trans-cores", default="1200,2100", help="the two core clocks for the clock-switch test (MHz)")
    ap.add_argument("--trans-mems", default="6001,8001", help="the two memory clocks for the clock-switch test (MHz)")
    ap.add_argument("--trans-reps", type=int, default=20, help="clock changes per clock for --transition")
    ap.add_argument("--no-transition", action="store_true", help="leave the clock-switch test out of the quick check")
    ap.add_argument("--period", type=float, default=0.05, help="sensor sampling period in seconds")
    ap.add_argument("--no-trace", action="store_true", help="do not store the raw sensor samples")
    ap.add_argument("--out-dir", help="folder for the result files (default: next to this script)")
    ap.add_argument("--live-log", help="also write every printed line to this file at once (used by main.py)")
    args = ap.parse_args(argv)
    os_setup()
    global LIVE
    if args.live_log:
        try:
            LIVE = open(args.live_log, "a", encoding="utf-8")
        except OSError:
            LIVE = None
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    try:
        if args.summary:
            summary(args.summary)
        elif args.reanalyse:
            reanalyse(args.reanalyse, max(0.0, args.settle))
        elif args.transition:
            transition(args)
        elif args.dry_run or args.model or args.resume:
            if args.resume:
                restore(args)
            sweep(args)
        else:
            ap.print_help()
    except Stop as exc:
        say("")
        say("STOPPED: %s" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
