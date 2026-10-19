#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACE hardware measurements - start here.

In PyCharm:  open this folder as a project, open main.py and press Run. Type a number in the
             Run window at the bottom and press Enter.
In a terminal:  python main.py           shows the menu
                python main.py quick     runs one item directly (quick, full, 3b, 8b, switch,
                                         tables, download)

The first run downloads llama.cpp and the model (about 2 GB). Before a measurement Windows asks
for administrator rights, because changing GPU clocks needs them. The measurement then runs in
a second window, and everything it prints is also shown here.

Results are written into the folder "results". Send the .zip file named at the end.
"""
import glob
import json
import os
import shlex
import sys
import time

import pace_hw_sweep as engine
import pace_setup as setup
from pace_hw_sweep import Stop, say

HERE = os.path.dirname(os.path.abspath(__file__))
DONE = "[[pace-hw finished with code "
engine.ADMIN = ("The driver did not allow changing the GPU clocks: this run has no administrator rights.\n"
                "  Close PyCharm, right-click the PyCharm icon and choose 'Run as administrator'.\n"
                "  Then open this project again and run main.py.")

MENU = [
    ("quick", "Quick check", "about 4 minutes; do this first"),
    ("full", "Full sweep, 1B model", "about 45 minutes"),
    ("3b", "Short sweep, 3B model", "about 25 minutes; downloads 3.4 GB the first time"),
    ("8b", "Short sweep, 8B model", "about 40 minutes; downloads 4.9 GB the first time"),
    ("switch", "Clock-switch time", "about 2 minutes"),
    ("tables", "Show the result tables again", "no measurement"),
    ("download", "Download all files now and measure later", "about 10 GB with the two larger models"),
]
MODEL_OF = {"quick": "1b", "full": "1b", "switch": "1b", "3b": "3b", "8b": "8b"}


def ask(prompt):
    """A typed answer, or None if nothing can be typed here."""
    try:
        return input(prompt).strip()
    except EOFError:
        return None


def choose():
    say("PACE hardware measurements")
    say("")
    for i, (_, title, note) in enumerate(MENU, 1):
        say("  %d  %-42s %s" % (i, title, note))
    say("  0  Exit")
    say("")
    while True:
        answer = ask("Type a number and press Enter: ")
        if answer is None:
            say("")
            say("Nothing can be typed in this window. Start an item directly, for example:  python main.py quick")
            return None
        answer = answer.lower()
        if answer in ("0", "q", "exit", ""):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(MENU):
            return MENU[int(answer) - 1][0]
        if answer in [m[0] for m in MENU]:
            return answer
        say("Please type a number from 0 to %d." % len(MENU))


# ----------------------------------------------------------------------------- earlier results
def sweeps(prefix):
    """Result files of sweeps (not quick checks, not clock-switch files), newest first."""
    files = [f for f in glob.glob(os.path.join(setup.RESULTS, "pace_hw_*.csv"))
             if not os.path.basename(f).startswith(("pace_hw_quick_", "pace_hw_transition_"))
             and not f.endswith("_transition.csv") and os.path.basename(f).startswith(prefix)]
    return sorted(files, key=os.path.getmtime, reverse=True)


def unfinished(model_key):
    """The newest sweep of this model that was stopped before the end, with its number of measurements."""
    stem = engine.safe(os.path.splitext(setup.MODELS_INFO[model_key]["file"])[0])
    for path in sweeps("pace_hw_" + stem + "_"):
        raw = os.path.splitext(path)[0] + "_raw.jsonl"
        try:
            with open(raw, encoding="utf-8") as fh:
                kinds = [json.loads(line).get("kind") for line in fh if line.strip()]
        except (OSError, ValueError):
            continue
        if "end" in kinds:
            return None, 0                                        # the newest sweep is complete
        if kinds.count("run") > 1:
            return path, kinds.count("run")
    return None, 0


def show_tables():
    files = sorted(glob.glob(os.path.join(setup.RESULTS, "pace_hw_*.csv")), key=os.path.getmtime, reverse=True)
    files = [f for f in files if "transition" not in os.path.basename(f)]
    if not files:
        say("There are no results yet in %s" % setup.RESULTS)
        return 0
    say("Result files, newest first:")
    for i, f in enumerate(files[:9], 1):
        say("  %d  %s" % (i, os.path.basename(f)))
    answer = ask("Which one? Type a number and press Enter (Enter alone = 1): ") or "1"
    if not (answer.isdigit() and 1 <= int(answer) <= min(len(files), 9)):
        return 0
    return engine.main(["--summary", files[int(answer) - 1]])


# ----------------------------------------------------------------------------- running a measurement
def engine_options(action, resume):
    common = ["--bench", setup.bench_path(), "--out-dir", setup.RESULTS]
    common += shlex.split(os.environ.get("PACE_HW_EXTRA_ARGS", ""))               # for testing only
    if resume:
        return ["--resume", resume] + common
    model = ["--model", setup.model_path(MODEL_OF[action])]
    if action == "quick":
        return model + ["--quick"] + common
    if action == "switch":
        return model + ["--transition"] + common
    if action in ("3b", "8b"):
        return model + ["--grid", "cross", "--gen", "128"] + common
    return model + common


def run_here(action, resume, live):
    options = engine_options(action, resume) + (["--live-log", live] if live else [])
    if action in ("full", "3b", "8b"):
        engine.RESUME_HINT = "To continue later, run main.py again and choose the same item; it asks whether to continue."
    else:
        engine.RESUME_HINT = "What was measured is saved. Run main.py again to repeat it."
    try:
        return engine.main(options)
    except KeyboardInterrupt:
        say("Stopped by you.")
        return 1


def run_in_admin_window(action, resume):
    """Windows only: start this script again with administrator rights and show its output here."""
    live = os.path.join(setup.RESULTS, "live_%d_%d.log" % (os.getpid(), int(time.time())))
    open(live, "w", encoding="utf-8").close()
    params = [os.path.abspath(__file__), action, "--elevated", "--live-log", live] + (["--resume", resume] if resume else [])
    say("Changing GPU clocks needs administrator rights. Windows asks for them now: please choose 'Yes'.")
    say("(If no question appears, look for a flashing shield icon in the taskbar and click it.)")
    say("The measurement then runs in a second window. Leave that window open; its output is shown here too.")
    say("To stop the measurement, click into the second window and press Ctrl+C.")
    say("")
    handle = setup.launch_elevated(params, HERE)
    code, pos, quiet_since, text = None, 0, time.time(), ""
    try:
        while True:
            finished, exit_code = setup.process_finished(handle) if handle else (False, None)
            with open(live, encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                text += fh.read()
                pos = fh.tell()
            while "\n" in text:
                line, text = text.split("\n", 1)
                quiet_since = time.time()
                if line.startswith(DONE):
                    code = int("".join(ch for ch in line[len(DONE):] if ch.isdigit()) or 1)
                else:
                    print(line, flush=True)
            if code is not None:
                break
            if finished:                                           # the window was closed, or Python could not start
                say("")
                say("The measurement window has closed before the end%s." % (
                    "" if exit_code is None else " (code %s)" % exit_code))
                say("What was measured so far is saved. Every new run starts by returning the GPU clocks to default.")
                code = 1
                break
            if handle is None and time.time() - quiet_since > 1200:
                say("No news from the measurement window for 20 minutes; this window stops waiting.")
                code = 1
                break
            time.sleep(0.3)
    except KeyboardInterrupt:
        say("")
        say("This window stops showing the output. The measurement itself continues in the second window;")
        say("press Ctrl+C there to stop it.")
        return 1
    finally:
        for _ in range(40 if code is not None else 1):             # the second window closes a few seconds later
            try:
                os.remove(live)
                break
            except OSError:
                time.sleep(0.25)
    return code


def tidy():
    """Remove output files of earlier runs that were left behind by a window that was closed."""
    for f in glob.glob(os.path.join(setup.RESULTS, "live_*.log")):
        try:
            if time.time() - os.path.getmtime(f) > 3600:
                os.remove(f)
        except OSError:
            pass


def main():
    engine.os_setup()
    tidy()
    args = sys.argv[1:]
    elevated = "--elevated" in args
    live = args[args.index("--live-log") + 1] if "--live-log" in args[:-1] else None
    resume = args[args.index("--resume") + 1] if "--resume" in args[:-1] else None
    words = [a for a in args if not a.startswith("--") and a not in (live, resume)]
    code = 1
    try:
        if elevated and live:
            engine.LIVE = open(live, "a", encoding="utf-8")
        action = words[0].lower() if words else choose()
        if action is None:
            return 0
        if action not in [m[0] for m in MENU]:
            raise Stop("Unknown item '%s'. Use one of: %s" % (action, ", ".join(m[0] for m in MENU)))
        if action == "tables":
            return show_tables()
        say("")
        if action == "download":
            setup.prepare(["1b"])
            if (ask("Also download the 3B model (3.4 GB) and the 8B model (4.9 GB) now? [y/N]: ") or "n").lower()[0] == "y":
                setup.prepare(["3b", "8b"], quiet=True)
            say("All files are there. Run main.py again to measure.")
            return 0
        setup.prepare([MODEL_OF[action]])
        if not elevated and not resume and action in ("full", "3b", "8b"):
            old, count = unfinished(MODEL_OF[action])
            if old:
                say("An earlier sweep of this model was stopped after %d measurements:" % count)
                say("  %s" % os.path.basename(old))
                if (ask("Continue it? [Y = continue it / n = start a new one]: ") or "y").lower()[0] != "n":
                    resume = old
                say("")
        need_window = os.name == "nt" and not setup.is_admin() and "--no-elevate" not in args
        if os.environ.get("PACE_HW_TEST_FORCE_WINDOW") and not elevated:              # for testing only
            need_window = True
        if need_window and not elevated:
            code = run_in_admin_window(action, resume)
        else:
            code = run_here(action, resume, None if elevated else live)
    except Stop as exc:
        say("")
        say("STOPPED: %s" % exc)
        code = 2
    except KeyboardInterrupt:
        say("")
        say("Stopped by you.")
        code = 1
    except Exception:                                              # show the error in both windows before closing
        import traceback
        say("")
        say("An unexpected error happened. Please send this text:")
        for line in traceback.format_exc().splitlines():
            say(line)
        code = 3
    finally:
        if elevated:
            if engine.LIVE is not None:
                try:
                    engine.LIVE.write("%s%d]]\n" % (DONE, code))
                    engine.LIVE.flush()
                except (OSError, ValueError):
                    pass
            print("\nThis window closes in a few seconds. The same text is in the first window.", flush=True)
            time.sleep(6)
    return code


if __name__ == "__main__":
    sys.exit(main())
