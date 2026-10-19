#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pace_setup.py - first-run setup for the PACE hardware measurements

"""
import hashlib
import http.client
import importlib
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

from pace_hw_sweep import Stop, say

ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(ROOT, "downloads")
TOOLS = os.path.join(ROOT, "tools", "llama.cpp")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")

# ----------------------------------------------------------------------------- what is downloaded
# llama.cpp is fixed to one release. Sizes and checksums are those of the official files on GitHub.
LLAMA_TAG = "b11425"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/%s/" % LLAMA_TAG
# Public GitHub mirror sites, used only if github.com does not answer or is very slow.
# A file from a mirror is accepted only if its checksum is the official one.
GITHUB_MIRRORS = ["https://ghfast.top/", "https://gh-proxy.com/"]
PACKAGES = [
    {"file": "llama-%s-bin-win-cuda-12.4-x64.zip" % LLAMA_TAG, "size": 264468885,
     "sha256": "0d2984647bb9864d8f5a28495605c68f27824e5faf8ad99768fc8b7597745698"},
    {"file": "cudart-llama-bin-win-cuda-12.4-x64.zip", "size": 391443627,
     "sha256": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"},
]
# Files that must be in tools/llama.cpp after unpacking.
NEEDED = ["llama-bench.exe", "llama-bench-impl.dll", "llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cuda.dll",
          "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]
BENCH_NAME = "llama-bench.exe"

# Models: the same files are on ModelScope (fast inside China) and on Hugging Face.
MODELS_INFO = {
    "1b": {"repo": "bartowski/Llama-3.2-1B-Instruct-GGUF", "file": "Llama-3.2-1B-Instruct-Q8_0.gguf",
           "size": 1321083008, "sha256": "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"},
    "3b": {"repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
           "size": 3421899296, "sha256": "b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921"},
    "8b": {"repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
           "size": 4920739232, "sha256": "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"},
}
MODEL_SOURCES = [
    "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}",
    "https://modelscope.cn/models/{repo}/resolve/master/{file}",
    "https://hf-mirror.com/{repo}/resolve/main/{file}",
    "https://huggingface.co/{repo}/resolve/main/{file}",
]
PIP_INDEXES = [[], ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], ["-i", "https://mirrors.aliyun.com/pypi/simple/"]]

AGENT = "pace-hw-setup/1.0 (Python urllib)"
FIRST_TIMEOUT = 15        # seconds to wait for an address that has not sent anything yet
TIMEOUT = 40              # seconds without data before a running download is tried again
SLOW_BYTES_PER_S = 80000  # slower than this for SLOW_WINDOW seconds: try the next address
SLOW_WINDOW = 30
CHUNK = 1 << 20
SKIP_GPU_CHECK = False


class WrongFile(Exception):
    """This address does not have the file (or not the same file)."""


class TooSlow(Exception):
    """This address is too slow; another one may be faster."""


# ----------------------------------------------------------------------------- small helpers
def mb(n):
    return "%.0f MB" % (n / 1e6) if n < 1e9 else "%.2f GB" % (n / 1e9)


def host_of(url):
    return url.split("/")[2] if "//" in url else url


def sha256_of(path, label=None):
    h, done, total, t_show = hashlib.sha256(), 0, os.path.getsize(path), time.time()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(4 * CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if label and time.time() - t_show > 5:
                say("    checking %s: %d%%" % (label, 100 * done // max(total, 1)))
                t_show = time.time()
    return h.hexdigest()


def free_bytes(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


# ----------------------------------------------------------------------------- downloading
def _download(url, part, size, label, patient):
    """Fetch url into the file 'part', continuing a partial file. Raises WrongFile, TooSlow or OSError."""
    failures, seen = 0, 0                                        # seen: bytes this address has sent so far
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have > size:
            os.remove(part)
            have = 0
        if size and have == size:
            return
        got_now = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "*/*"})
            if have:
                req.add_header("Range", "bytes=%d-" % have)
            with urllib.request.urlopen(req, timeout=TIMEOUT if seen else FIRST_TIMEOUT) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                ctype = (resp.headers.get("Content-Type") or "").lower()
                clen = resp.headers.get("Content-Length") or ""
                crange = resp.headers.get("Content-Range") or ""
                if "text/html" in ctype:
                    raise WrongFile("it sent a web page instead of the file")
                if have and (status != 206 or not crange.replace(" ", "").startswith("bytes%d-" % have)):
                    have = 0                                   # the server ignored the request to continue
                    if status == 206:
                        raise OSError("unexpected answer to the request to continue")
                total = have + int(clen) if clen.isdigit() else None
                if size and total is None:                     # real file servers always say how large the file is
                    raise WrongFile("it did not say how large the file is")
                if size and total != size:
                    raise WrongFile("the file there has another size (%d instead of %d bytes)" % (total, size))
                t0 = t_show = t_slow = time.time()
                b_slow = 0
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        got_now += len(block)
                        now = time.time()
                        if now - t_show >= 4:
                            done, speed = have + got_now, got_now / max(now - t0, 1e-3)
                            left = (size - done) / speed if size and speed > 0 else None
                            say("    %s of %s (%d%%), %.1f MB/s%s" % (
                                mb(done), mb(size) if size else "?", 100 * done // size if size else 0, speed / 1e6,
                                "" if left is None else ", about %d min left" % (left // 60 + 1)))
                            t_show = now
                        if now - t_slow >= SLOW_WINDOW:
                            if not patient and (got_now - b_slow) / (now - t_slow) < SLOW_BYTES_PER_S:
                                raise TooSlow("less than %d kB/s" % (SLOW_BYTES_PER_S // 1000))
                            t_slow, b_slow = now, got_now
            have = os.path.getsize(part)
            if size and have < size:
                raise OSError("the connection closed early")
            return
        except (WrongFile, TooSlow):
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and os.path.exists(part):       # our partial file does not fit: start again
                os.remove(part)
                continue
            if exc.code in (400, 401, 403, 404, 410, 451):
                raise WrongFile("the server answered %d" % exc.code)
            problem = "the server answered %d" % exc.code
        except (urllib.error.URLError, http.client.HTTPException, socket.timeout, ssl.SSLError, OSError) as exc:
            problem = str(getattr(exc, "reason", exc)) or exc.__class__.__name__
        seen += got_now
        failures = 0 if got_now > 5 * CHUNK else failures + 1   # progress since the last problem: be patient again
        if failures >= (4 if seen else (2 if patient else 1)):   # an address that never answered is left quickly
            raise OSError(problem)
        say("    connection problem (%s); continuing" % problem[:80])
        time.sleep(max(1, 3 * failures))


def fetch(urls, dest, size, sha256, label):
    """Get one file: try the addresses in turn, continue partial downloads, check size and checksum."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(dest) and (not size or os.path.getsize(dest) == size):
        if not sha256 or sha256_of(dest, label) == sha256:
            return
        say("  %s is damaged; downloading it again" % os.path.basename(dest))
        os.remove(dest)
    elif os.path.exists(dest):
        os.replace(dest, part)                                   # incomplete file put there by hand: continue it
    free = free_bytes(os.path.dirname(dest))
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if free is not None and size and free < (size - have) + 200e6:
        raise Stop("Not enough free disk space for %s: %s needed, %s free on this drive." % (
            label, mb(size - have + 200e6), mb(free)))
    notes = []
    for rnd in (1, 2):                                           # second round: no switching away from slow addresses
        for url in urls:
            say("  %s from %s%s" % (label, host_of(url), " (continuing)" if os.path.exists(part) else ""))
            try:
                _download(url, part, size, label, patient=(rnd == 2))
            except WrongFile as exc:
                notes.append("%s: %s" % (host_of(url), exc))
                say("    not usable: %s" % exc)
                continue
            except TooSlow as exc:
                notes.append("%s: too slow" % host_of(url))
                say("    this address is slow (%s); trying another one" % exc)
                continue
            except OSError as exc:
                notes.append("%s: %s" % (host_of(url), str(exc)[:80]))
                say("    no success: %s" % str(exc)[:100])
                continue
            if sha256:
                say("    checking the file ...")
                if sha256_of(part, label) != sha256:
                    notes.append("%s: wrong checksum" % host_of(url))
                    say("    the checksum is wrong; this copy is thrown away")
                    os.remove(part)
                    continue
            os.replace(part, dest)
            say("    done")
            return
    raise Stop("Could not download %s automatically.\n"
               "  Please download this file with your browser:\n      %s\n"
               "  Put it (as it is, not unpacked) into this folder:\n      %s\n"
               "  Then run main.py again.\n  What was tried: %s" % (
                   label, urls_for_humans(urls), os.path.dirname(dest), "; ".join(list(dict.fromkeys(notes))[:6])))


def urls_for_humans(urls):
    """The addresses that are simple to open in a browser."""
    good = [u for u in urls if "/api/" not in u]
    return "\n      or  ".join(good[:3])


# ----------------------------------------------------------------------------- the three things we need
def ensure_python_package():
    try:
        importlib.import_module("pynvml")
        return
    except ImportError:
        pass
    say("Python package nvidia-ml-py: installing it into %s" % sys.executable)
    for extra in PIP_INDEXES:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "nvidia-ml-py"] + extra
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            importlib.invalidate_caches()
            try:
                importlib.import_module("pynvml")
                say("  installed")
                return
            except ImportError:
                break
    raise Stop("Could not install the Python package nvidia-ml-py.\n"
               "  In PyCharm: open the Terminal tab at the bottom and type\n      pip install nvidia-ml-py\n"
               "  Then run main.py again.")


def bench_path():
    return os.path.join(TOOLS, BENCH_NAME)


def llama_ready():
    return all(os.path.isfile(os.path.join(TOOLS, f)) and os.path.getsize(os.path.join(TOOLS, f)) > 0 for f in NEEDED)


def ensure_llama():
    if llama_ready():
        return
    if os.name != "nt":
        raise Stop("Automatic download of llama.cpp is prepared for Windows only.\n"
                   "  On Linux, build llama.cpp with CUDA and copy llama-bench into\n      %s" % TOOLS)
    total = sum(p["size"] for p in PACKAGES)
    say("llama.cpp %s for Windows with CUDA: %d files, %s (one time)" % (LLAMA_TAG, len(PACKAGES), mb(total)))
    free = free_bytes(ROOT)
    if free is not None and free < 2.4 * total:
        raise Stop("Not enough free disk space: about %s needed for llama.cpp, %s free." % (mb(2.4 * total), mb(free)))
    os.makedirs(TOOLS, exist_ok=True)
    for p in PACKAGES:
        official = LLAMA_URL + p["file"]
        zip_path = os.path.join(DOWNLOADS, p["file"])
        fetch([official] + [m + official for m in GITHUB_MIRRORS], zip_path, p["size"], p["sha256"], p["file"])
        say("  unpacking %s (this can take a minute)" % p["file"])
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(TOOLS)
        except (zipfile.BadZipFile, OSError) as exc:
            raise Stop("Could not unpack %s: %s\n  Delete the file and run main.py again." % (zip_path, exc))
    if not llama_ready():
        missing = [f for f in NEEDED if not os.path.isfile(os.path.join(TOOLS, f))]
        raise Stop("llama.cpp was unpacked, but these files are missing in %s:\n  %s" % (TOOLS, ", ".join(missing)))
    with open(os.path.join(TOOLS, "installed.json"), "w", encoding="utf-8") as fh:
        json.dump({"release": LLAMA_TAG, "packages": PACKAGES}, fh, indent=1)
    for p in PACKAGES:                                           # the unpacked files are enough; free the space
        try:
            os.remove(os.path.join(DOWNLOADS, p["file"]))
        except OSError:
            pass
    say("  llama.cpp is ready")


def model_path(key):
    return os.path.join(MODELS, MODELS_INFO[key]["file"])


def model_ready(key):
    m, path = MODELS_INFO[key], model_path(key)
    return os.path.isfile(path) and os.path.getsize(path) == m["size"] and os.path.isfile(path + ".sha256")


def ensure_model(key):
    m, path = MODELS_INFO[key], model_path(key)
    if model_ready(key):
        return
    os.makedirs(MODELS, exist_ok=True)
    by_hand = os.path.join(DOWNLOADS, m["file"])                 # a file the user downloaded by hand
    if os.path.isfile(by_hand) and not os.path.exists(path):
        shutil.move(by_hand, path)
    if not (os.path.isfile(path) and os.path.getsize(path) == m["size"]):
        say("Model %s: %s (one time)" % (m["file"], mb(m["size"])))
    urls = [u.format(repo=m["repo"], file=m["file"]) for u in MODEL_SOURCES]
    fetch(urls, path, m["size"], m["sha256"], m["file"])
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write("%s  %s\n" % (m["sha256"], m["file"]))


VC_RUNTIME = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def check_gpu_tools():
    if SKIP_GPU_CHECK:
        return
    if shutil.which("nvidia-smi") is None:
        raise Stop("nvidia-smi was not found, so there seems to be no NVIDIA driver on this computer.\n"
                   "  These measurements need an NVIDIA GPU with its driver installed.")
    if os.name == "nt" and sys.maxsize > 2 ** 32:               # llama.cpp needs these Windows files
        folder = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32")
        missing = [f for f in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
                   if not os.path.isfile(os.path.join(folder, f))]
        if missing:
            say("note: %s not found in %s. If llama-bench cannot start, install the Microsoft Visual C++\n"
                "      runtime from %s" % (", ".join(missing), folder, VC_RUNTIME))


def prepare(models, quiet=False):
    """Make sure the Python package, llama.cpp and the given models are there."""
    for d in (DOWNLOADS, MODELS, RESULTS):
        os.makedirs(d, exist_ok=True)
    first = not llama_ready() or not all(model_ready(k) for k in models)
    if first and not quiet:
        say("First run: the files that are still missing are downloaded now. This happens only once.")
        say("An interrupted download continues where it stopped when you run main.py again.")
        say("")
    check_gpu_tools()
    ensure_python_package()
    ensure_llama()
    for k in models:
        ensure_model(k)
    if first and not quiet:
        say("")
        say("Everything is ready.")
        say("")


# ----------------------------------------------------------------------------- administrator rights (Windows)
def is_admin():
    if os.name != "nt":
        return hasattr(os, "geteuid") and os.geteuid() == 0
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def launch_elevated(params, workdir):
    """Start 'python params...' with administrator rights in a new window (Windows asks the user).
    Returns a process handle, or None if the program was started but cannot be watched.
    Raises Stop if the user said no or Windows refused."""
    import ctypes
    from ctypes import wintypes
    line = subprocess.list2cmdline(params)
    try:
        class Info(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
                        ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
                        ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
                        ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
                        ("dwHotKey", wintypes.DWORD), ("hIcon", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]

        info = Info()
        info.cbSize = ctypes.sizeof(Info)
        info.fMask = 0x00000040                                  # SEE_MASK_NOCLOSEPROCESS: give us the process handle
        info.lpVerb, info.lpFile, info.lpParameters, info.lpDirectory, info.nShow = "runas", sys.executable, line, workdir, 1
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(Info)]
        shell32.ShellExecuteExW.restype = wintypes.BOOL
        if shell32.ShellExecuteExW(ctypes.byref(info)):
            return info.hProcess or None
        error = ctypes.get_last_error()
    except Exception:
        error = None
    if error == 1223:                                            # ERROR_CANCELLED: the user chose "No"
        raise Stop("You chose 'No' in the Windows question. Changing GPU clocks needs administrator rights,\n"
                   "  so the measurement cannot run without it. Run main.py again and choose 'Yes'.")
    if error is None:                                            # the modern call did not work: try the simple one
        try:
            import ctypes as c2
            if c2.windll.shell32.ShellExecuteW(None, "runas", sys.executable, line, workdir, 1) > 32:
                return None
        except Exception:
            pass
    raise Stop("Windows did not start the measurement with administrator rights (error %s).\n"
               "  Other way: close PyCharm, right-click the PyCharm icon, choose 'Run as administrator',\n"
               "  open this project again and run main.py." % error)


def process_finished(handle):
    """(finished, exit code) of a process started by launch_elevated."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    if k.WaitForSingleObject(handle, 0) != 0:                    # 0 = WAIT_OBJECT_0: the process has ended
        return False, None
    code = wintypes.DWORD(0)
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.GetExitCodeProcess(handle, ctypes.byref(code))
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle(handle)
    return True, code.value


# ----------------------------------------------------------------------------- for testing only
_test_file = os.environ.get("PACE_HW_TEST_SETTINGS")
if _test_file and os.path.isfile(_test_file):
    with open(_test_file, encoding="utf-8") as _fh:
        globals().update(json.load(_fh))
