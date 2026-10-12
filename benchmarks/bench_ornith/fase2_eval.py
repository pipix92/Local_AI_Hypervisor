"""FASE 2: lm_eval task suite on the Ornith GGUF via local-completions llama-server.

Capped at 200 examples per task. Each task is attempted independently so one
failure never costs the rest of the suite; errors are recorded verbatim.
"""
import json, os, subprocess, sys, time, urllib.request

SRV = r"C:/Users/User/AppData/Local/hermes/runtimes/llamacpp/b10964/cuda/llama-server.exe"
MODEL = sys.argv[1]
CTX = int(sys.argv[2])
DTYPE = sys.argv[3] if len(sys.argv) > 3 else "q8_0"
OUT = r"C:/Users/User/Documents/ai_tui/bench_ornith"
RES = f"{OUT}/results"
PORT = 18434
CAP_MIB = 7618
LIMIT = 200
PY = r"C:/ProgramData/chocolatey/bin/python3.13.exe"

TASKS = [
    ("mmlu_pro", 200),
    ("gpqa", 200),
    ("ifeval", 200),
    ("humaneval", 200),
]
# Recorded as absent-in-harness, NOT as a score:
UNAVAILABLE = ["livecodebench", "swebench", "terminalbench"]


def gpu():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip())


def health(timeout=500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


os.makedirs(RES, exist_ok=True)
log = open(f"{OUT}/srv_fase2_{DTYPE}_{CTX}.log", "w", encoding="utf-8", errors="replace")
srv = subprocess.Popen(
    [SRV, "-m", MODEL, "-c", str(CTX), "-ngl", "99", "-fa", "on",
     "-ctk", DTYPE, "-ctv", DTYPE, "-t", "8", "--port", str(PORT),
     "--host", "127.0.0.1", "-np", "1"],
    stdout=log, stderr=subprocess.STDOUT)

summary = {"model": MODEL, "ctx": CTX, "kv_dtype": DTYPE, "limit": LIMIT,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": {}, "unavailable": UNAVAILABLE}
try:
    if not health():
        summary["fatal"] = "llama-server never became healthy"
        json.dump(summary, open(f"{OUT}/fase2_summary.json", "w"), indent=1)
        sys.exit(1)
    vram_load = gpu()
    summary["vram_after_load_mib"] = vram_load
    summary["fits_cap"] = vram_load <= CAP_MIB
    print(f"server up, vram={vram_load} MiB fits_cap={summary['fits_cap']}")

    for task, n in TASKS:
        t0 = time.time()
        outfile = f"{RES}/{task}_{DTYPE}_{CTX}.json"
        print(f"\n=== {task} (limit {n}) ===", flush=True)
        cmd = [PY, "-m", "lm_eval", "--model", "local-completions",
               "--model_args", f"model={MODEL},num_concurrent=1,max_retries=0,dtype=q8_0",
               "--tasks", task, "--limit", str(n), "--batch_size", "1",
               "--output_path", outfile, "--seed", "1234"]
        try:
            pr = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                                cwd=OUT, encoding="utf-8", errors="replace")
            ok = pr.returncode == 0 and os.path.exists(outfile)
            rec = {"ok": ok, "returncode": pr.returncode, "seconds": round(time.time() - t0, 1),
                   "file": outfile}
            if ok:
                d = json.load(open(outfile, encoding="utf-8"))
                res = d.get("results", {}).get(task, {})
                rec["metrics"] = {k: v for k, v in res.items() if isinstance(v, (int, float))}
                rec["stderr"] = {k: v for k, v in
                                 d.get("n-shot", {}).items()} if "n-shot" in d else None
                agg = d.get("configs", {}).get("results", {})
                print("   metrics:", json.dumps(rec["metrics"])[:400])
            else:
                rec["stderr_tail"] = (pr.stderr or "")[-2500:]
                print("   FAILED rc=%s" % pr.returncode)
                print((pr.stderr or "")[-1200:])
            summary["tasks"][task] = rec
        except subprocess.TimeoutExpired:
            summary["tasks"][task] = {"ok": False, "err": f"timeout after 7200s",
                                      "seconds": round(time.time() - t0, 1)}
            print(f"   TIMEOUT after {round(time.time()-t0,1)}s")
        json.dump(summary, open(f"{OUT}/fase2_summary.json", "w"), indent=1)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
finally:
    srv.kill()
    srv.wait(timeout=60)
    log.close()
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    json.dump(summary, open(f"{OUT}/fase2_summary.json", "w"), indent=1)
print("\nDONE")
