"""FASE 1b - slope empirico: 2 punti di occupazione GPU per dtype -> MiB/token KV, poi ctx max sotto il 93%."""
import json, os, subprocess, time, urllib.request, urllib.error

RUNTIME = r"C:/Users/User/AppData/Local/hermes/runtimes/llamacpp/b10964/cuda"
MODEL   = r"D:\models\mradermacher\Qwen3.8-9B-Distill-uncensored-heretic-i1-GGUF\Qwen3.8-9B-Distill-uncensored-heretic.i1-IQ4_XS.gguf"
OUT     = r"C:/Users/User/Documents/ai_tui/bench_qwen38"
LIMIT   = 8192 * 0.93
RESULT  = os.path.join(OUT, "phase1b_ctx_probe.json")


def gpu_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return int(o.splitlines()[0])


def wait_free(thresh=350, tmo=180):
    t0 = time.time()
    while time.time() - t0 < tmo:
        if gpu_mib() < thresh:
            return True
        time.sleep(2)
    return False


def probe(ctx, ctk):
    cmd = [f"{RUNTIME}/llama-server.exe", "-m", MODEL, "-c", str(ctx), "-ngl", "99",
           "-fa", "on", "-ctk", ctk, "-ctv", ctk, "-t", "8", "--port", "18434",
           "--no-webui", "-np", "1"]
    lp = os.path.join(OUT, "logs", f"probe_c{ctx}_{ctk}.log")
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    lf = open(lp, "w", encoding="utf-8", errors="replace", buffering=1)
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=RUNTIME)
    ok, peak, base = False, 0, gpu_mib()
    t0 = time.time()
    while time.time() - t0 < 240:
        if p.poll() is not None:
            break
        peak = max(peak, gpu_mib())
        try:
            with urllib.request.urlopen("http://127.0.0.1:18434/health", timeout=3) as r:
                if r.status == 200 and b"Loading" not in r.read():
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(1)
    if ok:
        peak = max(peak, gpu_mib())
    if p.poll() is None:
        p.terminate()
        try: p.wait(timeout=60)
        except subprocess.TimeoutExpired: p.kill()
    wait_free()
    rec = {"dtype": ctk, "ctx": ctx, "started": ok, "gpu_idle_before_mib": base,
           "gpu_peak_mib": peak, "pct": round(100 * peak / 8192, 1), "under_cap": peak <= LIMIT}
    json.dump(rec, open(RESULT.replace(".json", "_last.json"), "w"), indent=2)
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    pts = []
    for ctk in ("q8_0", "f16", "q4_0"):
        for ctx in (32768, 65536):
            wait_free()
            pts.append(probe(ctx, ctk))
            time.sleep(3)

    # slope
    print("\n=== SLOPE (MiB per 1k token) ===", flush=True)
    slopes = {}
    for ctk in ("q8_0", "f16", "q4_0"):
        a = [x for x in pts if x["dtype"] == ctk and x["started"]]
        if len(a) < 2:
            slopes[ctk] = None; continue
        a.sort(key=lambda r: r["ctx"])
        d_mib = a[1]["gpu_peak_mib"] - a[0]["gpu_peak_mib"]
        d_ctx = a[1]["ctx"] - a[0]["ctx"]
        s = d_mib / d_ctx * 1000
        base = a[0]["gpu_peak_mib"] - a[0]["ctx"] * s / 1000
        slopes[ctk] = {"mib_per_1k_tok": round(s, 3), "base_mib": round(base, 1),
                       "ctx_max_under_cap": int((LIMIT - base) / s * 1000)}
        print(f"{ctk}: {s:.3f} MiB/1k  base={base:.0f} MiB  ctx_max@93%={int((LIMIT-base)/s*1000)}", flush=True)

    json.dump({"probes": pts, "slopes": slopes}, open(RESULT, "w"), indent=2)
    print("DONE", flush=True)
