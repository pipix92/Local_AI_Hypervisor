"""FASE 1b - KV self size autorevole via llama-bench + probing del ctx massimo reale sotto il 93%."""
import json, os, re, subprocess, time

RUNTIME = r"C:/Users/User/AppData/Local/hermes/runtimes/llamacpp/b10964/cuda"
MODEL   = r"D:\models\mradermacher\Qwen3.8-9B-Distill-uncensored-heretic-i1-GGUF\Qwen3.8-9B-Distill-uncensored-heretic.i1-IQ4_XS.gguf"
OUT     = r"C:/Users/User/Documents/ai_tui/bench_qwen38"
LIMIT   = 8192 * 0.93   # 7618.56 MiB


def bench(ctx, ctk, ctv, p=128, n=16):
    cmd = [f"{RUNTIME}/llama-bench.exe", "-m", MODEL, "-c", str(ctx), "-ngl", "99",
           "-fa", "on", "-ctk", ctk, "-ctv", ctv, "-t", "8", "-p", str(p), "-n", str(n), "-r", "1"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=RUNTIME, timeout=900)
    out = r.stdout + r.stderr
    rec = {"ctx": ctx, "ctk": ctk, "ctv": ctv, "raw": out.strip()[-3000:], "seconds": round(time.time() - t0, 1)}
    m = re.search(r"(\d+)\.\d+ GiB", out)
    # llama-bench prints: "| N | ctx | ctk | ctv | ... | pp512 | tg128 |"
    m2 = re.search(r"model size\s*=\s*([\d.]+)\s*MiB", out)
    m3 = re.search(r"kv self size\s*=\s*([\d.]+)\s*MiB", out)
    m4 = re.search(r"buffer size\s*=\s*([\d.]+)\s*MiB", out)
    if m2: rec["model_size_mib"] = float(m2.group(1))
    if m3: rec["kv_self_size_mib"] = float(m3.group(1))
    if m4: rec["buffer_size_mib"] = float(m4.group(1))
    return rec


def gpu_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return int(o.splitlines()[0])


if __name__ == "__main__":
    results = []
    print(f"VRAM idle: {gpu_mib()} MiB | limit {LIMIT:.0f} MiB (93%)\n")
    # KV self size per dtype at a fixed ctx (KV is linear in ctx -> one point is enough)
    for ctk in ("q8_0", "q4_0", "f16"):
        r = bench(65536, ctk, ctk)
        results.append(r)
        print(f"--- {ctk} @65536  kv={r.get('kv_self_size_mib')} MiB  "
              f"model={r.get('model_size_mib')} MiB  buf={r.get('buffer_size_mib')} MiB  ({r['seconds']}s)")
        if r.get("kv_self_size_mib") is None:
            print(r["raw"][-1200:])

    json.dump(results, open(os.path.join(OUT, "phase1b_kv_selfsize.json"), "w"), indent=2)

    # Probe real max ctx under the cap, per dtype, using measured GPU occupancy.
    print("\n=== PROBING max ctx (measured via nvidia-smi, real llama-server) ===")
    probes = []
    for ctk, cands in (("q8_0", [131072, 98304, 81920, 65536]),
                       ("q4_0", [196608, 163840, 131072]),
                       ("f16", [49152, 40960, 32768])):
        found = None
        for ctx in cands:
            cmd = [f"{RUNTIME}/llama-server.exe", "-m", MODEL, "-c", str(ctx), "-ngl", "99",
                   "-fa", "on", "-ctk", ctk, "-ctv", ctk, "-t", "8", "--port", "18434",
                   "--no-webui", "-np", "1"]
            lp = os.path.join(OUT, "logs", f"probe_c{ctx}_{ctk}.log")
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            lf = open(lp, "w", encoding="utf-8", errors="replace", buffering=1)
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=RUNTIME)
            ok, peak = False, 0
            t0 = time.time()
            while time.time() - t0 < 300:
                if p.poll() is not None:
                    break
                try:
                    import urllib.request
                    with urllib.request.urlopen("http://127.0.0.1:18434/health", timeout=3) as r:
                        if r.status == 200 and b"Loading" not in r.read():
                            ok = True
                            break
                except Exception:
                    pass
                peak = max(peak, gpu_mib())
                time.sleep(1)
            peak = max(peak, gpu_mib())
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=60)
                except subprocess.TimeoutExpired: p.kill()
            time.sleep(4)
            for _ in range(60):
                if gpu_mib() < 300: break
                time.sleep(1)
            free_after = gpu_mib()
            pct = 100 * peak / 8192
            rec = {"dtype": ctk, "ctx": ctx, "started": ok, "gpu_peak_mib": peak,
                   "pct_of_8192": round(pct, 1), "under_cap": peak <= LIMIT, "gpu_after_kill": free_after}
            probes.append(rec)
            print(f"  {ctk} ctx={ctx:6d} started={ok} peak={peak} MiB ({pct:.1f}%) "
                  f"{'OK' if peak <= LIMIT else 'OVER CAP'}  freed->{free_after}")
            if not ok or peak > LIMIT:
                break
            found = ctx
        print(f"  => max ctx for {ctk}: {found}")

    json.dump(probes, open(os.path.join(OUT, "phase1b_ctx_probe.json"), "w"), indent=2)
    print("\nDONE")
