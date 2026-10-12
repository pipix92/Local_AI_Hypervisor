"""FASE 0: find max ctx that fits under the 7618 MiB (93%) VRAM cap, per KV dtype.

Method: launch llama-server at a candidate ctx, wait for /health, read actual
peak VRAM from nvidia-smi, kill, and record. Walk DOWN from an analytic estimate
until a launch fails or VRAM exceeds the cap -> that value is the real max.

Never accepts a launch that offloads: we verify n_gpu_layers == 999 from the log.
"""
import json, os, re, subprocess, sys, time, urllib.request

SRV = r"C:/Users/User/AppData/Local/hermes/runtimes/llamacpp/b10964/cuda/llama-server.exe"
MODEL = sys.argv[1]
OUT = r"C:/Users/User/Documents/ai_tui/bench_ornith"
PORT = 18434
CAP_MIB = 7618          # 93% of 8192
BASELINE_MIB = None     # measured idle usage
THREADS = 8


def gpu_mib():
    o = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True).strip()
    used, total = [int(x) for x in o.split(",")]
    return used, total


def wait_health(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            if p.poll() is not None:
                return False
            time.sleep(2)
    return False


def probe(max_tokens=32):
    """force an actual inference pass so the full KV is materialised"""
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps({"prompt": "Hello, my name is", "n_predict": max_tokens,
                         "temperature": 0.0, "cache_prompt": False}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.time() - t0, r.get("timings", {})


p = None


def launch(ctx, ctk, ctv, tag, np=1, extra=None):
    global p
    log = open(f"{OUT}/srv_{tag}.log", "w", encoding="utf-8", errors="replace")
    cmd = [SRV, "-m", MODEL, "-c", str(ctx), "-ngl", "99", "-fa", "on",
           "-ctk", ctk, "-ctv", ctv, "-t", str(THREADS), "--port", str(PORT),
           "--host", "127.0.0.1", "-np", str(np), "--no-warmup"]
    if extra:
        cmd += extra
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    ok = wait_health()
    if not ok:
        txt = open(f"{OUT}/srv_{tag}.log", encoding="utf-8", errors="replace").read()
        p.wait(timeout=60)
        p = None
        log.close()
        return None, None, txt
    used, total = gpu_mib()
    dt, tm = probe()
    used2, _ = gpu_mib()
    txt = open(f"{OUT}/srv_{tag}.log", encoding="utf-8", errors="replace").read()
    p.kill()
    p.wait(timeout=60)
    p = None
    log.close()
    return {"ctx": ctx, "ctk": ctk, "ctv": ctv, "np": np, "vram_mib": used2,
            "kv_mib": None, "tok_s": tm.get("predicted_per_second"),
            "prompt_tok_s": tm.get("prompt_per_second"),
            "n_gpu_layers": int(m.group(1)) if (m := re.search(r"n_gpu_layers\s*=\s*(\d+)", txt)) else None,
            "offload_kv": bool(re.search(r"offload(?:ing)? (?:the )?KV", txt, re.I)),
            "fit_changed": bool(re.search(r"fit.*(reduc|adjust|chang)", txt, re.I)),
            "log": f"srv_{tag}.log"}, txt


def kill_stale():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True, text=True)
    time.sleep(6)


if __name__ == "__main__":
    kill_stale()
    time.sleep(4)
    idle, total = gpu_mib()
    BASELINE_MIB = idle
    print(f"idle VRAM = {idle} MiB / {total} MiB ; cap = {CAP_MIB} MiB")
    budget = CAP_MIB - idle
    print(f"budget for llama-server = {budget} MiB")

    # analytic KV rate from header: 8 full-attn layers, 4 kv heads, head_dim 256
    KV_LAYERS = 8
    bytes_f16 = KV_LAYERS * 4 * 256 * 2 * 2      # K+V, f16
    rate = {"f16": bytes_f16 / 1024,
            "q8_0": bytes_f16 * 0.53 / 1024,
            "q4_0": bytes_f16 * 0.27 / 1024}
    for k, v in rate.items():
        print(f"  KV {k:5s}: {v:.2f} MiB / 1k tok")

    results = {}
    for dtype in ["q4_0", "q8_0", "f16"]:
        ctk = "q4_0" if dtype == "q4_0" else dtype
        ctv = ctk
        # ladder descending from analytic max
        if dtype == "f16":
            ladder = [65536, 49152, 32768, 24576, 16384]
        elif dtype == "q8_0":
            ladder = [131072, 98304, 65536, 49152, 32768]
        else:
            ladder = [262144, 196608, 131072, 98304, 65536]
        best = None
        for ctx in ladder:
            r, txt = launch(ctx, ctk, ctv, f"f0_{dtype}_{ctx}")
            if r is None:
                print(f"[{dtype}] ctx={ctx}: LAUNCH FAILED")
                continue
            fits = r["vram_mib"] <= CAP_MIB and r["n_gpu_layers"] == 999 and not r["offload_kv"]
            print(f"[{dtype}] ctx={ctx}: vram={r['vram_mib']} MiB "
                  f"ngl={r['n_gpu_layers']} offload_kv={r['offload_kv']} "
                  f"tok/s={r['tok_s']} -> {'OK' if fits else 'NO'}")
            if fits:
                best = r
                break
            else:
                print("   detail:", r["log"])
        results[dtype] = best
        json.dump(results, open(f"{OUT}/fase0_maxctx.json", "w"), indent=1)
    kill_stale()
    print(json.dumps(results, indent=1))
