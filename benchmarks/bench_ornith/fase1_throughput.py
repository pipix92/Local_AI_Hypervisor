"""FASE 1: throughput at an IDENTICAL prompt across f16 / q8_0 / q4_0.

Per the vault rule: vary the dtype, HOLD the prompt. Prompt length changes make
runs incomparable, so the same text is used for all three and the server ctx is
held fixed at the smallest tested max so only the dtype differs.
"""
import json, subprocess, sys, time, urllib.request

SRV = r"C:/Users/User/AppData/Local/hermes/runtimes/llamacpp/b10964/cuda/llama-server.exe"
MODEL = sys.argv[1]
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
OUT = r"C:/Users/User/Documents/ai_tui/bench_ornith"
PORT = 18434
CAP_MIB = 7618
THREADS = 8

# ONE fixed prompt, ~2500 tokens of varied prose+code+math. Same bytes for all dtypes.
WORDS = ("the quick brown fox jumps over lazy dog while machine learning models "
         "transform sequences of tokens into embeddings and attention heads "
         "aggregate value vectors across sequence positions in parallel layers "
         "of a transformer block with residual connections and normalization "
         "compute def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2) "
         "the integral of x squared over zero to pi equals pi cubed divided by three "
         "memory bandwidth measured in gigabytes per second determines throughput "
         "context length tokens kilobytes megabytes gigabytes quantization bit width "
         "temperature sampling logits softmax probability distribution inference ")
PROMPT = ("Below is reference material.\n\n" + (WORDS * 60) +
          "\n\nQuestion: list three causes of thermal throttling in a GPU.\nAnswer:")


def gpu():
    o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"], text=True)
    return int(o.strip())


def health(timeout=400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def run(dtype, reps=3):
    log = open(f"{OUT}/f1_{dtype}.log", "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen([SRV, "-m", MODEL, "-c", str(CTX), "-ngl", "99", "-fa", "on",
                          "-ctk", dtype, "-ctv", dtype, "-t", str(THREADS),
                          "--port", str(PORT), "--host", "127.0.0.1", "-np", "1"],
                         stdout=log, stderr=subprocess.STDOUT)
    try:
        if not health():
            return {"dtype": dtype, "ok": False, "err": "server did not become healthy"}
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=json.dumps({"prompt": PROMPT, "n_predict": 128, "temperature": 0.0,
                             "cache_prompt": False, "seed": 42}).encode(),
            headers={"Content-Type": "application/json"})
        runs = []
        vram_peak = gpu()
        for _ in range(reps):
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            t = r["timings"]
            runs.append({
                "prompt_tokens": r.get("tokens_evaluated"),
                "gen_tokens": t.get("predicted_count"),
                "prompt_tok_s": t.get("prompt_per_second"),
                "gen_tok_s": t.get("predicted_per_second"),
                "prompt_ms": t.get("prompt_ms"),
                "predicted_ms": t.get("predicted_ms"),
            })
            vram_peak = max(vram_peak, gpu())
            time.sleep(2)
        return {"dtype": dtype, "ok": True, "ctx": CTX, "vram_mib": vram_peak,
                "fits_cap": vram_peak <= CAP_MIB, "runs": runs,
                "gen_tok_s_mean": sum(x["gen_tok_s"] for x in runs) / len(runs),
                "prompt_tok_s_mean": sum(x["prompt_tok_s"] for x in runs) / len(runs),
                "prompt_tokens": runs[0]["prompt_tokens"]}
    finally:
        p.kill()
        p.wait(timeout=60)
        log.close()
        time.sleep(6)


subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
time.sleep(6)
idle = gpu()
print(f"idle={idle} MiB  cap={CAP_MIB}  ctx={CTX}")

res = {"idle_mib": idle, "ctx": CTX, "prompt_len_chars": len(PROMPT), "dtypes": {}}
for d in ["f16", "q8_0", "q4_0"]:
    if d == "q4_0":
        pass
    r = run(d)
    res["dtypes"][d] = r
    print(f"{d:6s} vram={r.get('vram_mib')} fits={r.get('fits_cap')} "
          f"gen={r.get('gen_tok_s_mean')} t/s prompt={r.get('prompt_tok_s_mean')} t/s "
          f"ptoks={r.get('prompt_tokens')}")
    json.dump(res, open(f"{OUT}/fase1_throughput.json", "w"), indent=1)
subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
print(json.dumps(res, indent=1))
