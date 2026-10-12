#!/usr/bin/env python3
"""
Benchmark KV / context per la TUI ai_tui su Miky (RTX 5050 8GB).
Usa llama-server direttamente (non LM Studio) per avere il controllo completo dei flag.

Configurazioni testate per ogni modello:
  vram-kv  : KV in VRAM, f16            -> massima velocita', tetto contesto ~16k
  ram-kv   : KV in RAM (-nkvo), f16     -> contesto lungo, penalita' di banda
  ram-kv-q8: KV in RAM (-nkvo), q8_0    -> meta' memoria ma su RAM = dequant per token
  shift    : come ram-kv + --ctx-shift  -> rolling window, tetto dichiarato alto

Contesti: 8192, 16384, 32768, 65536, 131072
Metriche: tok/s, TTFT, VRAM libera, tempo di load
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error

PORT_BASE = 18600
MODELS_DIR = "D:/LLM models"
OUT = "D:/ai_tui/mikye_kv_results.json"

# llama-server CUDA piu' recente presente
BACKENDS = os.path.expandvars(
    r"%USERPROFILE%\.lmstudio\extensions\backends")


def find_llama():
    cands = []
    if os.path.isdir(BACKENDS):
        for d in os.listdir(BACKENDS):
            if "cuda" in d.lower():
                p = os.path.join(BACKENDS, d, "llama-server.exe")
                if os.path.exists(p):
                    ver = re.findall(r"(\d+)\.(\d+)\.(\d+)", d)
                    cands.append(((tuple(int(x) for x in ver[0]) if ver else (0,)), p))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None


LLAMA = find_llama()

MODELS = {
    "gemma4-12b-qat": {
        "path": f"{MODELS_DIR}/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/"
                f"Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        "mtp": f"{MODELS_DIR}/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/"
               f"mtp-gemma-4-12B-it.gguf",
    },
    "spark-4b": {
        "path": f"{MODELS_DIR}/XHToken/Spark-X2.5-4B-GGUF/Spark-X2.5-4B-Q4_K_M.gguf",
    },
    "qwen3.5-9b": {
        "path": f"{MODELS_DIR}/bigatuna/Qwen3.5-9b-Sushi-Coder-GGUF/Qwen3.5-9b-Sushi-Coder-Q4_K_M.gguf",
    },
}

# (label, flag KV, flag shift)
CONFIGS = [
    ("vram-kv",    ["-ctk", "f16",  "-ctv", "f16"], False),
    ("ram-kv",     ["-nkvo", "-ctk", "f16", "-ctv", "f16"], False),
    ("ram-kv-q8",  ["-nkvo", "-ctk", "q8_0", "-ctv", "q8_0"], False),
    ("shift",      ["-nkvo", "-ctk", "f16", "-ctv", "f16", "--ctx-shift"], True),
]
CONTEXTS = [8192, 16384, 32768, 65536, 131072]

results = []


def nvidia():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        t, u, f = [int(x) for x in o.split(",")]
        return {"total": t, "used": u, "free": f}
    except Exception:
        return None


def kill_all():
    for exe in ("llama-server.exe", "llama_server.exe"):
        subprocess.run(["taskkill", "/F", "/IM", exe, "/T"],
                       capture_output=True, text=True, timeout=20)
    time.sleep(5)


def start(gguf, ctx, kv_flags, shift, use_mtp=False, mtp=None):
    global PORT
    PORT += 1
    args = [LLAMA, "-m", gguf, "-c", str(ctx), "--port", str(PORT),
            "--host", "127.0.0.1", "-ngl", "99", "-fa", "on", "--no-warmup",
            "-t", "8", "-b", "2048"] + kv_flags
    if shift:
        args += ["--ctx-shift", "2048"]
    if use_mtp and mtp and os.path.exists(mtp):
        args += ["--model-draft", mtp, "--spec-type", "draft-mtp"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read()
            return proc, round(time.time() - _t0, 1)
        except Exception:
            if proc.poll() is not None:
                return None, None
    return None, None


_t0 = time.time()


def gen(prompt, max_tok=48, timeout=900):
    body = {"prompt": prompt, "n_predict": max_tok, "temperature": 0.1, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            try:
                d = json.loads(line[5:].strip())
            except Exception:
                continue
            if d.get("content"):
                if ttft is None:
                    ttft = (time.time() - t0) * 1000
                n += 1
    el = max(1e-6, (time.time() - t0) - (ttft or 0) / 1000)
    return {"ttft_ms": round(ttft or 0, 1), "tok_s": round(n / el, 2), "gen_tok": n}


PORT = PORT_BASE

if __name__ == "__main__":
    print("=" * 78)
    print(f"MIKY KV BENCHMARK - llama-server: {LLAMA}")
    print(f"modelli: {list(MODELS)}")
    print("=" * 78)
    print(f"  {'config':<12} {'ctx':>7}  {'load_s':>7} {'tok/s':>7} {'ttft_ms':>9} {'vram_free':>9}  stato")

    for mname, mcfg in MODELS.items():
        if not os.path.exists(mcfg["path"]):
            print(f"\n  SKIP {mname}: file non trovato")
            continue
        for cfg_label, kv_flags, shift in CONFIGS:
            for ctx in CONTEXTS:
                kill_all()
                _t0 = time.time()
                proc, load_s = start(mcfg["path"], ctx, kv_flags, shift,
                                     mtp=mcfg.get("mtp"))
                if proc is None:
                    st = "LOAD FALLITO (KV non entra o errore)"
                    print(f"  {cfg_label:<12} {ctx:>7}  {'-':>7} {'-':>7} {'-':>9} {'-':>9}  {st}", flush=True)
                    results.append({"model": mname, "config": cfg_label, "ctx": ctx,
                                    "status": "load_failed"})
                    continue
                v = nvidia()
                filler = "Analizza il sistema di telemetria e produci un report. " * 6000
                prompt = (filler + "\nRispondi: OK") if ctx >= 16384 else "Rispondi: OK"
                try:
                    r = gen(prompt)
                    row = {"model": mname, "config": cfg_label, "ctx": ctx,
                           "load_s": load_s, "vram_free": v["free"] if v else None, **r}
                    results.append(row)
                    print(f"  {cfg_label:<12} {ctx:>7}  {load_s:>7} {r['tok_s']:>7} "
                          f"{r['ttft_ms']:>9} {row['vram_free']:>9}  ok", flush=True)
                except Exception as e:
                    results.append({"model": mname, "config": cfg_label, "ctx": ctx,
                                    "err": str(e)[:120]})
                    print(f"  {cfg_label:<12} {ctx:>7}  {load_s:>7} ERRORE: {str(e)[:50]}", flush=True)
                json.dump(results, open(OUT, "w"), indent=2)

    kill_all()
    print(f"\nSalvato: {OUT}")
