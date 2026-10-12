#!/usr/bin/env python
"""
DRIVER LOCALE - Host Node, RTX 3050 8GB.
Testa le 3 modalita' KV con llama-server diretto (controllo completo delle flag):
  vram  : KV in VRAM              (default, -ctk f16 -ctv f16)
  ram   : KV in RAM               (-nkvo)
  full  : offload completo        (-nkvo -ngl 99, pesi in GPU, KV su host 32GB)
Poi testa la compressione Groq sul contesto.
"""
import json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_harness import CONTEXTS, PROBE_TOKENS, gen_probe, nvidia_local, groq_compress, estimate_tokens

# llama-server CUDA: i build in .lmstudio/extensions/backends sono STUB da 20KB
# (la logica sta in llama-server-impl.dll) e non partono da CLI. Il runtime
# Hermes invece espone un binario funzionante, verificato con --version.
HERMES_RUNTIME = r"C:\Users\User\AppData\Local\hermes\runtimes\llamacpp\b10964\cuda\llama-server.exe"
BACKENDS = r"C:\Users\User\.lmstudio\extensions\backends"


def find_llama():
    """Preferisce il runtime Hermes (binario reale). Fallback: build LM Studio
    standalone, ignorando gli stub <100KB che escono senza fare nulla."""
    if os.path.exists(HERMES_RUNTIME):
        return HERMES_RUNTIME
    cands = []
    for d in os.listdir(BACKENDS):
        p = os.path.join(BACKENDS, d, "llama-server.exe")
        if os.path.exists(p) and "cuda" in d.lower() and os.path.getsize(p) > 100_000:
            cands.append((d, p))
    cands.sort(reverse=True)
    return cands[0][1] if cands else None

LLAMA = find_llama()
print(f"llama-server: {LLAMA}")

# modelli: preferisci i piu' adatti a 8GB
MODELS = [
    (r"D:\models\yuxinlu1\gemma-4-12B-coder-fable5-composer2.5-v1-GGUF\gemma4-coding-Q3_K_M.gguf", "gemma4-12b-coder-fable5"),
    (r"D:\models\lmstudio-community\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf", "qwen3.5-4b"),
    (r"D:\models\XHToken\Spark-X2.5-4B-GGUF\Spark-X2.5-4B-Q4_K_M.gguf", "spark-x2.5-4b"),
]
MODELS = [(p, n) for p, n in MODELS if os.path.exists(p)]

MODES = {
    "vram": [],              # KV in VRAM (default llama.cpp)
    "ram":  ["-nkvo"],       # KV offload su host
    "full": ["-nkvo"],       # + ngl 99 sotto
}

results = []
port = 18500
proc = None


def kill():
    global proc
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        proc = None
    time.sleep(4)


def start_server(gguf, ctx, mode):
    global proc, port
    kill()
    port += 1
    args = [LLAMA, "-m", gguf, "-c", str(ctx), "--port", str(port),
            "--host", "127.0.0.1", "-ngl", "99",
            "-ctk", "f16", "-ctv", "f16", "--flash-attn", "on", "--no-warmup"]
    args += MODES[mode]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # attendi che sia pronto
    import urllib.request
    for _ in range(120):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).read()
            return True
        except Exception:
            if proc.poll() is not None:
                return False
    return False


def vram_free():
    return nvidia_local()


if __name__ == "__main__":
    print("=" * 78)
    print(f"DRIVER LOCALE - RTX 3050 | modelli: {[n for _, n in MODELS]}")
    print("=" * 78)

    for gguf, name in MODELS:
        for mode in MODES:
            for ctx in CONTEXTS:
                label = f"{name} | kv={mode}"
                try:
                    if not start_server(gguf, ctx, mode):
                        results.append({"label": label, "ctx": ctx, "err": "server non partito"})
                        print(f"  {label:<46} ctx={ctx:>6} SERVER NON PARTITO", flush=True)
                        continue
                    v = vram_free()
                    filler = "Analizza il sistema di telemetria e produci un report. " * 6000
                    prompt = filler + "\nRispondi: OK" if ctx >= 16384 else "Rispondi solo: OK"
                    r = gen_probe(f"http://127.0.0.1:{port}", name, prompt, timeout=600)
                    row = {"label": label, "ctx": ctx, "model": name, "mode": mode,
                           "vram_free": v["free"] if v else None, **r}
                    results.append(row)
                    print(f"  {label:<46} ctx={ctx:>6} tok/s={r['tok_s']:>6} "
                          f"ttft={r['ttft_ms']:>8} vram_free={row['vram_free']}", flush=True)
                except Exception as e:
                    results.append({"label": label, "ctx": ctx, "err": str(e)[:100]})
                    print(f"  {label:<46} ctx={ctx:>6} ERR {str(e)[:70]}", flush=True)

    kill()

    # ---- compressione Groq: quanto riduce un contesto reale
    print("\n" + "=" * 78)
    print("COMPRESSIONE GROQ su contesto di agente")
    print("=" * 78)
    real_ctx = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "ai_tui", "tui_app.py"), encoding="utf-8", errors="ignore").read(60000)
    s, meta = groq_compress(real_ctx)
    if s:
        comp_info = {"orig_chars": len(real_ctx), "comp_chars": len(s),
                     "orig_tok_est": estimate_tokens(real_ctx), "comp_tok_est": estimate_tokens(s),
                     "groq_latency_s": meta,
                     "compression_ratio": round(estimate_tokens(s) / max(1, estimate_tokens(real_ctx)), 3)}
        print(f"  originale: {comp_info['orig_chars']} char (~{comp_info['orig_tok_est']} tok)")
        print(f"  compresso: {comp_info['comp_chars']} char (~{comp_info['comp_tok_est']} tok)")
        print(f"  ratio: {comp_info['compression_ratio']}  latenza Groq: {meta}s")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "groq_compressed_sample.txt"), "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(f"  compressione fallita: {meta}")
        comp_info = {"err": meta}

    json.dump({"results": results, "groq": comp_info},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_local_results.json"), "w"),
              indent=2)
    print("\nSalvato: kv_local_results.json")
