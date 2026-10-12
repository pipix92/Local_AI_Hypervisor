#!/usr/bin/env python
"""
HARNESS COMUNE di benchmark KV/context - usato sia sul Host Node (RTX 3050) sia
sul PC di Miky (RTX 5050) tramite SSH, cosi' i due risultati sono confrontabili.

Misura per ogni combinazione modello x modalita' KV x context:
  - tok/s, TTFT (ms), VRAM usata (MiB)
  - quanto della finestra di contesto regge PRIMA di degradare
  - effetto della compressione Groq sul contesto in ingresso

Modalita' KV:
  vram  -> KV in VRAM   (noKvOffload=true, il default che misurammo prima)
  ram   -> KV in RAM    (-nkvo in llama-server, flag: offload KV to CPU)
  full  -> -ngl 99 + KV in RAM = "offload completo" (pesi tutti in GPU, KV su host)

Uso:
  python kv_harness.py --mode local  --models cfg.json
  python kv_harness.py --mode remote --models cfg.json   (via SSH + API LM Studio)
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.parse, argparse

# ---------------------------------------------------------------- config
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile non e' piu' disponibile su questa chiave (model_not_found).
# qwen/qwen3.8-27b: 131k di contesto, scelto per comprimere blocchi lunghi.
GROQ_MODEL = "qwen/qwen3.8-27b"
ENV_FILE = r"C:\Users\User\AppData\Local\hermes\.env"

CONTEXTS = [8192, 16384, 32768, 65536]
PROBE_TOKENS = 48          # token da generare per la misura
FILL_TOKENS = 6000         # riempimento per testare la tenuta della finestra


def groq_key():
    try:
        for line in open(ENV_FILE, encoding="utf-8", errors="ignore"):
            m = re.match(r"\s*GROQ_API_KEY\s*=\s*(\S+)", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def groq_compress(text, max_tokens=1200):
    """Chiede a Groq di condensare il contesto preservando fatti, tool e decisioni."""
    key = groq_key()
    if not key:
        return None, "no-key"
    # Groq rifiuta con 413 i payload sopra ~1MB: un contesto da 60k token lo supera.
    # Si spezza in blocchi da ~11k caratteri, compressi indipendentemente.
    if len(text) > 11000:
        parts = [text[i:i + 11000] for i in range(0, len(text), 11000)]
        merged, lat = [], 0.0
        for i, part in enumerate(parts):
            out, l = groq_compress(part, max_tokens)
            lat += l
            if out:
                merged.append(f"[parte {i+1}/{len(parts)}] {out}")
        if not merged:
            return None, "compressione fallita su tutti i blocchi"
        return "\n".join(merged), round(lat, 2)
    prompt = (
        "Comprimi questo contesto di agente. Tieni: obiettivo corrente, file/percorsi, "
        "stati, errori, decisioni prese, TODO aperti, tool disponibili. Scarta: ripetizioni, "
        "log verbosi, testo già risolto. Max 900 parole. Solo il riassunto.\n\n" + text
    )
    try:
        body = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.2}
        req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json",
                                              # errore 1010 = Cloudflare bot check: senza UA python
                                              # la richiesta viene bloccata prima di arrivare all'API
                                              "User-Agent": "curl/8.5.0",
                                              "Accept": "application/json"},
                                     method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read())
        out = d["choices"][0]["message"]["content"]
        return out, round(time.time() - t0, 2)
    except Exception as e:
        return None, str(e)[:90]


# ---------------------------------------------------------------- measuring
def estimate_tokens(text):
    return max(1, len(text) // 4)


def gen_probe(port_or_base, model, prompt, gen_tok=PROBE_TOKENS, timeout=900):
    """Genera in streaming e restituisce TTFT + tok/s.

    chat_template_kwargs.enable_thinking=false e' OBBLIGATORIO: senza, i modelli
    Qwen3.5/Spark/Gemma4 consumano l'intero max_tokens in reasoning_content e
    restituiscono content vuoto (finish_reason=length) -> tok/s 0.0.
    """
    url = (port_or_base if str(port_or_base).startswith("http")
           else f"http://127.0.0.1:{port_or_base}")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": gen_tok, "temperature": 0.1, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                if line == "data: [DONE]":
                    break
                continue
            try:
                d = json.loads(line[5:].strip())
            except Exception:
                continue
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = (time.time() - t0) * 1000
                n += 1
    el = max(1e-6, (time.time() - t0) - (ttft or 0) / 1000)
    return {"ttft_ms": round(ttft or 0, 1), "tok_s": round(n / el, 2), "gen_tok": n}


def nvidia_local():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        t, u, f = [int(x) for x in o.split(",")]
        return {"total": t, "used": u, "free": f}
    except Exception:
        return None


def nvidia_remote(ssh_key, host, user):
    sh = ('export PATH="/mnt/c/Windows/System32:$PATH"\n'
          '/mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=memory.total,memory.used,'
          'memory.free --format=csv,noheader,nounits\n')
    try:
        p = subprocess.run(["ssh", "-i", ssh_key, "-o", "IdentitiesOnly=yes", "-o",
                            "StrictHostKeyChecking=no", f"{user}@{host}", "bash -s"],
                           input=sh, capture_output=True, text=True, timeout=60)
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                return {"total": int(parts[0]), "used": int(parts[1]), "free": int(parts[2])}
    except Exception:
        pass
    return None


def out(row):
    print(f"  {row['label']:<46} ctx={row.get('ctx'):>6} "
          f"tok/s={row.get('tok_s', '-'):>6} ttft={row.get('ttft_ms', '-'):>8} "
          f"vram_free={row.get('vram_free', '-')}", flush=True)
    return row


def run_matrix(load_fn, unload_fn, vram_fn, model, port, base_url, results, tag):
    """Cycle through contexts with the given KV mode already applied by load_fn."""
    for ctx in CONTEXTS:
        try:
            unload_fn()
            load_fn(ctx)
            time.sleep(3)
            v = vram_fn()
            row = {"tag": tag, "model": model, "ctx": ctx, "vram_free": v["free"] if v else None}
            if ctx >= 16384:
                filler = "Analizza il sistema di telemetria e produci un report. " * FILL_TOKENS
                row.update(gen_probe(base_url or port, model, filler + "\nRispondi: OK"))
            else:
                row.update(gen_probe(base_url or port, model, "Rispondi solo: OK"))
            results.append(out(row))
        except Exception as e:
            results.append({"tag": tag, "model": model, "ctx": ctx, "err": str(e)[:90]})
            print(f"  {tag} ctx={ctx} ERRORE: {str(e)[:90]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "remote"], required=True)
    ap.add_argument("--models", required=True, help="JSON con la lista dei modelli da testare")
    ap.add_argument("--out", default="kv_results.json")
    args = ap.parse_args()
    print(f"Harness {args.mode} - context {CONTEXTS}")
