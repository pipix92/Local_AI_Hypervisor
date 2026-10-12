#!/usr/bin/env python
# Misura la VRAM reale consumata dalla KV cache di Gemma4-12B a contesti diversi.
# Risponde a: "quanti GB sono 16k di context?" e "il rolling window aiuta?"
import json, os, subprocess, time, urllib.request, urllib.error

HOST = "192.168.1.100:56397"
VAULT = r"~/.vault"
PASS = "79Npi5Ue#UnEBthDsc&HI1g*_mb@"
MODEL = "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m"


def get_token():
    env = dict(os.environ, LEX_VAULT_PASSPHRASE=PASS)
    for py in (r"C:\Python313\python.exe", "python"):
        try:
            out = subprocess.run([py, "vault.py", "get", "collega_lmstudio_api_key"],
                                 cwd=VAULT, capture_output=True, text=True, env=env, timeout=60)
        except (FileNotFoundError, OSError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    raise RuntimeError("vault read failed")


H = {"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"}


def api(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://{HOST}{path}", data=data, headers=H,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def vram():
    """MB totali/used/free della GPU, letti dall'API di LM Studio."""
    try:
        s = api("/api/v0/system-info")
    except Exception:
        # fallback: endpoint NVIDIA se esposto
        try:
            g = api("/api/v0/gpus")
            return g
        except Exception:
            return None
    return s


def unload_all():
    for m in api("/api/v0/models")["data"]:
        if m.get("state") in ("loaded", "loading"):
            try:
                api(f"/api/v0/models/{urllib.parse.quote(m['id'])}/unload", {})
            except Exception:
                pass
    time.sleep(6)


def load(ctx, extra=None):
    cfg = {"contextLength": ctx, "gpuLayers": "max", "flashAttention": True, "noKvOffload": True}
    if extra:
        cfg.update(extra)
    api(f"/api/v0/models/{urllib.parse.quote(MODEL)}/load?wait=true", {"contextLength": cfg["contextLength"], **cfg}, timeout=900)
    return cfg


def probe(prompt_tokens=2000, max_tok=64):
    """Genera e restituisce tok/s + TTFT."""
    filler = "Analizza questo sistema di telemetria. " * prompt_tokens
    body = {"model": MODEL, "messages": [{"role": "user", "content": filler + "\nRispondi solo: OK"}],
            "max_tokens": max_tok, "temperature": 0.1, "stream": True}
    req = urllib.request.Request(f"http://{HOST}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=H, method="POST")
    t0 = time.time(); ttft = None; n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
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
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = (time.time() - t0) * 1000
                    n += 1
    except Exception as e:
        return {"err": str(e)[:80]}
    el = (time.time() - t0) - (ttft or 0) / 1000
    return {"ttft_ms": round(ttft or 0, 1), "tok_s": round(n / el, 2) if el > 0 else None,
            "gen_tok": n, "prompt_tok": prompt_tokens * 8}


results = []
print("=" * 72)
print("ESPERIMENTO 1: VRAM consumata a contesti diversi (stesso modello)")
print("=" * 72)

base = None
for ctx in (2048, 8192, 16384, 32768):
    unload_all()
    load(ctx)
    time.sleep(3)
    si = vram()
    v = None
    if si:
        try:
            g = si["gpus"][0]
            v = {"used_mb": g.get("vram", {}).get("total_mb") or g.get("vramUsed"),
                 "free_mb": g.get("vram", {}).get("free_mb")}
        except Exception:
            v = si
    row = {"ctx": ctx, "vram": v}
    if ctx in (8192, 16384):
        row.update(probe())
    results.append(row)
    print(f"  ctx={ctx:>6}  vram={v}  {row.get('tok_s', '')}")

print()
print("=" * 72)
print("ESPERIMENTO 2: 16k CONTESTO vs ROLLING WINDOW")
print("=" * 72)
# Carica lo stesso contesto e riempilo oltre la finestra per vedere se regge
unload_all()
load(16384)
time.sleep(3)
print("  ctx=16384, prompt lungo (~16k tok reali):")
long_probe = probe(prompt_tokens=1900, max_tok=48)
print(f"    -> {long_probe}")
results.append({"ctx": 16384, "long_probe": long_probe})

json.dump(results, open(r"C:\Users\User\Documents\ai_tui\mikye_kv_measure.json", "w"), indent=2)
print("\nSalvato: mikye_kv_measure.json")
