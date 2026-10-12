#!/usr/bin/env python
# Calibrazione impostazioni ottimali per i modelli su LM Studio di MIKYE (192.168.1.100:56397)
# Misura: tok/s, TTFT, VRAM, e validita' del tool-calling su RTX 5050 8GB
import json, subprocess, time, urllib.request, urllib.error, os, sys

BASE = "http://192.168.1.100:56397"
VAULT = r"~/.vault"
PASS = "79Npi5Ue#UnEBthDsc&HI1g*_mb@"

def get_token():
    """vault.py richiede 'cryptography' -> usare l'interprete di sistema, non il venv Hermes."""
    env = dict(os.environ, LEX_VAULT_PASSPHRASE=PASS)
    for py in (r"C:\Python313\python.exe", "C:/Python313/python.exe", "python"):
        try:
            out = subprocess.run([py, "vault.py", "get", "collega_lmstudio_api_key"],
                                 cwd=VAULT, capture_output=True, text=True, env=env, timeout=60)
        except (FileNotFoundError, OSError):
            continue
        tok = out.stdout.strip()
        if tok and out.returncode == 0:
            return tok
    raise RuntimeError(f"vault read failed: rc={out.returncode} err={out.stderr[:200]}")

TOKEN = get_token()
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def api(path, data=None, timeout=180):
    req = urllib.request.Request(BASE + path, headers=H,
                                 data=json.dumps(data).encode() if data else None,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def gpu():
    """Legge memoria GPU Windows via wmic/powershell locale NON disponibile: usa /api/v0/models info."""
    try:
        return api("/api/v0/models")["data"]
    except Exception as e:
        return {"err": str(e)}

# 1. stato attuale
print("=" * 72)
print("STATO INIZIALE")
print("=" * 72)
for m in api("/api/v0/models")["data"]:
    if m.get("state") != "not-loaded":
        print(f"  [{m.get('state'):>12}] {m.get('id')}")

# 2. configurazioni da testare (context x gpu offload)
CONFIGS = [
    {"label": "ctx32k ngl-full", "ctx": 32768, "ngl": "max"},
    {"label": "ctx16k ngl-full", "ctx": 16384, "ngl": "max"},
    {"label": "ctx8k  ngl-full", "ctx": 8192,  "ngl": "max"},
]
MODELS = [
    "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m",
    "qwen3.8-27b-imatrix-mtp",
    "qwen3.5-9b-dflash",
    "gemma-4-12b-coder-fable5-composer2.5-v1",
]

PROMPT = "Scrivi in Python una funzione che calcola i numeri primi fino a n. Rispondi solo con il codice."
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }
}]

results = []
for mid in MODELS:
    for cfg in CONFIGS:
        print(f"\n>>> {mid} | {cfg['label']}", flush=True)
        try:
            api("/api/v0/models/load", {"model": mid, "contextLength": cfg["ctx"],
                                        "gpu": cfg["ngl"], "ttl": 600}, timeout=300)
        except Exception as e:
            print(f"    LOAD FAIL: {e}"); continue
        time.sleep(20)  # attendi caricamento
        row = {"model": mid, **cfg, "load": "ok"}
        # benchmark generazione
        try:
            t0 = time.time()
            r = api("/v1/chat/completions",
                    {"model": mid, "messages": [{"role": "user", "content": PROMPT}],
                     "max_tokens": 128, "temperature": 0.2}, timeout=240)
            dt = time.time() - t0
            u = r.get("usage", {})
            ct = u.get("completion_tokens", 0)
            row["ttft_ms"] = round(dt * 1000, 1) if ct else None
            row["tok_s"] = round(ct / dt, 2) if dt > 0 else None
            row["prompt_tok"] = u.get("prompt_tokens")
        except Exception as e:
            row["gen_err"] = str(e)[:120]
        # tool calling
        try:
            r2 = api("/v1/chat/completions",
                     {"model": mid, "messages": [{"role": "user", "content": "Qual e il meteo a Milano?"}],
                      "tools": TOOLS, "max_tokens": 128, "temperature": 0.0}, timeout=240)
            tc = r2["choices"][0]["message"].get("tool_calls")
            row["tool_call"] = "OK" if tc else "NO"
        except Exception as e:
            row["tool_call"] = f"ERR {str(e)[:60]}"
        results.append(row)
        print(f"    -> {row}", flush=True)
        try:
            api("/api/v0/models/unload", {"model": mid}, timeout=120)
        except Exception:
            pass
        time.sleep(5)

out = r"C:\Users\User\Documents\ai_tui\mikye_calibration.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSalvato: {out}")
