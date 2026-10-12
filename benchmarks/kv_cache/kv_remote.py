#!/usr/bin/env python
"""
DRIVER REMOTO - PC di Miky, RTX 5050 8GB, via API LM Studio (192.168.1.100:56397).
LM Studio non espone il flag -nkvo, quindi la modalita' 'ram' si ottiene con
noKvOffload=false (KV offload su host) e 'vram' con noKvOffload=true.
'full' non e' distinguibile via API: gpuLayers=max copre l'offload dei pesi.
"""
import json, os, subprocess, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_harness import CONTEXTS, gen_probe, nvidia_remote, groq_compress, estimate_tokens

HOST = "192.168.1.100:56397"
BASE = f"http://{HOST}"
VAULT = r"~/.vault"
PASS = "79Npi5Ue#UnEBthDsc&HI1g*_mb@"
SSH_KEY = r"~/.ssh/id_rsa_cluster"
SSH_HOST, SSH_USER = "192.168.1.100", "mikye"


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


def api(path, payload=None, timeout=1200):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=H,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def api_safe(path, payload=None, timeout=1200, default=None):
    """api() che non alza: dopo un load fallito LM Studio può rispondere 500
    per qualche secondo, e serve poter comunque scaricare e proseguire."""
    try:
        return api(path, payload, timeout)
    except Exception:
        time.sleep(5)
        try:
            return api(path, payload, timeout)
        except Exception:
            return default


RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_remote_results.json")


def flush():
    """Salvataggio incrementale: un crash a ctx=65k non deve far perdere le ore
    di misura precedenti."""
    json.dump({"results": results, "groq": comp_info}, open(RESULTS_PATH, "w"), indent=2)


import urllib.request

MODELS = [
    "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m",
    "gemma-4-12b-coder-fable5-composer2.5-v1",
]
MODES = {"vram": True, "ram": False}   # True = noKvOffload (KV in VRAM)

results = []
comp_info = {}


def unload_all():
    d = api_safe("/api/v0/models", default={"data": []})
    for m in d.get("data", []):
        if m.get("state") in ("loaded", "loading"):
            api_safe(f"/api/v0/models/{urllib.parse.quote(m['id'])}/unload", {})
    time.sleep(8)


def vram_free():
    return nvidia_remote(SSH_KEY, SSH_HOST, SSH_USER)


if __name__ == "__main__":
    print("=" * 78)
    print("DRIVER REMOTO - RTX 5050 (Miky) | modelli: " + ", ".join(MODELS))
    print("=" * 78)

    def ensure_loaded(model, ctx, no_kv_offload):
        """LM Studio su Miky non espone piu' /api/v0/models/<id>/load: alla prima
        richiesta di chat carica il modello da solo. Si pre-carica con una
        richiesta minima e si verifica lo stato prima di misurare."""
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "temperature": 0.1}
        t0 = time.time()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=H, method="POST")
        with urllib.request.urlopen(req, timeout=1800) as r:
            r.read()
        # carica e aspetta che si assesti
        for _ in range(90):
            st = api_safe("/api/v0/models", default={})
            state = next((m.get("state") for m in st.get("data", []) if m.get("id") == model), "?")
            if state == "loaded":
                return True, round(time.time() - t0, 1)
            time.sleep(2)
        return False, round(time.time() - t0, 1)

    for model in MODELS:
        for mode, no_kv_offload in MODES.items():
            for ctx in CONTEXTS:
                label = f"{model.split('@')[0][:34]} | kv={mode}"
                try:
                    unload_all()
                    ok, load_s = ensure_loaded(model, ctx, no_kv_offload)
                    if not ok:
                        results.append({"label": label, "ctx": ctx, "model": model, "mode": mode,
                                        "load_state": "timeout", "load_s": load_s,
                                        "note": "caricamento non completato"})
                        flush()
                        print(f"  {label:<44} ctx={ctx:>6} LOAD TIMEOUT ({load_s}s)", flush=True)
                        continue
                    v = vram_free()
                    filler = "Analizza il sistema di telemetria e produci un report. " * 6000
                    prompt = filler + "\nRispondi: OK" if ctx >= 16384 else "Rispondi solo: OK"
                    r = gen_probe(BASE, model, prompt, timeout=900)
                    row = {"label": label, "ctx": ctx, "model": model, "mode": mode,
                           "noKvOffload": no_kv_offload, "load_s": load_s,
                           "vram_free": v["free"] if v else None, **r}
                    results.append(row)
                    flush()
                    print(f"  {label:<44} ctx={ctx:>6} tok/s={r['tok_s']:>6} "
                          f"ttft={r['ttft_ms']:>8} vram_free={row['vram_free']}", flush=True)
                except Exception as e:
                    results.append({"label": label, "ctx": ctx, "err": str(e)[:100]})
                    flush()
                    print(f"  {label:<44} ctx={ctx:>6} ERR {str(e)[:70]}", flush=True)

    unload_all()

    print("\n" + "=" * 78)
    print("COMPRESSIONE GROQ (stessa suite, eseguita qui per il confronto)")
    print("=" * 78)
    ctx_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai_tui", "tui_app.py")
    real_ctx = open(ctx_file, encoding="utf-8", errors="ignore").read(60000)
    s, meta = groq_compress(real_ctx)
    if s:
        info = {"orig_tok_est": estimate_tokens(real_ctx), "comp_tok_est": estimate_tokens(s),
                "ratio": round(estimate_tokens(s) / max(1, estimate_tokens(real_ctx)), 3),
                "latency_s": meta}
        print(f"  {info['orig_tok_est']} tok -> {info['comp_tok_est']} tok  "
              f"(ratio {info['ratio']})  in {meta}s")
    else:
        info = {"err": meta}
        print(f"  fallita: {meta}")

    json.dump({"results": results, "groq": info}, open(RESULTS_PATH, "w"), indent=2)
    print("\nSalvato: kv_remote_results.json")
