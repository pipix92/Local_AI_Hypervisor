#!/usr/bin/env python
"""Trova l'endpoint corretto per caricare un modello con context specifico su LM Studio."""
import json, os, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

VAULT = r"~/.vault"
PASS = "79Npi5Ue#UnEBthDsc&HI1g*_mb@"
BASE = "http://192.168.1.100:56397"
M = "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m"


def token():
    env = dict(os.environ, LEX_VAULT_PASSPHRASE=PASS)
    for py in (r"C:\Python313\python.exe", "python"):
        try:
            r = subprocess.run([py, "vault.py", "get", "collega_lmstudio_api_key"],
                               cwd=VAULT, capture_output=True, text=True, env=env, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            continue
    raise SystemExit("no vault")


H = {"Authorization": f"Bearer {token()}", "Content-Type": "application/json"}


def call(path, payload=None, method=None, timeout=240):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=H,
                                 method=method or ("POST" if data else "GET"))
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "ignore")
            return r.status, body[:220], round(time.time() - t0, 1)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")[:220], round(time.time() - t0, 1)
    except Exception as e:
        return None, str(e)[:150], round(time.time() - t0, 1)


print("=" * 70)
print("1) versione / info server")
for p in ["/api/v0/models", "/v1/models"]:
    s, b, t = call(p)
    print(f"  {p:<24} {s}  {b[:110]}")

print()
print("2) tentativi endpoint load (ctx 16384)")
cfg = {"contextLength": 16384, "gpuLayers": "max", "flashAttention": True, "noKvOffload": True}
q = urllib.parse.quote(M, safe="")
tries = [
    (f"/api/v0/models/{q}/load?wait=true", cfg),
    ("/api/v0/models/load", {"model": M, **cfg}),
    (f"/api/v0/models/{q}/load", cfg),
]
for path, payload in tries:
    s, b, t = call(path, payload, timeout=600)
    ok = "OK" if (s == 200 and "Unexpected" not in b) else "no"
    print(f"  [{ok}] {s} {path[:56]:<56} {t}s  {b[:90]}")

print()
print("3) stato dopo i tentativi")
s, b, t = call("/api/v0/models")
if s == 200:
    d = json.loads(b) if b.startswith("{") else None
    for m in json.loads(urllib.request.urlopen(
            urllib.request.Request(BASE + "/api/v0/models", headers=H), timeout=60).read())["data"]:
        if m["id"] == M:
            print("  ", json.dumps(m, ensure_ascii=False)[:300])
