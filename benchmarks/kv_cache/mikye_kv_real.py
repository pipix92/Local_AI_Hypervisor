#!/usr/bin/env python
# Misura empirica della KV cache: stessa identica configurazione, cambia solo il context.
# La differenza di VRAM tra due contesti = byte per token della KV cache.
import json, os, subprocess, time, urllib.request, urllib.parse

HOST = "192.168.1.100:56397"
VAULT = r"~/.vault"
PASS = "79Npi5Ue#UnEBthDsc&HI1g*_mb@"
SSH_KEY = r"~/.ssh/id_rsa_cluster"
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


def api(path, payload=None, timeout=900, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://{HOST}{path}", data=data, headers=H,
                                 method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def vram_mb():
    """(total, used, free) in MiB, letti sul PC di Miky via SSH+nvidia-smi."""
    sh = 'export PATH="/mnt/c/Windows/System32:$PATH"\n/mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits\n'
    p = subprocess.run(["ssh", "-i", SSH_KEY, "-o", "IdentitiesOnly=yes", "-o",
                        "StrictHostKeyChecking=no", "mikye@192.168.1.100", "bash -s"],
                       input=sh, capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        nums = [x.strip() for x in line.split(",")]
        if len(nums) == 3 and nums[0].isdigit():
            return int(nums[0]), int(nums[1]), int(nums[2])
    return None


def unload_all():
    for m in api("/api/v0/models")["data"]:
        if m.get("state") in ("loaded", "loading"):
            try:
                api(f"/api/v0/models/{urllib.parse.quote(m['id'])}/unload", {})
            except Exception:
                pass
    time.sleep(8)


def load_ctx(ctx):
    api(f"/api/v0/models/{urllib.parse.quote(MODEL)}/load?wait=true",
        {"contextLength": ctx, "gpuLayers": "max", "flashAttention": True, "noKvOffload": True},
        timeout=1200)
    time.sleep(4)


if __name__ == "__main__":
    print("=" * 74)
    print("MISURA KV CACHE REALE - Gemma4-12B QAT Q4_K_M su RTX 5050 8GB")
    print("=" * 74)

    unload_all()
    idle = vram_mb()
    print(f"\n  BASELINE (nessun modello): total={idle[0]} used={idle[1]} free={idle[2]} MiB\n")

    rows = []
    for ctx in (2048, 8192, 16384, 32768):
        unload_all()
        load_ctx(ctx)
        v = vram_mb()
        model_mb = v[1] - idle[1]
        per_tok = model_mb * 1024 * 1024 / ctx if ctx else 0
        rows.append({"ctx": ctx, "used_mb": v[1], "free_mb": v[2],
                     "model_mb": model_mb, "bytes_per_tok": round(per_tok, 1),
                     "fits": v[2] > 200})
        print(f"  ctx={ctx:>6}  usati={v[1]:>5} MiB  liberi={v[2]:>5} MiB  "
              f"modello={model_mb:>5} MiB  KV={per_tok/1024:.1f} KiB/tok  "
              f"{'OK' if v[2] > 200 else 'TESA'}")

    print("\n  --- Stima GB di contesto per varie finestre ---")
    base = [r for r in rows if r["ctx"] == 2048][0]
    kv_per_tok = base["bytes_per_tok"]
    for ctx in (16384, 32768, 65536, 131072, 262144):
        mb = kv_per_tok * ctx / 1024 / 1024
        print(f"    {ctx:>7} tok = {mb:>8.0f} MiB KV  ({mb/1024:.2f} GB)")

    json.dump({"baseline": idle, "rows": rows, "kv_bytes_per_tok": kv_per_tok},
              open(r"C:\Users\User\Documents\ai_tui\mikye_kv_real.json", "w"), indent=2)
    print("\n  Salvato: mikye_kv_real.json")
