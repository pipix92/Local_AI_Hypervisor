#!/usr/bin/env python3
"""Benchmark KV "overhead + griglia" in un colpo solo, per PC con 8 GB VRAM.

IL PROBLEMA CHE RISOLVE
-----------------------
La taratura dell'overhead (CUDA + desktop + runtime) e' il numero che decide
tutto: se sbagli di 500 MiB, salti meta' dei modelli per un errore di aritmetica
e non perche' non ci stiano. Quindi l'overhead viene MISURATO qui, non assunto.

FASE 1 - calibrazione dell'overhead
  Si avvia un modello piccolo a contesto 2048 e si legge la VRAM usata. La KV
  di quel contesto e' pochi MiB, quindi usati - pesi - KV = overhead. La KV e'
  calcolata con la geometria REALE letta dall'header GGUF (gguf_geom.py), non
  con una stima a mano: sull'architettura ibrida la stima sbagliava del 40%.

FASE 2 - griglia
  Per ogni modello e ogni tipo di KV, calcola la memoria necessaria e la
  confronta col budget REALE. Se non entra, SKIP: nessun processo avviato.
  Se entra, avvia UN server alla volta e misura.

Fase 3: scrive kv_overhead_grid.json e stampa la tabella finale.
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gguf_geom import geometry          # geometria REALE dall'header GGUF

# ------------------------------------------------------------------ parametri
_ap = argparse.ArgumentParser()
_ap.add_argument("--exe", default="")
_ap.add_argument("--models-root", default=r"D:\LLM models")
_ap.add_argument("--ref-model", default="", help="modello piccolo per calibrare l'overhead")
_ap.add_argument("--kv", default="f16,q8_0,q5_1,q4_1,q4_0")
_ap.add_argument("--ctx", default="8192,16384,32768,65536")
_ap.add_argument("--headroom", type=int, default=700, help="MiB liberi richiesti DOPO il caricamento")
_ap.add_argument("--port", type=int, default=18841)
_ap.add_argument("--out", default=os.path.join(HERE, "kv_overhead_grid.json"))
_ap.add_argument("--min-size-gb", type=float, default=2.0)
_ap.add_argument("--max-size-gb", type=float, default=16.0)
_A = _ap.parse_args()

KVS = [k.strip() for k in _A.kv.split(",") if k.strip()]
CTXS = [int(c) for c in _A.ctx.split(",") if c.strip()]
PORT = _A.port

# bytes-per-elemento -> quanti "slot" occupa nella KV (usato da gguf_geom)
SLOT = {"f16": 2, "f32": 2, "bf16": 2, "q8_0": 1,
        "q5_1": 0.5, "q5_0": 0.5, "q4_1": 0.5, "q4_0": 0.5}
# sopralluoco sui KV a 5 bit: lo slot reale e' 0.3125 ma si arrotonda a 0.5,
# quindi il calcolo resta prudenziale (sovrastima la memoria necessaria).
SLOT[".5_prudente"] = 0.5


def gpu_mib():
    """(usata, totale, libera) in MiB. nvidia-smi e' l'unica fonte affidabile."""
    try:
        r = subprocess.run(["nvidia-smi.exe",
                            "--query-gpu=memory.used,memory.total,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, t, f = [int(x) for x in r.stdout.strip().split(",")]
        return u, t, f
    except Exception:
        return -1, -1, -1


def kill_llama():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    for _ in range(60):
        time.sleep(2)
        if not port_open():
            return
    return


def port_open(p=None):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", p or PORT))
        return True
    except Exception:
        return False
    finally:
        s.close()


def start(exe, gguf, ctx, kv):
    kill_llama()
    args = [exe, "-m", gguf, "-c", str(ctx), "--port", str(PORT),
            "--host", "127.0.0.1", "-ngl", "99",
            "-ctk", kv, "-ctv", kv, "-fa", "on", "--no-warmup"]
    wd = os.path.dirname(exe)
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         cwd=wd if os.path.isdir(wd) else None)
    for _ in range(90):
        time.sleep(2)
        if port_open():
            return p
    try:
        p.kill()
    except Exception:
        pass
    return None


def ready(tries=150):
    """Il server risponde 503 per tutto il caricamento: /v1/models va 200 a fine."""
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def measure(reps=3, prompt_chars=16000, gen_tok=64):
    """tok/s dai timings del server (predicted_per_second). Nessuna sottrazione."""
    best = None
    body = {"model": "x",
            "messages": [{"role": "user", "content": "Rispondi con un numero. " * (prompt_chars // 28)}],
            "max_tokens": gen_tok, "temperature": 0.2, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(reps):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
        except Exception as e:
            print(f"      misura fallita: {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue
        tm = d.get("timings") or {}
        n, ms = tm.get("predicted_n"), tm.get("predicted_ms")
        tps = tm.get("predicted_per_second")
        if not n or not ms or n < 2:
            continue
        if tps is None:
            tps = n * 1000.0 / ms
        if best is None or tps > best[0]:
            best = (round(float(tps), 2), tm.get("prompt_n"), n)
    return best


# File che NON sono modelli standalone: draft per speculazione, proiezioni
# multimodali, encoder di token. Vanno esclusi per NOME e per dimensione:
# un MTP puo' pesare 240 MB ma un draft anche 1,8 GB, e nessuno dei due si
# avvia da solo con -m.
NOT_A_MODEL = ("mmproj", "draft", "mtp", "medias", "encoder", "vision",
               "mmadapter", "lora", "adapter")


def discover(root, min_gb, max_gb, ref="", quiet=False):
    """Modelli standalone, dalla piu' piccola alla piu' grande."""
    out, skipped = [], []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".gguf"):
                continue
            p = os.path.join(dp, fn)
            low = fn.lower()
            try:
                gb = os.path.getsize(p) / 1073741824.0
            except OSError:
                continue
            # prima per nome, poi per dimensione: sotto i 2 GB non e' un modello
            # che vale la pena avviare, di solito e' un MTP o un draft
            bad = next((k for k in NOT_A_MODEL if k in low), None)
            if bad:
                skipped.append((fn, gb, f"nome contiene '{bad}'"))
                continue
            # modelli sharded: se esiste la parte 2 manca un pezzo e il server
            # non parte. Se esiste la 1 senza la 2 (modello rinominato) prova
            # comunque, ma segnala.
            if re.search(r"-of-\d+", low):
                parts = sorted(x for x in os.listdir(dp)
                               if x.lower().startswith(fn.split("-of-")[0].lower())
                               and re.search(r"-of-\d+", x.lower()))
                if len(parts) < 2:
                    skipped.append((fn, gb, f"sharded: trovate {len(parts)} parti"))
                    continue
            if gb < min_gb:
                skipped.append((fn, gb, f"sotto {min_gb} GB"))
                continue
            if gb > max_gb:
                skipped.append((fn, gb, f"sopra {max_gb} GB"))
                continue
            out.append((p, gb))
    out.sort(key=lambda t: t[1])
    if ref:
        out.sort(key=lambda t: (0 if os.path.basename(t[0]) == ref else 1, t[1]))
    if not quiet:
        print(f"  {len(out)} modelli standalone, {len(skipped)} esclusi")
        for fn, gb, why in sorted(skipped, key=lambda s: s[1]):
            print(f"    escluso {gb:5.2f} GB  {fn[:58]:58} ({why})")
    return out


def calibrate(exe, root):
    """FASE 1: misura l'overhead reale su un modello piccolo a ctx 2048."""
    print("=" * 74)
    print("FASE 1 - calibrazione dell'overhead")
    print("=" * 74)
    cands = discover(root, 0.3, 3.0)
    if _A.ref_model:
        cands = [(p, g) for p, g in cands if os.path.basename(p) == _A.ref_model] or cands
    if not cands:
        print("NESSUN MODELLO PICCOLO TROVATO per calibrare: non posso misurare l'overhead.")
        return None
    path, gb = cands[0]
    g = geometry(path)
    if "error" in g:
        print(f"header incompleto su {os.path.basename(path)}: {g['error']}")
        return None
    print(f"modello di riferimento: {os.path.basename(path)} ({gb:.2f} GB)")
    print(f"  arch={g['arch']}  {g['n_attn_layers']}/{g['n_layer']} layer attn  "
          f"kv={g['n_head_kv']} head")
    _, tot, free0 = gpu_mib()
    print(f"  VRAM prima: {free0} MiB liberi su {tot}")
    p = start(exe, path, 2048, "f16")
    if not p or not ready():
        print("  il modello di riferimento non si e' avviato: calibrazione impossibile")
        return None
    used, tot, free1 = gpu_mib()
    kv_mib = g["mib_per_1k"][2] * 2048 / 1000.0
    wg = os.path.getsize(path) / 1048576.0
    overhead = used - wg - kv_mib
    print(f"  VRAM dopo:  {free1} MiB liberi  (usati {used})")
    print(f"  pesi {wg:.0f} + KV(f16,2048) {kv_mib:.0f} -> OVERHEAD = {overhead:.0f} MiB")
    try:
        p.kill()
    except Exception:
        pass
    kill_llama()
    return {"overhead_mib": round(overhead), "ref_model": os.path.basename(path),
            "weights_mib": round(wg), "kv_at_2048_mib": round(kv_mib, 1),
            "total_mib": tot}


def main():
    used, tot, free = gpu_mib()
    print(f"GPU: {used} MiB usati, {free} liberi su {tot}\n")
    if used > 1200:
        print(f"Attenzione: {used} MiB gia' occupati. Serve VRAM libera per la calibrazione.")
        kill_llama()

    cal = calibrate(_A.exe, _A.models_root)
    if not cal:
        return
    OH = cal["overhead_mib"]

    models = discover(_A.models_root, _A.min_size_gb, _A.max_size_gb)
    if not models:
        print("nessun modello trovato")
        return
    print(f"\n{len(models)} modelli, {len(KVS)} tipi di KV, contesti {CTXS}")
    print(f"overhead usato per tutti i calcoli: {OH} MiB")

    out = {"calibration": cal, "gpu": {"total_mib": tot}, "runs": [], "skipped": []}
    for mp, gb in models:
        mname = os.path.basename(mp)
        g = geometry(mp)
        if "error" in g:
            print(f"\n--- {mname}: header incompleto, lo salto")
            out["skipped"].append({"model": mname, "reason": g["error"]})
            continue
        wg = os.path.getsize(mp) / 1048576.0
        budget = tot - OH - wg
        print(f"\n{'='*74}")
        print(f"{mname}  ({gb:.2f} GB)  arch={g['arch']}  "
              f"{g['n_attn_layers']}/{g['n_layer']}L attn  kv={g['n_head_kv']}h")
        print(f"  KV f16 {g['mib_per_1k'][2]:.0f} MiB/1k | budget {budget:.0f} MiB "
              f"(utilizzabile {budget-_A.headroom:.0f})")
        if g.get("warn"):
            print(f"  attenzione: {g['warn']}")

        for kv in KVS:
            per_1k = g["mib_per_1k"][SLOT[kv]]
            for ctx in CTXS:
                need = per_1k * ctx / 1000.0
                # SKIP PRIMA DI PROVARE
                if need > budget - _A.headroom:
                    out["skipped"].append({"model": mname, "kv": kv, "ctx": ctx,
                                           "kv_mib": round(need),
                                           "budget_mib": round(budget)})
                    continue
                print(f"  misuro {kv:6} ctx={ctx:6} (KV {need:5.0f} MiB) ...", end="", flush=True)
                p = start(_A.exe, mp, ctx, kv)
                if not p:
                    print(" server non partito")
                    out["skipped"].append({"model": mname, "kv": kv, "ctx": ctx, "reason": "no start"})
                    continue
                if not ready():
                    print(" timeout in attesa del modello")
                    try:
                        p.kill()
                    except Exception:
                        pass
                    out["skipped"].append({"model": mname, "kv": kv, "ctx": ctx, "reason": "load timeout"})
                    continue
                _, _, free_after = gpu_mib()
                m = measure()
                if not m:
                    print(" nessuna risposta")
                    out["skipped"].append({"model": mname, "kv": kv, "ctx": ctx, "reason": "no response"})
                else:
                    tps, ptok, gtok = m
                    print(f" {tps:6.2f} tok/s  (liberi {free_after})")
                    out["runs"].append({"model": mname, "gb": round(gb, 2), "kv": kv, "ctx": ctx,
                                        "decode_tps": tps, "kv_mib": round(need),
                                        "budget_mib": round(budget), "free_after": free_after,
                                        "prompt_tok": ptok, "gen_tok": gtok})
                try:
                    p.kill()
                except Exception:
                    pass
                time.sleep(3)

    with open(_A.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 74)
    print("RISULTATI")
    print("=" * 74)
    if out["runs"]:
        print(f"{'modello':38} {'KV':6} {'ctx':>7} {'MiB':>7} {'tok/s':>8} {'liberi':>8}")
        for r in sorted(out["runs"], key=lambda r: (r["model"], r["kv"], r["ctx"])):
            print(f"{r['model'][:38]:38} {r['kv']:6} {r['ctx']:>7} "
                  f"{r['kv_mib']:>7} {r['decode_tps']:>8.2f} {r['free_after']:>8}")
    else:
        print("nessun test eseguito")
    print(f"\nJSON: {_A.out}")
    kill_llama()


if __name__ == "__main__":
    main()
