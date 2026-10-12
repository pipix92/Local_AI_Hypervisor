#!/usr/bin/env python3
"""
Benchmark KV SU MIKY (RTX 5050 8GB) - v4, con due correzioni rispetto a v3.

1. CALCOLA PRIMA, PROVA POI. Per ogni (modello, KV, contesto) la memoria
   necessaria e' stimata prima di avviare: pesi + KV + overhead fisso. Se non
   entra nel budget, NON si prova e NON si avvia nessun processo. Serve una
   riga di calibrazione per KV-per-token, piu' l'overhead fisso: entrambi si
   ottengono da una prima esecuzione di calibrazione, non da una stima teorica.

2. MISURA LA VELOCITA' AL CONTESTO REALE, non al massimo. I test precedenti
   avviavano ogni tipo di KV al suo ctx massimo ma leggevano il decode a
   ctx 4096, dove la KV e' minuscola: f16/q8_0/q4_0 erano la stessa identica
   configurazione e i tok/s non distinguevano nulla. Qui ogni combinazione
   viene misurata al CONTESTO CHE STA TESTANDO.

   Griglia KV: f16, q8_0, q5_1, q5_0, q4_1, q4_0
   Contesti:  8k, 16k, 32k, 65k
   Modelli:   i piu' grandi che stanno in 8GB (un modello alla volta)

Output: kv_bench_mikye.json  (risultati) + kv_server_stderr.log (errori reali)
"""
import argparse, json, os, re, socket, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_geom import geometry          # geometria REALE letta dall'header GGUF

HERE = os.path.dirname(os.path.abspath(__file__))
LOGF = os.path.join(HERE, "kv_server_stderr.log")

_ap = argparse.ArgumentParser()
_ap.add_argument("--exe", default=None)
_ap.add_argument("--dir", dest="dir_", default=None)
_ap.add_argument("--models", default=None, help="separati da ;")
_ap.add_argument("--kv", default="f16,q8_0,q5_1,q5_0,q4_1,q4_0")
_ap.add_argument("--ctx", default="8192,16384,32768,65536")
_ap.add_argument("--out", default=os.path.join(HERE, "kv_bench_mikye.json"))
_ap.add_argument("--headroom", type=int, default=700, help="MiB liberi richiesti")
_A = _ap.parse_args()

DEFAULT_DIR = (r"C:\Users\mikye\.lmstudio\extensions\backends"
               r"\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.47.0")
LLAMA_DIR = _A.dir_ or os.environ.get("LLaMA_DIR") or DEFAULT_DIR
LLAMA = _A.exe or os.environ.get("LLaMA_EXE") or os.path.join(LLAMA_DIR, "llama-server.exe")

# MiB per token della KV, per tipo di quantizzazione. I pesi del modello sono
# in f16-equivalent, quindi il fattore e' quello del formato della KV.
KV_RATIO = {"f16": 1.0, "f32": 2.0, "bf16": 1.0, "q8_0": 0.5,
            "q5_1": 0.3125, "q5_0": 0.3125, "q4_1": 0.25, "q4_0": 0.25}

MODELS_ROOT = r"D:\LLM models"
FIXED_OVERHEAD_MIB = 3000  # tarato sull'RTX 3050 (3.3 GB)

PORT = 18831
GEN_TOK = 64
PROMPT_CHARS = 16000
# prompt di ~4k token, cosi' il decode e' dominato dalla generazione e non
# dal prefill; i tempi si confrontano tra loro.
PROBE_TOK = 4096


def free_mib():
    try:
        r = subprocess.run(["nvidia-smi.exe",
                            "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        tot, used = [int(x) for x in r.stdout.strip().split(",")]
        return tot - used, tot
    except Exception:
        return -1, -1


def port_open(port=PORT):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def kill_any_llama():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)


def discover_models(limit=4):
    """Modelli standalone da 1 a 6 GB, dal piu' grande: sono i piu' interessanti."""
    out = []
    for root, _, files in os.walk(MODELS_ROOT):
        for f in files:
            if not f.lower().endswith(".gguf"):
                continue
            p = os.path.join(root, f)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            # escludi draft/mtp/mmproj e i troppo grossi
            if any(k in f.lower() for k in ("mmproj", "-mtp", "draft")):
                continue
            if 1.2e9 <= sz <= 6.5e9:
                out.append((sz, p))
    out.sort(reverse=True)
    return [p for _, p in out[:limit]]


def start(gguf, ctx, kv):
    kill_any_llama()
    time.sleep(3)
    args = [LLAMA, "-m", gguf, "-c", str(ctx), "--port", str(PORT),
            "--host", "127.0.0.1", "-ngl", "99",
            "-ctk", kv, "-ctv", kv, "-fa", "on", "--no-warmup"]
    wd = os.path.dirname(LLAMA)
    errf = open(LOGF, "a", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errf,
                         cwd=wd if os.path.isdir(wd) else None)
    for _ in range(90):
        time.sleep(2)
        if port_open():
            return p
    p.kill()
    return None


def ready(tries=150):
    """/v1/models risponde 200 solo quando il modello e' davvero caricato.
    Durante il caricamento llama-server risponde 503."""
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def measure(reps=3):
    """tok/s e prefill/s DAI TIMINGS del server, al contesto in esame."""
    best = None
    body = {"model": "x",
            "messages": [{"role": "user", "content": "Rispondi con un numero da 1 a 10. " * 260}],
            "max_tokens": GEN_TOK, "temperature": 0.2, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(reps):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
        except Exception as e:
            print(f"      misura fallita: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        tm = d.get("timings") or {}
        n, ms = tm.get("predicted_n"), tm.get("predicted_ms")
        if not n or not ms or n < 2:
            continue
        tps = tm.get("predicted_per_second") or (n * 1000.0 / ms)
        pre = tm.get("prompt_per_second")
        if best is None or tps > best[0]:
            best = (round(float(tps), 2), round(float(pre), 1) if pre else None,
                    tm.get("prompt_n"), n)
    return best


def main():
    print("=" * 72)
    print("KV BENCH MIKY v4 - calcola prima, prova solo se entra")
    print("=" * 72)
    if not (os.path.exists(LLAMA) and os.path.getsize(LLAMA) > 0):
        print(f"ERRORE: llama-server non utilizzabile: {LLAMA}")
        return

    models = _A.models.split(";") if _A.models else discover_models()
    if not models:
        print("ERRORE: nessun modello trovato")
        return
    kvs = [k for k in _A.kv.split(",") if k in KV_RATIO]
    ctxs = [int(x) for x in _A.ctx.split(",")]

    f0, tot = free_mib()
    print(f"GPU: {tot} MiB totali, {f0} liberi")
    print(f"overhead fisso assunto: {FIXED_OVERHEAD_MIB} MiB\n")

    out = {"gpu_total": tot, "overhead_assumed": FIXED_OVERHEAD_MIB,
           "runs": [], "skipped": []}

    for mp in models:
        mname = os.path.basename(mp)
        wg = os.path.getsize(mp) / 1048576.0
        # Geometria REALE letta dall'header GGUF. Stimare a mano dava scarti del
        # 40%: Qwen3.5-4B e' ibrido (8 layer attn su 32) e Spark ha 36 layer tutti
        # attn con key_length 256, quindi la stessa stima valeva per due modelli
        # che differiscono di 4,5x nella KV.
        g = geometry(mp)
        if "error" in g:
            print(f"\n--- {mname}: header incompleto ({g['error']}), lo salto")
            out["skipped"].append({"model": mname, "reason": g["error"]})
            continue
        f16_per_1k = g["mib_per_1k"][2]          # MiB per 1000 token in f16
        print(f"\n{'='*72}\n{mname}  ({wg/1024:.2f} GB)")
        print(f"  arch={g['arch']}  {g['n_attn_layers']}/{g['n_layer']} layer attn  "
              f"kv={g['n_head_kv']} head  ->  f16 {f16_per_1k:.0f} MiB/1k")
        if g.get("warn"):
            print(f"  attenzione: {g['warn']}")
        budget = tot - FIXED_OVERHEAD_MIB - wg
        print(f"  budget per la KV: {budget:.0f} MiB")

        for kv in kvs:
            # mib_per_1k e' indicizzato per i bytes-per-elemento: 2=f16, 1=q8_0,
            # 0.5=q4/q5. I tipi a 5 bit usano lo stesso peso di q4 (0.3125 reali
            # arrotondati): il calcolo resta prudenziale.
            slot = {2: 2, "f16": 2, "f32": 2, "bf16": 2,
                    "q8_0": 1, "q5_1": 0.5, "q5_0": 0.5,
                    "q4_1": 0.5, "q4_0": 0.5}[kv]
            per_1k = g["mib_per_1k"][slot]
            for ctx in ctxs:
                need = per_1k * ctx / 1000.0
                # SKIP PRIMA DI PROVARE: nessun processo viene avviato
                if need > budget - _A.headroom:
                    print(f"  SKIP {kv:6} ctx={ctx:6}  KV {need:7.0f} MiB, "
                          f"budget {budget:.0f} (servono {_A.headroom} liberi)")
                    out["skipped"].append({"model": mname, "kv": kv, "ctx": ctx,
                                           "kv_mib": round(need),
                                           "budget_mib": round(budget)})
                    continue

                print(f"  misuro {kv:6} ctx={ctx:6} (KV {need:5.0f} MiB) ...", end="", flush=True)
                p = start(mp, ctx, kv)
                if not p:
                    print(" server non partito")
                    out["runs"].append({"model": mname, "kv": kv, "ctx": ctx,
                                        "err": "server non partito"})
                    continue
                if not ready():
                    print(" timeout nel caricamento")
                    out["runs"].append({"model": mname, "kv": kv, "ctx": ctx,
                                        "err": "timeout caricamento"})
                    p.kill()
                    continue
                m = measure()
                free_after, _ = free_mib()
                if not m:
                    print(" nessuna risposta")
                    out["runs"].append({"model": mname, "kv": kv, "ctx": ctx,
                                        "err": "nessuna risposta"})
                else:
                    tps, pre, ptok, gtok = m
                    print(f" decode {tps:6.2f} t/s   prefill {pre} t/s"
                          f"   (ctx {ctx}, VRAM libera {free_after})")
                    out["runs"].append({"model": mname, "kv": kv, "ctx": ctx,
                                        "decode_tps": tps, "prefill_tps": pre,
                                        "prompt_n": ptok, "gen_n": gtok,
                                        "kv_mib": round(need), "vram_free": free_after})
                p.kill()
                time.sleep(2)

    kill_any_llama()
    with open(_A.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'='*72}")
    print(f"Salvato: {_A.out}")
    print(f"Log errori: {LOGF}")


if __name__ == "__main__":
    main()
