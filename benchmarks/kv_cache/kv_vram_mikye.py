#!/usr/bin/env python3
"""
Benchmark KV in VRAM - gira su Miky (RTX 5050 8GB) via WSL.

Perche' questo script e non il TUI: il llama-server di Miky e' uno STUB da
20KB (la logica sta in llama-server-impl.dll) e non parte da CLI. Il runtime
Hermes b10964 e' un binario funzionante e si puo' copiare.

METODO: calcola il contesto PRIMA di testarlo, gira UN SOLO processo alla
volta, svuota la VRAM fra un test e l'altro, e misura dai `timings` del
server (predicted_per_second) senza sottrazioni.
"""
import argparse, json, os, re, shutil, socket, subprocess, sys, time, urllib.request

# Percorsi sovrascrivibili da CLI: l'eseguibile di LM Studio e' solo un loader da
# 20 KB (la logica sta in llama-server-impl.dll), quindi filtrare per dimensione
# e' sbagliato. Il runner PowerShell passa il backend che ha verificato.
DEFAULT_DIR = (r"C:\Users\mikye\.lmstudio\extensions\backends"
               r"\llama.cpp-win-x86_64-nvidia-cuda-avx2-2.47.0")
_ap = argparse.ArgumentParser()
_ap.add_argument("--exe", default=None)
_ap.add_argument("--dir", dest="dir_", default=None)
_ap.add_argument("--model", default=r"D:\LLM models\XHToken\Spark-X2.5-4B-GGUF\Spark-X2.5-4B-Q4_K_M.gguf")
_ap.add_argument("--out", default=None)
_A = _ap.parse_args()

LLAMA_DIR = _A.dir_ or os.environ.get("LLaMA_DIR") or DEFAULT_DIR
LLAMA = _A.exe or os.environ.get("LLaMA_EXE") or os.path.join(LLAMA_DIR, "llama-server.exe")
MODEL = _A.model
PORT = 18821
SPEED_CTX = 4096          # contesto per la misura di velocita'
PROMPT_TOK_CHARS = 16000  # ~4k token di prompt
CTX_TO_TEST = [8192, 16384, 32768, 65536]
KV_TYPES = ["f16", "q8_0", "q4_0"]
HEADROOM_MIB = 800        # VRAM libera minima richiesta per partire
OUT = _A.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_vram_mikye.json")
LOGF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_server_stderr.log")


def free_mib():
    """Legge la VRAM libera. nvidia-smi e' l'unica fonte affidabile:
    LM Studio non espone /api/v0/system-info."""
    try:
        r = subprocess.run(["nvidia-smi.exe",
                            "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        tot, used = [int(x) for x in r.stdout.strip().split(",")]
        return tot - used, tot
    except Exception as e:
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


def start(llama, gguf, ctx, kv):
    kill_any_llama()
    time.sleep(4)
    args = [llama, "-m", gguf, "-c", str(ctx), "--port", str(PORT),
            "--host", "127.0.0.1", "-ngl", "99",
            "-ctk", kv, "-ctv", kv, "-fa", "on", "--no-warmup"]
    # cwd OBBLIGATORIA sulla cartella del backend: i llama-server di LM Studio
    # caricano le DLL wrapper (llm_engine, lmstudiocore, ggml_llamacpp) per
    # nome relativo. Senza cwd il processo muore in silenzio.
    wd = os.path.dirname(llama)
    # NON deviare stderr: quattro esecuzioni di fila si sono fermate a 30s con
    # "nessuna risposta" e senza alcun errore visibile, perche' sia qui che in
    # measure() le eccezioni erano silenziate. Ora l'errore va nel log.
    errf = open(LOGF, "a", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errf,
                         cwd=wd if os.path.isdir(wd) else None)
    for _ in range(90):
        time.sleep(2)
        if port_open():
            return p
    p.kill()
    return None


def ready(port=PORT, tries=150):
    """Aspetta che il server abbia FINITO di caricare il modello.

    Non basta che la porta sia aperta: llama-server la apre subito e risponde
    503 "loading model" per tutti i secondi in cui legge i 2,4 GB da disco. Il
    controllo giusto e' GET /v1/models, che risponde 200 solo quando il modello
    e' pronto. Senza questo, 503 -> "nessuna risposta" su ogni combinazione.
    """
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def measure(reps=3):
    """tok/s dai timings del server. Nessuna sottrazione: il server espone
    predicted_n / predicted_ms / predicted_per_second."""
    best = None
    body = {"model": "x",
            "messages": [{"role": "user", "content": "Rispondi con un numero da 1 a 10. " * 300}],
            "max_tokens": 64, "temperature": 0.2, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(reps):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
        except Exception as e:
            # NON silenziare: riporta il motivo, altrimenti "nessuna risposta"
            # resta incomprensibile. Prima questo except faceva continue e
            # quattro tentativi di fila si sono fermati qui senza diagnosi.
            print(f"      misura fallita: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        tm = d.get("timings") or {}
        n, ms, tps = tm.get("predicted_n"), tm.get("predicted_ms"), tm.get("predicted_per_second")
        if not n or not ms or n < 2:
            continue
        if tps is None:
            tps = n * 1000.0 / ms
        # best e' una TUPLA (tps, prompt_n, n): confrontare tps > best
        # confrontava un float con una tupla -> TypeError a ogni misura.
        if best is None or tps > best[0]:
            best = (round(float(tps), 2), tm.get("prompt_n"), n)
    return best


def main():
    # MODEL arriva gia' risolto da argparse (default: Spark-X2.5-4B su D:\LLM models).
    # Il vecchio os.walk("/mnt/d/LLM models") falliva perche' lo script gira sotto
    # Windows, dove quel path Unix non esiste: dava sempre "modello non trovato".
    gguf = MODEL
    if not gguf or not os.path.exists(gguf):
        # fallback: qualunque GGUF da 1 a 5 GB su D:\LLM models
        alt = r"D:\LLM models"
        for root, _, files in os.walk(alt):
            for f in files:
                p = os.path.join(root, f)
                if f.lower().endswith(".gguf"):
                    try:
                        sz = os.path.getsize(p)
                    except OSError:
                        continue
                    if 1e9 < sz < 5e9:
                        gguf = p
                        break
            if gguf:
                break
    if not gguf or not os.path.exists(gguf):
        print("ERRORE: modello non trovato")
        print(f"  cercato: {MODEL}")
        print(f"  esiste:  {os.path.exists(MODEL) if MODEL else 'n/d'}")
        return
    if not (os.path.exists(LLAMA) and os.path.getsize(LLAMA) > 0):
        print(f"ERRORE: llama-server non utilizzabile: {LLAMA}")
        return

    size_gb = os.path.getsize(gguf) / 1073741824
    print(f"modello : {os.path.basename(gguf)}  ({size_gb:.2f} GB)")
    print(f"llama   : {LLAMA}")
    f0, tot = free_mib()
    print(f"VRAM    : {f0} MiB liberi su {tot}\n")

    results = {"model": os.path.basename(gguf), "gb": round(size_gb, 2), "runs": []}

    for kv in KV_TYPES:
        for ctx in CTX_TO_TEST:
            f, _ = free_mib()
            if f < HEADROOM_MIB:
                print(f"  SKIP {kv} ctx={ctx}: VRAM occupata ({f} MiB liberi)")
                continue
            p = start(LLAMA, gguf, ctx, kv)
            if not p:
                print(f"  FAIL {kv} ctx={ctx}: server non partito")
                results["runs"].append({"kv": kv, "ctx": ctx, "err": "server non partito"})
                continue
            # Aspetta che il modello sia davvero pronto, non solo che la porta
            # sia aperta: durante il caricamento llama-server risponde 503.
            print(f"  carico {kv} ctx={ctx} ...", end="", flush=True)
            t_wait = time.time()
            if not ready():
                print(" timeout in attesa del modello")
                results["runs"].append({"kv": kv, "ctx": ctx, "err": "timeout caricamento"})
                p.kill()
                continue
            print(f" pronto in {int(time.time()-t_wait)}s", flush=True)
            free_after, _ = free_mib()
            m = measure()
            if not m:
                print(f"  FAIL {kv} ctx={ctx}: nessuna risposta")
                results["runs"].append({"kv": kv, "ctx": ctx, "err": "nessuna risposta"})
            else:
                tps, ptok, gtok = m
                print(f"  OK   {kv:5s} ctx={ctx:6d}  {tps:6.2f} tok/s  "
                      f"prompt={ptok}  VRAM libera dopo={free_after} MiB")
                results["runs"].append({"kv": kv, "ctx": ctx, "tok_s": tps,
                                        "prompt_tok": ptok, "free_mib": free_after})
            kill_any_llama()
            time.sleep(5)

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nSalvato: {OUT}")


if __name__ == "__main__":
    main()
