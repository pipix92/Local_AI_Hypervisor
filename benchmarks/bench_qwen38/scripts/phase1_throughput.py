"""FASE 1 - throughput: same prompt across KV dtypes. Reports TTFT, tok/s, and KV MiB from server log."""
import json, os, re, signal, subprocess, sys, time, urllib.request, urllib.error

RUNTIME = r"C:/Users/User/AppData/Local/hermes/runtimes/llamacpp/b10964/cuda"
MODEL   = r"D:\models\mradermacher\Qwen3.8-9B-Distill-uncensored-heretic-i1-GGUF\Qwen3.8-9B-Distill-uncensored-heretic.i1-IQ4_XS.gguf"
OUT     = r"C:/Users/User/Documents/ai_tui/bench_qwen38"
PORT    = 18434
LIMIT_MIB = 7618.6

# Identical prompt for every dtype run (fixed text, fixed length, deterministic).
PROMPT = (
    "Write a short Python function that checks whether a given string is a valid IPv4 address. "
    "It must handle exactly four dot-separated decimal octets in range 0-255 with no leading zeros, "
    "and reject anything else. Show the code first, then two lines explaining the edge cases.\n\n"
) * 12  # repeat to get a prompt long enough to make prefill measurable

MAX_TOK = 256


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def wait_health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200 and b"Loading" not in r.read():
                    return time.time() - t0
        except urllib.error.HTTPError as e:
            pass
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early rc={proc.returncode}\n{logtail()}")
        time.sleep(1.0)
    raise TimeoutError("server not healthy in time\n" + logtail())


def logtail(n=60):
    try:
        with open(logpath, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return "<no log>"


def kill_server():
    global proc
    if proc and proc.poll() is None:
        proc.send_signal(signal.CTRL_BREAK_EVENT) if False else proc.terminate()
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=20)
    for _ in range(60):
        u = gpu_used_mib()
        if u < 300:
            return u
        time.sleep(1)
    return gpu_used_mib()


def parse_log(logfile):
    info = {}
    pats = {
        "kv_self_size":   r"kv self size\s*:\s*([\d.]+)\s*MiB",
        "kv_unified":     r"KV self size\s*:\s*([\d.]+)\s*MiB",
        "compute_buffer": r"compute buffer size\s*:\s*([\d.]+)\s*MiB",
        "buffer_size":    r"buffer size\s*:\s*([\d.]+)\s*MiB",
        "model_size":     r"model size\s*:\s*([\d.]+)\s*MiB",
        "ctx_shift":      r"n_ctx\s*:\s*(\d+)",
        "n_gpu_layers":   r"n_gpu_layers\s*:\s*(\d+)",
        "kv_type":        r"kv_cache_type\s*:\s*(\S+)",
        "flash_attn":     r"flash_attn\s*:\s*(\S+)",
    }
    txt = open(logfile, "r", encoding="utf-8", errors="replace").read()
    for k, p in pats.items():
        m = re.findall(p, txt)
        if m:
            info[k] = m[-1] if isinstance(m[-1], str) else m[-1]
    info["_raw_kv_lines"] = [l.strip() for l in txt.splitlines()
                            if re.search(r"kv|KV|buffer size|model size|n_ctx|n_gpu_layers", l)][-40:]
    return info


def run_once(ctx, ctk, ctv):
    """Start server, measure throughput on fixed prompt, kill, return record."""
    global proc, logpath
    logpath = os.path.join(OUT, "logs", f"server_c{ctx}_{ctk}.log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    cmd = [f"{RUNTIME}/llama-server.exe", "-m", MODEL, "-c", str(ctx), "-ngl", "99",
           "-fa", "on", "-ctk", ctk, "-ctv", ctv, "-t", "8",
           "--port", str(PORT), "--no-webui", "-np", "1"]
    lf = open(logpath, "w", encoding="utf-8", errors="replace", buffering=1)
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=RUNTIME)
    rec = {"ctx": ctx, "ctk": ctk, "ctv": ctv, "cmd": " ".join(cmd[1:])}
    try:
        ready = wait_health()
        rec["load_seconds"] = round(ready, 1)
        rec["gpu_used_after_load_mib"] = gpu_used_mib()
        rec["log"] = parse_log(logpath)

        # throughput: streaming request on the fixed prompt
        body = json.dumps({"prompt": PROMPT, "max_tokens": MAX_TOK,
                           "temperature": 0.0, "stream": True, "cache_prompt": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        t0 = time.time(); ttft = None; n_tok = 0
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                ch = d.get("choices", [{}])[0]
                txt = ch.get("text", "")
                if txt:
                    if ttft is None:
                        ttft = time.time() - t0
                    n_tok += 1
        total = time.time() - t0
        rec["ttft_s"] = round(ttft, 3) if ttft else None
        rec["gen_seconds"] = round(total - (ttft or 0), 3)
        rec["gen_tokens_streamed"] = n_tok
        rec["decode_tok_per_s"] = round(n_tok / (total - (ttft or 0)), 3) if n_tok and total > ttft else None
        rec["prompt_chars"] = len(PROMPT)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/tokenize", timeout=60) as r:
                pass
        except Exception:
            pass
        rec["peak_gpu_mib"] = gpu_used_mib()
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = repr(e)
        rec["log_tail"] = logtail(80)
    finally:
        rec["gpu_after_kill_mib"] = kill_server()
        rec["log_file"] = logpath
    return rec


if __name__ == "__main__":
    print(f"VRAM pre-test: {gpu_used_mib()} MiB")
    results = []
    # f16 must use -fa off (CUDA FA requires f16/q8_0 KV); test all three per spec.
    for ctk, ctv, fa in (("q8_0", "q8_0", "on"), ("q4_0", "q4_0", "on"), ("f16", "f16", "off")):
        print(f"\n=== run dtype={ctk} fa={fa} ===", flush=True)
        r = run_once(65536, ctk, ctv)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=2)[:2500], flush=True)
        print(f"gpu after kill: {r['gpu_after_kill_mib']} MiB", flush=True)
        time.sleep(3)

    json.dump(results, open(os.path.join(OUT, "phase1_throughput.json"), "w"), indent=2)
    print("\nDONE -> phase1_throughput.json")
