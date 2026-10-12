#!/usr/bin/env python3
"""
KV-type grid benchmark for local GGUFs on a single 8 GB card.

One model at a time, one process at a time, VRAM emptied between runs.

MEASURED BASELINE to reproduce (RTX 3050 8GB, smol-tools-4b-32k-q4_k_m.gguf):
  q8_0 = 155648 tok max @ 44.4 t/s | f16 = 98304 @ 47.1 | q4_0 = 262144 @ 39.7
  fixed overhead 3.306 MiB, hybrid SSM full_attention_interval=4, 8 KV layers of 32

HARD RULES (never violate):
  - NEVER -nkvo  : KV in system RAM is 8.2x slower (3.5 t/s)
  - NEVER q4_0 KV in RAM (0.96 t/s)
  - -ngl 99     : if a model does not fit, SKIP it, never offload KV
  - one process at a time, killed between runs

Per KV type two runs:
  A) fixed ctx (REF_CTX)  -> speed comparable across models AND KV types
  B) max feasible ctx     -> capacity, with bisection down on alloc failure
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_geom import geometry   # noqa: E402

EXE = r"C:\Users\User\AppData\Local\hermes\runtimes\llamacpp\b10964\cuda\llama-server.exe"
PORT = 18821
# f16 and q8_0/q4_0 were measured before; the rest are the mid-range parameter grid sweep.
KV_TYPES = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
BPE = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 1, "q5_1": 0.625, "q5_0": 0.5625,
       "q4_1": 0.5, "q4_0": 0.5, "iq4_nl": 0.5}
HEADROOM_MIB = 900        # refuse to start below this (desktop + CUDA ctx live here)
CTX_MULTIPLE = 256        # llama.cpp allocates ctx in blocks


# ---------------------------------------------------------------- gpu / server

def gpu_mib():
    try:
        r = subprocess.run(["nvidia-smi.exe", "--query-gpu=memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        tot, used = [int(x) for x in r.stdout.strip().split(",")]
        return tot - used, tot, used
    except Exception:
        return -1, -1, -1


def kill_llama():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    for _ in range(20):
        time.sleep(1)
        if not port_open():
            return True
    return not port_open()


def port_open():
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", PORT))
        return True
    except Exception:
        return False
    finally:
        s.close()


def api(path, body=None, timeout=30):
    url = f"http://127.0.0.1:{PORT}{path}"
    try:
        if body is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def start_server(gguf, ctx, kv):
    """Launch llama-server with the known-good flags. Returns proc or None.

    -np 1 is mandatory, not a tuning choice. The default n_slots=4 splits the
    context across 4 parallel slots, so the KV cache is sized for ctx/4 useful
    tokens and a quarter of the allocation is reserved for slots that never
    run: q8_0 maxes at 110336 that way instead of the 155648 the baseline
    measured with a single slot.
    """
    kill_llama()
    args = [EXE, "-m", gguf, "-c", str(ctx), "--port", str(PORT), "--host", "127.0.0.1",
            "-ngl", "99", "-np", "1", "-ctk", kv, "-ctv", kv, "-fa", "on", "--no-warmup"]
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "server_%s.log" % kv), "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                         cwd=os.path.dirname(EXE))
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(2)
        if p.poll() is not None:
            return None                      # died: bad KV type, OOM, or bad ctx
        if port_open():
            pr = api("/props", timeout=10)
            if pr:
                # /props answers while the weights are still being uploaded:
                # the first inference then returns no timings and every
                # measurement in the cell reads -1. Poll a real 1-token
                # generation until the model actually answers.
                if warm_ready():
                    return p
    p.kill()
    return None


def warm_ready(timeout=180):
    """True once the model can actually generate (not just serve /props)."""
    end = time.time() + timeout
    while time.time() < end:
        d = api("/v1/chat/completions",
                {"model": "x",
                 "messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": 1, "temperature": 0, "stream": False},
                timeout=60)
        if d and (d.get("timings") or {}).get("predicted_n"):
            return True
        time.sleep(2)
    return False


def server_report():
    """Real n_ctx and KV allocation the server came up with."""
    pr = api("/props", timeout=10)
    if not pr:
        return {}
    md = pr.get("default_generation_settings", {}) or {}
    return {"n_ctx": md.get("n_ctx"),
            "model": (pr.get("model_path") or "").split("\\")[-1]}


def gpu_total_mib():
    return gpu_mib()[1]


def gpu_used_mib():
    return gpu_mib()[2]


def resident(proc, min_headroom=400):
    """True only if the KV cache is really in VRAM.

    The llama.cpp allocator does NOT fail when the KV cache exceeds VRAM: on
    Windows/WDDM it maps the overflow into system RAM and the server still
    answers, just slowly. q8_0 at ctx=262144 on this card is 8192 MiB of KV on
    an 8192 MiB card, and it loaded "successfully" at 12.51 tok/s - a 3.9x
    collapse versus 48.61 tok/s resident. Free-VRAM-after-load is therefore the
    only trustworthy signal; a silent accept is not.
    """
    if proc is None or proc.poll() is not None:
        return False
    used = gpu_used_mib()
    if used < 0:
        return True                     # cannot measure: trust the allocator
    total = gpu_total_mib()
    if total > 0 and (total - used) < min_headroom:
        return False
    return True


def measure(reps=3, prompt_chars=16000):
    """tok/s straight from the server timings; no subtraction needed.

    The prompt carries a unique nonce per rep. Without it the server's prompt
    cache serves reps 2..N from VRAM: prompt_n drops to 4, prefill time goes to
    zero, and every rep after the first measures cache replay, not the model.
    """
    best = None
    for i in range(reps):
        nonce = "run%04d " % i
        body = {"model": "x",
                "messages": [{"role": "user",
                              "content": nonce + "Rispondi con un numero da 1 a 10. " * 300}],
                "max_tokens": 64, "temperature": 0.2, "stream": False,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        d = api("/v1/chat/completions", body, timeout=300)
        if not d:
            continue
        tm = d.get("timings") or {}
        n, ms = tm.get("predicted_n"), tm.get("predicted_ms")
        if not n or not ms or n < 2:
            continue
        tps = tm.get("predicted_per_second") or (n * 1000.0 / ms)
        if best is None or tps > best[0]:
            best = (round(float(tps), 2), tm.get("prompt_n"), n)
    return best


# ---------------------------------------------------------------- kv / ctx math

def kv_mib_per_1k(geo, kv):
    d = geo["key_length"] + (geo["value_length"] or geo["key_length"])
    return 2 * geo["n_attn_layers"] * geo["n_head_kv"] * d * BPE.get(kv, 1) / 1048576 * 1000


def max_ctx_estimate(geo, kv, free_mib):
    """Starting point for the ctx search, from measured arithmetic.

    Only a STARTING POINT. The reserve below is deliberately small: the
    baseline proved a q8_0 155648 context fits in 8192 MiB with weights at
    2924 MiB (KV 4864 + 3.306 fixed overhead), leaving ~400 MiB for the
    desktop. Probing upward from a too-conservative estimate loses capacity,
    so the search is allowed to climb past this and let the allocator refuse.
    """
    kv_rate = kv_mib_per_1k(geo, kv)          # MiB per 1000 tokens
    if kv_rate <= 0:
        return 0
    budget = free_mib - geo["weights_mib"] - 500
    return int(budget / kv_rate * 1000) // CTX_MULTIPLE * CTX_MULTIPLE


def probe_max_ctx(gguf, geo, kv, free_mib, cap_ctx, est):
    """Largest ctx whose KV cache is genuinely resident in VRAM.

    Geometric climb from est, then bisect the gap. The allocator is NOT the
    judge: on WDDM it accepts an oversized KV by mapping the overflow into
    system RAM and keeps serving (262144 on this card: "accepted", 10 tok/s).
    resident() is the judge, measured after the model is loaded and answering.
    """
    if est <= 0:
        return None, "nessun ctx stimabile"
    p = start_server(gguf, est, kv)
    if not p:
        # estimate itself does not fit: halve down to find the floor
        lo, cur = 0, est
        for _ in range(14):
            cur = (cur // 2 // CTX_MULTIPLE) * CTX_MULTIPLE
            if cur < 1024:
                return None, "KV non entra nemmeno a 1024"
            p = start_server(gguf, cur, kv)
            if resident(p):
                lo = cur
                kill_llama()
                break
        if lo == 0:
            return None, "KV non entra a nessun ctx provato"
    else:
        if not resident(p):
            kill_llama()
            return None, ("stima %d gia' in spill (VRAM non basta)" % est)
        kill_llama()
        lo = est
    # climb only while the context still FITS in VRAM, not merely while the
    # allocator accepts it
    hi_fail = cap_ctx
    cur = lo
    while cur < cap_ctx:
        nxt = min(cap_ctx, int(cur * 1.6) // CTX_MULTIPLE * CTX_MULTIPLE)
        if nxt <= cur:
            break
        p = start_server(gguf, nxt, kv)
        if not (p and resident(p)):
            hi_fail = nxt
            if p:
                kill_llama()
            break
        lo = nxt
        kill_llama()
        cur = nxt
    if lo >= cap_ctx:
        return cap_ctx, None
    # bisect the (fits, fails) gap
    while hi_fail - lo > CTX_MULTIPLE * 64:
        mid = (lo + hi_fail) // 2 // CTX_MULTIPLE * CTX_MULTIPLE
        if mid <= lo or mid >= hi_fail:
            break
        p = start_server(gguf, mid, kv)
        if p and resident(p):
            lo = mid
            kill_llama()
        else:
            if p:
                kill_llama()
            hi_fail = mid
    return lo, None


# ---------------------------------------------------------------- runs

def run_one(gguf, geo, kv, ctx, label):
    free_before, tot, used = gpu_mib()
    if free_before < HEADROOM_MIB:
        return {"kv": kv, "ctx": ctx, "run": label, "err": "VRAM occupata (%d MiB)" % free_before}
    p = start_server(gguf, ctx, kv)
    if not p:
        return {"kv": kv, "ctx": ctx, "run": label, "err": "server non partito (KV type o ctx rifiutato)"}
    rep = server_report()
    m = measure()
    free_after, _, _ = gpu_mib()
    kill_llama()
    if not m:
        return {"kv": kv, "ctx": ctx, "run": label, "err": "nessuna risposta",
                "n_ctx": rep.get("n_ctx")}
    tps, ptok, gtok = m
    rec = {"kv": kv, "ctx": ctx, "run": label, "tok_s": tps,
           "prompt_tok": ptok, "gen_tok": gtok, "n_ctx": rep.get("n_ctx"),
           "free_mib_before": free_before, "free_mib_after": free_after}
    rec["kv_alloc_mib"] = round(geo["n_attn_layers"] and
                                (ctx / 1000.0) * kv_mib_per_1k(geo, kv) or 0, 1)
    return rec


# A max-ctx cell is only valid if generation speed is close to the same KV type
# measured at the reference context. Below this ratio the KV is in system RAM
# and the number is meaningless.
SPILL_RATIO = 0.55


def measure_is_resident(rec, ref_tps):
    """True if rec's speed is consistent with a VRAM-resident KV.

    The allocator and nvidia-smi both lie about a WDDM spill: the server loads,
    reports a valid n_ctx, and can leave ~500 MiB free, yet the KV pages live
    in system RAM and generation collapses (f16 at 262144 on the reference
    model: 9.45 tok/s versus 49.26 at ctx 8192, a 5.2x drop, and 262144 in f16
    is 16384 MiB of KV on an 8192 MiB card - physically impossible).
    Speed is the only signal that reflects where the pages actually are, so
    residency is decided by comparison against the ref-context baseline.
    """
    if "err" in rec or not ref_tps:
        return False
    return rec.get("tok_s", 0) >= SPILL_RATIO * ref_tps


def bench_model(gguf, ref_ctx, cap_ctx, out, kvs):
    geo = geometry(gguf)
    if "error" in geo:
        print("  SKIP %s: %s" % (os.path.basename(gguf), geo["error"]))
        return
    size_gb = geo["weights_mib"] / 1024
    print("\n=== %s  (%.2f GB, arch=%s, %d/%d layer con KV%s) ==="
          % (geo["file"], size_gb, geo["arch"], geo["n_attn_layers"], geo["n_layer"],
             ", ibrido SSM" if geo["hybrid_ssm"] else ""))
    free, tot, used = gpu_mib()
    print("    pesi %d MiB | VRAM %d MiB liberi su %d | ctx_train %s"
          % (geo["weights_mib"], free, tot, geo.get("ctx_train")))
    if geo["weights_mib"] + HEADROOM_MIB > free:
        print("    SKIP: i pesi da soli non entrano con headroom (%d MiB necessari)"
              % (geo["weights_mib"] + HEADROOM_MIB))
        out["models"].append({"file": geo["file"], "skipped": "pesi non entrano",
                              "weights_mib": geo["weights_mib"], "free_mib": free})
        return
    runs = []
    for kv in kvs:
        est = max_ctx_estimate(geo, kv, free)
        print("  -- KV %-5s  stima max ctx %7d" % (kv, est))
        # A) speed at a fixed context -> comparable across models and KV types
        rc = min(ref_ctx, est) if est else ref_ctx
        ra = run_one(gguf, geo, kv, rc, "ref")
        if "err" in ra:
            print("     ref  ctx=%-6d  SKIP: %s" % (rc, ra["err"]))
        else:
            print("     ref  ctx=%-6d  %6.2f tok/s  (prompt %s, gen %s, n_ctx %s)"
                  % (rc, ra["tok_s"], ra["prompt_tok"], ra["gen_tok"], ra["n_ctx"]))
        runs.append(ra)
        # B) capacity: largest ctx that BOTH allocates AND generates at
        # ref-context speed. probe_max_ctx only knows about allocation.
        ref_tps = ra.get("tok_s") if "err" not in ra else None
        best, err = probe_max_ctx(gguf, geo, kv, free, cap_ctx, est)
        if ref_tps is None:
            print("     max  SKIP: nessun ref valido, non posso distinguere spill")
        else:
            rb = None
            while best and best > ref_ctx:
                rb = run_one(gguf, geo, kv, best, "max")
                if measure_is_resident(rb, ref_tps):
                    break
                print("     max  ctx=%-6d  %6.2f tok/s  -> SPILL (rifiutato)"
                      % (best, rb.get("tok_s", -1)))
                kill_llama()
                best = (int(best * 0.8) // CTX_MULTIPLE) * CTX_MULTIPLE
                rb = None
            if not best or best <= ref_ctx:
                print("     max  SKIP: %s" % (err or "nessun ctx sopra il ref regge"))
            else:
                if rb is None:
                    rb = run_one(gguf, geo, kv, best, "max")
                if "err" in rb:
                    print("     max  ctx=%-6d  SKIP: %s" % (best, rb["err"]))
                else:
                    print("     max  ctx=%-6d  %6.2f tok/s  (stima %d, VRAM dopo %d MiB)"
                          % (best, rb["tok_s"], est, rb["free_mib_after"]))
                runs.append(rb)
        # persist after every cell: a crash mid-grid must not lose the sweep
        out["models"][-1]["runs"] = runs
        out["models"][-1]["kv_mib_per_1k"] = {k: round(kv_mib_per_1k(geo, k), 2) for k in kvs}
        json.dump(out, open(out["_path"], "w"), indent=1)
    kill_llama()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--kv", nargs="+", default=KV_TYPES)
    ap.add_argument("--ref-ctx", type=int, default=8192)
    ap.add_argument("--cap-ctx", type=int, default=262144)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "kv_grid_results.json"))
    a = ap.parse_args()
    out = {"ref_ctx": a.ref_ctx, "kv_types": a.kv, "models": []}
    out["_path"] = a.out
    for m in a.models:
        if not os.path.exists(m):
            print("SKIP non esiste: %s" % m)
            continue
        geo = geometry(m)
        entry = {"file": geo.get("file", os.path.basename(m)),
                 "weights_mib": geo.get("weights_mib"), "arch": geo.get("arch"),
                 "n_layer": geo.get("n_layer"), "n_attn_layers": geo.get("n_attn_layers"),
                 "n_head_kv": geo.get("n_head_kv"), "key_length": geo.get("key_length"),
                 "hybrid_ssm": geo.get("hybrid_ssm"), "ctx_train": geo.get("ctx_train"),
                 "runs": []}
        out["models"].append(entry)
        try:
            bench_model(m, a.ref_ctx, a.cap_ctx, out, a.kv)
        except Exception as e:
            print("  ERRORE su %s: %r" % (os.path.basename(m), e))
            kill_llama()
    out.pop("_path", None)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nSalvato: %s" % a.out)


if __name__ == "__main__":
    main()
