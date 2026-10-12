#!/usr/bin/env python3
"""Cheap acceptance check: does this build take <kv> for the KV cache at all?
Wrong/unsupported types are rejected at load, so a 4k context tells us in ~20s
per type instead of running the full grid and finding out 20 minutes later."""
import sys, time
sys.path.insert(0, r"C:\Users\User\Documents\ai_tui")
import kv_grid_bench as B

GG = r"D:/models/enfuse/smol-tools-4b-32k/smol-tools-4b-32k-q4_k_m.gguf"
geo = B.geometry(GG)
for kv in ("f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "q3_k"):
    B.kill_llama(); time.sleep(1)
    t0 = time.time()
    p = B.start_server(GG, 4096, kv)
    if p:
        r = B.measure(reps=1)
        tps = r[0] if r else -1
        print("%-8s ACCETTATO  %6.2f tok/s  (%.0fs)" % (kv, tps, time.time() - t0))
    else:
        print("%-8s RIFIUTATO  (%.0fs)" % (kv, time.time() - t0))
    B.kill_llama()
