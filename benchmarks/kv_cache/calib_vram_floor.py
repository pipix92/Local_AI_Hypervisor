#!/usr/bin/env python3
"""Calibrate the VRAM floor: what free-VRAM after load separates a resident
KV from a WDDM spill into system RAM. Prints tok/s + free MiB per ctx."""
import sys, time
sys.path.insert(0, r"C:\Users\User\Documents\ai_tui")
import kv_grid_bench as B

GG = r"D:/models/enfuse/smol-tools-4b-32k/smol-tools-4b-32k-q4_k_m.gguf"
geo = B.geometry(GG)
print("gpu_used_mib helper:", B.gpu_used_mib(), " total:", B.gpu_total_mib())
for ctx in (98304, 155648, 196608, 262144):
    B.kill_llama(); time.sleep(1)
    p = B.start_server(GG, ctx, "q8_0")
    if not p:
        print("%-8d START FAILED" % ctx); continue
    used = B.gpu_used_mib()
    r = B.measure(reps=1)
    if r is None:
        print("%-8d measure failed" % ctx)
    else:
        tps = r[0] if isinstance(r, tuple) else r["tok_s"]
        print("%-8d %7.2f tok/s   used %5d MiB  free %5d MiB"
              % (ctx, tps, used, (B.gpu_total_mib() - used)))
    B.kill_llama()
