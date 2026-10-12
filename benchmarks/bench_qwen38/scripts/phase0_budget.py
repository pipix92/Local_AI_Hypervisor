"""FASE 0 - calcolo preventivo budget VRAM per Qwen3.8-9B i1-IQ4_XS su RTX 3050 8GB."""
import json

# --- Geometria letta dal GGUF header [MISURATO] ---
BLOCK_COUNT        = 33
FULL_ATTN_INTERVAL = 4      # qwen35.full_attention_interval
N_HEAD_KV          = 4       # qwen35.attention.head_count_kv
KEY_LEN            = 256     # qwen35.attention.key_length
VAL_LEN            = 256     # qwen35.attention.value_length
NATIVE_CTX         = 262144  # qwen35.context_length

# layer full-attention: 0,4,8,...,32  -> floor((N-1)/interval)+1
N_FULL_ATTN = (BLOCK_COUNT - 1) // FULL_ATTN_INTERVAL + 1
N_LINEAR    = BLOCK_COUNT - N_FULL_ATTN

# ggml bytes-per-value per quant
BPV = {
    "f16":  2.0,
    "q8_0": (2 + 32) / 32,   # fp16 scale + 32 int8 = 34B/32? -> ggml: d(fp16)+32*int8 = 33B
    "q4_0": (2 + 16) / 32,   # fp16 scale + 16 packed nibbles = 18B per 32
}
BPV["q8_0"] = (2 + 32) / 32  # 1.0625 -> d(fp16)=2 + 32 int8 = 34 bytes per 32
# NOTE: ggml q8_0 block is exactly 34 bytes (fp16 d + 32 int8)? verify below.

def kv_bytes_per_token(dtype):
    """K + V per token, only over full-attention layers."""
    bpv = BPV[dtype]
    per_layer = 2 * N_HEAD_KV * (KEY_LEN + VAL_LEN) / 2 * bpv
    return per_layer * N_FULL_ATTN

# --- Budget VRAM [da specifica] ---
VRAM_TOTAL   = 8192.0
VRAM_LIMIT   = VRAM_TOTAL * 0.93          # 7618.56 MiB
WEIGHT_BYTES = 5326418560
WEIGHT_MIB   = WEIGHT_BYTES / (1024 ** 2)
COMPUTE_HI   = 700.0
COMPUTE_LO   = 400.0
SAFETY       = 700.0

kv_budget_hi = VRAM_LIMIT - WEIGHT_MIB - COMPUTE_HI - SAFETY
kv_budget_lo = VRAM_LIMIT - WEIGHT_MIB - COMPUTE_LO - SAFETY

out = {
    "n_full_attn_layers": N_FULL_ATTN,
    "n_linear_layers": N_LINEAR,
    "vram_limit_mib": round(VRAM_LIMIT, 1),
    "weight_mib": round(WEIGHT_MIB, 1),
    "kv_budget_mib_hi": round(kv_budget_hi, 1),
    "kv_budget_mib_lo": round(kv_budget_lo, 1),
    "per_dtype": {},
}

print(f"Full-attention layers: {N_FULL_ATTN}  | linear(ssm): {N_LINEAR}")
print(f"VRAM total {VRAM_TOTAL:.0f} MiB | limit 93% = {VRAM_LIMIT:.1f} MiB")
print(f"Weight = {WEIGHT_MIB:.1f} MiB")
print(f"KV budget: {kv_budget_hi:.0f} MiB (compute 700)  ..  {kv_budget_lo:.0f} MiB (compute 400)")
print()
print(f"{'dtype':6} {'MiB/1k tok':>11} {'ctx@700':>9} {'ctx@400':>9}")
for dt in ("f16", "q8_0", "q4_0"):
    bpt = kv_bytes_per_token(dt)
    mib_1k = bpt * 1000 / (1024 ** 2)
    ctx_hi = int(kv_budget_hi * 1024 ** 2 / bpt)
    ctx_lo = int(kv_budget_lo * 1024 ** 2 / bpt)
    out["per_dtype"][dt] = {
        "bytes_per_token": round(bpt, 1),
        "mib_per_1k_tokens": round(mib_1k, 3),
        "ctx_max_budget700": ctx_hi,
        "ctx_max_budget400": ctx_lo,
    }
    print(f"{dt:6} {mib_1k:11.3f} {ctx_hi:9d} {ctx_lo:9d}")

# verify weights fit at all
print()
if WEIGHT_MIB + 700 > VRAM_LIMIT:
    print("SKIP: il peso da solo non entra nel budget")
else:
    print(f"Peso + overhead minimo (700) = {WEIGHT_MIB + 700:.0f} MiB  ->  {100*(WEIGHT_MIB+700)/VRAM_TOTAL:.1f}% di VRAM  [OK]")

print()
for c in (16384, 32768, 65536, 131072):
    bpt = kv_bytes_per_token("q8_0")
    used = WEIGHT_MIB + COMPUTE_HI + c * bpt / 1024 ** 2
    print(f"q8_0 ctx={c:6d}: KV={c*bpt/1024**2:8.1f} MiB | totale(compute700)={used:8.1f} MiB = {100*used/VRAM_TOTAL:5.1f}%")

json.dump(out, open(r"C:/Users/User/Documents/ai_tui/bench_qwen38/scripts/phase0_budget.json", "w"), indent=2)
