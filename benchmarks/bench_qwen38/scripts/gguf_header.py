import struct, sys, json

PATH = r"D:\models\mradermacher\Qwen3.8-9B-Distill-uncensored-heretic-i1-GGUF\Qwen3.8-9B-Distill-uncensored-heretic.i1-IQ4_XS.gguf"

# GGUF value type sizes
STRUCT = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}

f = open(PATH, "rb")
magic = f.read(4)
ver = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
print(f"magic={magic!r} version={ver} n_tensors={n_tensors} n_kv={n_kv}")

def read_string():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def read_value(t):
    if t == 8:
        return read_string()
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_value(et) for _ in range(n)]
    fmt, size = STRUCT[t]
    vals = struct.unpack("<" + fmt * 1, f.read(size))
    return vals[0]

kv = {}
for _ in range(n_kv):
    key = read_string()
    t = struct.unpack("<I", f.read(4))[0]
    kv[key] = read_value(t)

f.close()

interesting = [
    "general.architecture", "general.name", "general.file_type", "general.quantization_version",
    "context_length", "block_count", "full_attention_interval", "attention.head_count",
    "attention.head_count_kv", "embedding_length", "feed_forward_length",
    "attention.key_length", "attention.value_length", "attention.layer_norm_rms_epsilon",
    "attention.causal_attn", "rope.dimension_count", "ssm.*", "*.rope_freq_base",
    "expert_.*", "expert_used_count", "*.expert_count",
]
for k in sorted(kv):
    for pat in interesting:
        if pat.endswith("*") and k.startswith(pat[:-1]):
            print(f"{k} = {kv[k]}")
            break
        elif pat == k:
            print(f"{k} = {kv[k]}")
            break

with open(r"C:/Users/User/Documents/ai_tui/bench_qwen38/scripts/gguf_kv.json", "w", encoding="utf-8") as o:
    json.dump({k: v for k, v in kv.items()}, o, indent=2, default=str)
print("\nDUMPED all kv -> scripts/gguf_kv.json")
