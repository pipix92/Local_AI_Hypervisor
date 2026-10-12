import json, struct, sys

p = sys.argv[1]
f = open(p, "rb")
magic = f.read(4)
ver = struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]
print("magic", magic, "ver", ver, "n_tensors", n_tensors, "n_kv", n_kv)

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "h", 3: "H", 4: "i", 5: "I", 6: "f", 7: "B",
       10: "Q", 11: "q", 12: "d"}


def pack(n, et):
    """little-endian format for n elements of scalar type et"""
    return "<" + FMT[et] * n


def rstr():
    n = struct.unpack("<Q", f.read(8))[0]
    assert n < 200 * 1024 * 1024, f"str len {n} desync at offset {f.tell()}"
    return f.read(n).decode("utf-8", "replace")


kv = {}
for i in range(n_kv):
    k = rstr()
    t = struct.unpack("<I", f.read(4))[0]
    if t == 8:
        v = rstr()
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        ln = struct.unpack("<Q", f.read(8))[0]
        assert ln < 50_000_000, f"{k}: arr len {ln} desync at {f.tell()}"
        if et == 8:
            v = [rstr() for _ in range(ln)]
        else:
            v = list(struct.unpack(pack(ln, et), f.read(ln * SZ[et])))
    else:
        v = struct.unpack(FMT[t], f.read(SZ[t]))[0]
    kv[k] = v

print("\n=== ALL META ===")
for k, v in kv.items():
    if isinstance(v, list):
        v = f"[list of {len(v)}]"
    print(f"  {k} = {v}")

print("\n=== GEOMETRY ===")
arch = kv.get("general.architecture", "qwen3")
for suf in ["context_length", "embedding_length", "block_count", "attention.head_count",
            "attention.head_count_kv", "attention.key_length", "attention.value_length"]:
    key = arch + "." + suf
    if key in kv:
        print(f"  {key} = {kv[key]}")

nc = kv.get(arch + ".block_count", 0)
nkv = kv.get(arch + ".attention.head_count_kv", 1)
hd = kv.get(arch + ".attention.key_length") or kv.get(arch + ".embedding_length", 0) // max(kv.get(arch + ".attention.head_count", 1), 1)
nl = nc * nkv * hd * 2 * 2  # K+V, 2 bytes f16
print(f"\n  KV bytes/token = {nl}  ({nl/1024:.2f} KiB/tok)")
for name, bits in [("f16", 16), ("q8_0", 8.5), ("q4_0", 4.5)]:
    per = nl * bits / 16 / 1024
    print(f"  {name:5s}: {per:.3f} MiB/1k tok")

json.dump(kv, open(sys.argv[2], "w"), indent=1, default=str)
print(f"\nwrote {sys.argv[2]}")
