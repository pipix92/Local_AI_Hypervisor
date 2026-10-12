#!/usr/bin/env python3
"""
Read GGUF metadata WITHOUT loading the model: n_layer, n_head_kv, key_length,
full_attention_interval, and file size. Used to compute the KV budget BEFORE
launching anything, so the grid starts inside the feasible range instead of
burning minutes on a load that cannot fit.

Formula for a hybrid SSM+attention GGUF:
  n_attn_layers = ceil(n_layer / full_attention_interval)
  mib_per_1k_tok = 2 (K,V) * n_attn_layers * n_head_kv * key_length * bpe / 1MiB * 1000
  if full_attention_interval is absent -> every layer keeps KV (dense attention).
"""
import json
import re
import struct
import sys


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_val(f, t):
    if t == 0:   # UINT8
        return struct.unpack("<B", f.read(1))[0]
    if t == 1:   # INT8
        return struct.unpack("<b", f.read(1))[0]
    if t == 2:   # UINT16
        return struct.unpack("<H", f.read(2))[0]
    if t == 3:   # INT16
        return struct.unpack("<h", f.read(2))[0]
    if t == 4:   # UINT32
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:   # INT32
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:   # FLOAT32
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:   # BOOL
        return struct.unpack("<B", f.read(1))[0]
    if t == 8:   # STRING
        return _read_str(f)
    if t == 9:   # ARRAY
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_val(f, et) for _ in range(n)]
    if t == 10:  # UINT64
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:  # INT64
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:  # FLOAT64
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError("gguf type %d" % t)


def read_meta(path, keep_substr=()):
    """Return {key: value} for keys containing any of keep_substr.

    Matching is on SUBSTRINGS, not exact names: every arch-scoped key is
    prefixed with the architecture name (qwen35.block_count, not
    block_count), and the arch is not known until the header is parsed. An
    exact-match filter silently drops every real key and yields
    "header incompleto" for a perfectly valid file.
    """
    out = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF: %s" % magic)
        struct.unpack("<I", f.read(4))     # version
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            key = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            if keep_substr and not any(s in key for s in keep_substr):
                _read_val(f, t)   # must still consume to stay aligned
                continue
            out[key] = _read_val(f, t)
    out["_n_tensors"] = n_tensors
    return out


WANTED = ("architecture", "general.name", "file_type", "block_count",
          "head_count", "head_count_kv", "key_length", "value_length",
          "embedding_length", "full_attention_interval", "ssm",
          "context_length", "rope.dimension_count")


def _num(m, arch, key):
    """Fetch a numeric metadata value, or None.

    Two traps here. Substring matching can hit an ARRAY-valued key, and some
    archs store a scalar as a 1-element list: nemotron_h publishes
    attention.head_count_kv = [42] where qwen35 publishes 4. A list is coerced
    when it holds exactly one number, and treated as absent otherwise, so one
    model never crashes the sweep and never silently reports a wrong KV size.
    """
    v = m.get("%s.%s" % (arch, key), m.get(key))
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)) \
            and not isinstance(v[0], bool):
        return v[0]
    return None


def count_attn_layers(path, n_layer):
    """Count layers that own a K (or V) projection tensor.

    A linear-attention/SSM layer owns no K tensor, so the layers that do are
    exactly the KV-bearing ones. Walks the GGUF tensor-info table (after the
    KV metadata) reading name + shape only. Falls back to n_layer if the table
    cannot be walked.
    """
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return n_layer
            struct.unpack("<I", f.read(4))            # version
            struct.unpack("<Q", f.read(8))            # n_tensors (metadata side)
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                _read_str(f)
                (t,) = struct.unpack("<I", f.read(4))
                _read_val(f, t)
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            layers = set()
            for _ in range(n_tensors):
                name = _read_str(f)
                (ndims,) = struct.unpack("<I", f.read(4))
                for _d in range(ndims):
                    f.read(8)                         # shape dims
                f.read(4)                             # ggml type
                f.read(8)                             # offset
                m = _BLK_KV_RE.search(name)
                if m:
                    layers.add(int(m.group(1) or m.group(2)))
            return len(layers) or n_layer
    except Exception:
        return n_layer


_BLK_KV_RE = re.compile(r"blk\.(\d+)\.[^/]*?(?:^|[_.])(?:k|v)(?:[_.]|$)|"
                        r"blk\.(\d+)\.attn")


def geometry(path):
    import os
    m = read_meta(path, WANTED)
    arch = m.get("general.architecture", "?")
    n_layer = _num(m, arch, "block_count")
    n_kv = _num(m, arch, "attention.head_count_kv")
    kv_arr = m.get("%s.attention.head_count_kv" % arch) or m.get("attention.head_count_kv")
    kv_per_layer = None
    if isinstance(kv_arr, list) and len(kv_arr) > 1:
        # Per-layer array, not a scalar: nemotron_h publishes 42 entries with
        # 0 on the SSM layers and 8 on the 4 attention layers. Count only the
        # non-zero ones; the total is the number of layers that hold a KV.
        live = [v for v in kv_arr if isinstance(v, (int, float)) and v]
        if live:
            kv_per_layer = {
                "layers_with_kv": len(live),
                "kv_heads": max(live),
                "distinct": sorted(set(live)),
            }
            n_kv = max(live)
    kl = _num(m, arch, "attention.key_length")
    vl = _num(m, arch, "attention.value_length")
    fai = _num(m, arch, "full_attention_interval")
    has_ssm = any(".ssm." in k for k in m)
    if n_layer is None or n_kv is None or kl is None:
        return {"file": os.path.basename(path), "error": "header incompleto",
                "raw": {k: v for k, v in m.items() if not isinstance(v, (list, str))}}
    n_attn = n_layer if fai in (None, 0) else -(-n_layer // fai)
    warn = []
    if has_ssm and fai is None:
        n_attn = kv_per_layer["layers_with_kv"] if kv_per_layer else count_attn_layers(path, n_layer)
        warn.append("full_attention_interval assente: %d/%d layer con KV" % (n_attn, n_layer))
    d = kl + (vl or kl)        # K and V are separate tensors; absent v_len -> assume symmetric
    g = {
        "file": os.path.basename(path),
        "arch": arch,
        "n_layer": n_layer,
        "n_head_kv": n_kv,
        "key_length": kl,
        "value_length": vl,
        "full_attention_interval": fai,
        "hybrid_ssm": bool(fai) or has_ssm,
        "n_attn_layers": n_attn,
        "kv_per_layer": kv_per_layer,
        "weights_mib": round(os.path.getsize(path) / 1048576),
        "ctx_train": _num(m, arch, "context_length"),
        "warn": warn,
    }
    g["mib_per_1k"] = {2: round(2 * n_attn * n_kv * d * 2 / 1048576 * 1000, 2),
                       1: round(2 * n_attn * n_kv * d * 1 / 1048576 * 1000, 2),
                       0.5: round(2 * n_attn * n_kv * d * 0.5 / 1048576 * 1000, 2)}
    return g


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(json.dumps(geometry(p), ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"file": p, "error": repr(e)}))
