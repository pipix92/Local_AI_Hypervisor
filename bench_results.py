"""
bench_results.py - lettura dei risultati di benchmark dalla cartella
consolidata /mnt/lm_models/DEBIAN_FOLDER/benchmarks/ (tab "Benchmark").

* .json : formati diversi fra loro (liste, dict con "results", dict per
  modello con "runs" annidati...). Qui si cammina la struttura e si prende
  ogni dict che contiene una metrica di velocita' (tok/s) o un errore,
  ereditando il nome del modello dal livello sopra. Niente stime: i numeri
  sono quelli scritti nei file.
* .db   : prima si legge lo schema (PRAGMA table_info) e solo poi si
  interroga, con le colonne che esistono davvero.
* bench-logs/ : elenco dei log grezzi (nome, dimensione, data).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BENCH_DIR = "/mnt/lm_models/DEBIAN_FOLDER/benchmarks"
# Ripiego (solo se la cartella consolidata manca): gli originali, nella
# cartella del modulo stesso (non un percorso assoluto scritto a mano,
# cosi' sopravvive a uno spostamento della TUI).
FALLBACK_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_FILES = ["grid_ref.json", "mikye_optimal_presets.json", "mikye_calibration.json",
               "kv_local_results.json", "lcb_values.json", "target_benchmark_results.json",
               "benchmark_results.json", "benchmark_report.json"]

SPEED_KEYS = ("tok_s", "speed_tok_s", "generation_speed_tok_s", "tokens_per_second", "gen_tok_s")
TTFT_KEYS = ("ttft_ms", "ttft")
MODEL_KEYS = ("model", "model_name", "name", "file", "filename")
CTX_KEYS = ("ctx", "n_ctx", "ctx_size", "contextLength")
KV_KEYS = ("kv", "cache", "kv_type", "mode")


@dataclass
class BenchRow:
    source: str
    model: str
    label: str = ""
    ctx: Optional[int] = None
    kv: str = ""
    tok_s: Optional[float] = None
    ttft_ms: Optional[float] = None
    status: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(d: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _walk(obj: Any, source: str, model: str, label: str, out: List[BenchRow]) -> None:
    if isinstance(obj, list):
        for it in obj:
            _walk(it, source, model, label, out)
        return
    if not isinstance(obj, dict):
        return
    m = _first(obj, MODEL_KEYS)
    cur_model = str(m) if isinstance(m, (str, int, float)) else model
    speed = _first(obj, SPEED_KEYS)
    err = obj.get("err") or obj.get("error") or obj.get("error_detail")
    if speed is not None or err:
        ctx = _num(_first(obj, CTX_KEYS))
        out.append(BenchRow(
            source=source, model=cur_model or "?",
            label=str(obj.get("label") or label or ""),
            ctx=int(ctx) if ctx else None,
            kv=str(_first(obj, KV_KEYS) or ""),
            tok_s=_num(speed), ttft_ms=_num(_first(obj, TTFT_KEYS)),
            status=("ERR " + str(err)[:50]) if err else str(obj.get("status") or obj.get("load") or "ok"),
        ))
    for k, v in obj.items():
        if isinstance(v, (dict, list)):
            sub_model = cur_model
            # dict indicizzato per nome modello ({"modello.gguf": {...}})
            if isinstance(v, dict) and k not in ("_misurato", "runs", "results", "models") \
                    and not k.startswith("_") and not cur_model:
                sub_model = k
            sub_label = label
            if k.startswith("ctx") or k == "_misurato":
                sub_label = (label + " " + k).strip()
            if isinstance(v, dict) and k.startswith("ctx") and not _first(v, CTX_KEYS):
                v = dict(v, ctx=_num(k[3:].replace("k", "")) * 1024 if k[3:].replace("k", "").isdigit() else None)
            _walk(v, source, sub_model, sub_label, out)


def read_json(path: str) -> List[BenchRow]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows: List[BenchRow] = []
    src = os.path.basename(path)
    if src == "lcb_values.json" and isinstance(data, dict):
        # punteggi LiveCodeBench: niente tok/s, solo il valore
        for k, v in data.items():
            rows.append(BenchRow(source=src, model=k, label="LiveCodeBench",
                                 status=f"LCB {v}" if v is not None else "LCB n/d"))
        return rows
    _walk(data, src, "", "", rows)
    return rows


def db_schema(path: str) -> Dict[str, List[str]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tabs = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [c[1] for c in con.execute(f"PRAGMA table_info('{t}')")] for t in tabs}
    finally:
        con.close()


def read_db(path: str) -> List[BenchRow]:
    schema = db_schema(path)
    rows: List[BenchRow] = []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for t, cols in schema.items():
            want = [c for c in ("model_name", "filename", "quantization", "speed_tok_s",
                                "ttft_ms", "status", "error_detail", "timestamp",
                                "vram_used_mb") if c in cols]
            if not want:
                continue
            for rec in con.execute(f"SELECT {', '.join(want)} FROM '{t}' ORDER BY rowid DESC LIMIT 500"):
                d = dict(zip(want, rec))
                rows.append(BenchRow(
                    source=f"{os.path.basename(path)}:{t}",
                    model=str(d.get("model_name") or d.get("filename") or "?"),
                    label=str(d.get("quantization") or ""),
                    tok_s=_num(d.get("speed_tok_s")), ttft_ms=_num(d.get("ttft_ms")),
                    status=str(d.get("status") or "") + (f" {d['error_detail']}"[:40] if d.get("error_detail") else ""),
                    extra={"timestamp": d.get("timestamp"), "vram_mb": d.get("vram_used_mb")},
                ))
    finally:
        con.close()
    return rows


class BenchResults:
    def __init__(self, base: str = BENCH_DIR) -> None:
        self.base = base if os.path.isdir(base) else FALLBACK_DIR
        self.rows: List[BenchRow] = []
        self.sources: Dict[str, Dict[str, Any]] = {}
        self.logs: List[Dict[str, Any]] = []
        self.loaded_at = 0.0
        self.sort_key = "tok_s"
        self.filter_idx = 0          # 0 = tutte le sorgenti
        self.cursor = 0

    def load(self) -> None:
        rows: List[BenchRow] = []
        sources: Dict[str, Dict[str, Any]] = {}
        names = sorted(set(BENCH_FILES) | {n for n in os.listdir(self.base)
                                           if n.endswith((".json", ".db"))}) \
            if os.path.isdir(self.base) else []
        for n in names:
            p = os.path.join(self.base, n)
            if not os.path.isfile(p):
                sources[n] = {"ok": False, "error": "assente", "rows": 0}
                continue
            try:
                if n.endswith(".db"):
                    got = read_db(p)
                    sources[n] = {"ok": True, "rows": len(got), "schema": db_schema(p)}
                else:
                    got = read_json(p)
                    sources[n] = {"ok": True, "rows": len(got)}
                rows.extend(got)
            except Exception as exc:
                sources[n] = {"ok": False, "error": str(exc)[:80], "rows": 0}
        logs = []
        ld = os.path.join(self.base, "bench-logs")
        if os.path.isdir(ld):
            for n in os.listdir(ld):
                p = os.path.join(ld, n)
                if os.path.isfile(p):
                    st = os.stat(p)
                    logs.append({"name": n, "size": st.st_size, "mtime": st.st_mtime})
        logs.sort(key=lambda x: -x["mtime"])
        self.rows, self.sources, self.logs = rows, sources, logs
        self.loaded_at = time.time()

    def source_names(self) -> List[str]:
        return [n for n, s in self.sources.items() if s.get("rows")]

    def view(self) -> List[BenchRow]:
        names = self.source_names()
        rows = self.rows
        if self.filter_idx and names:
            want = names[(self.filter_idx - 1) % len(names)]
            rows = [r for r in rows if r.source.split(":")[0] == want]
        if self.sort_key == "tok_s":
            rows = sorted(rows, key=lambda r: -(r.tok_s or -1))
        elif self.sort_key == "ttft":
            rows = sorted(rows, key=lambda r: (r.ttft_ms is None, r.ttft_ms or 0))
        elif self.sort_key == "model":
            rows = sorted(rows, key=lambda r: r.model.lower())
        return rows

    def filter_label(self) -> str:
        names = self.source_names()
        if not self.filter_idx or not names:
            return "tutte"
        return names[(self.filter_idx - 1) % len(names)]

    def cycle_filter(self, step: int = 1) -> str:
        n = len(self.source_names()) + 1
        self.filter_idx = (self.filter_idx + step) % max(n, 1)
        self.cursor = 0
        return self.filter_label()

    def cycle_sort(self) -> str:
        order = ["tok_s", "ttft", "model"]
        self.sort_key = order[(order.index(self.sort_key) + 1) % len(order)]
        return self.sort_key
