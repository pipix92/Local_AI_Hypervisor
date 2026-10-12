#!/usr/bin/env python3
"""
lmbench.py - lm_eval 0.4.13 benchmark module for the ai_tui dashboard.

Three independent capabilities, none of which touch the GPU by themselves:

1. TASK INVENTORY. The real task list is read from
   ``lm_eval.tasks.TaskManager().all_tasks`` on the interpreter that actually
   has lm_eval installed. Nothing here is a hardcoded guess: a name that is not
   returned by TaskManager is reported as ABSENT, not silently offered.
   Verified on this machine: 14677 tasks in 0.4.13, where ``humaneval``
   exists (10 variants) and ``livecodebench`` does not exist at all.

2. CONTEXT CALCULATOR. Reads a GGUF header only (no weights mapped, no CUDA
   context) and reports KV bytes per 1k tokens in f16 / q8_0 / q4_0, then the
   max context that fits a given VRAM budget. Delegates to the already
   battle-tested ``gguf_geom.geometry`` so the hybrid-SSM and per-layer-KV-head
   traps stay handled in one place.

3. DRY RUN. ``--dry-run`` prints the ctx x KV matrix and the resolved task
   names, then exits. It starts no server, imports no model, and never touches
   the GPU. This is the default-safe path and the one the TUI calls to preview.

lm_eval import is slow (~70 s here) and is therefore lazy + cached to
``bench_tasks.json``: the inventory is harvested once, then served from disk.
"""

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_JSON = os.path.join(HERE, "bench_tasks.json")

# Interpreter that carries lm_eval 0.4.13 on this box. Resolved at runtime by
# _find_lm_eval_python() when the current interpreter does not have it.
FALLBACK_LM_EVAL_PYTHONS = (
    "python3.13",
    r"C:\Users\User\AppData\Roaming\Python\Python313\python.exe",
    r"C:\Users\User\Documents\jev_test\lmenv\Scripts\python.exe",
)

# KV bytes per element, the three formats the TUI exposes.
KV_FORMATS: Dict[str, float] = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}

# Fractions of a VRAM budget that stay free for weights + compute buffers.
VRAM_HEADROOM_DEFAULT = 0.80
# Overhead llama.cpp adds on top of raw KV (allocator + attention scratch).
KV_RUNTIME_OVERHEAD = 1.10

# Curated "classici" set: the standard academic suite a local GGUF is graded
# against. Each entry is (task, one-line description). A name absent from the
# real TaskManager list is dropped at harvest time and reported under
# "absenti" so the JSON never advertises a task that cannot be run.
CLASSIC_SUITE: Tuple[Tuple[str, str], ...] = (
    ("mmlu", "57-knowledge / reasoning MMLU (default grouped subtasks)"),
    ("mmlu_flan_n_shot_generative", "MMLU few-shot generative variant"),
    ("hellaswag", "Commonsense NLI, 4-way multiple choice"),
    ("arc_challenge", "ARC science questions, hard split"),
    ("arc_easy", "ARC science questions, easy split"),
    ("winogrande", "Pronoun/blank resolution"),
    ("piqa", "Physical commonsense how-to"),
    ("openbookqa", "Open-book science facts"),
    ("boolq", "Yes/no reading comprehension"),
    ("sciq", "Science exam multiple choice"),
    ("lambada", "Last-word prediction, perplexity-style"),
    ("logiqa", "Logical reading comprehension (4-way)"),
    ("drop", "Discrete reasoning over paragraphs"),
    ("gsm8k", "Grade-school math, chain-of-thought"),
    ("gsm8k_cot", "GSM8K with explicit CoT prompting"),
    ("hendrycks_math", "Hendrycks MATH (LaTeX answers)"),
    ("truthfulqa_mc2", "Truthfulness, multi-choice"),
    ("humaneval", "Python function synthesis, pass@1"),
    ("humaneval_infilling", "Python infilling-style synthesis"),
)

# Curated "agentici" set: tasks that stress instruction following, tool-shaped
# output, long context and multi-turn behaviour rather than raw knowledge.
AGENTIC_SUITE: Tuple[Tuple[str, str], ...] = (
    ("ifeval", "Verifiable instruction following"),
    ("bbh", "BIG-Bench Hard, 23 hard reasoning tasks"),
    ("bbh_cot_fewshot", "BBH with few-shot chain-of-thought"),
    ("truthfulqa_mc1", "Single-choice truthfulness"),
    ("agieval", "AGIEval exam suite"),
    ("cruxeval_input", "CRUXEval minimal-pair input reconstruction"),
    ("cruxeval_output", "CRUXEval minimal-pair output"),
    ("mmlu_pro", "MMLU-Pro, 10-option harder MMLU"),
    ("squadv2", "Extractive QA v2, F1 scored (real name: squadv2)"),
    ("gpqa_diamond_n_shot", "GPQA diamond subset, n-shot"),
    ("gpqa_main_n_shot", "GPQA main subset, n-shot"),
    ("blimp", "BLiMP syntactic minimal pairs"),
    ("leaderboard", "Open LLM leaderboard aggregation (real name: leaderboard)"),
)

ALL_SUITE = CLASSIC_SUITE + AGENTIC_SUITE


# =====================================================================
# LOCATING THE INTERPRETER THAT HAS lm_eval
# =====================================================================
def _probe_lm_eval(python_exe: str) -> Optional[Tuple[str, str]]:
    """Return (version, interpreter) if lm_eval imports, else None.

    Runs a subprocess so the heavy import never pollutes this process, and sets
    CUDA_VISIBLE_DEVICES='' defensively: importing lm_eval must never allocate a
    CUDA context on a machine where another agent owns the GPU.
    """
    code = "import lm_eval;print(lm_eval.__version__)"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONWARNINGS"] = "ignore"
    try:
        r = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True, text=True, timeout=180, env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = (r.stdout or "").strip().splitlines()
    return (out[-1].strip(), python_exe) if out else None


def find_lm_eval_python() -> Optional[Tuple[str, str]]:
    """Locate an interpreter exposing lm_eval. Returns (version, exe) or None."""
    seen = []
    first = sys.executable or "python"
    for exe in (first,) + FALLBACK_LM_EVAL_PYTHONS:
        if not exe or exe in seen:
            continue
        seen.append(exe)
        hit = _probe_lm_eval(exe)
        if hit:
            return hit
    return None


# =====================================================================
# TASK INVENTORY (real names, harvested from TaskManager)
# =====================================================================
def harvest_tasks(python_exe: Optional[str] = None) -> Dict[str, Any]:
    """Ask a real lm_eval TaskManager for the task list and classify it.

    Runs in a subprocess: TaskManager() scans the whole YAML tree and takes
    ~70 s on this machine, which must not block the TUI's render loop.
    """
    if python_exe is None:
        found = find_lm_eval_python()
        if not found:
            return {"error": "lm_eval non installato su alcun interprete noto"}
        python_exe = found[1]

    child = r'''
import json, sys
import lm_eval
from lm_eval.tasks import TaskManager
tm = TaskManager()
tasks = sorted(set(tm.all_tasks))
json.dump({"version": lm_eval.__version__,
           "python": sys.executable,
           "tasks": tasks}, sys.stdout)
'''
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONWARNINGS"] = "ignore"
    t0 = time.time()
    try:
        r = subprocess.run(
            [python_exe, "-c", child],
            capture_output=True, text=True, timeout=900, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"error": "TaskManager(): timeout dopo 900 s"}
    if r.returncode != 0:
        return {"error": "TaskManager() fallito: %s" % (r.stderr or "")[-400:]}

    payload = json.loads(r.stdout)
    available = set(payload["tasks"])

    classici, agentici, assenti = [], [], []
    for name, desc in ALL_SUITE:
        if name in available:
            (classici if name in {n for n, _ in CLASSIC_SUITE} else agentici).append(
                {"task": name, "desc": desc}
            )
        else:
            assenti.append({"task": name, "desc": desc, "motivo": "non in TaskManager().all_tasks"})

    return {
        "lm_eval_version": payload["version"],
        "python": payload["python"],
        "harvested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harvest_seconds": round(time.time() - t0, 1),
        "totale_task_disponibili": len(available),
        "classici": classici,
        "agentici": agentici,
        "assenti": assenti,
        "all_tasks_sample": sorted(available)[:200],
        "all_tasks": sorted(available),
        "_all_tasks": sorted(available),
    }


def load_tasks(refresh: bool = False) -> Dict[str, Any]:
    """Serve the inventory from bench_tasks.json, harvesting only if needed."""
    if not refresh and os.path.isfile(TASKS_JSON):
        try:
            with open(TASKS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["source"] = "cache:%s" % os.path.basename(TASKS_JSON)
            return data
        except (OSError, ValueError):
            pass
    data = harvest_tasks()
    if "error" not in data:
        write_tasks_file(data)
        data["source"] = "live:TaskManager"
    return data


def write_tasks_file(data: Optional[Dict[str, Any]] = None) -> str:
    """Write bench_tasks.json (strips the private _all_tasks dump)."""
    if data is None:
        data = load_tasks()
    out = {k: v for k, v in data.items() if k not in ("_all_tasks", "source")}
    with open(TASKS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return TASKS_JSON


def task_exists(name: str) -> bool:
    """True only if the name is in the real TaskManager list."""
    data = load_tasks()
    allt = data.get("_all_tasks") or data.get("all_tasks") or []
    return name in allt


# =====================================================================
# CONTEXT CALCULATOR (GGUF header only, no GPU)
# =====================================================================
_MSYS_DRIVE = re.compile(r"^/([A-Za-z])(/.*)$")


def normalize_path(path: str) -> str:
    """Accept MSYS/bash paths (/d/models/x.gguf) and return a native path.

    The TUI is driven from git-bash but python is the native Windows build,
    which does not translate /d/... at all. Normalising here keeps every caller
    (CLI, TUI, test) from having to know which shell it was typed in.
    """
    if os.path.isfile(path):
        return path

    m = _MSYS_DRIVE.match(path)
    if m:
        cand = "%s:\\%s" % (m.group(1).upper(),
                            m.group(2).lstrip("/").replace("/", "\\"))
        if os.path.isfile(cand):
            return cand
    return path


def gguf_geometry(path: str) -> Dict[str, Any]:
    """Read GGUF header geometry via the shared gguf_geom module."""
    sys.path.insert(0, HERE)
    import gguf_geom  # noqa: E402  (local module, path set above)

    return gguf_geom.geometry(normalize_path(path))


def kv_matrix(geom: Dict[str, Any], ctx_values: Iterable[int]) -> Dict[str, Any]:
    """Return {kv_format: {ctx: mib}} plus per-format max ctx in budget.

    mib_per_1k keys in gguf_geom are bytes-per-element (2=f16, 1=q8_0, 0.5=q4_0).
    """
    per1k = geom.get("mib_per_1k") or {}
    if not per1k:
        return {"error": "geometria incompleta: mib_per_1k assente"}
    elems = {2.0: "f16", 1.0: "q8_0", 0.5: "q4_0"}

    rows: Dict[str, Dict[int, float]] = {}
    for bpe, name in elems.items():
        base = per1k.get(bpe) or per1k.get(int(bpe))
        if not base:
            continue
        rows[name] = {int(c): round(base * c / 1000.0, 1) for c in ctx_values}
    return rows


def max_contexts(geom: Dict[str, Any], budget_mib: float,
                 headroom: float = VRAM_HEADROOM_DEFAULT) -> Dict[str, Any]:
    """Max ctx that fits in budget_mib, per KV format.

    budget_mib is the TOTAL VRAM available to the process; the KV cache may use
    headroom of it, the rest pays for weights and the compute buffer.
    """
    kv_budget = budget_mib * headroom
    per1k = geom.get("mib_per_1k") or {}
    if not per1k:
        return {"error": "geometria incompleta: mib_per_1k assente"}

    elems = {2.0: "f16", 1.0: "q8_0", 0.5: "q4_0"}
    out: Dict[str, Any] = {}
    weights_mib = float(geom.get("weights_mib") or 0)
    for bpe, name in elems.items():
        base = per1k.get(bpe) or per1k.get(int(bpe))
        if not base:
            continue
        # free = what is left after the weights are resident, floored at 0
        free = max(0.0, kv_budget - weights_mib)
        usable = free / KV_RUNTIME_OVERHEAD if free > 0 else 0.0
        ctx = int(usable / base * 1000.0) if base else 0
        out[name] = {
            "mib_per_1k_tok": base,
            "ctx_max": max(0, ctx),
            "ctx_max_umili": max(0, (max(0, ctx) // 1024) * 1024),
            "kv_budget_mib": round(kv_budget, 1),
            "spazio_dopo_pesi_mib": round(free, 1),
            "fitta": bool(ctx > 0),
        }
    return {
        "budget_mib": round(budget_mib, 1),
        "headroom": headroom,
        "kv_budget_mib": round(kv_budget, 1),
        "pesi_mib": round(weights_mib, 1),
        "formati": out,
    }


# =====================================================================
# DRY RUN MATRIX
# =====================================================================
def dry_run_matrix(model_path: str, budget_mib: float,
                   ctx_values: Optional[List[int]] = None,
                   tasks: Optional[List[str]] = None) -> Dict[str, Any]:
    """Full ctx x KV matrix for one model. Reads the header, loads nothing."""
    ctx_values = ctx_values or [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    geom = gguf_geometry(model_path)
    if "error" in geom:
        return {"model": model_path, "error": geom["error"]}

    inv = load_tasks()
    tasks = tasks or [t["task"] for t in inv.get("classici", [])]

    return {
        "model": model_path,
        "geometry": geom,
        "ctx_values": list(ctx_values),
        "matrice_ctx_kv": kv_matrix(geom, ctx_values),
        "ctx_max": max_contexts(geom, budget_mib),
        "task_risolti": tasks,
        "lm_eval": inv.get("lm_eval_version", "?"),
        "n_task_disponibili": inv.get("totale_task_disponibili"),
    }


def print_dry_run(res: Dict[str, Any], show_tasks: bool = True) -> None:
    """Human-readable ctx x KV matrix. Prints only; starts nothing."""
    if "error" in res:
        print("[!] %s" % res["error"])
        return

    g = res["geometry"]
    print("=" * 78)
    print("DRY-RUN lmbench  |  nessun server, nessun modello caricato, GPU intatta")
    print("=" * 78)
    print("Modello      : %s" % res["model"])
    print("Arch         : %s   layers=%s   KV-heads=%s" % (g.get("arch"), g.get("n_layer"), g.get("n_head_kv")))
    print("key/value len: %s / %s   layer con KV=%s   hybrid_ssm=%s" % (
        g.get("key_length"), g.get("value_length"),
        g.get("n_attn_layers"), g.get("hybrid_ssm")))
    if g.get("full_attention_interval"):
        print("full_attention_interval: %s" % g["full_attention_interval"])
    for w in g.get("warn") or []:
        print("[warn] %s" % w)
    print("Pesi su disco: %s MiB   ctx allenamento: %s" % (g.get("weights_mib"), g.get("ctx_train")))
    print("lm_eval      : %s   task disponibili: %s" % (res.get("lm_eval"), res.get("n_task_disponibili")))

    print("\n--- MATRICE ctx x KV (MiB occupati dal solo KV cache) ---")
    rows = res["matrice_ctx_kv"]
    ctxs = sorted({c for r in rows.values() for c in r})
    header = "%-22s" % "ctx" + "".join("%14s" % ("%d tok" % c) for c in ctxs)
    print(header)
    for fmt in ("f16", "q8_0", "q4_0"):
        if fmt not in rows:
            continue
        line = "%-22s" % ("KV %s" % fmt)
        for c in ctxs:
            line += "%14s" % ("%.0f" % rows[fmt][c])
        print(line)

    cm = res["ctx_max"]
    print("\n--- CTX MASSIMO entro budget VRAM ---")
    print("Budget: %s MiB (headroom %.0f%% -> pool KV %s MiB, pesi %s MiB)" % (
        cm["budget_mib"], cm["headroom"] * 100, cm["kv_budget_mib"], cm["pesi_mib"]))
    for fmt, d in cm["formati"].items():
        mark = "OK " if d["fitta"] else "NO "
        print("  [%s] KV %-5s -> ctx max %10s tok  (%s MiB/1k tok)" % (
            mark, fmt, format(d["ctx_max"], ","), d["mib_per_1k_tok"]))

    if show_tasks and res.get("task_risolti"):
        print("\n--- TASK lm_eval che verrebbero lanciati (dry-run, non eseguiti) ---")
        inv = load_tasks()
        by = {t["task"]: t["desc"] for t in inv.get("classici", []) + inv.get("agentici", [])}
        for t in res["task_risolti"]:
            print("  - %-28s %s" % (t, by.get(t, "")))
    print("\n[ok] dry-run completato: solo lettura header GGUF, GPU non interrogata.")


# =====================================================================
# COMMAND-LINE DRY RUN ENTRY POINT
# =====================================================================
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="lmbench.py",
        description="lm_eval 0.4.13 benchmark per ai_tui (nessun uso GPU in dry-run)",
    )
    p.add_argument("--model", help="percorso .gguf per il calcolatore di context")
    p.add_argument("--budget-mib", type=float, default=4096.0,
                   help="budget VRAM in MiB per il calcolo del ctx massimo")
    p.add_argument("--headroom", type=float, default=VRAM_HEADROOM_DEFAULT,
                   help="frazione del budget lasciata al KV cache")
    p.add_argument("--ctx", type=int, nargs="*",
                   help="lista di contesti da mettere in matrice")
    p.add_argument("--tasks", nargs="*", help="task lm_eval da risolvere (default: suite classici)")
    p.add_argument("--refresh", action="store_true",
                   help="ri-usa TaskManager() invece di bench_tasks.json")
    p.add_argument("--refresh-json", action="store_true",
                   help="riscrive bench_tasks.json e esce")
    p.add_argument("--list-tasks", action="store_true", help="stampa l'inventario e esce")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="calcola e stampa senza avviare nulla (default)")
    args = p.parse_args(argv)

    if args.refresh_json:
        found = find_lm_eval_python()
        if not found:
            print("[!] lm_eval non trovato su alcun interprete noto.")
            return 2
        print("[*] Interprete: %s (lm_eval %s)" % (found[1], found[0]))
        data = harvest_tasks(found[1])
        if "error" in data:
            print("[!] %s" % data["error"])
            return 2
        path = write_tasks_file(data)
        print("[ok] %s scritto: %d task, %d classici, %d agentici, %d assenti" % (
            path, data["totale_task_disponibili"], len(data["classici"]),
            len(data["agentici"]), len(data["assenti"])))
        return 0

    if args.list_tasks:
        found = find_lm_eval_python()
        inv = harvest_tasks(found[1]) if found else load_tasks()
        if "error" in inv:
            print("[!] %s" % inv["error"])
            return 2
        print("lm_eval %s | %d task in TaskManager().all_tasks" % (
            inv.get("lm_eval_version"), inv.get("totale_task_disponibili", 0)))
        for key, title in (("classici", "CLASSICI"), ("agentici", "AGENTICI")):
            print("\n== %s (%d) ==" % (title, len(inv.get(key, []))))
            for t in inv.get(key, []):
                print("  %-30s %s" % (t["task"], t["desc"]))
        if inv.get("assenti"):
            print("\n== NON PRESENTI in 0.4.13 (%d) ==" % len(inv["assenti"]))
            for t in inv["assenti"]:
                print("  %-30s %s" % (t["task"], t["motivo"]))
        return 0

    if not args.model:
        p.error("--model è obbligatorio (oppure usa --list-tasks / --refresh-json)")

    res = dry_run_matrix(args.model, args.budget_mib, args.ctx, args.tasks)
    if "error" not in res and args.headroom != VRAM_HEADROOM_DEFAULT:
        res["ctx_max"] = max_contexts(res["geometry"], args.budget_mib, args.headroom)
    print_dry_run(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
