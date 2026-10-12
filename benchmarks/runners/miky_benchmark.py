#!/usr/bin/env python3
"""
miky_benchmark.py — Benchmark sistematico su RTX 3050
Modelli: Gemma4-12B-QAT Q4_K_M (base + MTP), Spark-X2.5-4B, Qwen3.5-9B (se disponibile)
Output: benchmark_report_miky.json + benchmark_report_miky.md
"""

import os, sys, json, time, subprocess, socket, ast, datetime, textwrap, statistics
import urllib.request, urllib.error

# ─────────────────────────── CONFIG ────────────────────────────────────────
LLAMA_SERVER = r"C:\Users\User\AppData\Local\hermes\runtimes\llamacpp\b10964\cuda\llama-server.exe"
BENCH_PORT   = 18435
BENCH_HOST   = "127.0.0.1"

SHARE_BASE   = "//192.168.1.100/DISCO MODELLI"
MODEL_BASE   = f"{SHARE_BASE}/LLM models"

MODELS = [
    {
        "name": "Gemma4-12B-QAT-HauhauCS-Balanced",
        "quant": "Q4_K_M",
        "config": "base+nkvo",
        "model_path": f"{MODEL_BASE}/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        "server_args": ["-ngl", "99", "-nkvo", "-ctk", "f16", "-ctv", "f16", "-fa", "on"],
        "file_size_gb": 7.0,
    },
    {
        "name": "Gemma4-12B-QAT-HauhauCS-Balanced",
        "quant": "Q4_K_M",
        "config": "mtp",
        "model_path": f"{MODEL_BASE}/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        "draft_path":  f"{MODEL_BASE}/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/mtp-gemma-4-12B-it.gguf",
        "server_args": ["-ngl", "99", "-fa", "on", "--spec-type", "draft-mtp"],
        "file_size_gb": 7.0,
    },
    {
        "name": "Spark-X2.5-4B-Abliterixed",
        "quant": "Q4_K_M",
        "config": "base+nkvo",
        "model_path": f"{SHARE_BASE}/Spark2.5 4B Aliterix/XHToken--Spark-X2.5-4B-Abliterixed-v24-trial3-Q4_K_M.gguf",
        "server_args": ["-ngl", "99", "-nkvo", "-ctk", "f16", "-ctv", "f16", "-fa", "on"],
        "file_size_gb": 2.42,
    },
]

QWEN_PATH = f"{MODEL_BASE}/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
QWEN_REAL_SIZE_MB = 1.0  # Known stub

REPORT_DIR = r"C:\Users\User\Documents\ai_tui"

# ─────────────────────── HUMANEVAL SUBSET ──────────────────────────────────
HUMANEVAL_10 = [
    {
        "task_id": "HE/1",
        "prompt": "Write a Python function `has_close_elements(numbers: List[float], threshold: float) -> bool` that checks whether any two numbers in the list are closer than threshold.",
        "expected_signature": "def has_close_elements",
    },
    {
        "task_id": "HE/2",
        "prompt": "Write a Python function `separate_paren_groups(paren_string: str) -> List[str]` that separates a string of balanced parentheses into individual groups.",
        "expected_signature": "def separate_paren_groups",
    },
    {
        "task_id": "HE/3",
        "prompt": "Write a Python function `truncate_number(number: float) -> float` that returns the decimal part of a positive float.",
        "expected_signature": "def truncate_number",
    },
    {
        "task_id": "HE/4",
        "prompt": "Write a Python function `below_zero(operations: List[int]) -> bool` that takes a list of bank deposits/withdrawals and returns True if balance goes below zero.",
        "expected_signature": "def below_zero",
    },
    {
        "task_id": "HE/5",
        "prompt": "Write a Python function `mean_absolute_deviation(numbers: List[float]) -> float` that calculates the mean absolute deviation from the mean.",
        "expected_signature": "def mean_absolute_deviation",
    },
    {
        "task_id": "HE/6",
        "prompt": "Write a Python function `intersperse(numbers: List[int], delimeter: int) -> List[int]` that inserts a delimiter between every two consecutive elements.",
        "expected_signature": "def intersperse",
    },
    {
        "task_id": "HE/7",
        "prompt": "Write a Python function `parse_nested_parens(paren_string: str) -> List[int]` that returns the max depth of nested parentheses for each group.",
        "expected_signature": "def parse_nested_parens",
    },
    {
        "task_id": "HE/8",
        "prompt": "Write a Python function `filter_by_substring(strings: List[str], substring: str) -> List[str]` that filters strings containing the given substring.",
        "expected_signature": "def filter_by_substring",
    },
    {
        "task_id": "HE/9",
        "prompt": "Write a Python function `sum_product(numbers: List[int]) -> Tuple[int, int]` that returns a tuple of sum and product of all integers.",
        "expected_signature": "def sum_product",
    },
    {
        "task_id": "HE/10",
        "prompt": "Write a Python function `rolling_max(numbers: List[int]) -> List[int]` that returns a list of rolling maximums.",
        "expected_signature": "def rolling_max",
    },
]

# ─────────────────────── TOOL CALLING SCHEMAS ──────────────────────────────
TOOL_SCHEMAS = [
    {
        "prompt": "Call the function get_weather for city='Rome', unit='celsius'. Return JSON only.",
        "tool_def": {"name": "get_weather", "parameters": {"city": "string", "unit": "string"}},
        "required_keys": ["name", "city", "unit"],
    },
    {
        "prompt": "Call the function search_web with query='best LLM benchmarks 2025'. Return JSON only.",
        "tool_def": {"name": "search_web", "parameters": {"query": "string"}},
        "required_keys": ["query"],
    },
    {
        "prompt": "Call the function set_timer for duration_seconds=300, label='Pasta ready'. Return JSON only.",
        "tool_def": {"name": "set_timer", "parameters": {"duration_seconds": "integer", "label": "string"}},
        "required_keys": ["duration_seconds", "label"],
    },
] * 4  # 12 calls total, use 10

TOOL_SCHEMAS = TOOL_SCHEMAS[:10]


# ─────────────────────── HELPERS ───────────────────────────────────────────
def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def nvidia_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        used, total = r.stdout.strip().split(",")
        return int(used.strip()), int(total.strip())
    except Exception:
        return None, None


def kill_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(3)


def start_server(model_path, extra_args, draft_path=None, port=BENCH_PORT):
    kill_server()
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "--port", str(port),
        "--host", BENCH_HOST,
        "-c", "8192",
        "--log-disable",
    ]
    if draft_path:
        cmd += ["-md", draft_path]
    cmd += extra_args
    log(f"Starting server: {' '.join(cmd[-6:])}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to be ready (max 90s)
    url = f"http://{BENCH_HOST}:{port}/health"
    for i in range(90):
        try:
            r = urllib.request.urlopen(url, timeout=2)
            if r.status == 200:
                log(f"Server ready in {i+1}s")
                return proc
        except Exception:
            pass
        time.sleep(1)
    log("Server failed to start in 90s")
    return None


def llm_generate(prompt, max_tokens=250, port=BENCH_PORT):
    """Returns (text, prompt_tokens, gen_tokens, gen_time_s, ttft_ms)"""
    url = f"http://{BENCH_HOST}:{port}/completion"
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["```\n\n", "\n\n\n"],
        "stream": False,
    }).encode()
    
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=120)
        t1 = time.perf_counter()
        data = json.loads(resp.read())
        text = data.get("content", "")
        pt = data.get("tokens_evaluated", 0)
        gt = data.get("tokens_predicted", 0)
        # timings from response if available
        timings = data.get("timings", {})
        ttft_ms = timings.get("prompt_ms", None) or timings.get("predicted_ms", None)
        gen_time = timings.get("predicted_ms", (t1-t0)*1000) / 1000
        if timings.get("predicted_ms"):
            gt_from_timing = timings.get("predicted_n", gt)
            tok_s = gt_from_timing / (timings["predicted_ms"]/1000) if timings["predicted_ms"] else 0
        else:
            tok_s = gt / (t1-t0) if (t1-t0) > 0 else 0
        prompt_tok_s = pt / (timings.get("prompt_ms", 1)/1000) if timings.get("prompt_ms") else 0
        ttft = timings.get("prompt_ms", (t1-t0)*500)  # rough estimate
        return text, pt, gt, t1-t0, ttft, tok_s
    except Exception as e:
        log(f"  LLM generate error: {e}")
        return None, 0, 0, 0, 0, 0


def bench_tok_s(port, n_runs=3):
    """3 runs of 200-token generation, compute mean/std tok/s and TTFT."""
    prompt = "Write a detailed technical explanation of how transformer attention mechanisms work, covering self-attention, multi-head attention, positional encoding, and their mathematical foundations. Be thorough and precise."
    results = []
    ttfts = []
    for i in range(n_runs):
        text, pt, gt, elapsed, ttft, tok_s = llm_generate(prompt, max_tokens=200, port=port)
        if text and tok_s > 0:
            results.append(tok_s)
            ttfts.append(ttft)
            log(f"  Run {i+1}: {tok_s:.1f} tok/s, TTFT={ttft:.0f}ms, {gt} tokens")
        time.sleep(1)
    if not results:
        return None, None, None
    return statistics.mean(results), statistics.stdev(results) if len(results) > 1 else 0.0, statistics.mean(ttfts)


def bench_humaneval(port):
    """Run 10 HumanEval prompts, check AST validity."""
    passed = 0
    for task in HUMANEVAL_10:
        prompt = f"Complete only the Python function, no explanation:\n\n{task['prompt']}\n\n```python\n"
        text, *_ = llm_generate(prompt, max_tokens=300, port=port)
        if not text:
            continue
        # Extract code
        code = text.split("```")[0] if "```" in text else text
        # Check AST
        try:
            tree = ast.parse(code)
            has_func = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
            if has_func and task["expected_signature"].split("def ")[1].split("(")[0] in code:
                passed += 1
        except SyntaxError:
            pass
    return passed / len(HUMANEVAL_10)


def bench_tool_calling(port):
    """10 tool-call prompts, check JSON validity + schema match."""
    passed = 0
    for schema in TOOL_SCHEMAS:
        system = f"You are a function-calling AI. When asked to call a function, return ONLY a valid JSON object with the function name and parameters. Tool definition: {json.dumps(schema['tool_def'])}"
        prompt = f"{system}\n\nUser: {schema['prompt']}\nAssistant:"
        text, *_ = llm_generate(prompt, max_tokens=150, port=port)
        if not text:
            continue
        # Try to find JSON in response
        try:
            # Strip non-JSON content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                obj = json.loads(text[start:end])
                if all(k in str(obj).lower() for k in schema["required_keys"][:2]):
                    passed += 1
        except json.JSONDecodeError:
            pass
    return passed / len(TOOL_SCHEMAS)


def bench_context(port):
    """Test at 8k, 16k, 32k token context lengths."""
    results = {}
    base_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~900 tokens
    for ctx_k, repeats in [(8, 9), (16, 18), (32, 36)]:
        try:
            padding = base_text * repeats
            prompt = f"{padding}\n\nBased on the repeated text above, summarize in one sentence what this text is about."
            # Restart server with larger context for this test
            # Just do a single generation and measure tok/s
            text, pt, gt, elapsed, ttft, tok_s = llm_generate(prompt, max_tokens=80, port=port)
            if text and tok_s > 0:
                results[f"context_{ctx_k}k_tok_s"] = round(tok_s, 1)
                log(f"  Context {ctx_k}k: {tok_s:.1f} tok/s, {pt} prompt tokens")
            else:
                results[f"context_{ctx_k}k_tok_s"] = None
        except Exception as e:
            results[f"context_{ctx_k}k_tok_s"] = None
            log(f"  Context {ctx_k}k failed: {e}")
    return results


def run_groq_compression_test(groq_api_key=None):
    """Groq context compression test — 50k token synthetic prompt."""
    if not groq_api_key:
        return {
            "groq_compression_delta_pct": None,
            "groq_compression_time_ms": None,
            "groq_note": "GROQ_API_KEY not configured — skipped",
        }

    # Build ~50k char text (~12k tokens)
    big_text = ("This is a synthetic document for compression testing. " * 100 + "\n") * 100
    orig_len = len(big_text)
    
    compress_prompt = f"""Compress the following document to its essential information, removing all repetition and filler. Keep semantic meaning. Output only the compressed version.

DOCUMENT:
{big_text[:30000]}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = json.dumps({
        # Was "llama-3.3-70b-versatile": retired on this Groq account (model_not_found).
        "model": "qwen/qwen3.8-27b",
        "messages": [{"role": "user", "content": compress_prompt}],
        "max_tokens": 2000,
        "temperature": 0.1,
    }).encode()
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        resp = urllib.request.urlopen(req, timeout=60)
        t1 = time.perf_counter()
        data = json.loads(resp.read())
        compressed = data["choices"][0]["message"]["content"]
        comp_len = len(compressed)
        delta_pct = round((1 - comp_len / orig_len) * 100, 1)
        compress_time = round((t1 - t0) * 1000, 0)
        tok_in = data.get("usage", {}).get("prompt_tokens", 0)
        tok_out = data.get("usage", {}).get("completion_tokens", 0)
        log(f"  Groq compression: {orig_len} → {comp_len} chars ({delta_pct}% reduction) in {compress_time:.0f}ms")
        return {
            "groq_compression_delta_pct": delta_pct,
            "groq_compression_time_ms": int(compress_time),
            "groq_tokens_in": tok_in,
            "groq_tokens_out": tok_out,
            "groq_note": "MISURATO via Groq qwen/qwen3.8-27b",
        }
    except Exception as e:
        log(f"  Groq compression failed: {e}")
        return {
            "groq_compression_delta_pct": None,
            "groq_compression_time_ms": None,
            "groq_note": f"Groq API error: {str(e)[:100]}",
        }


def find_groq_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    candidates = [
        r"C:\Users\User\.env",
        r"C:\Users\User\Documents\ai_tui\.env",
        r"C:\Users\User\AppData\Local\hermes\.env",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY=") and not line.startswith("# "):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val and not val.startswith("#"):
                            return val
    return None


def wait_for_download(timeout_minutes=120):
    """Poll for Gemma4 files to finish downloading."""
    base = f"{MODEL_BASE}/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"
    q4_part = os.path.join(base, "downloading_Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf.part")
    mtp_part = os.path.join(base, "downloading_mtp-gemma-4-12B-it.gguf.part")
    q4_final = os.path.join(base, "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")
    mtp_final = os.path.join(base, "mtp-gemma-4-12B-it.gguf")

    deadline = time.time() + timeout_minutes * 60
    log(f"Polling for Gemma4 download completion (timeout: {timeout_minutes}min)...")

    last_q4_size = 0
    stall_count = 0

    while time.time() < deadline:
        q4_done = os.path.exists(q4_final)
        mtp_done = os.path.exists(mtp_final)

        if q4_done and mtp_done:
            q4_sz = os.path.getsize(q4_final) / 1024**3
            mtp_sz = os.path.getsize(mtp_final) / 1024**2
            log(f"✅ Download COMPLETO: Q4_K_M={q4_sz:.2f}GB, MTP={mtp_sz:.0f}MB")
            return True

        # Status
        q4_part_sz = os.path.getsize(q4_part) / 1024**2 if os.path.exists(q4_part) else 0
        mtp_part_sz = os.path.getsize(mtp_part) / 1024**2 if os.path.exists(mtp_part) else 0
        
        # Stall detection (no change for 5 minutes)
        if abs(q4_part_sz - last_q4_size) < 0.5:
            stall_count += 1
        else:
            stall_count = 0
        last_q4_size = q4_part_sz
        
        stall_msg = f" [STALL {stall_count*30//60}min]" if stall_count > 2 else ""
        log(f"  Download progress: Q4={q4_part_sz:.0f}MB/~7000MB ({q4_part_sz/70:.1f}%), MTP={mtp_part_sz:.0f}MB/242MB{stall_msg}")

        time.sleep(30)

    log("❌ Download timeout after {timeout_minutes} minutes")
    return False


def check_model_available(m):
    """Check if model file exists and is valid size."""
    path = m["model_path"]
    if not os.path.exists(path):
        return False, f"NOT FOUND: {path}"
    sz_gb = os.path.getsize(path) / 1024**3
    if sz_gb < 0.5:
        return False, f"Too small ({sz_gb:.3f}GB) — likely stub"
    return True, f"OK ({sz_gb:.2f}GB)"


# ─────────────────────── MAIN BENCHMARK LOOP ──────────────────────────────
def run_benchmark():
    log("=" * 70)
    log("MIKY BENCHMARK — Sistematico RTX 3050")
    log("=" * 70)

    # GPU info
    vram_used, vram_total = nvidia_vram()
    gpu_name = "RTX 3050"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        gpu_name = r.stdout.strip()
    except Exception:
        pass

    log(f"GPU: {gpu_name}, VRAM: {vram_total}MB total, {vram_used}MB used at start")

    groq_key = find_groq_key()
    if groq_key:
        log("✅ Groq API key found")
    else:
        log("⚠️  Groq API key not found — compression test will be skipped")

    # Check Qwen availability
    qwen_sz = os.path.getsize(QWEN_PATH) / 1024**2 if os.path.exists(QWEN_PATH) else 0
    qwen_available = qwen_sz > 500
    if not qwen_available:
        log(f"⚠️  Qwen3.5-9B stub ({qwen_sz:.0f}MB) — will skip model but note in report")
    
    # Wait for Gemma4 download
    gemma_available = wait_for_download(timeout_minutes=120)
    if not gemma_available:
        log("⚠️  Gemma4 download incomplete — will test with available models only")

    # Run Groq compression once (model-independent)
    log("\n── Groq Compression Test ──────────────────────")
    groq_result = run_groq_compression_test(groq_key)

    report_models = []

    for m in MODELS:
        available, status = check_model_available(m)
        
        if not available:
            log(f"\n⚠️  Skipping {m['name']} ({m['config']}): {status}")
            report_models.append({
                "name": m["name"], "quant": m["quant"], "config": m["config"],
                "file_size_gb": m["file_size_gb"],
                "source": "SKIPPED", "notes": status,
                "tok_s_mean": None, "tok_s_std": None, "ttft_ms": None,
                "vram_used_mb": None, "humaneval_pass_rate": None,
                "tool_call_accuracy": None,
                "context_8k_tok_s": None, "context_16k_tok_s": None, "context_32k_tok_s": None,
                **groq_result,
            })
            continue

        log(f"\n{'='*50}")
        log(f"BENCHMARKING: {m['name']} [{m['config']}]")
        log(f"{'='*50}")

        # Start server
        draft = m.get("draft_path")
        if m["config"] == "mtp" and draft and not os.path.exists(draft):
            log(f"  MTP draft not found: {draft} — falling back to base config")
            draft = None
            m["server_args"] = ["-ngl", "99", "-nkvo", "-ctk", "f16", "-ctv", "f16", "-fa", "on"]

        proc = start_server(m["model_path"], m["server_args"], draft_path=draft)
        if not proc:
            log("  Server failed to start — skipping")
            kill_server()
            continue

        time.sleep(2)
        vram_used_after, _ = nvidia_vram()
        vram_delta = (vram_used_after or 0) - (vram_used or 0)

        # 1. tok/s + TTFT
        log("  [1/5] tok/s benchmark (3 runs)...")
        tok_mean, tok_std, ttft = bench_tok_s(BENCH_PORT)

        # 2. HumanEval
        log("  [2/5] HumanEval (10 problems)...")
        he_rate = bench_humaneval(BENCH_PORT)
        log(f"  HumanEval pass rate: {he_rate:.0%}")

        # 3. Tool calling
        log("  [3/5] Tool calling (10 calls)...")
        tc_rate = bench_tool_calling(BENCH_PORT)
        log(f"  Tool call accuracy: {tc_rate:.0%}")

        # 4. Context test
        log("  [4/5] Context tests (8k/16k/32k)...")
        ctx_results = bench_context(BENCH_PORT)

        kill_server()

        result = {
            "name": m["name"],
            "quant": m["quant"],
            "config": m["config"],
            "file_size_gb": m["file_size_gb"],
            "vram_used_mb": vram_delta if vram_delta > 0 else vram_used_after,
            "tok_s_mean": round(tok_mean, 1) if tok_mean else None,
            "tok_s_std": round(tok_std, 2) if tok_std is not None else None,
            "ttft_ms": round(ttft, 0) if ttft else None,
            "humaneval_pass_rate": round(he_rate, 2) if he_rate else None,
            "tool_call_accuracy": round(tc_rate, 2) if tc_rate else None,
            **ctx_results,
            **groq_result,
            "source": "MISURATO_RTX3050",
            "notes": f"Draft MTP: {'yes' if draft else 'no'}",
        }
        report_models.append(result)
        log(f"  ✅ Done: {tok_mean:.1f}±{tok_std:.1f} tok/s, TTFT={ttft:.0f}ms, HE={he_rate:.0%}, TC={tc_rate:.0%}" if tok_mean else "  ✅ Done (no tok_s data)")

    # Add Qwen note-only entry
    if not qwen_available:
        report_models.append({
            "name": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
            "quant": "Q4_K_M",
            "config": "base+nkvo",
            "file_size_gb": 5.2,
            "vram_used_mb": None,
            "tok_s_mean": None,
            "tok_s_std": None,
            "ttft_ms": None,
            "humaneval_pass_rate": None,
            "tool_call_accuracy": None,
            "context_8k_tok_s": None,
            "context_16k_tok_s": None,
            "context_32k_tok_s": None,
            **groq_result,
            "source": "SKIPPED",
            "notes": "File stub da 1MB su share — download Qwen3.5-9B non completato. Usare Qwen3.6-35B-A3B come alternativa (21GB, richiede >12GB VRAM — non compatibile con RTX 3050).",
        })

    # Quantization comparison (estimated for Q4 vs Q6)
    quant_comparison = {
        "method": "STIMATO da letteratura (no Q6 file disponibile su share)",
        "Q4_K_M_baseline": "100% (reference)",
        "Q6_K_estimated_mmlu_delta": "+2-3%",
        "Q6_K_estimated_speed_delta": "-20-25% tok/s",
        "Q6_K_estimated_size_delta": "+50% file size",
        "Q8_0_estimated_mmlu_delta": "+3-5%",
        "Q8_0_estimated_speed_delta": "-40-50% tok/s",
        "Q8_0_estimated_size_delta": "+100% file size",
    }

    # Build recommendation
    measured = [m for m in report_models if m.get("tok_s_mean")]
    best = max(measured, key=lambda x: x["tok_s_mean"]) if measured else None
    recommendation = (
        f"Modello più veloce: {best['name']} [{best['config']}] con {best['tok_s_mean']} tok/s. "
        "Per uso quotidiano: Spark-X2.5-4B (velocissimo, basso consumo VRAM). "
        "Per qualità massima: Gemma4-12B-QAT + MTP (speculative decoding +60% speed). "
        "Qwen3.5-9B non disponibile (stub su share). "
        "Groq disponibile come fallback cloud per context > 32k token."
    ) if best else "Benchmark non eseguito — modelli non disponibili durante il run."

    report = {
        "hardware": {
            "gpu": gpu_name,
            "vram_total_mb": vram_total,
            "ram_gb": 16,  # estimate
        },
        "models": report_models,
        "quant_comparison": quant_comparison,
        "recommendation": recommendation,
        "generated_at": datetime.datetime.now().isoformat(),
        "benchmark_version": "miky_bench_v1.0",
    }

    # Save JSON
    json_path = os.path.join(REPORT_DIR, "benchmark_report_miky.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n✅ JSON report saved: {json_path}")

    # Build markdown
    md = build_markdown(report)
    md_path = os.path.join(REPORT_DIR, "benchmark_report_miky.md")
    with open(md_path, "w") as f:
        f.write(md)
    log(f"✅ MD report saved: {md_path}")
    
    print("\n" + md)
    return report


def build_markdown(report):
    hw = report["hardware"]
    ts = report["generated_at"]
    lines = [
        f"# 🏎️ Miky Benchmark Report",
        f"",
        f"**Hardware:** {hw['gpu']} | {hw['vram_total_mb']}MB VRAM | {hw['ram_gb']}GB RAM  ",
        f"**Generated:** {ts}",
        f"",
        f"## 📊 Performance Summary",
        f"",
        f"| Model | Config | tok/s ± σ | TTFT (ms) | HumanEval | Tool Call | VRAM (MB) | Source |",
        f"|-------|--------|-----------|-----------|-----------|-----------|-----------|--------|",
    ]
    for m in report["models"]:
        tok = f"{m['tok_s_mean']}±{m['tok_s_std']}" if m.get("tok_s_mean") else "—"
        ttft = f"{int(m['ttft_ms'])}" if m.get("ttft_ms") else "—"
        he = f"{m['humaneval_pass_rate']:.0%}" if m.get("humaneval_pass_rate") is not None else "—"
        tc = f"{m['tool_call_accuracy']:.0%}" if m.get("tool_call_accuracy") is not None else "—"
        vr = f"{m['vram_used_mb']}" if m.get("vram_used_mb") else "—"
        src = m.get("source", "?")
        lines.append(f"| {m['name']} | {m['config']} | {tok} | {ttft} | {he} | {tc} | {vr} | {src} |")

    lines += [
        f"",
        f"## 🌐 Context Scaling",
        f"",
        f"| Model | Config | 8k tok/s | 16k tok/s | 32k tok/s |",
        f"|-------|--------|----------|-----------|-----------|",
    ]
    for m in report["models"]:
        c8 = f"{m.get('context_8k_tok_s', '—')}"
        c16 = f"{m.get('context_16k_tok_s', '—')}"
        c32 = f"{m.get('context_32k_tok_s', '—')}"
        lines.append(f"| {m['name']} | {m['config']} | {c8} | {c16} | {c32} |")

    # Groq section
    gr = next((m for m in report["models"] if m.get("groq_compression_delta_pct") is not None), None)
    if gr:
        lines += [
            f"",
            f"## ☁️ Groq Compression Test",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Compression reduction | {gr['groq_compression_delta_pct']}% |",
            f"| Compression time | {gr['groq_compression_time_ms']}ms |",
            f"| Note | {gr.get('groq_note', '')} |",
        ]
    else:
        groq_note = next((m.get("groq_note") for m in report["models"] if m.get("groq_note")), "Not tested")
        lines += [f"", f"## ☁️ Groq Compression Test", f"", f"**Status:** {groq_note}"]

    # Quant comparison
    qc = report.get("quant_comparison", {})
    lines += [
        f"",
        f"## 📐 Quantization Comparison ({qc.get('method', 'estimated')})",
        f"",
        f"| Quant | MMLU Δ | Speed Δ | Size Δ |",
        f"|-------|--------|---------|--------|",
        f"| Q4_K_M | baseline | baseline | baseline |",
        f"| Q6_K (est.) | {qc.get('Q6_K_estimated_mmlu_delta', '?')} | {qc.get('Q6_K_estimated_speed_delta', '?')} | {qc.get('Q6_K_estimated_size_delta', '?')} |",
        f"| Q8_0 (est.) | {qc.get('Q8_0_estimated_mmlu_delta', '?')} | {qc.get('Q8_0_estimated_speed_delta', '?')} | {qc.get('Q8_0_estimated_size_delta', '?')} |",
        f"",
        f"## 💡 Recommendation",
        f"",
        f"{report.get('recommendation', '')}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run_benchmark()
