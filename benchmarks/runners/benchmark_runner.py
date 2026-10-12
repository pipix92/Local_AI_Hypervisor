"""
benchmark_runner.py - Automated Benchmark Execution Harness for AI TUI Backend
Tests top local GGUF models on RTX 3050 8GB using ModelTestRunner & LlamaServerManager.
"""

import os
import sys
import time
import json
import psutil
from typing import List, Dict, Any

# Ensure backend path
BACKEND_DIR = r"C:\Users\User\Documents\ai_tui"
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend import (
    LlamaServerManager,
    ModelTestRunner,
    ModelConfig,
    ModelConfigManager,
    HardwareMonitor,
    LocalModelInfo,
    ModelTestResult
)

MODELS_TO_BENCHMARK = [
    {
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "path": r"D:\models\lmstudio-community\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf",
        "enable_thinking": False,
        "temperature": 0.2
    },
    {
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "path": r"D:\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf",
        "enable_thinking": False,
        "temperature": 0.2
    },
    {
        "filename": "Spark-X2.5-4B-Q4_K_M.gguf",
        "path": r"D:\models\XHToken\Spark-X2.5-4B-GGUF\Spark-X2.5-4B-Q4_K_M.gguf",
        "enable_thinking": False,
        "temperature": 0.2
    },
    {
        "filename": "Qwen3-Code-Reasoning-4B.Q5_K_M.gguf",
        "path": r"D:\models\GetSoloTech\Qwen3-Code-Reasoning-4B-GGUF\Qwen3-Code-Reasoning-4B.Q5_K_M.gguf",
        "enable_thinking": False,
        "temperature": 0.2
    },
    {
        "filename": "smol-tools-4b-32k-q4_k_m.gguf",
        "path": r"D:\models\enfuse\smol-tools-4b-32k\smol-tools-4b-32k-q4_k_m.gguf",
        "enable_thinking": False,
        "temperature": 0.2
    },
    {
        "filename": "mica-v0.1-4b-Q4_K_M.gguf",
        "path": r"D:\models\mica-v0.1-4b-Q4_K_M\mic\mica-v0.1-4b-Q4_K_M.gguf",
        "enable_thinking": False,
        "temperature": 0.1
    }
]


def kill_existing_llama_servers():
    """Ensure no background llama-server is occupying GPU VRAM or port 18434."""
    for p in psutil.process_iter(["pid", "name"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "llama-server" in name:
                p.kill()
        except Exception:
            pass
    time.sleep(1.5)


def run_full_suite():
    print("=" * 70)
    print("  AI TUI BACKEND - AUTOMATED MODEL BENCHMARK SUITE")
    print(f"  Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    mgr = LlamaServerManager()
    hw = HardwareMonitor()
    runner = ModelTestRunner(endpoint_url="http://127.0.0.1:18434")

    bin_path = mgr.find_llama_server_binary()
    print(f"[*] Detected llama-server: {bin_path}\n")

    results_data: List[Dict[str, Any]] = []

    for target in MODELS_TO_BENCHMARK:
        fn = target["filename"]
        fp = target["path"]
        if not os.path.exists(fp):
            print(f"[!] Warning: Model file not found at {fp}, skipping...")
            continue

        print("-" * 70)
        print(f"[*] Preparing benchmark for: {fn}")
        print(f"    Path: {fp}")

        # Kill any prior servers to maintain isolated VRAM footprint
        print("[*] Ensuring clean VRAM state...")
        kill_existing_llama_servers()

        # Launch server
        print("[*] Launching llama-server on port 18434 (ctx: 8192, ngl: 99, flash_attn: True, -nkvo: True)...")
        launch_res = mgr.start_server(
            model_path=fp,
            port=18434,
            ctx_size=8192,
            kv_dtype="q4_0",
            no_kv_offload=True,
            ctx_shift=True,
            flash_attn=True,
            n_gpu_layers=99,
            threads=8,
            binary_path=bin_path
        )

        if not launch_res.get("success"):
            print(f"[!] Failed to launch server: {launch_res.get('error')}")
            continue

        pid = launch_res.get("pid")
        print(f"[*] Process spawned (PID: {pid}). Waiting for /health online...")

        # Poll health
        online = False
        for attempt in range(40):
            time.sleep(1.0)
            h = mgr.check_health("llama-server")
            if h.get("status") == "online":
                online = True
                break

        if not online:
            print(f"[!] Server timed out during initialization.")
            mgr.stop_server()
            kill_existing_llama_servers()
            continue

        print("[✓] Server is ONLINE and responsive.")

        # Construct model info and config
        sz_gb = os.path.getsize(fp) / (1024 ** 3)
        model_info = LocalModelInfo(
            filename=fn,
            filepath=fp,
            rel_path=fn,
            author="Local",
            size_bytes=os.path.getsize(fp),
            size_gb=round(sz_gb, 2),
            quant_type="Q4_K_M",
            vram_fit_status="🟢 100% VRAM (<6GB)",
            is_recommended=True,
            last_modified="2026"
        )

        config = ModelConfig(
            model_filename=fn,
            temperature=target.get("temperature", 0.3),
            enable_thinking=target.get("enable_thinking", True)
        )

        print("[*] Running ModelTestRunner benchmark suite (AST Code Gen + JSON Tool Calling)...")
        test_res = runner.run_benchmark(model_info, config, hw)

        print(f"    Status: {test_res.status}")
        print(f"    Generation Speed: {test_res.generation_speed_tok_s:.1f} tok/s")
        print(f"    Time To First Token (TTFT): {test_res.ttft_ms:.1f} ms")
        print(f"    Total Tokens Generated: {test_res.total_tokens}")
        print(f"    Total Time: {test_res.total_time_s:.2f} s")
        print(f"    Python Code Syntax (AST Valid): {test_res.code_syntax_valid}")
        print(f"    Palindrome Function Test: {'PASSED' if test_res.code_test_passed else 'FAILED'}")
        print(f"    Tool Calling JSON Valid: {test_res.tool_json_valid}")
        print(f"    Tool Action Target Match: {'PASSED' if test_res.tool_test_passed else 'FAILED'}")
        print(f"    VRAM Allocated: {test_res.vram_used_mb:.1f} MB")

        res_dict = {
            "model_name": fn,
            "model_path": fp,
            "size_gb": round(sz_gb, 2),
            "status": test_res.status,
            "generation_speed_tok_s": test_res.generation_speed_tok_s,
            "ttft_ms": test_res.ttft_ms,
            "total_tokens": test_res.total_tokens,
            "total_time_s": test_res.total_time_s,
            "code_syntax_valid": test_res.code_syntax_valid,
            "code_test_passed": test_res.code_test_passed,
            "tool_json_valid": test_res.tool_json_valid,
            "tool_test_passed": test_res.tool_test_passed,
            "vram_used_mb": test_res.vram_used_mb,
            "raw_code_output": test_res.raw_code_output,
            "raw_tool_output": test_res.raw_tool_output
        }
        results_data.append(res_dict)

        # Stop server
        print("[*] Terminating model server to release GPU VRAM...\n")
        mgr.stop_server()
        kill_existing_llama_servers()
        time.sleep(1.0)

    # Save to JSON
    out_file = os.path.join(BACKEND_DIR, "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"[✓] Benchmark completed. Results saved to: {out_file}\n")

    # Print Summary Table
    print("=" * 95)
    print(f"{'Model':<35} | {'Size':<8} | {'Tok/s':<8} | {'TTFT (ms)':<10} | {'AST Valid':<10} | {'Tool JSON':<10} | {'VRAM (MB)':<10}")
    print("-" * 95)
    for r in results_data:
        ast_str = "PASS" if r["code_syntax_valid"] and r["code_test_passed"] else "FAIL"
        tool_str = "PASS" if r["tool_json_valid"] and r["tool_test_passed"] else "FAIL"
        print(f"{r['model_name']:<35} | {r['size_gb']:<5.2f} GB | {r['generation_speed_tok_s']:<8.1f} | {r['ttft_ms']:<10.1f} | {ast_str:<10} | {tool_str:<10} | {r['vram_used_mb']:<10.1f}")
    print("=" * 95)


if __name__ == "__main__":
    run_full_suite()
