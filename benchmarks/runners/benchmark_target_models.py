"""
benchmark_target_models.py - Precision Benchmark and Optimization Suite
Benchmarks Gemma 4 12B and Qwen3.8-35B MoE on RTX 3050 (8GB VRAM) + 32GB RAM Host.
"""

import os
import sys
import time
import json
import psutil

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

MODELS = [
    {
        "name": "Gemma-4-12B-Coder-Fable5",
        "filename": "gemma4-coding-Q3_K_M.gguf",
        "path": r"D:\models\yuxinlu1\gemma-4-12B-coder-fable5-composer2.5-v1-GGUF\gemma4-coding-Q3_K_M.gguf",
        "ngl": 99,
        "ctx_size": 8192,
        "no_kv_offload": False,
        "kv_dtype": "f16",
        "temperature": 0.6,
        "top_p": 0.90,
        "top_k": 64,
        "min_p": 0.05,
        "repeat_penalty": 1.10,
        "enable_thinking": True,
        "threads": 8,
        "system_prompt": "You are Gemma 4, an advanced AI reasoning and coding model."
    },
    {
        "name": "Qwen3.8-35B-A3B-Distill-APEX",
        "filename": "Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf",
        "path": r"D:\models\IsValorum\Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF\Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf",
        "ngl": 24,
        "ctx_size": 8192,
        "no_kv_offload": True,
        "kv_dtype": "f16",
        "temperature": 0.6,
        "top_p": 0.90,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "enable_thinking": True,
        "threads": 8,
        "system_prompt": "You are Qwen3.8-35B-A3B-Distill APEX, an expert AI software engineer and reasoning system."
    }
]


def kill_all_servers():
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if "llama-server" in (p.info.get("name") or "").lower():
                p.kill()
        except Exception:
            pass
    time.sleep(1.5)


def run():
    print("=" * 75)
    print("  AUTOMATED BENCHMARK & OPTIMIZATION: GEMMA 4 12B vs QWEN 3.8 35B MoE")
    print(f"  Hardware: NVIDIA GeForce RTX 3050 (8GB) | 32GB RAM Host")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    mgr = LlamaServerManager()
    hw = HardwareMonitor()
    runner = ModelTestRunner(endpoint_url="http://127.0.0.1:18434")
    bin_path = mgr.find_llama_server_binary()
    print(f"[*] Llama-server binary: {bin_path}\n")

    results = []

    for item in MODELS:
        fn = item["filename"]
        fp = item["path"]
        name = item["name"]

        print("-" * 75)
        print(f"[*] Benchmarking: {name}")
        print(f"    File: {fn}")
        print(f"    Path: {fp}")
        sz_gb = os.path.getsize(fp) / (1024 ** 3)
        print(f"    Size on disk: {sz_gb:.2f} GB")
        print(f"    Config: ngl={item['ngl']}, nkvo={item['no_kv_offload']}, kv_dtype={item['kv_dtype']}, ctx={item['ctx_size']}")

        print("[*] Flushing VRAM and terminating previous servers...")
        kill_all_servers()

        snap_before = hw._collect_now()
        print(f"    Pre-launch VRAM used: {snap_before.gpu.vram_used_mb:.1f} MB, Free: {snap_before.gpu.vram_free_mb:.1f} MB")

        launch_res = mgr.start_server(
            model_path=fp,
            port=18434,
            ctx_size=item["ctx_size"],
            kv_dtype=item["kv_dtype"],
            no_kv_offload=item["no_kv_offload"],
            ctx_shift=True,
            flash_attn=True,
            n_gpu_layers=item["ngl"],
            threads=item["threads"],
            binary_path=bin_path
        )

        if not launch_res.get("success"):
            print(f"[!] Launch failed: {launch_res.get('error')}")
            continue

        print(f"[*] Process PID {launch_res.get('pid')} spawned. Polling health...")
        online = False
        for attempt in range(60):
            time.sleep(1.0)
            h = mgr.check_health("llama-server")
            if h.get("status") == "online":
                online = True
                break
            if attempt % 10 == 0 and attempt > 0:
                print(f"    Waiting for model load ({attempt}s)...")

        if not online:
            print("[!] Server failed to come online within 60s.")
            mgr.stop_server()
            kill_all_servers()
            continue

        snap_loaded = hw._collect_now()
        vram_loaded_mb = snap_loaded.gpu.vram_used_mb
        vram_delta_mb = vram_loaded_mb - snap_before.gpu.vram_used_mb
        print(f"[✓] Server ONLINE! Loaded VRAM: {vram_loaded_mb:.1f} MB (Delta: +{vram_delta_mb:.1f} MB)")

        model_info = LocalModelInfo(
            filename=fn,
            filepath=fp,
            rel_path=fn,
            author="Local",
            size_bytes=os.path.getsize(fp),
            size_gb=round(sz_gb, 2),
            quant_type="Q3_K_M" if "Q3" in fn else "MoE",
            vram_fit_status="🟢 100% VRAM" if item["ngl"] >= 99 else "🟡 RAM Offload (MoE)",
            is_recommended=True,
            last_modified="2026"
        )

        config = ModelConfig(
            model_filename=fn,
            temperature=item["temperature"],
            top_p=item["top_p"],
            top_k=item["top_k"],
            min_p=item["min_p"],
            repeat_penalty=item["repeat_penalty"],
            enable_thinking=item["enable_thinking"],
            context_length=item["ctx_size"],
            ngl=item["ngl"],
            flash_attention=True,
            no_kv_offload=item["no_kv_offload"],
            system_prompt=item["system_prompt"]
        )

        print("[*] Running Live ModelTestRunner suite (AST Palindrome Code Gen + JSON Tool Calling)...")
        test_res = runner.run_benchmark(model_info, config, hw)

        print(f"    Status: {test_res.status}")
        print(f"    Speed: {test_res.generation_speed_tok_s:.2f} tok/s")
        print(f"    TTFT: {test_res.ttft_ms:.1f} ms")
        print(f"    Total Tokens: {test_res.total_tokens}")
        print(f"    Total Time: {test_res.total_time_s:.2f} s")
        print(f"    Code AST Valid: {test_res.code_syntax_valid}")
        print(f"    Code Test Passed: {test_res.code_test_passed}")
        print(f"    Tool JSON Valid: {test_res.tool_json_valid}")
        print(f"    Tool Test Passed: {test_res.tool_test_passed}")
        print(f"    VRAM During Test: {test_res.vram_used_mb:.1f} MB")

        res_entry = {
            "name": name,
            "filename": fn,
            "path": fp,
            "size_gb": round(sz_gb, 2),
            "ngl": item["ngl"],
            "no_kv_offload": item["no_kv_offload"],
            "speed_tok_s": test_res.generation_speed_tok_s,
            "ttft_ms": test_res.ttft_ms,
            "tokens": test_res.total_tokens,
            "total_time_s": test_res.total_time_s,
            "ast_valid": test_res.code_syntax_valid,
            "code_passed": test_res.code_test_passed,
            "tool_valid": test_res.tool_json_valid,
            "tool_passed": test_res.tool_test_passed,
            "vram_used_mb": test_res.vram_used_mb,
            "vram_delta_mb": vram_delta_mb,
            "raw_code_output": test_res.raw_code_output,
            "raw_tool_output": test_res.raw_tool_output
        }
        results.append(res_entry)

        print("[*] Stopping server and cooling down...")
        mgr.stop_server()
        kill_all_servers()
        time.sleep(1.5)

    print("\n" + "=" * 90)
    print("  FINAL BENCHMARK COMPARISON MATRIX (RTX 3050 8GB)")
    print("=" * 90)
    print(f"{'Model Name':<32} | {'Size':<8} | {'Tok/s':<8} | {'TTFT (ms)':<10} | {'AST Valid':<10} | {'Tool JSON':<10} | {'VRAM MB':<10}")
    print("-" * 90)
    for r in results:
        ast_str = "PASS" if r["ast_valid"] and r["code_passed"] else "FAIL"
        tool_str = "PASS" if r["tool_valid"] and r["tool_passed"] else "FAIL"
        print(f"{r['name']:<32} | {r['size_gb']:<5.2f} GB | {r['speed_tok_s']:<8.2f} | {r['ttft_ms']:<10.1f} | {ast_str:<10} | {tool_str:<10} | {r['vram_used_mb']:<10.1f}")
    print("=" * 90)

    # Save to disk
    out_path = os.path.join(BACKEND_DIR, "target_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Benchmark results saved to {out_path}")

    return results

if __name__ == "__main__":
    run()
