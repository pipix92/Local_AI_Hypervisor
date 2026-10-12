import os
import json
import re

preset_dir = r"C:\Users\User\.lmstudio\config-presets"
model_configs_path = r"C:\Users\User\Documents\ai_tui\model_configs.json"
models_dir = r"D:\models"

# Remove obsolete draft presets
obsolete = [
    "mtp-gemma-4-E4B-it-Q8_0.preset.json",
    "Qwen3 6 35b a3b DSpark 0 12.preset.json",
    "Qwen3 6 35b a3b DSPARK 300MB.preset.json",
    "sss.preset.json",
    "qwen.preset.json"
]
for obs in obsolete:
    p = os.path.join(preset_dir, obs)
    if os.path.exists(p):
        try:
            os.remove(p)
            print(f"Removed obsolete preset: {obs}")
        except Exception as e:
            print(f"Error removing {obs}: {e}")

MODELS_SPEC = [
    {
        "filename": "Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Mini.gguf",
        "title": "Carnice Qwen3.6 MoE 35B APEX",
        "author": "mudler",
        "category": "MoE Reasoning & Coding",
        "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 24, "thinking": True,
        "system": "You are Carnice Qwen3.6 MoE 35B APEX, an expert AI software engineer and reasoning system."
    },
    {
        "filename": "GigaChat-20B-A3B-instruct-v1.5-abliterated.i1-IQ3_M.gguf",
        "title": "GigaChat 20B A3B Instruct Uncensored",
        "author": "mradermacher",
        "category": "Multilingual Reasoning & Uncensored",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 28, "thinking": False,
        "system": "You are GigaChat 20B A3B Instruct, a versatile, uncensored multilingual AI assistant."
    },
    {
        "filename": "Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.i1-IQ3_M.gguf",
        "title": "Huihui Qwen3 Coder 30B A3B",
        "author": "mradermacher",
        "category": "Deterministic MoE Coding Specialist",
        "temp": 0.2, "top_p": 0.85, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 32768, "ngl": 24, "thinking": False,
        "system": "You are Huihui Qwen3 Coder 30B A3B, an uncensored deterministic software engineering AI. Write clean, bug-free code."
    },
    {
        "filename": "Jev-Style-Qwen3.5-2B-Decision-Q8_0.gguf",
        "title": "Jev-Style Qwen3.5 2B Decision",
        "author": "chaoliangUNSW",
        "category": "Ultra-Fast Decision & Gating",
        "temp": 0.1, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 8192, "ngl": 99, "thinking": False,
        "system": "You are a ultra-fast routing, decision, and gating model. Output concise, structured decisions."
    },
    {
        "filename": "L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q2_k.gguf",
        "title": "Llama 3.2 8X3B MoE Dark Champion",
        "author": "DavidAU",
        "category": "MoE Creative & Uncensored Assistant",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 32, "thinking": False,
        "system": "You are Llama 3.2 8X3B MoE Dark Champion, an uncensored AI assistant."
    },
    {
        "filename": "Lily-7B-Instruct-v0.2.Q5_K_M.gguf",
        "title": "Lily 7B Cybersecurity Specialist",
        "author": "segolilylabs",
        "category": "Cybersecurity & Security Specialist",
        "temp": 0.3, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": False,
        "system": "You are Lily 7B Cybersecurity Specialist, an expert AI in cybersecurity, vulnerability analysis, and security research."
    },
    {
        "filename": "MiniCPM5-2B-Base.IQ4_XS.gguf",
        "title": "MiniCPM5 2B Base",
        "author": "mradermacher",
        "category": "Lightweight Base / Completion",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 16384, "ngl": 99, "thinking": False,
        "system": "You are MiniCPM5 2B, an ultra-compact and efficient language model."
    },
    {
        "filename": "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf",
        "title": "NVIDIA Nemotron-3 Nano 4B",
        "author": "lmstudio-community",
        "category": "Enterprise Reasoning & Agent",
        "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are NVIDIA Nemotron-3 Nano 4B, a high-performance reasoning model."
    },
    {
        "filename": "Qwen3-1.7B-uncensored.i1-IQ4_NL.gguf",
        "title": "Qwen3 1.7B Uncensored",
        "author": "mradermacher",
        "category": "Compact Uncensored Assistant",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 16384, "ngl": 99, "thinking": False,
        "system": "You are an uncensored, concise assistant."
    },
    {
        "filename": "Qwen3-18B-A3B-Stranger-Thoughts-Abliterated-Uncensored-MAX-D_AU-IQ4_XS.gguf",
        "title": "Qwen3 18B A3B Stranger Thoughts",
        "author": "DavidAU",
        "category": "Deep Reasoning & Uncensored CoT",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 28, "thinking": True,
        "system": "You are Qwen3 18B A3B Stranger Thoughts, an uncensored AI with deep analytical reasoning."
    },
    {
        "filename": "Qwen3-4B-Thinking-2507-Q4_K_M.gguf",
        "title": "Qwen3 4B Thinking",
        "author": "lmstudio-community",
        "category": "Dense Reasoning & Step-by-Step CoT",
        "temp": 0.6, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are Qwen3 4B Thinking, a focused step-by-step reasoning model."
    },
    {
        "filename": "Qwen3-Code-Reasoning-4B.Q5_K_M.gguf",
        "title": "Qwen3 Code Reasoning 4B",
        "author": "GetSoloTech",
        "category": "Coding & Reasoning Specialist",
        "temp": 0.3, "top_p": 0.85, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are Qwen3 Code Reasoning 4B, an expert AI programmer and code analyzer."
    },
    {
        "filename": "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
        "title": "Qwen3-VL 4B Instruct",
        "author": "lmstudio-community",
        "category": "Vision-Language & Multimodal",
        "temp": 0.5, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": False,
        "system": "You are Qwen3-VL 4B Instruct, an advanced multimodal and visual reasoning AI."
    },
    {
        "filename": "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "title": "Qwen3.5 2B Uncensored Aggressive",
        "author": "HauhauCS",
        "category": "Compact Uncensored Assistant",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 16384, "ngl": 99, "thinking": False,
        "system": "You are Qwen3.5 2B Uncensored, a helpful and unrestricted AI."
    },
    {
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "title": "Qwen3.5 4B Instruct",
        "author": "lmstudio-community",
        "category": "General Instruction & Agent",
        "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": False,
        "system": "You are Qwen3.5 4B, a versatile and helpful AI assistant."
    },
    {
        "filename": "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "title": "Qwen3.5 4B Uncensored Aggressive",
        "author": "HauhauCS",
        "category": "Uncensored Instruction & Agent",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": False,
        "system": "You are Qwen3.5 4B Uncensored, an unrestricted AI assistant."
    },
    {
        "filename": "Qwen3.6-27B-A3B-Coder-IQ3_M.gguf",
        "title": "Qwen3.6 27B A3B Coder",
        "author": "ManniX-ITA",
        "category": "High-End MoE Coding Specialist",
        "temp": 0.2, "top_p": 0.85, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 32768, "ngl": 24, "thinking": False,
        "system": "You are Qwen3.6 27B A3B Coder, an advanced software engineering and algorithm specialist."
    },
    {
        "filename": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "title": "Qwen3.6 35B A3B Uncensored Aggressive",
        "author": "HauhauCS",
        "category": "Massive MoE Uncensored General",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 20, "thinking": False,
        "system": "You are Qwen3.6 35B A3B Uncensored, an advanced high-capacity AI."
    },
    {
        "filename": "Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf",
        "title": "Qwen3.8 27B iMatrix NVFP4",
        "author": "cdiamond",
        "category": "Dense High-Precision Reasoning",
        "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 20, "thinking": True,
        "system": "You are Qwen3.8 27B iMatrix NVFP4, a state-of-the-art dense reasoning model."
    },
    {
        "filename": "Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf",
        "title": "Qwen3.8 35B A3B Distill APEX",
        "author": "IsValorum",
        "category": "Premier MoE Reasoning & Coding System",
        "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 24, "thinking": True,
        "system": "You are Qwen3.8-35B-A3B-Distill APEX, an expert AI software engineer and reasoning system."
    },
    {
        "filename": "Spark-X2.5-4B-Q4_K_M.gguf",
        "title": "Spark X2.5 4B",
        "author": "XHToken",
        "category": "High-Speed Decision & Filtering",
        "temp": 0.1, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 8192, "ngl": 99, "thinking": False,
        "system": "You are Spark X2.5 4B, a high-speed deterministic decision and filtering assistant."
    },
    {
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "title": "Google Gemma 4 E2B Instruct Q4_K_M",
        "author": "lmstudio-community",
        "category": "Lightweight Agentic Reasoner",
        "temp": 0.6, "top_p": 0.9, "top_k": 64, "min_p": 0.05, "rep_pen": 1.1,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are Gemma 4 E2B, an advanced AI reasoning and coding model."
    },
    {
        "filename": "gemma-4-E2B-it-Q5_K_S.gguf",
        "title": "Google Gemma 4 E2B Instruct Q5_K_S",
        "author": "unsloth",
        "category": "High-Precision Compact Reasoner",
        "temp": 0.6, "top_p": 0.9, "top_k": 64, "min_p": 0.05, "rep_pen": 1.1,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are Gemma 4 E2B, an advanced AI reasoning and coding model."
    },
    {
        "filename": "gemma4-coding-Q3_K_M.gguf",
        "title": "Gemma 4 12B Coder Fable5",
        "author": "yuxinlu1",
        "category": "Dedicated 12B Coding & Reasoner",
        "temp": 0.3, "top_p": 0.85, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are Gemma 4 Coder Fable5, a high-performance reasoning and code generation AI."
    },
    {
        "filename": "jevk5-2b-v0.2-Q8_0.gguf",
        "title": "JevK5 2B Decision v0.2",
        "author": "alibiserikbay",
        "category": "Ultra-Fast Decision & Gating",
        "temp": 0.1, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 8192, "ngl": 99, "thinking": False,
        "system": "You are JevK5 2B, an ultra-fast decision, routing, and filtering model."
    },
    {
        "filename": "jevk5-4b-v0.3-Q5_K_M.gguf",
        "title": "JevK5 4B Decision v0.3",
        "author": "alibiserikbay",
        "category": "High-Precision Decision & Gating",
        "temp": 0.1, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 8192, "ngl": 99, "thinking": False,
        "system": "You are JevK5 4B, an advanced decision, routing, and validation model."
    },
    {
        "filename": "mica-v0.1-4b-Q4_K_M.gguf",
        "title": "Mica 4B News & Decision Filter",
        "author": "sky7350",
        "category": "High-Speed News & Context Filter",
        "temp": 0.1, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 8192, "ngl": 99, "thinking": False,
        "system": "You are Mica 4B, a high-speed news classifier, context filter, and routing model."
    },
    {
        "filename": "minicpm5-2b-abliterated-Q5_K_M.gguf",
        "title": "MiniCPM5 2B Abliterated",
        "author": "mondk",
        "category": "Compact Uncensored Assistant",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 16384, "ngl": 99, "thinking": False,
        "system": "You are MiniCPM5 2B Abliterated, an uncensored compact AI assistant."
    },
    {
        "filename": "mlabonne_Qwen3-1.7B-abliterated-IQ4_NL.gguf",
        "title": "Qwen3 1.7B Abliterated IQ4_NL",
        "author": "bartowski",
        "category": "Compact Uncensored Assistant",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 16384, "ngl": 99, "thinking": False,
        "system": "You are an uncensored, concise assistant."
    },
    {
        "filename": "mlabonne_Qwen3-1.7B-abliterated-Q4_K_M.gguf",
        "title": "Qwen3 1.7B Abliterated Q4_K_M",
        "author": "bartowski",
        "category": "Compact Uncensored Assistant",
        "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 16384, "ngl": 99, "thinking": False,
        "system": "You are an uncensored, concise assistant."
    },
    {
        "filename": "ornith-1.0-9b-Q4_K_M.gguf",
        "title": "Ornith 1.0 9B",
        "author": "deepreinforce-ai",
        "category": "Reinforcement Reasoning Specialist",
        "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.05,
        "ctx": 32768, "ngl": 99, "thinking": True,
        "system": "You are Ornith 1.0 9B, an advanced reinforcement-trained reasoning model."
    },
    {
        "filename": "qwen2.5-coder-7b-instruct-q5_0-00001-of-00002.gguf",
        "title": "Qwen 2.5 Coder 7B Instruct",
        "author": "Qwen",
        "category": "Deterministic Coder Specialist",
        "temp": 0.2, "top_p": 0.85, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 32768, "ngl": 99, "thinking": False,
        "system": "You are a deterministic coding assistant. Write clean, bug-free code."
    },
    {
        "filename": "smol-tools-4b-32k-q4_k_m.gguf",
        "title": "Smol Tools 4B 32k",
        "author": "enfuse",
        "category": "Function Calling & Tool Specialist",
        "temp": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 32768, "ngl": 99, "thinking": False,
        "system": "You are Smol Tools 4B, an expert function calling and tool execution model."
    },
    {
        "filename": "spark-x2.5-4b-rizzo-flow-lora-q4_k_m.gguf",
        "title": "Spark X2.5 4B Rizzo Flow",
        "author": "rizzoaiacademy",
        "category": "Workflow Decision & Flow Engine",
        "temp": 0.1, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "rep_pen": 1.0,
        "ctx": 8192, "ngl": 99, "thinking": False,
        "system": "You are Spark X2.5 Rizzo Flow, an automated workflow decision and routing AI."
    }
]

model_configs = {}
for spec in MODELS_SPEC:
    fn = spec["filename"]
    cfg = {
        "model_filename": fn,
        "temperature": spec["temp"],
        "top_p": spec["top_p"],
        "top_k": spec["top_k"],
        "min_p": spec["min_p"],
        "repeat_penalty": spec["rep_pen"],
        "context_length": spec["ctx"],
        "ngl": spec["ngl"],
        "flash_attention": True,
        "no_kv_offload": True,
        "enable_thinking": spec["thinking"],
        "system_prompt": spec["system"],
        "custom_flags": "--ctx-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512",
        "source": "calibrated_rtx3050",
        "last_updated": "2026-09-28 00:25:00"
    }
    model_configs[fn] = cfg

    # Export LM Studio Preset
    clean_id = re.sub(r"[^a-zA-Z0-9-]", "-", spec["title"].lower()).strip("-")
    preset_data = {
        "identifier": f"@local:{clean_id}",
        "name": f"{spec['title']} (Calibrated RAM-KV)",
        "changed": False,
        "operation": {
            "fields": [
                {"key": "llm.prediction.temperature", "value": spec["temp"]},
                {"key": "llm.prediction.topP", "value": spec["top_p"]},
                {"key": "llm.prediction.topK", "value": spec["top_k"]},
                {"key": "llm.prediction.minP", "value": spec["min_p"]},
                {"key": "llm.prediction.repeatPenalty", "value": spec["rep_pen"]},
                {"key": "llm.prediction.reasoning.enableThinking", "value": spec["thinking"]},
                {"key": "llm.prediction.systemPrompt", "value": spec["system"]}
            ]
        },
        "load": {
            "fields": [
                {"key": "llm.load.llama.flashAttention", "value": True},
                {"key": "llm.load.llama.contextLength", "value": spec["ctx"]},
                {"key": "llm.load.llama.gpuLayers", "value": spec["ngl"]},
                {"key": "llm.load.llama.noKvOffload", "value": True},
                {"key": "llm.load.llama.cpuThreads", "value": 8}
            ]
        }
    }
    
    # Save preset by Title
    preset_path1 = os.path.join(preset_dir, f"{spec['title']}.preset.json")
    with open(preset_path1, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, indent=2)
        
    # Also save preset by filename (without .gguf extension)
    preset_path2 = os.path.join(preset_dir, f"{os.path.splitext(fn)[0]}.preset.json")
    with open(preset_path2, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, indent=2)

with open(model_configs_path, "w", encoding="utf-8") as f:
    json.dump(model_configs, f, indent=2)

print(f"Wrote {len(model_configs)} model configs to {model_configs_path}")
print(f"Generated calibrated presets in {preset_dir}")

# Generate D:/models/model_index.json
index_path = os.path.join(models_dir, "model_index.json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump({"total_models": len(MODELS_SPEC), "models": MODELS_SPEC}, f, indent=2)
print(f"Generated {index_path}")
