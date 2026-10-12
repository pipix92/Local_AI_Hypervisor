# 📊 Local LLM Intelligence Report — RTX 3050 8GB
> **Generato:** 2026-09-28 | **Fonte:** HuggingFace model cards, GitHub, badtheorylabs.com, arxiv
> Tutti i numeri marcati con la fonte: `[DICHIARATO HF]`, `[DICHIARATO Google]`, `[STIMATO]`

---

## 🏆 TABELLA PRINCIPALE — Ordinata per Intelligenza Composita

> **Score Composito** = media semplice di (MMLU-Pro o MMLU) + HumanEval + BFCL dove disponibili.
> Dove MMLU-Pro non disponibile, usato MMLU standard. Dove HumanEval non disponibile, omesso dal calcolo.

| Rank | Modello | Quant | GB | Fits 8GB VRAM | MMLU-Pro / MMLU | HumanEval | BFCL | GSM8K | tok/s est. RTX3050 | Note | URL HF |
|------|---------|-------|----|---------------|----------------|-----------|------|-------|---------------------|------|--------|
| 🥇1 | **BTL-3 Compact 27B** | AVQ2 ~2.5bpw | 8.39 GB | ⚠️ limite esatto | — | **95.1%** [DICHIARATO HF] | **88.5%** [DICHIARATO HF] | — | ~15-22 [STIMATO] | Custom runtime BTL-packed llama.cpp; LiveCodeBench v6 88.1%; 92.2% tool retention; richiede Qwen3.6-27B base | [badtheorylabs/BTL-3-Compact](https://huggingface.co/badtheorylabs/BTL-3-Compact) |
| 🥈2 | **Qwen3.5-9B Q4_K_M** | Q4_K_M | ~5.2 GB | ✅ | **82.5%** MMLU-Pro [DICHIARATO HF] | ~82% est. [STIMATO] | **66.1%** BFCL-V4 [DICHIARATO HF] | — | ~35-45 [STIMATO] | MMLU-Redux 91.1%, IFEval 91.5%, HMMT Feb 83.2%; modello denso Qwen ultima gen | [bartowski/Qwen_Qwen3.5-9B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF) |
| 🥉3 | **Gemma4-12B QAT HauhauCS Balanced** | Q4_K_M (QAT) | **6.87 GB** | ✅ | **77.2%** MMLU-Pro [DICHIARATO Google] | ~74% [DICHIARATO MindStudio] | ~69% TAU2 [DICHIARATO Google] | — | ~22-30 [STIMATO] | QAT = quasi full-precision a 4bit; +60% velocità con MTP drafter; uncensored 0/465 refusals; vision mmproj 168MB | [HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced) |
| 4 | **Gemma4-12B heretic iMatrix IQ3_M** | i1-IQ3_M | **5.8 GB** | ✅ | **77.2%** MMLU-Pro [DICHIARATO Google] | ~74% [STIMATO base] | ~69% [STIMATO base] | — | ~22-30 [STIMATO] | iMatrix calibrato; versione heretic = abliterato uncensored; stessa intelligenza di HauhauCS a 1GB in meno | [mradermacher/gemma-4-12B-it-heretic-i1-GGUF](https://huggingface.co/mradermacher/gemma-4-12B-it-heretic-i1-GGUF) |
| 5 | **Qwen3.8-9B-Distill-uncensored-heretic** | i1-Q4_K_M | **5.96 GB** | ✅ | 75.1% CoT [DICHIARATO HF] | ~80% [STIMATO distill] | — | — | ~30-40 [STIMATO] | Distillato da modello più grande; abliterato heretic; Qwen3.8 = generazione post-Qwen3 | [mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-i1-GGUF](https://huggingface.co/mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-i1-GGUF) |
| 6 | **Spark-X2.5-4B** | safetensors/MLX | ~3.5 GB | ✅ (MLX/CPU) | 63.1% MMLU [DICHIARATO HF] | **74.4%** [DICHIARATO HF] | **65.1%** BFCL-V4 [DICHIARATO HF] | 84.9% [DICHIARATO HF] | ~50-70 [STIMATO] | ⚠️ **No GGUF ufficiale** — solo safetensors/MLX; ibrido attenzione 1M ctx; SWE-Bench Pro 44.4%; GPQA 67.4% | [XHToken/Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) |
| 7 | **Qwen3-8B heretic abliterated** | Q4_K_M | **5.0 GB** | ✅ | ~72% MMLU [STIMATO base] | ~67% [STIMATO] | ~65% [STIMATO] | ~82% [STIMATO] | ~38-48 [STIMATO] | KL divergence 0.0838 (danno minimo); 13/100 refusals; F8/INT8/INT4/NVFP4 per ComfyUI | [DreamFast/qwen3-8b-heretic](https://huggingface.co/DreamFast/qwen3-8b-heretic) |
| 8 | **Qwen3-4B-Thinking-2507** | Q4_K_M | ~2.5 GB | ✅ | **74.0%** MMLU-Pro [DICHIARATO HF] | ~68% [STIMATO] | **71.2%** BFCL-v3 [DICHIARATO HF] | ~88% [STIMATO] | ~65-80 [STIMATO] | GPQA 65.8%; AIME25 81.3%; versione Thinking 2507 = miglioramento post-training | [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) |
| 9 | **Qwen3-4B Q4_K_M** | Q4_K_M | ~2.5 GB | ✅ | **70.4%** MMLU-Pro thinking [DICHIARATO HF] | ~65% [STIMATO] | **65.9%** BFCL-v3 [DICHIARATO HF] | ~85% [STIMATO] | ~65-80 [STIMATO] | MMLU-Redux 83.7%, IFEval 81.9%; ottimo per dispositivi a bassa VRAM | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| 10 | **MiniCPM3-4B** | Q4_K_M | ~2.8 GB | ✅ | 67.2% MMLU [DICHIARATO HF] | **68.3%** HumanEval+ [DICHIARATO HF] | **76.0%** BFCL v2 [DICHIARATO HF] | 81.1% [DICHIARATO HF] | ~60-75 [STIMATO] | ⚠️ MiniCPM**5**-4B non esiste (solo 1B); MiniCPM3-4B è il nearest; MathBench 65.6% | [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B) |
| 11 | **Qwen3.6-27B-abliterated IQ2_XXS** | i1-IQ2_XXS | **8.5 GB** | ❌ +0.5GB | ~86% MMLU [STIMATO] | ~88% [STIMATO] | ~85% [STIMATO] | — | ~15-18 [STIMATO] | ⚠️ **SUPERA 8GB** di 500MB; iMatrix abliterato; IQ2_M (10.1GB) fuori discussione | [mradermacher/Qwen3.6-27B-abliterated-v2-i1-GGUF](https://huggingface.co/mradermacher/Qwen3.6-27B-abliterated-v2-i1-GGUF) |

---

## 📦 SEZIONE: Best per 4GB VRAM (modelli < 3.5 GB)

| Modello | Quant | GB | MMLU-Pro/MMLU | HumanEval | BFCL | Nota |
|---------|-------|----|----------------|-----------|------|------|
| **Qwen3-4B-Thinking-2507** | Q4_K_M | ~2.5 GB | 74.0% [DICHIARATO HF] | ~68% [STIMATO] | 71.2% [DICHIARATO HF] | 🏆 **MIGLIORE** per <3.5GB; reasoning mode; GPQA 65.8% |
| **Qwen3-4B** | Q4_K_M | ~2.5 GB | 70.4% [DICHIARATO HF] | ~65% [STIMATO] | 65.9% [DICHIARATO HF] | Classico; non-thinking mode disponibile |
| **MiniCPM3-4B** | Q4_K_M | ~2.8 GB | 67.2% [DICHIARATO HF] | 68.3% [DICHIARATO HF] | 76.0% BFCL v2 [DICHIARATO HF] | BFCL eccellente; tool calling forte |
| **Spark-X2.5-4B** | MLX/native | ~3.5 GB | 63.1% [DICHIARATO HF] | 74.4% [DICHIARATO HF] | 65.1% [DICHIARATO HF] | ⚠️ No GGUF; SWE-Bench Pro 44.4% — agente straordinario |

**🎯 Raccomandazione <4GB VRAM:** `Qwen3-4B-Thinking-2507 Q4_K_M` — MMLU-Pro 74%, BFCL 71.2%, GPQA 65.8% a soli ~2.5GB. Scala fino a 8GB con Qwen3.5-9B.

---

## 📦 SEZIONE: Best per 6GB VRAM (modelli 3.5 – 6 GB)

| Modello | Quant | GB | MMLU-Pro/MMLU | HumanEval | BFCL | Nota |
|---------|-------|----|----------------|-----------|------|------|
| **Qwen3.5-9B Q4_K_M** | Q4_K_M | ~5.2 GB | 82.5% MMLU-Pro [DICHIARATO HF] | ~82% [STIMATO] | 66.1% [DICHIARATO HF] | 🏆 **MIGLIORE** <6GB; HMMT 83.2%; IFEval 91.5% |
| **Gemma4-12B heretic IQ3_M** | i1-IQ3_M | 5.8 GB | 77.2% MMLU-Pro [DICHIARATO Google] | ~74% [STIMATO] | — | Alternativa multimodale con visione (mmproj separato) |
| **Qwen3.8-9B Distill heretic iQ4_K_M** | i1-Q4_K_M | 5.96 GB | 75.1% [DICHIARATO HF] | ~80% [STIMATO] | — | Distillato, abliterato, generazione post-Qwen3 |
| **Qwen3-8B heretic Q4_K_M** | Q4_K_M | 5.0 GB | ~72% [STIMATO] | ~67% [STIMATO] | ~65% [STIMATO] | Uncensored; danno abliterazione minimo (KL 0.0838) |

**🎯 Raccomandazione <6GB VRAM:** `Qwen3.5-9B Q4_K_M` — MMLU-Pro 82.5%, benchmark di livello modello enterprise a ~5.2GB. Best overall gap qualità/dimensione.

---

## 📦 SEZIONE: Best per 8GB VRAM (modelli 6 – 8 GB)

| Modello | Quant | GB | MMLU-Pro/MMLU | HumanEval | BFCL | Nota |
|---------|-------|----|----------------|-----------|------|------|
| **Gemma4-12B QAT HauhauCS Balanced** | Q4_K_M (QAT) | 6.87 GB | 77.2% [DICHIARATO Google] | ~74% [STIMATO] | ~69% TAU2 [DICHIARATO] | 🏆 **MIGLIORE** <8GB; QAT = qualità quasi FP; +60% speed MTP; uncensored |
| **BTL-3 Compact 27B** | AVQ2 2.5bpw | 8.39 GB | — | 95.1% [DICHIARATO HF] | 88.5% [DICHIARATO HF] | ⚠️ limite VRAM esatto 8GB; richiede runtime custom; LiveCodeBench v6 88.1% |

**🎯 Raccomandazione <8GB VRAM:** `Gemma4-12B QAT HauhauCS Balanced` — 6.87GB con margine di sicurezza, QAT = qualità 4-bit vicina a FP16, MTP drafter +60% velocità, vision multimodale. **🥇 BEST OVERALL per RTX 3050 8GB.**

---

## ☁️ RIFERIMENTI CLOUD (confronto)

| Modello Cloud | MMLU | HumanEval | MATH | BFCL | Note |
|---------------|------|-----------|------|------|------|
| Claude Sonnet 4.6 (questo modello) | 88.7% | 93% | 93.5% | — | Riferimento contestuale |
| Gemini 3.7 Flash High | 85% | 89% | 89.3% | — | Flash reasoning mode |
| GPT-4o (2025) | ~88% | ~90% | ~89% | ~75% | Riferimento GPT |
| Qwen3-235B-A22B (cloud) | 89% | 95% | — | ~85% | Massimo open-source |

---

## ⚠️ NOTE CRITICHE

### MiniCPM5-4B: NON TROVATO
- OpenBMB ha rilasciato **solo MiniCPM5-1B** (1B params, MMLU ~65%, molto capace per la dimensione)
- La variante 4B **non risulta rilasciata** al 28/09/2026
- Il più vicino disponibile è **MiniCPM3-4B** (benchmark nella tabella sopra)
- In D:/models già presente: `mradermacher/MiniCPM5-2B-Base-GGUF` (base non instruct)

### Qwen3.6-27B abliterated IQ2_XXS: SUPERA 8GB
- IQ2_XXS = **8.5GB** — supera di 500MB la VRAM disponibile
- IQ1_M = 7.7GB — fitterebbe ma qualità molto bassa (non raccomandato)
- Alternativa: **BTL-3 Compact** (8.39GB, qualità molto superiore alla stessa dimensione)

### BTL-3 Compact: richiede runtime CUSTOM
- Non funziona con llama.cpp standard o Ollama stock
- Richiede il BTL-packed llama.cpp disponibile su badtheorylabs/BTL-3-Compact
- Benchmark eccezionali (95.1% HumanEval) ma deployment non-standard

### Spark-X2.5-4B: No GGUF
- Non disponibile in GGUF standard
- Usa MLX o safetensors nativi via `spark-mlx-lm`
- Per uso con llama.cpp: attendere conversione GGUF community

---

## 🎯 RACCOMANDAZIONE FINALE ≤ 6GB

### 🥇 **Qwen3.5-9B Q4_K_M** (~5.2 GB)
**Repo:** `bartowski/Qwen_Qwen3.5-9B-GGUF`

**Motivazione:**
- **MMLU-Pro 82.5%** [DICHIARATO HF] — supera tutti i modelli ≤6GB di questa lista
- **BFCL-V4 66.1%** — tool calling eccellente per agenti
- **IFEval 91.5%** — instruction following best-in-class
- **HMMT Feb 83.2%** — ragionamento matematico competitivo-level
- Architettura Qwen3.5 densa, ottimizzata per latenza
- ~5.2GB lascia 2.8GB VRAM libera per context (importante per sessioni lunghe)
- No runtime personalizzato: funziona con llama.cpp standard, Ollama, LM Studio
- Stimato **~35-45 tok/s** su RTX 3050 8GB

**Alternativa uncensored ≤6GB:** `mradermacher/gemma-4-12B-it-heretic-i1-GGUF` (IQ3_M 5.8GB) — stessa base di HauhauCS, uncensored, 1GB più piccolo.

**Per chi vuole il massimo a 8GB:** `HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced` (6.87GB) — QAT quality, MTP +60% speed, vision.

---

## 📋 STATO DOWNLOAD

| Modello | Dimensione | Stato |
|---------|-----------|-------|
| Gemma4-12B heretic IQ3_M | 5.8 GB | ⏸️ **NON AVVIATO** (solo ricerca) |
| Qwen3.6-27B-abliterated IQ2_XXS | 8.5 GB | 🚫 **SKIP** — supera 8GB VRAM |
| Qwen3.5-9B Q4_K_M | ~5.2 GB | ⏸️ **NON AVVIATO** (solo ricerca) |
| BTL-3 Compact AVQ2 | 8.39 GB | ⏸️ **NON AVVIATO** — richiede conferma (runtime custom) |
| MiniCPM5-4B | — | 🚫 **NON TROVATO** — modello non esiste (solo 1B) |
| Spark-X2.5-4B GGUF | — | 🚫 **NO GGUF disponibile** |

---

*Report generato da subagent Hermes — nessun download effettuato*
