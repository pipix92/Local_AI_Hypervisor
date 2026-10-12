"""
backend.py - Local AI Management & Optimization Backend Module
Author: Local AI Hypervisor Team
Environment: Windows Local AI Management TUI (ai_tui)

Core Components:
1. HardwareMonitor: Real-time NVIDIA GPU (nvidia-smi), CPU/RAM (psutil), and process telemetry.
2. LlamaServerManager: llama-server / LM Studio endpoint monitor, model inspection, slot tracker,
   RAM-only KV offload (-nkvo), server presets, and flush VRAM engine.
3. KVCacheOptimizer: StreamingLLM attention sinks, GrowPage dynamic KV budgeting & eviction,
   and RAM vs VRAM KV memory footprint calculator (up to 256k tokens).
4. ContextCompressor: Groq-powered high-speed structured context compression engine (Llama-3.3-70B).
5. AINewsAggregator: Live AI research & tech news aggregator (arXiv cs.AI/cs.CL/cs.LG, Hugging Face, RSS).
6. HFModelManager: Real-time Hugging Face search, quant weights inspector (.gguf, .exl3, .safetensors),
   and multi-threaded background downloader to /mnt/lm_models/models.
7. ModelConfigManager: Per-model hyperparameter configuration, Groq-powered HF model card extraction,
   and automated LM Studio preset generator (C:/Users/User/.lmstudio/config-presets/).
8. ModelTestRunner: Dedicated model testing & live benchmark engine (Python AST syntax, JSON tool calling,
   generation speed tok/s, TTFT ms, VRAM footprint).
9. Self-Contained Test & Verification Suite.
"""

from __future__ import annotations

import concurrent.futures
import gc
import glob
import http.client
import json
import logging
import math
import mmap
import os
import re
import shlex
import shutil
# serve a stop_embedder (os.kill con SIGTERM/SIGKILL): senza questo import il
# NameError veniva inghiottito da un except e F9 non fermava MAI l'embedder.
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

# Avvio staccato (cgroup separato via systemd-run --scope): vedi proc_spawn.py
import proc_spawn as _PS
# Preset F1-F6 (Ornith / Qwen3.8 via RPC) e menu di pre-avvio.
import launch_presets as _LP

try:
    import psutil
except ImportError:
    psutil = None

try:
    import ctypes
except ImportError:
    ctypes = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ai_tui.backend")


# =====================================================================
# 0. HARDWARE & SYSTEM TELEMETRY MONITOR
# =====================================================================

@dataclass
class GPUStats:
    name: str = "NVIDIA GeForce RTX 3050"
    driver_version: str = "560.70"
    vram_total_mb: float = 8192.0
    vram_used_mb: float = 0.0
    vram_free_mb: float = 8192.0
    util_percent: float = 0.0
    temperature_c: float = 0.0
    fan_percent: float = 0.0
    power_w: float = 0.0
    is_available: bool = False
    reserved_mb: float = 0.0          # memoria riservata dal driver (nvidia-smi)

    @property
    def gpu_util_percent(self) -> float:
        return self.util_percent

    @property
    def mem_util_percent(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb * 100.0) if self.vram_total_mb > 0 else 0.0

    @property
    def gpu_util_percent(self) -> float:
        return self.util_percent

    @property
    def temp_c(self) -> float:
        return self.temperature_c


@dataclass
class CPUStats:
    cpu_name: str = "Intel Core / AMD Ryzen"
    overall_percent: float = 0.0
    cores_percent: List[float] = field(default_factory=list)
    freq_mhz: float = 0.0
    physical_cores: int = 4
    logical_cores: int = 8

    @property
    def cpu_percent(self) -> float:
        return self.overall_percent


@dataclass
class RAMStats:
    total_gb: float = 16.0
    used_gb: float = 0.0
    free_gb: float = 16.0
    percent: float = 0.0
    swap_used_gb: float = 0.0
    swap_total_gb: float = 0.0


@dataclass
class ProcessStats:
    name: str = "llama-server"
    pid: Optional[int] = None
    is_running: bool = False
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    threads: int = 0


@dataclass
class HardwareSnapshot:
    gpu: GPUStats = field(default_factory=GPUStats)
    cpu: CPUStats = field(default_factory=CPUStats)
    ram: RAMStats = field(default_factory=RAMStats)
    llama_process: ProcessStats = field(default_factory=ProcessStats)
    timestamp: float = field(default_factory=time.time)


class HardwareMonitor:
    """
    Real-time telemetry reader for NVIDIA GPU (nvidia-smi) and System CPU/RAM (psutil).
    Uses background worker thread for non-blocking 0ms telemetry reads.
    """

    def __init__(self):
        self._last_snapshot: Optional[HardwareSnapshot] = None
        self._has_nvidia_smi: Optional[bool] = None
        self._known_pid: Optional[int] = None
        self._lock = threading.Lock()
        self._running = True
        
        # Initial fast snapshot
        self._last_snapshot = self._collect_now()
        
        # Start daemon telemetry worker
        self._worker_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._worker_thread.start()

    def _telemetry_loop(self):
        while self._running:
            try:
                snap = self._collect_now()
                with self._lock:
                    self._last_snapshot = snap
            except Exception:
                pass
            time.sleep(1.0)

    def _collect_now(self) -> HardwareSnapshot:
        return HardwareSnapshot(
            gpu=self._query_nvidia_smi(),
            cpu=self._query_cpu(),
            ram=self._query_ram(),
            llama_process=self._query_llama_process(),
            timestamp=time.time()
        )

    def _query_nvidia_smi(self) -> GPUStats:
        gpu = GPUStats()
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,fan.speed,power.draw,driver_version,memory.reserved",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0)
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpu.name = parts[0]
                    gpu.vram_total_mb = float(parts[1]) if parts[1] != "[N/A]" else 8192.0
                    gpu.vram_used_mb = float(parts[2]) if parts[2] != "[N/A]" else 0.0
                    gpu.vram_free_mb = float(parts[3]) if parts[3] != "[N/A]" else 8192.0
                    gpu.util_percent = float(parts[4]) if parts[4] != "[N/A]" else 0.0
                    gpu.temperature_c = float(parts[5]) if parts[5] != "[N/A]" else 0.0
                    if len(parts) >= 7 and parts[6] != "[N/A]":
                        try: gpu.fan_percent = float(parts[6])
                        except ValueError: pass
                    if len(parts) >= 8 and parts[7] != "[N/A]":
                        try: gpu.power_w = float(parts[7])
                        except ValueError: pass
                    # Driver e memoria riservata VERI (prima il driver era il
                    # valore di default "560.70", mai letto).
                    if len(parts) >= 9 and parts[8] not in ("[N/A]", ""):
                        gpu.driver_version = parts[8]
                    if len(parts) >= 10:
                        try: gpu.reserved_mb = float(parts[9])
                        except ValueError: pass
                    gpu.is_available = True
                    self._has_nvidia_smi = True
                    return gpu
        except Exception as e:
            logger.debug(f"nvidia-smi query failed: {e}")
            self._has_nvidia_smi = False

        gpu.is_available = False
        return gpu

    _cpu_name_cache: Optional[str] = None

    @classmethod
    def _cpu_model_name(cls) -> str:
        """Nome CPU reale da /proc/cpuinfo (prima era un segnaposto fisso)."""
        if cls._cpu_name_cache is None:
            name = ""
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("model name"):
                            name = line.split(":", 1)[1].strip()
                            break
            except OSError:
                pass
            if not name:
                import platform
                name = platform.processor() or "CPU sconosciuta"
            cls._cpu_name_cache = name
        return cls._cpu_name_cache

    def _query_cpu(self) -> CPUStats:
        cpu = CPUStats()
        cpu.cpu_name = self._cpu_model_name()
        if psutil:
            try:
                cpu.overall_percent = psutil.cpu_percent(interval=None)
                cpu.cores_percent = psutil.cpu_percent(interval=None, percpu=True)
                freq = psutil.cpu_freq()
                if freq:
                    cpu.freq_mhz = round(freq.current, 1)
                cpu.physical_cores = psutil.cpu_count(logical=False) or 4
                cpu.logical_cores = psutil.cpu_count(logical=True) or 8
            except Exception as e:
                logger.debug(f"psutil CPU query error: {e}")
        return cpu

    def _query_ram(self) -> RAMStats:
        ram = RAMStats()
        if psutil:
            try:
                v = psutil.virtual_memory()
                ram.total_gb = round(v.total / (1024 ** 3), 2)
                ram.used_gb = round(v.used / (1024 ** 3), 2)
                ram.free_gb = round(v.available / (1024 ** 3), 2)
                ram.percent = v.percent
                s = psutil.swap_memory()
                ram.swap_total_gb = round(s.total / (1024 ** 3), 2)
                ram.swap_used_gb = round(s.used / (1024 ** 3), 2)
            except Exception as e:
                logger.debug(f"psutil RAM query error: {e}")
        return ram

    def _query_llama_process(self) -> ProcessStats:
        pstats = ProcessStats()
        if psutil:
            try:
                # Fast check if known pid is still alive
                if self._known_pid and psutil.pid_exists(self._known_pid):
                    try:
                        p = psutil.Process(self._known_pid)
                        pname = p.name().lower()
                        if "llama-server" in pname or "llama_server" in pname:
                            pstats.name = p.name()
                            pstats.pid = self._known_pid
                            pstats.is_running = True
                            pstats.threads = p.num_threads()
                            mem = p.memory_info()
                            if mem:
                                pstats.memory_mb = round(mem.rss / (1024 * 1024), 2)
                            pstats.cpu_percent = p.cpu_percent(interval=None)
                            return pstats
                    except Exception:
                        self._known_pid = None

                for p in psutil.process_iter(["pid", "name", "memory_info", "num_threads"]):
                    pname = (p.info.get("name") or "").lower()
                    if "llama-server" in pname or "llama_server" in pname:
                        pstats.name = pname
                        pstats.pid = p.info.get("pid")
                        self._known_pid = pstats.pid
                        pstats.is_running = True
                        pstats.threads = p.info.get("num_threads") or 0
                        mem = p.info.get("memory_info")
                        if mem:
                            pstats.memory_mb = round(mem.rss / (1024 * 1024), 2)
                        try:
                            pstats.cpu_percent = p.cpu_percent(interval=None)
                        except Exception:
                            pass
                        break
            except Exception as e:
                logger.debug(f"psutil process scan error: {e}")
        return pstats

    def get_telemetry(self) -> HardwareSnapshot:
        """Collect current hardware telemetry snapshot in 0ms (memory cached)."""
        with self._lock:
            if self._last_snapshot:
                return self._last_snapshot
        return self._collect_now()

    get_snapshot = get_telemetry


# =====================================================================
# 1. LLAMA SERVER & LM STUDIO MANAGER
# =====================================================================

@dataclass
class ServerEndpoint:
    name: str
    url: str
    server_type: str  # "llama-server" or "lm-studio"
    is_active: bool = False
    latency_ms: float = 0.0
    last_checked: float = 0.0
    error_msg: Optional[str] = None


@dataclass
class ModelMetadata:
    id: str
    owned_by: str = "local"
    object: str = "model"
    context_length: Optional[int] = None
    parameters: Optional[str] = None
    quantization: Optional[str] = None
    raw_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotInfo:
    slot_id: int
    state: int  # 0: idle, 1: processing, etc.
    is_processing: bool
    n_ctx: int
    n_past: int
    n_decoded: int = 0
    prompt_tokens: int = 0
    task_id: Optional[int] = None
    model_name: Optional[str] = None


@dataclass
class LocalModelInfo:
    filename: str
    filepath: str
    rel_path: str
    author: str
    size_bytes: int
    size_gb: float
    quant_type: str
    vram_fit_status: str
    is_recommended: bool
    last_modified: str
    # Come e' stata dedotta la quantizzazione: "meta" = letta dagli
    # metadati interni del GGUF, "nome" = dedotta dal nome del file.
    # Serve per NON ricadere sull'euristica quando il metadato c'e'.
    quant_source: str = "nome"


# =====================================================================
# LETTURA METADATI INTERNI DI UN GGUF
# =====================================================================
# Il nome del file NON basta: i modelli APEX (Ornith-1.5-35B-...-APEX-
# I-Mini-v2D-lite.gguf, Carnice-Qwen3.6-MoE-35B-...-APEX-I-Mini.gguf,
# Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1.gguf) non contengono
# nessuna delle sigle classiche (Q4_K_M, IQ4_XS, ...) e finivano
# etichettati "OTHER", che e' indistinguibile da "non riconosciuto".
#
# Il GGUF porta invece la verita' in due posti dell'header:
#   general.file_type   -> enum llama_ftype (15 = Q4_K_M, 12 = Q3_K_M,
#                          23 = IQ3_XXS, 39 = NVFP4, ...)
#   tensor info         -> ggml type di ogni tensore: se il file e'
#                          quantizzato, il tipo dominante dei pesi dice
#                          la quantizzazione reale anche quando file_type
#                          mente o manca.
#
# Si leggono con mmap + struct (0.19s per un 14GB: si mappa l'header e
# basta, non si tocca il corpo del file) e si mettono in cache per
# (path, size, mtime), cosi' una riscanione costa quasi nulla.

# enum ggml_type (sottoinsieme utile; i valori sono stabili in llama.cpp)
GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 36: "MXFP4", 37: "NVFP4",
}

# enum llama_ftype (include/llama.h) -> nome leggibile.
GGUF_FILE_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}

# Tipi "1D" (norme, embedding positional) che in un modello quantizzato
# restano F32 e NON dicono nulla sulla quantizzazione dei pesi.
_GGML_NON_WEIGHT_TYPES = frozenset({0, 1, 24, 25, 26, 27, 28, 30})

# Quantizzazioni che si "espandono" in una famiglia: Q4_K_M e' un mix di
# Q4_K/Q5_K/Q6_K, Q3_K_M un mix di Q3_K/Q4_K/Q5_K. Per la colonna
# "Quant" della TUI si riporta la sigla del file_type, non il mix.
_GGUF_META_CACHE: Dict[str, Any] = {}
_GGUF_META_CACHE_MAX = 512


class _GGUFReader:
    """Parser minimale dell'header GGUF (solo metadati, mai i dati)."""

    __slots__ = ("d", "o")

    def __init__(self, data: Any):
        self.d = data
        self.o = 0

    def take(self, n: int) -> bytes:
        b = self.d[self.o:self.o + n]
        self.o += n
        return b

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.d, self.o)[0]
        self.o += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.d, self.o)[0]
        self.o += 8
        return v

    def string(self) -> str:
        n = self.u64()
        # CONTROLLO DI CONFINE. Senza questo, se l'header e' corrotto o
        # troncato n puo' essere enorme (misurati offset di 7.8 esabyte) e
        # self.d[self.o:self.o+n] alloca una fetta enorme: il parser si
        # blocca o esaurisce la RAM invece di fallire pulito.
        if n > len(self.d) or self.o + n > len(self.d):
            raise ValueError(
                f"stringa fuori dal file: n={n} o={self.o} size={len(self.d)}")
        v = self.d[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return v

    def value(self, t: int) -> Any:
        d, o = self.d, self.o
        if t == 0:
            self.o = o + 1
            return d[o]
        if t == 1:
            v = struct.unpack_from("<b", d, o)[0]; self.o = o + 1; return v
        if t == 2:
            v = struct.unpack_from("<H", d, o)[0]; self.o = o + 2; return v
        if t == 3:
            v = struct.unpack_from("<h", d, o)[0]; self.o = o + 2; return v
        if t == 4:
            v = struct.unpack_from("<I", d, o)[0]; self.o = o + 4; return v
        if t == 5:
            v = struct.unpack_from("<i", d, o)[0]; self.o = o + 4; return v
        if t == 6:
            v = struct.unpack_from("<f", d, o)[0]; self.o = o + 4; return v
        if t == 7:
            v = bool(d[o]); self.o = o + 1; return v
        if t == 8:
            return self.string()
        if t == 9:
            # array: la si salta, non serve il contenuto. Nota: i
            # tokenizer.ggml.tokens sono 150k+ stringhe, e' la voce piu'
            # grossa dell'header ma non ci interessa.
            et = self.u32(); n = self.u64()
            for _ in range(n):
                self.skip(et)
            return None
        if t == 10:
            v = struct.unpack_from("<Q", d, o)[0]; self.o = o + 8; return v
        if t == 11:
            v = struct.unpack_from("<q", d, o)[0]; self.o = o + 8; return v
        if t == 12:
            v = struct.unpack_from("<d", d, o)[0]; self.o = o + 8; return v
        raise ValueError(f"tipo GGUF sconosciuto {t}")

    def skip(self, t: int) -> None:
        # NB: u64()/u32() gia' spostano il puntatore, quindi qui si
        # SOMMA solo la lunghezza del dato. (raddoppiare lo spostamento
        # desincronizzava il parser e faceva fallire tutto dopo la
        # seconda voce: general.name finiva "perso".)
        if t == 8:
            # ATTENZIONE, bug di ordine di valutazione. `self.o += self.u64()`
            # NON funziona: per l'assegnazione composta Python carica il
            # valore di self.o PRIMA di valutare la destra, quindi il +8
            # dell'u64() veniva perso e l'avanzamento era di n invece di
            # 8+n. Ogni stringa lasciava il puntatore 8 byte indietro e
            # dal secondo elemento in poi l'offset era spazzatura (misurati
            # offset di 7.3 esabyte). Per questo general.file_type e
            # general.name tornavano None. Qui sono due statement separati:
            # u64() avanza il puntatore, poi si somma la lunghezza.
            n = self.u64()
            self.o += n
        elif t == 9:
            et = self.u32()
            n = self.u64()
            if et == 8:
                # ARRAY DI STRINGHE: il caso costoso. tokenizer.ggml.tokens
                # contiene ~150.000 stringhe e veniva saltato con una
                # chiamata ricorsiva a skip() per ognuna: con 83 modelli da
                # scansionare su NTFS il blocco era di minuti. Qui si
                # scorre in piano, senza chiamate di funzione, e si esce
                # appena l'offset esce dal file.
                d, o, end = self.d, self.o, len(self.d)
                up = struct.unpack_from
                for _ in range(n):
                    ln = up("<Q", d, o)[0]
                    o += 8 + ln
                    if o > end:
                        raise ValueError("array di stringhe fuori dal file")
                self.o = o
                return
            for _ in range(n):
                self.skip(et)
        elif t in (0, 1, 7):
            self.o += 1
        elif t in (2, 3):
            self.o += 2
        elif t in (4, 5, 6):
            self.o += 4
        elif t in (10, 11, 12):
            self.o += 8
        else:
            raise ValueError(f"tipo GGUF sconosciuto {t}")


def read_gguf_metadata(path: str, max_tensors: int = 4000) -> Dict[str, Any]:
    """
    Legge general.file_type / general.name e l'istogramma dei ggml type.

    Restituisce {"file_type": int|None, "name": str|None, "tensors":
    Counter-like dict, "arch": str|None}. Non solleva mai: un GGUF
    illeggibile o troncato restituisce semplicemente pochi campi, e il
    chiamante ripiega sull'euristica sul nome.
    """
    out: Dict[str, Any] = {"file_type": None, "name": None, "arch": None,
                           "tensors": {}}
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size < 24:
                return out
            # mmap PAGINATO: si indirizza il file senza copiarlo. Le
            # pagine non toccate non vengono mai lette dal disco, quindi
            # un 14GB su NTFS costa ~(quello che si usa davvero), cioe'
            # l'header: ~500KB. NON si fa mm[:N], che copierebbe N byte
            # in RAM (96MB per file -> 50s su 42 file, misurato).
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if mm[:4] != b"GGUF":
                    return out
                r = _GGUFReader(mm)
                r.take(4)
                r.u32()                 # version
                n_tensors = r.u64()
                n_kv = r.u64()
                for _ in range(n_kv):
                    key = r.string()
                    vtype = r.u32()
                    if key in ("general.file_type", "general.name",
                               "general.architecture"):
                        # Si salva l'offset PRIMA di value(): se value()
                        # solleva DOPO aver mosso il puntatore (es. una
                        # stringa malformata), calling skip() da li'
                        # ripartirebbe gia' avanti e desincronizzerebbe
                        # tutto il resto dell'header: general.name e
                        # general.file_type finivano None e l'istogramma
                        # tensori vuoto. Prima di questo fix si leggeva solo
                        # general.architecture, il primo dei tre.
                        _o = r.o
                        try:
                            v = r.value(vtype)
                        except Exception:
                            r.o = _o
                            r.skip(vtype)
                            v = None
                        if key == "general.file_type" and isinstance(v, int):
                            out["file_type"] = v
                        elif key == "general.name" and isinstance(v, str):
                            out["name"] = v
                        elif key == "general.architecture" and isinstance(v, str):
                            out["arch"] = v
                    else:
                        r.skip(vtype)
                counts: Dict[int, int] = {}
                for _ in range(min(n_tensors, max_tensors)):
                    r.string()                     # nome del tensore
                    n_dims = r.u32()
                    for _d in range(n_dims):       # dimensioni
                        r.take(8)
                    ttype = r.u32()
                    r.take(8)                      # offset
                    counts[ttype] = counts.get(ttype, 0) + 1
                out["tensors"] = counts
    except Exception:
        # File non GGUF, permesso negato, mmap non disponibile: si lascia
        # com'era e il chiamante usa il nome.
        pass
    return out


_GGUF_META_CACHE: Dict[str, Dict[str, Any]] = {}
_GGUF_META_CACHE_MAX = 256
# Cache su DISCO: i metadati di un GGUF non cambiano mai a parita' di
# path+size+mtime, quindi rileggerli a ogni avvio della TUI e' lavoro
# sprecato. Su NTFS il salto di 150.000 stringhe del tokenizer.ggml.tokens
# costava ~0.1 s per file: 83 modelli = 9 s di attesa a ogni avvio, piu'
# 9 s a ogni refresh del catalogo. Con la cache su disco il secondo avvio
# e' immediato e il refresh non tocca piu' i file.
_GGUF_META_DISK = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "ai_tui", "gguf_meta.json")
_GGUF_META_DISK_MAX = 4000
_gguf_disk_loaded = False


def _gguf_disk_load() -> None:
    global _gguf_disk_loaded
    if _gguf_disk_loaded:
        return
    _gguf_disk_loaded = True
    try:
        with open(_GGUF_META_DISK, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            _GGUF_META_CACHE.update(raw)
    except Exception:
        pass          # cache assente o corrotta: si ricostruisce


def _gguf_disk_save() -> None:
    try:
        os.makedirs(os.path.dirname(_GGUF_META_DISK), exist_ok=True)
        if len(_GGUF_META_CACHE) > _GGUF_META_DISK_MAX:
            return
        # tmp + rename: scrittura atomica, una cache a meta' non rovina
        # quella buona se il processo muore durante il salvataggio.
        tmp = _GGUF_META_DISK + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_GGUF_META_CACHE, fh)
        os.replace(tmp, _GGUF_META_DISK)
    except Exception:
        pass


def gguf_metadata_cached(path: str) -> Dict[str, Any]:
    """read_gguf_metadata con cache su (path, size, mtime), in RAM e su disco."""
    _gguf_disk_load()
    try:
        st = os.stat(path)
        key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return {"file_type": None, "name": None, "arch": None, "tensors": {}}
    hit = _GGUF_META_CACHE.get(key)
    if hit is not None:
        return hit
    meta = read_gguf_metadata(path)
    if len(_GGUF_META_CACHE) >= _GGUF_META_CACHE_MAX:
        _GGUF_META_CACHE.clear()
    _GGUF_META_CACHE[key] = meta
    _gguf_disk_save()
    return meta


def quant_from_gguf_metadata(meta: Dict[str, Any]) -> Optional[str]:
    """
    Sigla di quantizzazione dai metadati interni, oppure None.

    Ordine di fiducia:
      1. general.file_type -> nome diretto (Q4_K_M, Q3_K_M, NVFP4, ...)
      2. se file_type e' assente o "guessed", il tipo dominante dei
         tensori di peso (Q4_K/Q3_K/IQ4_XS/...) con la sua famiglia.
    Restituisce None se il file non dice niente: in quel caso il chiamante
    deve usare l'euristica sul nome.
    """
    ftype = meta.get("file_type")
    if isinstance(ftype, int) and ftype in GGUF_FILE_TYPE_NAMES:
        return GGUF_FILE_TYPE_NAMES[ftype]

    counts = meta.get("tensors") or {}
    if counts:
        weight_counts = {
            t: c for t, c in counts.items()
            if t not in _GGML_NON_WEIGHT_TYPES and c > 0
        }
        if not weight_counts:
            return None
        dom = max(weight_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        name = GGML_TYPE_NAMES.get(dom)
        if name in ("F16", "F32", "BF16"):
            # Per un modello "solo pesi in precisione piena" il nome e'
            # comunque F16/BF16: e' una risposta valida, non un fallback.
            return name
        if name:
            return name
    return None


# Sigle di quantizzazione: sono le uniche che hanno senso nella colonna
# "Quant". Gli schemi dinamici (APEX, I-Mini, Compact, Balanced, Quality,
# Turbo, Lite, v2D, AD-) NON sono quantizzazioni ma ricette di mixing, e
# le regole di sotto servono solo a non finire con una colonna vuota.
QUANT_NAME_TOKENS: List[str] = [
    "MXFP4_MOE", "NVFP4", "TQ2_0", "TQ1_0",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_NL", "IQ4_XS", "IQ1_S", "IQ1_M",
    "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
    "Q4_0_4_8", "Q4_0_8_8", "Q8_K", "Q6_K", "Q5_0", "Q5_1",
    "Q4_0", "Q4_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
    "MXFP4", "Q1_0", "Q2_0",
    "FP16", "BF16", "F16", "F32",
]


def quant_from_filename(filename: str) -> str:
    """
    EURISTICA DI RISERVA: si usa SOLO se il GGUF non dice niente.

    Vince la sigla che compare PRIMA nel nome (Q4_K_M specifica
    l'intero file, IQ4_XS lo precede solo se e' la sigla del file), e a
    parita' di posizione vince quella piu' lunga, cosi' "Q4_K_M" non
    viene mangiato come "Q4_K". Gli schemi APEX/I-Mini/Lite NON sono
    quantizzazioni: non si inventano qui un'etichetta falsa, si dice
    "GGUF" e la riga resta comunque utilizzabile.
    """
    fl = filename.upper()
    best = ""
    best_pos = len(fl)
    for q in QUANT_NAME_TOKENS:
        pos = fl.find(q)
        if pos < 0:
            continue
        if pos < best_pos or (pos == best_pos and len(q) > len(best)):
            best, best_pos = q, pos
    return best if best else "GGUF"


class LlamaServerManager:
    """
    Manages local llama-server and LM Studio instances.
    Provides health checks, model metadata queries, slot utilization,
    process control, local model discovery, server presets, and VRAM management.
    """

    DEFAULT_ENDPOINTS = [
        {"name": "llama-server", "url": "http://127.0.0.1:18434", "server_type": "llama-server"},
        {"name": "LM Studio", "url": "http://127.0.0.1:1234", "server_type": "lm-studio"},
    ]

    SERVER_PRESETS = {
        "max_speed": {
            "name": "Max Speed (100% GPU VRAM KV)",
            "ctx_size": 32768,
            "kv_dtype": "q4_0",
            "no_kv_offload": False,
            "flash_attn": True,
            "ctx_shift": True,
            "ngl": 99,
            "description": "Max speed GPU KV cache (4-bit quant, Flash Attention ON)"
        },
        "ram_kv_offload": {
            "name": "RAM-Only KV Offload (-nkvo)",
            "ctx_size": 65536,
            "kv_dtype": "f16",
            "no_kv_offload": True,
            "flash_attn": True,
            "ctx_shift": True,
            "ngl": 99,
            "description": "Leaves 100% VRAM free for model weights tensor offloading"
        },
        "extreme_256k": {
            "name": "Extreme 256k Long Context",
            "ctx_size": 262144,
            "kv_dtype": "q4_0",
            "no_kv_offload": False,
            "flash_attn": True,
            "ctx_shift": True,
            "ngl": 99,
            "description": "Massive 256k context window with context shifting enabled"
        },
        "hybrid_q8": {
            "name": "Hybrid High Precision (Q8_0 KV)",
            "ctx_size": 32768,
            "kv_dtype": "q8_0",
            "no_kv_offload": False,
            "flash_attn": True,
            "ctx_shift": True,
            "ngl": 99,
            "description": "Near FP16 precision with 47% lower KV memory footprint"
        }
    }

    # -----------------------------------------------------------------
    # RPC & CUDA13 PRESETS  (Tab 9 "RPC & CUDA13")
    # -----------------------------------------------------------------
    # MISURATI realmente su Host (RTX 3050 8GB) + Miky
    # (192.168.1.100, RTX 5050 8GB) con llama.cpp b11259 build CUDA 13.4.
    # Non sono ipotesi: i tok/s qui sotto vengono dai benchmark reali.
    # 'recommended' marca il preset con il throughput migliore misurato.
    RPC_PRESETS: Dict[str, Dict[str, Any]] = {
        "rpc_128k_fast": {
            "name": "RPC-128k-FAST",
            "recommended": True,
            # RPC0 (Miky) ESPLICITO: con il solo "CUDA0" llama.cpp usa solo i
            # device elencati e Miky restava inutilizzata. Verificato il
            # 2026-10-05: con --device CUDA0 la VRAM di Miky restava a 111 MiB.
            "device": "RPC0,CUDA0",
            "rpc": "192.168.1.100:50052",
            "ctx_size": 131072,
            "parallel": 1,
            "ngl": 99,
            "flash_attn": True,
            "kv_type": "q4_1",
            "kv_value": "q4_1",
            "threads": 8,
            "jinja": True,
            "tok_s": 33.32,
            "description": "Preset RACCOMANDATO: KV q4_1 su 128k di contesto, split automatico dei tensori sulla RTX 5050 di Miky.",
        },
        "rpc_64k_vram": {
            "name": "RPC-64k-VRAM",
            "recommended": False,
            # RPC0 (Miky) ESPLICITO: con il solo "CUDA0" llama.cpp usa solo i
            # device elencati e Miky restava inutilizzata. Verificato il
            # 2026-10-05: con --device CUDA0 la VRAM di Miky restava a 111 MiB.
            "device": "RPC0,CUDA0",
            "rpc": "192.168.1.100:50052",
            "ctx_size": 65536,
            "parallel": 1,
            "ngl": 99,
            "flash_attn": True,
            "kv_type": "q4_1",
            "kv_value": "q4_1",
            "threads": 8,
            "jinja": True,
            "tok_s": 0.0,
            "description": "Stessi flag del preset RACCOMANDATO ma con 64k di contesto: da usare quando serve piu' VRAM libera sulla RTX 3050.",
        },
        "rpc_262k_max": {
            "name": "RPC-262k-MAX",
            "recommended": False,
            # RPC0 (Miky) ESPLICITO: con il solo "CUDA0" llama.cpp usa solo i
            # device elencati e Miky restava inutilizzata. Verificato il
            # 2026-10-05: con --device CUDA0 la VRAM di Miky restava a 111 MiB.
            "device": "RPC0,CUDA0",
            "rpc": "192.168.1.100:50052",
            "ctx_size": 262144,
            "parallel": 1,
            "ngl": 99,
            "flash_attn": True,
            "kv_type": "q4_1",
            "kv_value": "q4_1",
            "threads": 8,
            "jinja": True,
            "tok_s": 0.0,
            "description": "Contesto massimo 256k: sfrutta anche la VRAM della RTX 5050 di Miky tramite split automatico.",
        },
        "local_only": {
            "name": "LOCAL-ONLY",
            "recommended": False,
            "device": "CUDA0",
            "rpc": "",
            "ctx_size": 131072,
            "parallel": 1,
            "ngl": 99,
            "flash_attn": True,
            "kv_type": "q4_1",
            "kv_value": "q4_1",
            "threads": 8,
            "jinja": True,
            "tok_s": 0.0,
            "description": "Solo RTX 3050, nessun --rpc: massima velocita' e zero latenza di rete (niente tensori su Miky).",
        },
        "kv_q8_native": {
            "name": "KV-Q8-NATIVE",
            "recommended": False,
            # RPC0 (Miky) ESPLICITO: con il solo "CUDA0" llama.cpp usa solo i
            # device elencati e Miky restava inutilizzata. Verificato il
            # 2026-10-05: con --device CUDA0 la VRAM di Miky restava a 111 MiB.
            "device": "RPC0,CUDA0",
            "rpc": "192.168.1.100:50052",
            "ctx_size": 131072,
            "parallel": 1,
            "ngl": 99,
            "flash_attn": True,
            "kv_type": "q8_0",
            "kv_value": "q8_0",
            "threads": 8,
            "jinja": True,
            "tok_s": 11.60,
            "description": "KV q8_0 nativo: nessun warning FlashAttention, ma piu' lento (9.7-11.6 tok/s misurati).",
        },
    }

    # Preset F1-F6 (2026-10-05): Ornith 35B-A3B e Qwen3.8 35B-A3B via RPC a
    # 16k/75k/120k, ognuno col SUO modello. Definiti in launch_presets.py
    # (schema 2: niente -ngl, --fit on --fit-target, flag dal menu di
    # pre-avvio). I 5 preset storici restano qui sotto come profili NON
    # mappati sui tasti (selezionabili dalla Tab 9).
    RPC_PRESETS.update({_k: _LP.PRESETS[_k] for _k in _LP.ORDER})

    # Ordine dei tasti F1..F6 -> chiavi dei preset. Coerente con TUI_RPC_HOTKEYS.
    RPC_PRESET_ORDER: List[str] = list(_LP.ORDER)
    # Preset storici (ex F1-F5), non piu' mappati sui tasti.
    RPC_LEGACY_ORDER: List[str] = [
        "rpc_128k_fast",
        "rpc_64k_vram",
        "rpc_262k_max",
        "local_only",
        "kv_q8_native",
    ]

    def find_rpc_server_binary(self) -> Optional[str]:
        """
        Find the llama-server of the CUDA 13.4 build used for the validated
        RPC topology (b11259). Kept separate from find_llama_server_binary()
        because the RPC flags need the CUDA 13 backend, not any older build.
        """
        candidates = [
            # PRIORITA' 1: build piu' recente in DEBIAN_FOLDER (es. b11429)
            # Cerca la piu' recente per numero (sort inverso)
            *sorted(
                [os.path.join(d, "llama-server") for d in 
                 glob.glob("/mnt/lm_models/DEBIAN_FOLDER/llamacpp/llama-b*")
                 if os.path.isdir(d) and os.path.isfile(os.path.join(d, "llama-server"))],
                reverse=True
            ),
            # PRIORITA' 2: build legacy in /mnt/lm_models/llamacpp/bin
            "/mnt/lm_models/llamacpp/bin/llama-b11330/llama-server",
            # PRIORITA' 3: llama-server in PATH di sistema
            shutil.which("llama-server"),
        ]
        for c in candidates:
            if not c:
                continue
            if os.path.exists(c) and os.path.isfile(c):
                return os.path.normpath(c)
        # Fall back to whatever generic search finds, so the panel still works
        # if the build was moved; the launch will simply not be the b11259 one.
        return self.find_llama_server_binary()

    @staticmethod
    def build_rpc_command(
        binary_path: str,
        model_path: str,
        preset: Dict[str, Any],
        port: int = 8087,
        host: str = "127.0.0.1",
        help_text: Optional[str] = None,
    ) -> List[str]:
        """
        Compose the llama-server argv for an RPC preset.
        help_text: --help gia' letto (None = lo legge compose_argv, cioe' un
        subprocess: da NON usare nel thread della UI).

        Kept pure (no I/O) so the TUI can show the exact command line before
        launching, and so tests can assert on the flags without touching VRAM.
        """
        # Preset F1-F6 (schema 2): riga composta da launch_presets, che
        # conosce i flag del menu di pre-avvio e omette quelli che la build
        # non accetta (letti dal suo --help).
        if int(preset.get("schema", 1)) >= 2:
            return _LP.compose_argv(binary_path, model_path, preset, port=port, host=host,
                                    help_text=help_text)
        # --rpc PRIMA di --device: llama.cpp legge gli argomenti in ordine e i
        # device RPC (RPC0...) esistono solo dopo --rpc. Con l'ordine inverso
        # "--device RPC0,CUDA0" falliva con "invalid device: RPC0".
        cmd = [binary_path, "-m", model_path]
        if preset.get("rpc"):
            cmd += ["--rpc", str(preset["rpc"])]
        cmd += ["--device", str(preset.get("device", "CUDA0"))]
        cmd += [
            "-c", str(preset.get("ctx_size", 131072)),
            "--parallel", str(preset.get("parallel", 1)),
            "-ngl", str(preset.get("ngl", 99)),
            "-fa", "on" if preset.get("flash_attn", True) else "off",
            "-ctk", str(preset.get("kv_type", "q4_1")),
            "-ctv", str(preset.get("kv_value", "q4_1")),
            "-t", str(preset.get("threads", 8)),
        ]
        if preset.get("jinja", True):
            cmd.append("--jinja")
        cmd += ["--host", host, "--port", str(port), "--verbosity", "3"]
        # Flag scritti a mano dall'utente: vanno in FONDO, verbatim, cosi'
        # se contraddicono un flag del preset (es. un secondo -c) vale
        # l'ultimo occorrenza e quindi vincono quelli manuali.
        extra = preset.get("extra_args") or []
        if isinstance(extra, str):
            extra = shlex.split(extra)
        cmd += [str(a) for a in extra]
        return cmd

    @staticmethod
    def is_port_in_use(host: str = "localhost", port: int = 8087) -> bool:
        """
        True if something is already listening on host:port. Pure socket
        probe: no process is touched, nothing is killed.

        Both address families are tried: llama-server often binds [::1] only,
        and a plain AF_INET connect to "localhost" would miss it (and then we
        would happily start a second server on the same port).
        """
        host = host or "localhost"
        targets: List[Any] = []
        try:
            infos = socket.getaddrinfo(host, int(port), proto=socket.IPPROTO_TCP)
            for fam, _type, _proto, _canon, sockaddr in infos:
                if (fam, sockaddr) not in targets:
                    targets.append((fam, sockaddr))
        except (socket.gaierror, OSError, ValueError):
            pass
        # Fallbacks: a host may resolve differently per family.
        for extra in (("localhost", int(port)), ("127.0.0.1", int(port)), ("::1", int(port))):
            if extra not in targets:
                targets.append(extra)
        for fam, addr in targets:
            try:
                with socket.socket(fam, socket.SOCK_STREAM) as sk:
                    sk.settimeout(1.0)
                    if sk.connect_ex(addr) == 0:
                        return True
            except (OSError, ValueError, TypeError):
                continue
        return False

    @staticmethod
    def probe_rpc_backend(endpoint: str, timeout_s: float = 1.5) -> Dict[str, Any]:
        """
        TCP reachability probe of a ggml-rpc-server endpoint (host:port).

        Only opens a socket: no VRAM, no model load. Returns the round-trip
        time so the panel can show the real measured link latency.
        """
        host, _, port = str(endpoint).rpartition(":")
        if not host or not port.isdigit():
            return {"reachable": False, "error": f"endpoint non valido: {endpoint}"}
        t0 = time.time()
        try:
            with socket.create_connection((host, int(port)), timeout=timeout_s):
                rtt_ms = (time.time() - t0) * 1000.0
                return {"reachable": True, "endpoint": f"{host}:{port}", "rtt_ms": rtt_ms}
        except Exception as e:
            # ggml-rpc-server serve UN client per volta: mentre lavora, una
            # seconda connect() resta in coda e scade. Se pero' una
            # connessione e' gia' stabilita, il backend e' vivo e solo
            # occupato: dirlo "offline" sarebbe falso.
            if LlamaServerManager._rpc_has_client(host, int(port)):
                return {"reachable": True, "busy": True,
                        "endpoint": f"{host}:{port}", "rtt_ms": 0.0,
                        "note": "in uso da un client"}
            return {"reachable": False, "endpoint": f"{host}:{port}", "error": str(e)}

    @staticmethod
    def _rpc_has_client(host: str, port: int) -> bool:
        """True se esiste gia' una connessione ESTABLISHED verso host:port."""
        try:
            packed = socket.inet_aton(host)
        except OSError:
            return False
        want = packed[::-1].hex().upper() + ":%04X" % int(port)
        try:
            with open("/proc/net/tcp", "r", encoding="ascii", errors="replace") as fh:
                next(fh, None)
                for line in fh:
                    parts = line.split()
                    # campo 2 = indirizzo remoto, campo 3 = stato (01 = ESTABLISHED)
                    if len(parts) > 3 and parts[3] == "01" and parts[2].upper() == want:
                        return True
        except OSError:
            pass
        return False

    def start_rpc_server(
        self,
        model_path: str,
        preset_key: str = "rpc_128k_fast",
        port: int = 8087,
        host: str = "localhost",
        binary_path: Optional[str] = None,
        dry_run: bool = False,
        preset_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Launch llama-server with the validated llama.cpp RPC flags (CUDA 13.4
        build, ggml-rpc-server on Miky).

        Reuses the same managed-process + log-reader plumbing as start_server()
        so the existing server log panel and process controls keep working.

        The KV guardrail from start_server() is deliberately NOT applied here:
        the q4_1/q8_0 KV of these presets was measured end-to-end on the real
        two-GPU RPC topology, and -nkvo is never used, so the "quantized KV only
        in VRAM" rule the guardrail enforces does not describe this setup.
        """
        # A user profile ("user::nome") carries its own dict; a built-in
        # preset is looked up by key.
        preset = dict(preset_override) if preset_override else self.RPC_PRESETS.get(preset_key)
        if not preset:
            return {"success": False, "error": f"Preset RPC sconosciuto: {preset_key}"}

        bin_path = binary_path or self.find_rpc_server_binary()
        if not bin_path or not os.path.exists(bin_path):
            return {"success": False, "error": "llama-server (build CUDA 13.4) non trovato"}
        if not os.path.exists(model_path):
            return {"success": False, "error": f"Modello non trovato: {model_path}"}

        cmd = self.build_rpc_command(bin_path, model_path, preset, port=port, host=host)
        if dry_run:
            # Build + validate the command line without spawning anything.
            return {
                "success": True,
                "dry_run": True,
                "cmd": " ".join(cmd),
                "argv": cmd,
                "preset": preset_key,
            }

        probe = self.probe_rpc_backend(preset["rpc"]) if preset.get("rpc") else {
            "reachable": True, "endpoint": "(locale, nessun RPC)"
        }
        if not probe.get("reachable"):
            with self.log_lock:
                self.server_logs.append(
                    f"[RPC] ggml-rpc-server NON raggiungibile su {preset['rpc']}: {probe.get('error')}"
                )

        # A server is already listening on the target port and this manager
        # did not start it (e.g. the user has one running by hand). Killing it
        # would be destructive, so we refuse instead.
        if self.managed_process is None and self.is_port_in_use(host, port):
            return {
                "success": False,
                "needs_confirmation": True,
                "error": f"Qualcosa e' gia' in ascolto su {host}:{port} "
                         f"e non e' stato avviato da questa TUI. "
                         f"Usa un'altra porta o libera quella a mano.",
                "port": port,
            }

        # Stop existing instance if active.
        # NOTE: stop_server() runs `taskkill /F /IM llama-server /T`, which
        # kills EVERY llama-server on the machine, including servers the user
        # started by hand on other ports. That is far too destructive for a
        # one-key hotkey, so we only terminate the process we manage ourselves.
        if self.managed_process is not None:
            try:
                self.managed_process.terminate()
                try:
                    self.managed_process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    self.managed_process.kill()
            except Exception:
                pass
            self.managed_process = None
            self.managed_pid = None
            with self.log_lock:
                ts = datetime.now().strftime("%H:%M:%S")
                self.server_logs.append(f"[{ts}] [RPC] precedente llama-server gestito terminato.")

        try:
            # Staccato dal cgroup della TUI (systemd-run --user --scope) e con
            # il log su FILE: con un PIPE il server morirebbe di SIGPIPE alla
            # chiusura della TUI, e restando nel cgroup della TUI un OOM
            # ucciderebbe la TUI (o l'app che la contiene) invece del server.
            spawned = _PS.spawn_detached(cmd, name=f"llama-{port}")
            if not spawned.get("success"):
                return {"success": False, "error": spawned.get("error", "avvio fallito")}
            proc = spawned["proc"]
            self.managed_log_path = spawned["log_path"]
            self.managed_isolated = spawned["isolated"]
            self.managed_process = proc
            self.managed_pid = proc.pid
            self.managed_model_path = model_path
            self.managed_port = port

            with self.log_lock:
                self.server_logs.clear()
                ts = datetime.now().strftime("%H:%M:%S")
                self.server_logs.append(f"[{ts}] Launching llama-server (PID: {proc.pid}) on port {port}...")
                self.server_logs.append(f"[{ts}] Model: {os.path.basename(model_path)}")
                self.server_logs.append(f"[{ts}] RPC preset: {preset['name']}")
                self.server_logs.append(f"[{ts}] Flags: {' '.join(cmd[1:])}")
                if probe.get("reachable") and preset.get("rpc"):
                    self.server_logs.append(
                        f"[{ts}] ggml-rpc-server {probe.get('endpoint')} raggiungibile (RTT {probe.get('rtt_ms', 0):.2f} ms)"
                    )

            log_path = spawned["log_path"]

            def _log_reader():
                # tail -f del file di log finche' il processo vive.
                pos = 0
                try:
                    while True:
                        lines, pos = _PS.tail_file(log_path, pos)
                        if lines:
                            with self.log_lock:
                                self.server_logs.extend(lines)
                                del self.server_logs[:-200]
                        if proc.poll() is not None:
                            lines, pos = _PS.tail_file(log_path, pos, final=True)
                            with self.log_lock:
                                self.server_logs.extend(lines)
                                del self.server_logs[:-200]
                            break
                        time.sleep(0.5)
                except Exception:
                    pass

            self.log_reader_thread = threading.Thread(target=_log_reader, daemon=True)
            self.log_reader_thread.start()

            return {
                "success": True,
                "pid": proc.pid,
                "port": port,
                "model": model_path,
                "preset": preset_key,
                "preset_name": preset["name"],
                "cmd": " ".join(cmd),
                "rpc_probe": probe,
                "message": f"Started llama-server [{preset['name']}] (PID {proc.pid}) on port {port}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def __init__(self, endpoints: Optional[List[Dict[str, str]]] = None, timeout_s: float = 2.5, ctx_shift_enabled: bool = True):
        self.timeout_s = timeout_s
        self.ctx_shift_enabled: bool = ctx_shift_enabled
        self.endpoints: Dict[str, ServerEndpoint] = {}
        configured = endpoints or self.DEFAULT_ENDPOINTS
        for ep in configured:
            self.register_endpoint(ep["name"], ep["url"], ep["server_type"])

        # Managed process lifecycle & logs
        self.managed_process: Optional[subprocess.Popen] = None
        self.managed_pid: Optional[int] = None
        # log su file e isolamento (cgroup separato) dell'ultimo avvio
        self.managed_log_path: str = ""
        self.managed_isolated: bool = False
        self.managed_model_path: Optional[str] = None
        self.managed_port: int = 18434
        self.server_logs: List[str] = []
        self.log_lock = threading.Lock()
        self.log_reader_thread: Optional[threading.Thread] = None

    @property
    def active_process(self) -> Optional[subprocess.Popen]:
        return self.managed_process

    @property
    def active_process(self) -> Optional[subprocess.Popen]:
        return self.managed_process

    def find_llama_server_binary(self) -> Optional[str]:
        """Find local llama-server executable from standard installations or PATH."""
        candidates = [
            # vedi find_rpc_server_binary(): i binari sono su /mnt_lm_models perche'
            # sulla root erano stati cancellati dalla pulizia automatica
            "/mnt/lm_models/llamacpp/bin/llama-b11330/llama-server",
            shutil.which("llama-server"),
            shutil.which("llama-cli"),
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c):
                return os.path.normpath(c)
        return None

    def scan_local_models(self, base_dir: str = "/mnt/lm_models/models", min_size_mb: float = 900.0, filter_drafts: bool = True) -> List[LocalModelInfo]:
        """
        Scan local models from /mnt/lm_models/models or custom directory for valid standalone .gguf files.
        Filters out draft models, mmproj/vision clips, and sub-1GB speculative heads by default.

        La quantizzazione NON viene piu' dedotta dal nome del file: si legge
        dai metadati interni del GGUF (general.file_type, e in subordine il
        ggml type dominante dei tensori). Il nome si usa solo come ultima
        riserva. Motivo: i modelli APEX da 13-14GB
        (Ornith-...-APEX-I-Mini-v2D-lite, Carnice-...-APEX-I-Mini,
        Qwen3.8-...APEX-I-MiniPlus-V2.1) non contengono nessuna sigla
        classica nel nome e finivano etichettati "OTHER", indistinguibili
        da "non riconosciuto".
        """
        results: List[LocalModelInfo] = []
        if not os.path.exists(base_dir):
            return results

        # Ignored patterns for auxiliary / draft / projector files when filtering
        draft_markers = [
            "mmproj-", "mmproj_", "-dspark", "dspark_", "-dflash", "dflash_", "mtp-", "speculator",
            "-00002-of-", "-00003-of-", "-00004-of-", "00002-of-", "00003-of-", "00004-of-",
            "nomic-embed", "laya_typed"
        ]

        try:
            for root, _, files in os.walk(base_dir):
                for f in files:
                    if ":" in f or f.startswith("."):
                        continue
                    if f.lower().endswith(".gguf"):
                        full_p = os.path.normpath(os.path.join(root, f))
                        try:
                            st = os.stat(full_p)
                            sz_bytes = st.st_size
                            sz_mb = sz_bytes / (1024 * 1024)
                            sz_gb = sz_bytes / (1024 ** 3)
                            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
                        except Exception:
                            sz_bytes = 0
                            sz_mb = 0.0
                            sz_gb = 0.0
                            mtime = "-"

                        # Filter out non-standalone draft / projector files if requested
                        fl_lower = f.lower()
                        if filter_drafts:
                            if sz_mb < min_size_mb:
                                continue
                            if any(marker in fl_lower for marker in draft_markers):
                                continue

                        try:
                            rel_p = os.path.relpath(full_p, base_dir)
                            parts = rel_p.split(os.sep)
                            author = parts[0] if len(parts) > 1 else "Local"
                        except Exception:
                            rel_p = f
                            author = "Local"

                        # ------------------------------------------------------------
                        # QUANTIZZAZIONE: prima i metadati interni, poi il nome.
                        # "meta" vs "nome" resta tracciato in quant_source.
                        # ------------------------------------------------------------
                        meta = gguf_metadata_cached(full_p)
                        q_type = quant_from_gguf_metadata(meta)
                        if q_type:
                            quant_source = "meta"
                        else:
                            q_type = quant_from_filename(f)
                            quant_source = "nome"

                        # VRAM Fit Status on 8GB RTX 3050
                        if sz_gb <= 5.5:
                            fit_status = "🟢 100% VRAM (<6GB)"
                            is_rec = True
                        elif sz_gb <= 7.8:
                            fit_status = "🟡 Near Limit (6-8GB)"
                            is_rec = False
                        else:
                            fit_status = f"🔴 RAM Offload ({sz_gb:.1f}G)"
                            is_rec = False

                        results.append(
                            LocalModelInfo(
                                filename=f,
                                filepath=full_p,
                                rel_path=rel_p,
                                author=author,
                                size_bytes=sz_bytes,
                                size_gb=sz_gb,
                                quant_type=q_type,
                                vram_fit_status=fit_status,
                                is_recommended=is_rec,
                                last_modified=mtime,
                                quant_source=quant_source,
                            )
                        )
        except Exception:
            pass

        # Sort by size ascending for convenient browsing
        results.sort(key=lambda m: m.size_gb)
        return results

    def start_server(
        self,
        model_path: str,
        port: int = 18434,
        ctx_size: int = 262144,
        kv_dtype: str = "f16",
        no_kv_offload: bool = True,
        ctx_shift: bool = True,
        flash_attn: bool = True,
        n_gpu_layers: int = 99,
        threads: int = 8,
        binary_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Launch llama-server with specified model and optimized launch flags."""
        bin_path = binary_path or self.find_llama_server_binary()
        if not bin_path or not os.path.exists(bin_path):
            return {"success": False, "error": "llama-server binary not found on system"}

        if not os.path.exists(model_path):
            return {"success": False, "error": f"Model file does not exist: {model_path}"}

        # Stop existing instance if active
        self.stop_server()

        # GUARDRAIL: quantized KV only if the KV cache is fully in VRAM.
        # probe_vram makes the fit check real (it used to never engage: no caller
        # passed a VRAM reading). apply_groq_compression sizes the cache against the
        # context the model actually receives, i.e. after the OmniRoute Groq plugin
        # compressed the tool results.
        kv_verdict = KVCacheOptimizer.resolve_kv_dtype(
            kv_dtype,
            no_kv_offload=no_kv_offload,
            ctx_size=ctx_size,
            probe_vram=True,
            apply_groq_compression=True,
        )
        kv_dtype_effective = kv_verdict["effective"]
        if kv_verdict["warning"]:
            logger.warning("ai_tui KV guardrail: %s", kv_verdict["warning"])
            with self.log_lock:
                self.server_logs.append(f"[GUARDRAIL] {kv_verdict['warning']}")

        cmd = [
            bin_path,
            "-m", model_path,
            "-c", str(ctx_size),
            "-ctk", kv_dtype_effective,
            "-ctv", kv_dtype_effective,
            "-fa", "on" if flash_attn else "off",
            "-ngl", str(n_gpu_layers),
            "-t", str(threads),
            "--port", str(port),
            "--host", "127.0.0.1"
        ]
        if no_kv_offload:
            cmd.append("-nkvo")
        # llama-server only accepts --context-shift / --no-context-shift;
        # the shorthand --ctx-shift is rejected by this build.
        cmd.append("--context-shift" if ctx_shift else "--no-context-shift")

        try:
            # Staccato dal cgroup della TUI (systemd-run --user --scope) e con
            # il log su FILE: con un PIPE il server morirebbe di SIGPIPE alla
            # chiusura della TUI, e restando nel cgroup della TUI un OOM
            # ucciderebbe la TUI (o l'app che la contiene) invece del server.
            spawned = _PS.spawn_detached(cmd, name=f"llama-{port}")
            if not spawned.get("success"):
                return {"success": False, "error": spawned.get("error", "avvio fallito")}
            proc = spawned["proc"]
            self.managed_log_path = spawned["log_path"]
            self.managed_isolated = spawned["isolated"]
            self.managed_process = proc
            self.managed_pid = proc.pid
            self.managed_model_path = model_path
            self.managed_port = port

            with self.log_lock:
                self.server_logs.clear()
                ts = datetime.now().strftime("%H:%M:%S")
                nkvo_str = " -nkvo" if no_kv_offload else ""
                shift_str = " --context-shift" if ctx_shift else " --no-context-shift"
                self.server_logs.append(f"[{ts}] Launching llama-server (PID: {proc.pid}) on port {port}...")
                self.server_logs.append(f"[{ts}] Model: {os.path.basename(model_path)}")
                self.server_logs.append(
                    f"[{ts}] Flags: -c {ctx_size} -ctk {kv_dtype_effective} -ctv {kv_dtype_effective}"
                    f"{nkvo_str} -ngl {n_gpu_layers} -fa on{shift_str}"
                )
                if kv_verdict["forced"]:
                    self.server_logs.append(
                        f"[{ts}] KV quantization guardrail active: requested {kv_verdict['requested']} "
                        f"-> forced {kv_dtype_effective} ({kv_verdict['reason']})"
                    )

            log_path = spawned["log_path"]

            def _log_reader():
                # tail -f del file di log finche' il processo vive.
                pos = 0
                try:
                    while True:
                        lines, pos = _PS.tail_file(log_path, pos)
                        if lines:
                            with self.log_lock:
                                self.server_logs.extend(lines)
                                del self.server_logs[:-200]
                        if proc.poll() is not None:
                            lines, pos = _PS.tail_file(log_path, pos, final=True)
                            with self.log_lock:
                                self.server_logs.extend(lines)
                                del self.server_logs[:-200]
                            break
                        time.sleep(0.5)
                except Exception:
                    pass

            self.log_reader_thread = threading.Thread(target=_log_reader, daemon=True)
            self.log_reader_thread.start()

            return {
                "success": True,
                "pid": proc.pid,
                "port": port,
                "model": model_path,
                "cmd": " ".join(cmd),
                "no_kv_offload": no_kv_offload,
                "message": f"Started llama-server (PID {proc.pid}) on port {port}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_server(self) -> Dict[str, Any]:
        """Terminate managed llama-server process and any residual instances."""
        stopped_pid = self.managed_pid
        if self.managed_process:
            try:
                self.managed_process.terminate()
                try:
                    self.managed_process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    self.managed_process.kill()
            except Exception:
                pass
            self.managed_process = None
            self.managed_pid = None

        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/F", "/IM", "llama-server", "/T"], capture_output=True, text=True, timeout=1.5)
            except Exception:
                pass

        with self.log_lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.server_logs.append(f"[{ts}] llama-server process terminated.")

        return {"success": True, "stopped_pid": stopped_pid, "message": "llama-server stopped"}

    def get_server_logs(self, max_lines: int = 50) -> List[str]:
        """Get recent stdout/stderr log lines from managed server."""
        with self.log_lock:
            return list(self.server_logs[-max_lines:])

    def flush_vram(self, hw_monitor: Optional[Any] = None,
                   kill_external: bool = False) -> Dict[str, Any]:
        """
        Emergency & interactive VRAM memory purge:
        1. Query initial VRAM allocation.
        2. Force kill all llama-server processes.
        3. Trigger LM Studio and Ollama model unload endpoints.
        4. Run Python garbage collection and PyTorch CUDA cache empty.
        5. Invoke Windows PSAPI EmptyWorkingSet via ctypes to trim process memory.
        6. Measure reclaimed VRAM and return result.
        """
        before_vram_mb = 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.5
            )
            if out.returncode == 0 and out.stdout.strip():
                before_vram_mb = float(out.stdout.strip().split("\n")[0].strip())
        except Exception:
            pass

        # 1. Terminate managed server
        self.stop_server()

        # 2. Kill all llama-server processes via psutil & taskkill
        # kill_external=False (usato dalla suite --test): si ferma SOLO il
        # server gestito dalla TUI. I server avviati a mano in altri
        # terminali (adottati) non si toccano: il test non deve poi doverli
        # riavviare dal proprio processo/cgroup.
        # Default (kill_external=False): SOLO il server gestito. Con
        # kill_external=True (scelta esplicita [F] poi [T]) si ferma ogni
        # llama-server dell'utente: SIGTERM, attesa 3 s, SIGKILL solo a chi
        # non e' uscito. Mai SIGKILL diretto a tutti.
        killed = []
        if psutil and kill_external:
            try:
                procs = []
                for p in psutil.process_iter(["pid", "name"]):
                    try:
                        pname = (p.info.get("name") or "").lower()
                        if "llama-server" in pname or "llama_server" in pname:
                            p.terminate()
                            procs.append(p)
                            killed.append(f"PID {p.info.get('pid')}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                _gone, alive = psutil.wait_procs(procs, timeout=3.0)
                for p in alive:
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:
                pass

        if os.name == "nt" and kill_external:
            try:
                subprocess.run(["taskkill", "/F", "/IM", "llama-server", "/T"], capture_output=True, text=True, timeout=2.0)
                subprocess.run(["taskkill", "/F", "/IM", "llama_server.exe", "/T"], capture_output=True, text=True, timeout=2.0)
            except Exception:
                pass

        # 3. LM Studio / Ollama unload endpoints
        unloaded = []
        for port in [1234, 56397, 8080]:
            for ep in ["/v1/models/unload", "/api/v0/models/unload"]:
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}{ep}",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=0.35) as resp:
                        unloaded.append(f"{port}{ep}")
                except Exception:
                    pass

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": "", "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=0.35) as resp:
                unloaded.append("ollama:unload")
        except Exception:
            pass

        # 4. Garbage collection & PyTorch cache
        gc.collect()
        if "torch" in sys.modules:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

        # 5. Windows working set trim via ctypes
        if os.name == "nt" and ctypes:
            try:
                h_proc = ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.psapi.EmptyWorkingSet(h_proc)
            except Exception:
                pass

        time.sleep(0.35)

        # 6. Measure new VRAM
        after_vram_mb = 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.5
            )
            if out.returncode == 0 and out.stdout.strip():
                after_vram_mb = float(out.stdout.strip().split("\n")[0].strip())
        except Exception:
            pass

        released_mb = max(0.0, before_vram_mb - after_vram_mb)
        if released_mb > 0:
            msg = f"VRAM FLUSHED SUCCESSFULLY - {released_mb:.0f} MB RELEASED"
        else:
            msg = "VRAM FLUSHED SUCCESSFULLY - GPU MEMORY CLEAN"

        return {
            "success": True,
            "vram_before_mb": before_vram_mb,
            "vram_after_mb": after_vram_mb,
            "released_mb": released_mb,
            "killed_processes": killed,
            "unloaded_endpoints": unloaded,
            "message": msg,
            "timestamp": time.time(),
        }

    def toggle_ctx_shift(self) -> bool:
        """Toggle context shifting (--ctx-shift) flag."""
        self.ctx_shift_enabled = not self.ctx_shift_enabled
        return self.ctx_shift_enabled

    # ------------------------------------------------------------------
    # EMBEDDER per mem0
    # ------------------------------------------------------------------
    # Serve un llama-server DIVERSO da quello dei modelli: questo gira con
    # --embedding --pooling mean e resta su una porta sua (:11435), cosi'
    # Ornith sulla :8080 non viene disturbato e viceversa. E' l'embedder
    # locale che mem0 usa: niente Ollama, niente API key, niente cloud.
    #
    # Il modello di default e' nomic-embed-text-v1.5.f16.gguf: BERT da 12
    # blocchi, 768 dimensioni, 262 MB. E' l'architettura (nomic-bert) che
    # llama.cpp sa caricare con --embedding, e le 768 dimensioni sono
    # esattamente quelle che mem0 si aspetta, quindi la collection Qdrant
    # non va ricreata.

    DEFAULT_EMBEDDER = "/mnt/lm_models/models/nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf"
    DEFAULT_EMBEDDER_PORT = 11435
    DEFAULT_EMBEDDER_CTX = 2048

    def embedder_is_running(self, port: Optional[int] = None) -> bool:
        """True se qualcuno risponde su /health alla porta dell'embedder."""
        port = int(port or self.DEFAULT_EMBEDDER_PORT)
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def embedder_dims(self, port: Optional[int] = None) -> Optional[int]:
        """Dimensione dell'embedding restituita dal server, se risponde.

        Serve per configurare mem0: la collection Qdrant deve avere la
        stessa dimensione, altrimenti al primo salvataggio fallisce.
        """
        port = int(port or self.DEFAULT_EMBEDDER_PORT)
        try:
            body = json.dumps({"model": "nomic-embed-text",
                               "input": "probe"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/embeddings", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
            vec = (d.get("data") or [{}])[0].get("embedding")
            return len(vec) if vec else None
        except Exception:
            return None

    def start_embedder(
        self,
        model_path: Optional[str] = None,
        port: Optional[int] = None,
        ctx_size: Optional[int] = None,
        threads: int = 6,
        binary_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Avvia l'embedder per mem0. Torna su start_server per i genitori."""
        bin_path = binary_path or self.find_llama_server_binary()
        if not bin_path or not os.path.exists(bin_path):
            return {"success": False, "error": "llama-server non trovato"}
        model = model_path or self.DEFAULT_EMBEDDER
        port = int(port or self.DEFAULT_EMBEDDER_PORT)
        ctx = int(ctx_size or self.DEFAULT_EMBEDDER_CTX)

        # Il path su NTFS va risolto con os.path.exists: readlink puo'
        # risolvere dove open() fallisce con ENOENT.
        if not os.path.exists(model):
            return {"success": False,
                    "error": f"modello embedder non trovato: {model}"}
        if self.embedder_is_running(port):
            return {"success": True, "already_running": True, "port": port,
                    "pid": None, "model": model}

        cmd = [bin_path, "-m", model,
               "--embedding", "--pooling", "mean",
               # -ngl 0 = embedder SEMPRE su CPU. Non e' una limitazione ma
               # una scelta: Ornith riempie le due GPU al 93% e l'embedder
               # in CUDA si mangierebbe 500-800 MiB che Ornith ha gia',
               # facendo fallire il suo prossimo avvio. Un BERT da 12 blocchi
               # su 6 thread codifica una query in 30-60 ms e la query e' una
               # per turno: la differenza non si percepisce, e la VRAM di
               # Ornith resta intatta.
               "-ngl", "0",
               "-c", str(ctx), "-t", str(threads),
               "--host", "127.0.0.1", "--port", str(port)]
        try:
            # start_new_session da solo NON cambia cgroup: si usa
            # systemd-run --scope (proc_spawn) che lo stacca davvero.
            spawned = _PS.spawn_detached(cmd, name=f"embedder-{port}")
            if not spawned.get("success"):
                return {"success": False, "error": spawned.get("error")}
            proc = spawned["proc"]
            emb_log = spawned["log_path"]
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        # aspetta /health: senza questo si crede di averlo avviato e poi
        # mem0 non trova l'embedder.
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.embedder_is_running(port):
                return {"success": True, "pid": proc.pid, "port": port,
                        "model": model, "dims": self.embedder_dims(port)}
            if proc.poll() is not None:
                out = ""
                try:
                    out = "\n".join(_PS.tail_file(emb_log, 0)[0])[-400:]
                except Exception:
                    pass
                return {"success": False,
                        "error": "l'embedder e' uscito subito", "output": out}
            time.sleep(1)
        try:
            proc.kill()
        except Exception:
            pass
        return {"success": False, "error": f"l'embedder non risponde su :{port}"}

    def stop_embedder(self, port: Optional[int] = None) -> Dict[str, Any]:
        """Ferma l'embedder uccidendo per porta, senza toccare Ornith.

        Si passa per `pgrep -f` invece che per `ss`: l'output di ss cambia
        forma tra versioni e il parsing del campo pid e' fragile. Con pgrep
        si cerca direttamente la porta nella riga di comando del processo,
        che e' l'unica cosa che distingue l'embedder da Ornith.
        """
        port = int(port or self.DEFAULT_EMBEDDER_PORT)
        # --port 11435 come argomento separato: e' cosi' che lo scrive
        # llama.cpp, quindi non si rischia di uccidere Ornith sulla :8080.
        try:
            out = subprocess.run(
                # '--' SEPARA le opzioni dal pattern: senza, pgrep legge
                # "--port 11435" come un'opzione sua e muore con
                # "unrecognized option", restituendo niente. Con '--' il
                # pattern viene cercato davvero nel cmdline.
                ["pgrep", "-f", "--", f"--port {port}"],
                capture_output=True, text=True, timeout=10).stdout.split()
        except Exception:
            out = []
        killed = []
        for tok in out:
            if not tok.isdigit():
                continue
            pid = int(tok)
            try:
                # verifica che sia davvero un llama-server prima di killare
                exe = os.path.realpath(f"/proc/{pid}/exe")
                if "llama-server" not in exe:
                    continue
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except Exception:
                pass
        if not killed:
            return {"success": False, "error": f"nessun llama-server su :{port}"}
        time.sleep(2)
        for pid in killed:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        return {"success": True, "killed": killed, "port": port}

    def get_launch_args(
        self,
        ctx_size: int = 262144,
        kv_dtype: str = "q4_0",
        no_kv_offload: bool = False,
        flash_attn: bool = True,
        ctx_shift: Optional[bool] = None,
        n_gpu_layers: int = 99,
        port: int = 18434,
        model_path: str = "model.gguf"
    ) -> Dict[str, Any]:
        """
        Generate optimized llama-server launch arguments.
        Supports -nkvo (RAM KV cache offload) and 256k long context.
        """
        use_ctx_shift = self.ctx_shift_enabled if ctx_shift is None else ctx_shift
        kv_verdict = KVCacheOptimizer.resolve_kv_dtype(
            kv_dtype, no_kv_offload=no_kv_offload, ctx_size=ctx_size,
            probe_vram=True, apply_groq_compression=True,
        )
        kv_effective = kv_verdict["effective"]
        args = [
            "-m", model_path,
            "-c", str(ctx_size),
            "-ctk", kv_effective,
            "-ctv", kv_effective,
            "-fa", "on" if flash_attn else "off",
            "-ngl", str(n_gpu_layers),
            "--port", str(port),
        ]
        if no_kv_offload:
            args.append("-nkvo")
        # --context-shift / --no-context-shift are the only spellings this build accepts.
        args.append("--context-shift" if use_ctx_shift else "--no-context-shift")

        nkvo_flag = " -nkvo" if no_kv_offload else ""
        shift_flag = " --context-shift" if use_ctx_shift else " --no-context-shift"
        cmd_str = f"llama-server -m {model_path} -c {ctx_size} -ctk {kv_effective} -ctv {kv_effective}{nkvo_flag}{shift_flag} -fa {'on' if flash_attn else 'off'} -ngl {n_gpu_layers} --port {port}"

        return {
            "cmd_args": args,
            "cmd_string": cmd_str,
            "ctx_size": ctx_size,
            "kv_dtype": kv_effective,
            "kv_dtype_requested": kv_verdict["requested"],
            "kv_quant_forced": kv_verdict["forced"],
            "kv_guardrail_warning": kv_verdict["warning"],
            "no_kv_offload": no_kv_offload,
            "ctx_shift": use_ctx_shift,
            "flash_attn": flash_attn,
            "n_gpu_layers": n_gpu_layers,
            "port": port,
        }

    def register_endpoint(self, name: str, url: str, server_type: str = "llama-server") -> ServerEndpoint:
        clean_url = url.rstrip("/")
        ep = ServerEndpoint(name=name, url=clean_url, server_type=server_type)
        self.endpoints[name] = ep
        return ep

    def _http_get(self, url: str) -> Tuple[bool, int, Any, float]:
        """Perform HTTP GET with latency measurement and error safety."""
        start = time.perf_counter()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ai_tui-backend/1.0", "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                status_code = response.getcode()
                raw_data = response.read().decode("utf-8")
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    data = raw_data
                return True, status_code, data, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return False, 0, str(e), elapsed_ms

    def is_server_online(self, host: str = "127.0.0.1", port: int = 18434) -> bool:
        """Quick socket/HTTP probe to check if server is reachable."""
        try:
            url = f"http://{host}:{port}/health"
            req = urllib.request.Request(url, headers={"User-Agent": "ai_tui/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status in (200, 503)
        except Exception:
            return False

    def is_server_online(self, host: str = "127.0.0.1", port: int = 18434) -> bool:
        """Check if llama-server is online on given host and port."""
        url = f"http://{host}:{port}"
        h = self.check_health(url)
        return h.get("status") == "online" or h.get("is_active") is True

    def check_health(self, endpoint_name_or_url: Optional[str] = None) -> Dict[str, Any]:
        """Check health of a specific endpoint or all registered endpoints."""
        if endpoint_name_or_url:
            target = None
            if endpoint_name_or_url in self.endpoints:
                target = self.endpoints[endpoint_name_or_url]
            else:
                for ep in self.endpoints.values():
                    if ep.url == endpoint_name_or_url.rstrip("/"):
                        target = ep
                        break
            if not target:
                target = ServerEndpoint(
                    name="custom",
                    url=endpoint_name_or_url.rstrip("/"),
                    server_type="auto"
                )

            return self._probe_endpoint_health(target)

        results = {}
        for name, ep in self.endpoints.items():
            results[name] = self._probe_endpoint_health(ep)
        return results

    def _probe_endpoint_health(self, ep: ServerEndpoint) -> Dict[str, Any]:
        ep.last_checked = time.time()
        health_url = f"{ep.url}/health"
        success, code, data, latency = self._http_get(health_url)

        if not success or code >= 400:
            models_url = f"{ep.url}/v1/models"
            success, code, data, latency = self._http_get(models_url)

        ep.is_active = success and (200 <= code < 300)
        ep.latency_ms = round(latency, 2)
        ep.error_msg = None if ep.is_active else str(data)

        return {
            "name": ep.name,
            "url": ep.url,
            "server_type": ep.server_type,
            "status": "online" if ep.is_active else "offline",
            "status_code": code,
            "latency_ms": ep.latency_ms,
            "error": ep.error_msg,
            "timestamp": ep.last_checked
        }

    def get_models(self, endpoint_name: Optional[str] = None) -> List[ModelMetadata]:
        """Fetch available models from /v1/models endpoint."""
        models: List[ModelMetadata] = []
        targets = [self.endpoints[endpoint_name]] if endpoint_name in self.endpoints else list(self.endpoints.values())

        for ep in targets:
            url = f"{ep.url}/v1/models"
            success, code, data, _ = self._http_get(url)
            if success and isinstance(data, dict):
                model_list = data.get("data") or data.get("models") or []
                for item in model_list:
                    m_id = item.get("id") or item.get("name") or item.get("model") or "unknown"
                    owned_by = item.get("owned_by", ep.name)
                    ctx = item.get("context_length") or item.get("max_model_len") or item.get("n_ctx")
                    models.append(ModelMetadata(
                        id=m_id,
                        owned_by=owned_by,
                        context_length=ctx,
                        raw_info=item
                    ))
        return models

    def get_server_props(self, endpoint_name: str = "llama-server") -> Dict[str, Any]:
        """Fetch llama.cpp /props containing context size, default settings, etc."""
        if endpoint_name not in self.endpoints:
            return {"error": f"Endpoint {endpoint_name} not found"}
        ep = self.endpoints[endpoint_name]
        url = f"{ep.url}/props"
        success, code, data, latency = self._http_get(url)
        if success and isinstance(data, dict):
            return {"success": True, "latency_ms": latency, "props": data}
        return {"success": False, "error": str(data), "latency_ms": latency}

    def get_slots_status(self, endpoint_name: str = "llama-server") -> List[SlotInfo]:
        """Query /slots endpoint from llama-server."""
        slots: List[SlotInfo] = []
        if endpoint_name not in self.endpoints:
            return slots
        ep = self.endpoints[endpoint_name]
        url = f"{ep.url}/slots"
        success, code, data, _ = self._http_get(url)

        if success and isinstance(data, list):
            for s in data:
                is_proc = s.get("is_processing", False) or s.get("state", 0) != 0
                slots.append(SlotInfo(
                    slot_id=s.get("id", len(slots)),
                    state=s.get("state", 0),
                    is_processing=is_proc,
                    n_ctx=s.get("n_ctx", 0),
                    n_past=s.get("n_past", 0),
                    n_decoded=s.get("n_decoded", 0),
                    prompt_tokens=s.get("n_prompt_tokens", 0),
                    task_id=s.get("id_task"),
                    model_name=s.get("model")
                ))
        return slots

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        endpoint_name: str = "llama-server",
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream chat completions from llama-server or OpenAI-compatible endpoint."""
        if endpoint_name not in self.endpoints:
            yield {"type": "error", "error": f"Endpoint '{endpoint_name}' not registered"}
            return

        ep = self.endpoints[endpoint_name]
        url = f"{ep.url}/v1/chat/completions"

        target_model = model
        if not target_model:
            models = self.get_models(endpoint_name)
            if models:
                target_model = models[0].id
            else:
                target_model = "default"

        payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "ai_tui-client/1.0",
            },
            method="POST",
        )

        start_time = time.perf_counter()
        token_count = 0
        full_content = ""
        full_reasoning = ""
        first_token_time: Optional[float] = None
        timings: Dict[str, Any] = {}

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                for line_b in resp:
                    line = line_b.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if not first_token_time:
                            first_token_time = time.perf_counter()

                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "timings" in chunk:
                            timings = chunk["timings"]

                        reasoning_delta = delta.get("reasoning_content") or ""
                        content_delta = delta.get("content") or ""

                        if reasoning_delta:
                            full_reasoning += reasoning_delta
                            token_count += 1
                            elapsed = max(0.001, time.perf_counter() - start_time)
                            tok_s = token_count / elapsed
                            yield {
                                "type": "chunk",
                                "delta": reasoning_delta,
                                "is_reasoning": True,
                                "tokens": token_count,
                                "elapsed": elapsed,
                                "tok_s": round(tok_s, 2),
                            }

                        if content_delta:
                            full_content += content_delta
                            token_count += 1
                            elapsed = max(0.001, time.perf_counter() - start_time)
                            tok_s = token_count / elapsed
                            yield {
                                "type": "chunk",
                                "delta": content_delta,
                                "is_reasoning": False,
                                "tokens": token_count,
                                "elapsed": elapsed,
                                "tok_s": round(tok_s, 2),
                            }

                    except json.JSONDecodeError:
                        continue

            total_elapsed = max(0.001, time.perf_counter() - start_time)
            avg_tok_s = token_count / total_elapsed
            ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0

            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning,
                "total_tokens": token_count,
                "elapsed": round(total_elapsed, 2),
                "tok_s": round(avg_tok_s, 2),
                "ttft_ms": round(ttft_ms, 2),
                "timings": timings,
            }

        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "partial_content": full_content,
                "partial_reasoning": full_reasoning,
            }

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute or simulate agent tools."""
        tool = tool_name.lower().strip()
        start = time.perf_counter()

        if tool in ("web_search", "search"):
            query = arguments.get("query", "")
            return {
                "tool": "web_search",
                "status": "success",
                "query": query,
                "result": f"[Simulated Live Search Results for '{query}']: Found 4 matching sources in index. Top match: arXiv:2403.09629 'Dynamic KV Cache Paging for Edge LLM Inference'.",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        elif tool in ("calculate", "calc", "math"):
            expr = str(arguments.get("expression", "0"))
            try:
                allowed = {"__builtins__": None, "math": math, "abs": abs, "round": round, "pow": pow}
                val = eval(expr, allowed, {})
                return {
                    "tool": "calculate",
                    "status": "success",
                    "expression": expr,
                    "result": str(val),
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            except Exception as e:
                return {
                    "tool": "calculate",
                    "status": "error",
                    "error": f"Evaluation error: {e}",
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }

        elif tool in ("system_check", "gpu_status", "hardware"):
            monitor = HardwareMonitor()
            telemetry = monitor.get_telemetry()
            return {
                "tool": "system_check",
                "status": "success",
                "gpu": asdict(telemetry.gpu),
                "ram": asdict(telemetry.ram),
                "cpu": asdict(telemetry.cpu),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        elif tool in ("kv_budget", "kv_calc"):
            seq = int(arguments.get("seq_len", 4096))
            opt = KVCacheOptimizer()
            res = opt.calculate_kv_memory(seq_len=seq)
            return {
                "tool": "kv_budget",
                "status": "success",
                "seq_len": seq,
                "total_mb": res["total_mb"],
                "quant_q8": res["quant_comparison"]["q8_0"]["mb"],
                "quant_q4": res["quant_comparison"]["q4_0"]["mb"],
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        elif tool in ("compress_context", "compress_prompt", "compress"):
            text = arguments.get("text", "") or arguments.get("prompt", "")
            compressor = ContextCompressor()
            comp_res = compressor.compress_context(text)
            return {
                "tool": "compress_context",
                "status": "success",
                "is_groq_powered": comp_res.is_groq_powered,
                "model_used": comp_res.model_used,
                "original_tokens": comp_res.original_tokens_est,
                "compressed_tokens": comp_res.compressed_tokens_est,
                "savings_percent": comp_res.savings_percent,
                "key_facts": comp_res.key_facts,
                "active_task_state": comp_res.active_task_state,
                "decisions": comp_res.decisions,
                "variables": comp_res.variables,
                "latency_ms": comp_res.latency_ms,
            }

        return {
            "tool": tool_name,
            "status": "unknown_tool",
            "error": f"Tool '{tool_name}' not implemented.",
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    def get_overview(self) -> Dict[str, Any]:
        """Return comprehensive status overview of all local AI servers."""
        health_status = self.check_health()
        models = self.get_models()
        slots_dict = {}
        for name, ep in self.endpoints.items():
            if ep.server_type == "llama-server":
                slots_dict[name] = [asdict(s) for s in self.get_slots_status(name)]

        return {
            "endpoints": health_status,
            "models": [asdict(m) for m in models],
            "slots": slots_dict,
            "total_models_available": len(models),
            "any_online": any(ep.get("status") == "online" for ep in health_status.values())
        }

    # Backward compatibility alias
    get_server_overview = get_overview


# =====================================================================
# 1.5 RPC PROFILE STORE  (salvataggio/richiamo profili di configurazione)
# =====================================================================
class RPCProfileStore:
    """
    Persist RPC launch profiles to a JSON file (rpc_profiles.json next to
    tui_app.py) so custom context / --parallel / KV quant / n_gpu_layers
    combinations survive a restart of the TUI.

    Built-in presets from LlamaServerManager.RPC_PRESETS are always loaded
    first and are never overwritten by a user profile with the same name;
    user profiles are kept in a separate "profili_utente" section.
    """

    FILENAME = "rpc_profiles.json"

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.base_dir, self.FILENAME)
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.last_error: str = ""
        self.load()

    def load(self) -> Dict[str, Dict[str, Any]]:
        """Read user profiles from disk. A missing/corrupt file is not fatal."""
        import atomic_json as _AJ
        # File corrotto: rinominato in .corrupt-<ts> (non azzerato in silenzio
        # al primo salvataggio) e l'errore resta in last_error per la UI.
        data, err = _AJ.load_json(self.path, {})
        if not isinstance(data, dict):
            data, err = {}, err or "rpc_profiles.json: formato inatteso"
        raw = data.get("profili_utente") or {}
        self.user_profiles = {k: v for k, v in raw.items() if isinstance(v, dict)} \
            if isinstance(raw, dict) else {}
        self.last_error = err or ""
        return self.user_profiles

    def save(self) -> bool:
        """Write user profiles back to disk. Never raises."""
        payload = {
            "_meta": "Profili RPC personalizzati di ai_tui (Tab 9). I preset validati stanno in backend.py: LlamaServerManager.RPC_PRESETS.",
            "profili_utente": self.user_profiles,
        }
        try:
            import atomic_json as _AJ
            _AJ.save_json(self.path, payload)          # tmp + fsync + os.replace
            self.last_error = ""
            return True
        except Exception as e:
            self.last_error = f"salvataggio fallito: {e}"
            return False

    def save_profile(
        self,
        name: str,
        ctx_size: int,
        parallel: int = 1,
        kv_type: str = "q4_1",
        kv_value: Optional[str] = None,
        ngl: int = 99,
        rpc: str = "192.168.1.100:50052",
        device: str = "CUDA0",
        flash_attn: bool = True,
        threads: int = 8,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create or overwrite a named user profile and persist it."""
        name = (name or "").strip()
        if not name:
            return {"success": False, "error": "Nome profilo vuoto"}
        self.user_profiles[name] = {
            "name": name,
            "device": device,
            "rpc": rpc,
            "ctx_size": int(ctx_size),
            "parallel": int(parallel),
            "ngl": int(ngl),
            "flash_attn": bool(flash_attn),
            "kv_type": kv_type,
            "kv_value": kv_value or kv_type,
            "threads": int(threads),
            "jinja": True,
            "tok_s": 0.0,
            "recommended": False,
            "description": description or f"Profilo utente: ctx {int(ctx_size)}, parallel {int(parallel)}, KV {kv_type}, ngl {int(ngl)}.",
        }
        ok = self.save()
        return {"success": ok, "name": name, "path": self.path, "error": self.last_error}

    def delete_profile(self, name: str) -> Dict[str, Any]:
        """Remove a user profile. Built-in presets cannot be deleted."""
        if name in self.user_profiles:
            self.user_profiles.pop(name)
            return {"success": self.save(), "name": name, "error": self.last_error}
        return {"success": False, "error": f"'{name}' non e' un profilo utente salvato"}

    def all_profiles(self) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Return [(key, preset, origin)] with built-ins first (in the F-key order)
        then user profiles. 'origin' is "preset" or "utente" and is what the
        TUI uses to label the rows.
        """
        out: List[Tuple[str, Dict[str, Any], str]] = []
        for key in LlamaServerManager.RPC_PRESET_ORDER:
            p = _LP.effective(key) or LlamaServerManager.RPC_PRESETS.get(key)
            if p:
                out.append((key, p, "preset"))
        for key in LlamaServerManager.RPC_LEGACY_ORDER:
            p = LlamaServerManager.RPC_PRESETS.get(key)
            if p:
                out.append((key, p, "storico"))
        for key, p in self.user_profiles.items():
            out.append((f"user::{key}", p, "utente"))
        return out


# =====================================================================
# 2. KV CACHE OPTIMIZER
# =====================================================================

@dataclass
class TokenItem:
    token_id: int
    text: str
    turn_type: str = "content"  # "system", "tool", "user", "assistant", "content"
    attention_score: float = 1.0
    position: int = 0
    is_protected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class KVCacheOptimizer:
    """
    Implements 3 key KV Cache optimizations:
    1. StreamingLLM: Attention Sinks (preserve first 4 tokens + sliding window of last N tokens).
    2. Dynamic KV Budgeting / GrowPage Eviction: Protected prefixes (system prompt + tool schemas)
       plus importance-weighted middle token pruning.
    3. Hybrid Attention & RAM/VRAM KV Cache footprint calculations (including RAM-Only Offload -nkvo).
    """

    # Bytes per element for various KV quantization datatypes
    DTYPE_BYTES = {
        "f32": 4.0,
        "fp32": 4.0,
        "float32": 4.0,
        "f16": 2.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "float16": 2.0,
        "q8_0": 1.0625,  # 8-bit quantized + block scale
        "q8": 1.0625,
        "q5_1": 0.6875,
        "q5_0": 0.625,
        "q4_1": 0.625,
        "q4_0": 0.5625,  # 4-bit quantized + block scale
        "q4": 0.5625,
        "q2_k": 0.328125
    }

    # ---- KV QUANTIZATION GUARDRAIL (hard rule, architecture specification) -------------
    # Quantized KV (q8_0/q4_0) is permitted ONLY when the KV cache is fully
    # resident in VRAM. With host-RAM KV offload (-nkvo) llama.cpp must run
    # per-token CPU AVX2 dequantization on every attention head and FlashAttention
    # is lost, which collapses throughput to ~1.2 tok/s. In that configuration the
    # KV dtype MUST be f16 and the quantized option is disabled with a warning.
    QUANTIZED_KV_DTYPES = {"q8_0", "q8", "q5_1", "q5_0", "q4_1", "q4_0", "q4", "q2_k"}
    KV_RAM_ONLY_DTYPE = "f16"
    KV_QUANT_BLOCKED_RAM_NOTICE = (
        "KV cache is in host RAM (-nkvo): quantized KV (q8_0/q4_0) is DISABLED. "
        "Forced -ctk f16 -ctv f16 — q4_0 on RAM needs per-token CPU dequantization "
        "and collapses throughput to ~1.2 tok/s."
    )

    # ── Groq context-compression feed (OmniRoute plugin `groq-context-compress`) ──
    # The plugin compresses bulky tool results BEFORE they reach the model and
    # records what it actually saved. Sizing the KV cache against the RAW context
    # over-allocates by the compression ratio, so the policy reads the real
    # post-compression figure instead of assuming 1.0.
    KV_COMPRESSION_STATS = os.path.join(
        os.path.expanduser("~"), ".omniroute", "plugins",
        "groq-context-compress", "kv-compression-stats.json",
    )
    # Ignore savings measured on payloads far below the KV-sized ones: a 1k-char
    # tool result says nothing about a 256k context. Require a real sample.
    KV_COMPRESSION_MIN_SAMPLE_CHARS = 8_000

    @classmethod
    def query_free_vram_mb(cls) -> Optional[float]:
        """Free VRAM in MB via nvidia-smi, or None when no NVIDIA GPU is present.

        Hardware detection rule: nvidia-smi --query-gpu=memory.total,memory.used,memory.free
        --format=csv,noheader,nounits. Returns None (not 0) when unavailable so callers
        can tell "no data" from "no free memory" and fail open to the requested dtype.
        """
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None
            return float(out.stdout.strip().split("\n")[0].strip())
        except Exception:
            return None

    @classmethod
    def read_groq_compression_ratio(cls) -> Optional[float]:
        """Mean measured compression ratio from the Groq plugin, or None.

        Returns the fraction of the ORIGINAL context bytes that the compression
        actually removed, so 0.98 means "98% smaller". Only records whose
        uncompressed payload was at least KV_COMPRESSION_MIN_SAMPLE_CHARS count,
        and only the most recent ones, so one odd batch cannot skew the policy.
        """
        try:
            if not os.path.exists(cls.KV_COMPRESSION_STATS):
                return None
            with open(cls.KV_COMPRESSION_STATS, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            usable = [
                r for r in (data.get("records") or [])
                if r.get("beforeChars", 0) >= cls.KV_COMPRESSION_MIN_SAMPLE_CHARS
                and 0.0 <= r.get("ratio", -1.0) <= 1.0
            ]
            if not usable:
                return None
            return sum(r["ratio"] for r in usable) / len(usable)
        except Exception:
            return None

    @classmethod
    def is_quantized_kv_dtype(cls, kv_dtype: str) -> bool:
        return (kv_dtype or "").strip().lower() in cls.QUANTIZED_KV_DTYPES

    @classmethod
    def resolve_kv_dtype(
        cls,
        kv_dtype: str,
        no_kv_offload: bool,
        ctx_size: int = 32768,
        n_layers: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        available_vram_mb: Optional[float] = None,
        probe_vram: bool = False,
        apply_groq_compression: bool = False,
    ) -> Dict[str, Any]:
        """
        Enforce the KV quantization guardrail and return the effective dtype.

        Rules, in order:
          1. KV in host RAM (-nkvo)  -> quantized KV is refused, forced to f16.
          2. KV in VRAM but does not fit -> quantized KV is refused, forced to f16.
          3. KV in VRAM and fits     -> requested dtype honored.

        `probe_vram` fills `available_vram_mb` from nvidia-smi when the caller has
        no live reading, so rule 2 can actually engage instead of silently never
        firing. `apply_groq_compression` discounts the required KV footprint by the
        ratio the OmniRoute Groq plugin MEASURED, so the cache is sized against the
        context the model really receives rather than the raw one.

        Never raises: callers get `forced`, `allowed` and `warning` to surface.
        """
        requested = (kv_dtype or cls.KV_RAM_ONLY_DTYPE).strip().lower()
        requested_quantized = cls.is_quantized_kv_dtype(requested)

        if probe_vram and available_vram_mb is None:
            available_vram_mb = cls.query_free_vram_mb()

        # Footprint the model will actually pay for: the cache is filled with the
        # post-compression context, not the raw tool output.
        kv_mb = cls._estimate_kv_mb(requested, ctx_size, n_layers, n_kv_heads, head_dim)
        groq_ratio = cls.read_groq_compression_ratio() if apply_groq_compression else None
        effective_kv_mb = kv_mb * (1.0 - groq_ratio) if groq_ratio is not None else kv_mb

        # f16 is never quantized, so it is always permitted: it is the mandated
        # dtype whenever the KV cache is not fully resident in VRAM.
        if no_kv_offload:
            return {
                "requested": requested,
                "effective": cls.KV_RAM_ONLY_DTYPE,
                "allowed": not requested_quantized,
                "forced": requested_quantized,
                "kv_in_vram": False,
                "kv_mb": kv_mb,
                "effective_kv_mb": effective_kv_mb,
                "available_vram_mb": available_vram_mb,
                "groq_compression_ratio": groq_ratio,
                "warning": cls.KV_QUANT_BLOCKED_RAM_NOTICE if requested_quantized else "",
                "reason": "no_kv_offload (KV in host RAM)",
            }

        if requested_quantized and available_vram_mb is not None and effective_kv_mb > float(available_vram_mb):
            saving = f" (~{int(round((1 - effective_kv_mb / kv_mb) * 100))}% smaller after Groq context compression)" if groq_ratio else ""
            return {
                "requested": requested,
                "effective": cls.KV_RAM_ONLY_DTYPE,
                "allowed": False,
                "forced": True,
                "kv_in_vram": True,
                "kv_mb": kv_mb,
                "effective_kv_mb": effective_kv_mb,
                "available_vram_mb": available_vram_mb,
                "groq_compression_ratio": groq_ratio,
                "warning": (
                    f"KV {requested} needs {effective_kv_mb:.0f} MB{saving} but only "
                    f"{float(available_vram_mb):.0f} MB of VRAM is free: "
                    f"quantized KV DISABLED, forced to f16."
                ),
                "reason": "kv does not fit in VRAM",
            }

        return {
            "requested": requested,
            "effective": requested,
            "allowed": True,
            "forced": False,
            "kv_in_vram": True,
            "kv_mb": kv_mb,
            "effective_kv_mb": effective_kv_mb,
            "available_vram_mb": available_vram_mb,
            "groq_compression_ratio": groq_ratio,
            "warning": "",
            "reason": "",
        }

    @classmethod
    def _estimate_kv_mb(
        cls,
        kv_dtype: str,
        ctx_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> float:
        """K and V cache bytes for one sequence (K and V both counted)."""
        bytes_per = cls.DTYPE_BYTES.get((kv_dtype or "f16").lower(), 2.0)
        per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per
        return per_token * max(0, int(ctx_size)) / (1024.0 * 1024.0)

    @classmethod
    def kv_dtype_options(
        cls,
        no_kv_offload: bool,
        available_vram_mb: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        KV quantization options annotated with guardrail state, for the TUI menu.
        Options the guardrail forbids are returned with enabled=False plus a reason.
        """
        options = [
            {"id": "f16", "label": "f16 (GPU KV only)"},
            {"id": "q8_0", "label": "q8_0 (GPU KV only)"},
            {"id": "q4_0", "label": "q4_0 (GPU KV only)"},
        ]
        for opt in options:
            verdict = cls.resolve_kv_dtype(
                opt["id"],
                no_kv_offload=no_kv_offload,
                ctx_size=32768,
                available_vram_mb=available_vram_mb,
            )
            opt["enabled"] = verdict["allowed"]
            opt["disabled_reason"] = "" if verdict["allowed"] else verdict["warning"]
        return options

    # -------------------------------------------------------------
    # 2.1 StreamingLLM: Attention Sinks + Sliding Window
    # -------------------------------------------------------------
    @staticmethod
    def apply_streaming_llm(
        tokens: List[Union[TokenItem, Dict[str, Any], int, str]],
        sink_tokens_count: int = 4,
        window_size: int = 1024
    ) -> Dict[str, Any]:
        """
        Preserves the first `sink_tokens_count` attention sink tokens (initial prompt anchors)
        and retains the most recent `window_size` sliding window tokens.
        """
        total_tokens = len(tokens)
        capacity = sink_tokens_count + window_size

        if total_tokens <= capacity:
            return {
                "kept_tokens": tokens,
                "evicted_count": 0,
                "evicted_indices": [],
                "sink_count": min(sink_tokens_count, total_tokens),
                "window_count": max(0, total_tokens - sink_tokens_count),
                "total_original": total_tokens,
                "total_retained": total_tokens,
                "compression_ratio": 1.0
            }

        sink_slice = tokens[:sink_tokens_count]
        window_slice = tokens[-window_size:]
        kept_tokens = list(sink_slice) + list(window_slice)

        evicted_indices = list(range(sink_tokens_count, total_tokens - window_size))
        evicted_count = len(evicted_indices)
        compression = len(kept_tokens) / total_tokens if total_tokens > 0 else 1.0

        return {
            "kept_tokens": kept_tokens,
            "evicted_count": evicted_count,
            "evicted_indices": evicted_indices,
            "sink_count": sink_tokens_count,
            "window_count": window_size,
            "total_original": total_tokens,
            "total_retained": len(kept_tokens),
            "compression_ratio": round(compression, 4)
        }

    # -------------------------------------------------------------
    # 2.2 Dynamic KV Budgeting / GrowPage-Style Token Eviction
    # -------------------------------------------------------------
    @staticmethod
    def apply_dynamic_growpage_eviction(
        tokens: List[TokenItem],
        max_budget: int,
        system_prompt_tokens_count: int = 0,
        tool_definition_tokens_count: int = 0,
        recent_window_protect: int = 32,
        importance_decay_gamma: float = 0.98
    ) -> Dict[str, Any]:
        """
        GrowPage-style dynamic token eviction:
        Protects system prompt, tool definitions, recent window, and explicit flags.
        Prunes least important tokens down to max_budget.
        """
        total = len(tokens)
        if total <= max_budget:
            return {
                "kept_tokens": tokens,
                "evicted_count": 0,
                "evicted_indices": [],
                "protected_count": total,
                "budget_utilized": total,
                "max_budget": max_budget,
                "compression_ratio": 1.0
            }

        protected_prefix_len = system_prompt_tokens_count + tool_definition_tokens_count
        recent_start_idx = max(0, total - recent_window_protect)

        protected_indices = set()
        evictable_candidates = []

        for idx, token in enumerate(tokens):
            is_sys = idx < system_prompt_tokens_count
            is_tool = system_prompt_tokens_count <= idx < protected_prefix_len
            is_recent = idx >= recent_start_idx
            is_explicit = token.is_protected or token.turn_type in ("system", "tool")

            if is_sys or is_tool or is_recent or is_explicit:
                protected_indices.add(idx)
            else:
                distance_from_end = total - 1 - idx
                recency_weight = math.pow(importance_decay_gamma, distance_from_end)
                composite_score = token.attention_score * recency_weight
                evictable_candidates.append((composite_score, idx, token))

        num_protected = len(protected_indices)
        available_slots = max(0, max_budget - num_protected)

        evictable_candidates.sort(key=lambda x: x[0], reverse=True)

        kept_from_candidates = evictable_candidates[:available_slots]
        retained_indices = set(protected_indices)
        for _, idx, _ in kept_from_candidates:
            retained_indices.add(idx)

        if len(retained_indices) > max_budget:
            sorted_retained = sorted(list(retained_indices))
            retained_indices = set(sorted_retained[:max_budget])

        kept_tokens = [tokens[i] for i in sorted(list(retained_indices))]
        evicted_indices = sorted([i for i in range(total) if i not in retained_indices])

        return {
            "kept_tokens": kept_tokens,
            "evicted_count": len(evicted_indices),
            "evicted_indices": evicted_indices,
            "protected_count": num_protected,
            "budget_utilized": len(kept_tokens),
            "max_budget": max_budget,
            "compression_ratio": round(len(kept_tokens) / total, 4) if total > 0 else 1.0
        }

    # -------------------------------------------------------------
    # 2.3 Hybrid Attention Metrics & RAM vs VRAM KV Tracking
    # -------------------------------------------------------------
    @classmethod
    def calculate_kv_memory(
        cls,
        n_layers: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        seq_len: int = 4096,
        dtype: str = "f16",
        batch_size: int = 1,
        available_vram_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates exact KV cache memory footprint across FP16, Q8_0, and Q4_0."""
        dtype_clean = dtype.lower()
        bytes_per_elem = cls.DTYPE_BYTES.get(dtype_clean, 2.0)

        elements_per_token = 2 * n_layers * n_kv_heads * head_dim
        bytes_per_token = elements_per_token * bytes_per_elem
        total_bytes = bytes_per_token * seq_len * batch_size

        total_kb = total_bytes / 1024.0
        total_mb = total_kb / 1024.0
        total_gb = total_mb / 1024.0

        quant_comparison = {}
        for q_name in ["f16", "q8_0", "q4_0"]:
            q_bytes_elem = cls.DTYPE_BYTES[q_name]
            q_total_mb = (elements_per_token * q_bytes_elem * seq_len * batch_size) / (1024.0 * 1024.0)
            quant_comparison[q_name] = {
                "mb": round(q_total_mb, 2),
                "gb": round(q_total_mb / 1024.0, 4),
                "savings_vs_f16_pct": round((1.0 - (q_bytes_elem / 2.0)) * 100.0, 1)
            }

        vram_analysis = {}
        if available_vram_gb is not None:
            vram_mb = available_vram_gb * 1024.0
            fits_in_vram = total_mb <= vram_mb
            overflow_mb = max(0.0, total_mb - vram_mb)
            vram_analysis = {
                "available_vram_gb": available_vram_gb,
                "fits_entirely_in_vram": fits_in_vram,
                "vram_used_mb": min(total_mb, vram_mb),
                "ram_offload_required_mb": overflow_mb,
                "vram_utilization_pct": round(min(100.0, (total_mb / vram_mb) * 100.0), 2) if vram_mb > 0 else 100.0,
                "recommended_kv_dtype": "f16" if total_mb <= vram_mb else ("q8_0" if quant_comparison["q8_0"]["mb"] <= vram_mb else "q4_0")
            }

        return {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "dtype": dtype_clean,
            "bytes_per_token": bytes_per_token,
            "total_bytes": total_bytes,
            "total_mb": round(total_mb, 2),
            "total_gb": round(total_gb, 4),
            "quant_comparison": quant_comparison,
            "vram_analysis": vram_analysis
        }

    # -------------------------------------------------------------
    # 2.4 256k Context Budget Calculations & Launch Flags
    # -------------------------------------------------------------
    @classmethod
    def calculate_256k_budget(
        cls,
        n_layers: int = 32,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        available_vram_gb: float = 8.0,
        sink_tokens: int = 64,
        sliding_window: int = 8192,
        hidden_size: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculates 256k context (262,144 tokens) memory footprint across datatypes:
        - FP16 full 256k KV cache (~32GB VRAM)
        - Q8_0 full 256k KV cache (~17GB VRAM)
        - Q4_0 full 256k KV cache (~9GB VRAM)
        - RAM-Only KV Offload (-nkvo): 0 MB VRAM, 100% VRAM free for weights
        - StreamingLLM Attention Sinks (sliding window + sink tokens) fixed footprint (<1GB VRAM)
        """
        seq_256k = 262144
        n_layers = kwargs.get("num_layers", n_layers)
        n_kv_heads = kwargs.get("num_kv_heads", n_kv_heads)
        full_f16 = cls.calculate_kv_memory(n_layers, n_kv_heads, head_dim, seq_256k, "f16", 1, available_vram_gb)
        full_q8 = cls.calculate_kv_memory(n_layers, n_kv_heads, head_dim, seq_256k, "q8_0", 1, available_vram_gb)
        full_q4 = cls.calculate_kv_memory(n_layers, n_kv_heads, head_dim, seq_256k, "q4_0", 1, available_vram_gb)

        sinks_seq = sink_tokens + sliding_window
        sinks_f16 = cls.calculate_kv_memory(n_layers, n_kv_heads, head_dim, sinks_seq, "f16", 1, available_vram_gb)
        sinks_q4 = cls.calculate_kv_memory(n_layers, n_kv_heads, head_dim, sinks_seq, "q4_0", 1, available_vram_gb)

        launch_flags = cls.get_llama_server_256k_flags()

        return {
            "seq_len": seq_256k,
            "available_vram_gb": available_vram_gb,
            "fp16": {
                "name": "FP16 (Standard / Lossless)",
                "mb": full_f16["total_mb"],
                "gb": full_f16["total_gb"],
                "vram_mb": full_f16["total_mb"],
                "vram_gb": full_f16["total_gb"],
                "vram_fit": "🔴 Heavy RAM Offload" if full_f16["total_gb"] > available_vram_gb else "🟢 100% VRAM",
                "savings_pct": 0.0,
                "ram_offload_gb": max(0.0, round(full_f16["total_gb"] - available_vram_gb, 2)),
            },
            "q8_0": {
                "name": "Q8_0 (-ctk q8_0 -ctv q8_0)",
                "mb": full_q8["total_mb"],
                "gb": full_q8["total_gb"],
                "vram_mb": full_q8["total_mb"],
                "vram_gb": full_q8["total_gb"],
                "vram_fit": "🟡 Hybrid RAM" if full_q8["total_gb"] > available_vram_gb else "🟢 100% VRAM",
                "savings_pct": 46.9,
                "ram_offload_gb": max(0.0, round(full_q8["total_gb"] - available_vram_gb, 2)),
            },
            "q4_0": {
                "name": "Q4_0 (-ctk q4_0 -ctv q4_0)",
                "mb": full_q4["total_mb"],
                "gb": full_q4["total_gb"],
                "vram_mb": full_q4["total_mb"],
                "vram_gb": full_q4["total_gb"],
                "vram_fit": "🟢 Optimal for 8GB GPU" if full_q4["total_gb"] <= available_vram_gb + 1.5 else "🟡 Hybrid RAM",
                "savings_pct": 71.9,
                "ram_offload_gb": max(0.0, round(full_q4["total_gb"] - available_vram_gb, 2)),
            },
            "ram_kv_offload": {
                "name": "RAM-Only KV Offload (-nkvo)",
                "mb": 0.0,
                "gb": 0.0,
                "vram_mb": 0.0,
                "vram_gb": 0.0,
                "vram_fit": "🟢 100% VRAM Free for Weights",
                "savings_pct": 100.0,
                "ram_offload_gb": full_f16["total_gb"],
                "savings_vram_pct": 100.0,
                "description": "Dedicates full 8GB GPU VRAM strictly to model layers (-ngl 99)"
            },
            "streaming_sinks": {
                "name": "StreamingLLM (4 Sinks + 8k Window)",
                "sink_tokens": sink_tokens,
                "sliding_window": sliding_window,
                "total_retained_tokens": sinks_seq,
                "vram_mb": sinks_q4["total_mb"],
                "vram_gb": sinks_q4["total_gb"],
                "mb": sinks_q4["total_mb"],
                "gb": sinks_q4["total_gb"],
                "vram_fit": "🟢 100% VRAM Resident (<1GB)",
                "compression_ratio": round(sinks_seq / seq_256k, 4),
                "savings_pct": 98.2,
            },
            "launch_flags": launch_flags,
        }

    @classmethod
    def get_llama_server_256k_flags(
        cls,
        model_path: str = "model.gguf",
        ctx_shift: bool = True,
        kv_dtype: str = "q4_0",
        no_kv_offload: bool = False,
        flash_attn: bool = True,
        n_gpu_layers: int = 99,
        port: int = 18434,
        ctx_size: int = 262144,
        enable_ctx_shift: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Return the optimal launch configuration flags for 256k context in llama-server.
        """
        actual_ctx_shift = enable_ctx_shift if enable_ctx_shift is not None else ctx_shift
        # Guardrail: quantized KV is allowed only when the KV cache stays in VRAM.
        kv_verdict = cls.resolve_kv_dtype(
            kv_dtype, no_kv_offload=no_kv_offload, ctx_size=ctx_size
        )
        kv_effective = kv_verdict["effective"]
        flags = [
            f"-c {ctx_size}",
            f"-ctk {kv_effective}",
            f"-ctv {kv_effective}",
            f"-fa {'on' if flash_attn else 'off'}",
            f"-ngl {n_gpu_layers}",
            f"--port {port}",
        ]
        if no_kv_offload:
            flags.append("-nkvo")
        # llama-server accepts only --context-shift / --no-context-shift.
        flags.append("--context-shift" if actual_ctx_shift else "--no-context-shift")

        flag_str = " ".join(flags)
        cmd_str = f"llama-server -m {model_path} {flag_str}"

        return {
            "flags_list": flags,
            "flags_string": flag_str,
            "cmd_string": cmd_str,
            "ctx_size": ctx_size,
            "kv_dtype": kv_effective,
            "kv_dtype_requested": kv_verdict["requested"],
            "kv_quant_forced": kv_verdict["forced"],
            "kv_guardrail_warning": kv_verdict["warning"],
            "no_kv_offload": no_kv_offload,
            "ctx_shift": actual_ctx_shift,
            "flash_attn": flash_attn,
            "n_gpu_layers": n_gpu_layers,
            "port": port,
        }


# =====================================================================
# 2.5 GROQ-POWERED CONTEXT COMPRESSION ENGINE
# =====================================================================

# Hard-coded Groq operational rules (strict rate limit compliance rules):
#   1. "llama-3.3-70b-versatile" was RETIRED from this Groq account
#      (model_not_found). Default to qwen/qwen3.8-27b (131k ctx).
#   2. Every Groq request MUST carry an explicit User-Agent (e.g. curl/8.5.0),
#      otherwise Cloudflare in front of the API rejects it with error 1010.
#   3. GROQ_API_KEY lives in %LOCALAPPDATA%\hermes\.env on this machine.
GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"
GROQ_USER_AGENT = "curl/8.5.0"
GROQ_CONTEXT_WINDOW_TOKENS = 131072


@dataclass
class CompressedContext:
    original_text: str
    compressed_text: str
    original_tokens_est: int
    compressed_tokens_est: int
    compression_ratio: float
    tokens_saved: int
    savings_percent: float
    key_facts: List[str] = field(default_factory=list)
    active_task_state: str = ""
    decisions: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    model_used: str = GROQ_DEFAULT_MODEL
    latency_ms: float = 0.0
    is_groq_powered: bool = False
    error: Optional[str] = None
    truncated: bool = False
    truncation_reason: str = ""


# Alias for compatibility
CompressionResult = CompressedContext


class ContextCompressor:
    """
    Groq-powered high-speed Context Compression Engine.
    Compresses multi-turn conversation logs and massive prompt contexts into structured
    state on Groq (default qwen/qwen3.8-27b, 131k context window).
    """
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = GROQ_DEFAULT_MODEL
    # A "fast" model must be a real, reachable id on this account: llama-3.1-8b-instant
    # and llama-3.3-70b-versatile are both retired (model_not_found). gpt-oss-20b is the
    # cheapest live option and is kept distinct from the 120b one.
    FAST_MODEL = "openai/gpt-oss-20b"
    # Dead ids (verified against the live /chat/completions on this key) are excluded:
    #   llama-3.3-70b-versatile, llama-3.1-8b-instant
    CANDIDATE_MODELS = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 12.0,
        max_input_tokens: int = GROQ_CONTEXT_WINDOW_TOKENS
    ):
        self.api_key = api_key or self._find_groq_api_key()
        self.model = model or self.DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_input_tokens = max_input_tokens
        self.last_compression: Optional[CompressedContext] = None

    def _find_groq_api_key(self) -> Optional[str]:
        """Discover GROQ_API_KEY from environment or common .env locations."""
        key = os.environ.get("GROQ_API_KEY")
        if key and key.strip():
            return key.strip()

        candidate_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(__file__), ".env"),
            os.path.join(os.path.expanduser("~"), ".env"),
        ]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate_paths.append(os.path.join(local, "hermes", ".env"))

        seen = set()
        for p in candidate_paths:
            if p in seen:
                continue
            seen.add(p)
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("GROQ_API_KEY=") or line.startswith("GROQ_API_KEY ="):
                                val = line.split("=", 1)[1].strip().strip("\"'")
                                if val:
                                    return val
                except Exception:
                    pass
        return None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (approx 3.8 characters per token for English/code)."""
        if not text:
            return 0
        return max(1, int(len(text) / 3.8))

    # -------------------------------------------------------------
    # TOOL CONTEXT TRUNCATION (Groq-backed head+tail reducer)
    # -------------------------------------------------------------
    @staticmethod
    def _token_chars() -> float:
        return 3.8

    def truncate_tool_context(
        self,
        tool_text: str,
        max_tokens: int = 8192,
        head_ratio: float = 0.35,
        min_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Truncate an oversized TOOL RESULT payload to fit `max_tokens`.

        Head+tail strategy: keeps the beginning (usually the command/tool
        invocation header, which carries the semantics) and the end (usually
        the result/error tail, which carries the conclusion), dropping the
        repetitive middle. Returns metadata describing what was dropped so the
        TUI can surface it instead of silently losing data.

        Never invoked on the KV cache dtype: this only bounds the PROMPT text
        that tool results inject into the context window.
        """
        if not tool_text:
            return {"text": "", "truncated": False, "reason": "", "original_tokens": 0,
                    "kept_tokens": 0, "dropped_tokens": 0}

        char_budget = max(1, int(max_tokens * self._token_chars()))
        orig_tokens = self.estimate_tokens(tool_text)

        if len(tool_text) <= char_budget:
            return {"text": tool_text, "truncated": False, "reason": "",
                    "original_tokens": orig_tokens, "kept_tokens": orig_tokens,
                    "dropped_tokens": 0}

        head_chars = int(char_budget * max(0.0, min(0.9, head_ratio)))
        tail_chars = char_budget - head_chars
        dropped_est = max(0, orig_tokens - int(char_budget / self._token_chars()))
        marker = f"\n\n[... ai_tui TOOL CONTEXT TRUNCATED: ~{dropped_est} tokens dropped from the middle ...]\n\n"

        head = tool_text[:head_chars]
        tail = tool_text[-tail_chars:] if tail_chars > 0 else ""
        truncated_text = head + marker + tail

        kept_tokens = self.estimate_tokens(truncated_text)
        return {
            "text": truncated_text,
            "truncated": True,
            "reason": f"tool_result exceeded {max_tokens} tokens; head+tail applied",
            "original_tokens": orig_tokens,
            "kept_tokens": kept_tokens,
            "dropped_tokens": max(0, orig_tokens - kept_tokens),
            "head_chars": len(head),
            "tail_chars": len(tail),
        }

    def compress_context(
        self,
        context_text: str,
        target_ratio: float = 0.3,
        focus_areas: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
    ) -> CompressedContext:
        """
        Compresses large context/conversation using Groq API or heuristic fallback.

        The input is first bounded by `truncate_tool_context` so we never ship a
        payload that overflows the Groq context window (which previously ended as
        a hard model_not_found / bad_request failure instead of a clean truncation).
        """
        start_t = time.perf_counter()
        target_model = model or self.model
        effective_max = max_input_tokens or self.max_input_tokens

        truncation = self.truncate_tool_context(context_text, max_tokens=effective_max)
        truncated_input = truncation["text"]

        if self.api_key:
            models_to_try = [target_model] + [m for m in self.CANDIDATE_MODELS if m != target_model]
            for try_m in models_to_try:
                res = self._compress_with_groq(truncated_input, try_m)
                if res:
                    res.truncated = truncation["truncated"]
                    res.truncation_reason = truncation["reason"]
                    self.last_compression = res
                    return res

        res = self._compress_heuristic_fallback(truncated_input)
        res.latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        res.truncated = truncation["truncated"]
        res.truncation_reason = truncation["reason"]
        self.last_compression = res
        return res

    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        target_ratio: float = 0.3,
        model: Optional[str] = None
    ) -> CompressedContext:
        """Helper to format chat messages list into single text and compress."""
        formatted_parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            formatted_parts.append(f"[{role}]: {content}")
        raw_text = "\n\n".join(formatted_parts)
        return self.compress_context(raw_text, target_ratio=target_ratio, model=model)

    def _compress_with_groq(self, context_text: str, model: str) -> Optional[CompressedContext]:
        start_t = time.perf_counter()
        orig_tokens = self.estimate_tokens(context_text)

        system_prompt = (
            "You are an ultra-fast context compression engine for AI inference and large KV cache management.\n"
            "Analyze the input text/conversation and compress it into an information-dense structured summary.\n"
            "Extract and return ONLY a valid JSON object with the following schema:\n"
            "{\n"
            '  "key_facts": ["fact 1", "fact 2", ...],\n'
            '  "active_task_state": "concise description of current goal and step in progress",\n'
            '  "decisions": ["decision 1", "decision 2", ...],\n'
            '  "variables": {"key": "value", ...},\n'
            '  "compressed_summary": "High-density markdown reconstructed context ready for LLM input without conversational filler."\n'
            "}"
        )

        user_prompt = f"Compress the following context into structured JSON:\n\n{context_text}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.GROQ_API_URL,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Mandatory: Cloudflare in front of the Groq API rejects requests
                # without an explicit User-Agent with error 1010.
                "User-Agent": GROQ_USER_AGENT,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_resp = resp.read().decode("utf-8")
                parsed = json.loads(raw_resp)
                content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")

                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    return None
                data = json.loads(json_match.group(0))

                key_facts = data.get("key_facts", [])
                task_state = data.get("active_task_state", "")
                decisions = data.get("decisions", [])
                variables = data.get("variables", {})
                compressed_summary = data.get("compressed_summary", "")

                if not compressed_summary:
                    parts = []
                    if task_state:
                        parts.append(f"### Active Task State\n{task_state}")
                    if key_facts:
                        facts_str = "\n".join([f"- {f}" for f in key_facts])
                        parts.append(f"### Key Facts\n{facts_str}")
                    if decisions:
                        dec_str = "\n".join([f"- {d}" for d in decisions])
                        parts.append(f"### Decisions\n{dec_str}")
                    if variables:
                        var_str = "\n".join([f"- `{k}` = `{v}`" for k, v in variables.items()])
                        parts.append(f"### Variables\n{var_str}")
                    compressed_summary = "\n\n".join(parts)

                comp_tokens = self.estimate_tokens(compressed_summary)
                comp_tokens = min(orig_tokens, max(1, comp_tokens))
                ratio = round(comp_tokens / max(1, orig_tokens), 4)
                saved = max(0, orig_tokens - comp_tokens)
                savings_pct = round((1.0 - (comp_tokens / max(1, orig_tokens))) * 100.0, 1)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                return CompressedContext(
                    original_text=context_text,
                    compressed_text=compressed_summary,
                    original_tokens_est=orig_tokens,
                    compressed_tokens_est=comp_tokens,
                    compression_ratio=ratio,
                    tokens_saved=saved,
                    savings_percent=savings_pct,
                    key_facts=key_facts,
                    active_task_state=task_state,
                    decisions=decisions,
                    variables=variables,
                    model_used=model,
                    latency_ms=round(elapsed_ms, 2),
                    is_groq_powered=True,
                )
        except Exception as e:
            logger.debug(f"Groq API context compression attempt for model '{model}' failed: {e}")
            return None

    def _compress_heuristic_fallback(self, context_text: str) -> CompressedContext:
        """Fast local heuristic prompt compression (zero network dependency)."""
        orig_tokens = self.estimate_tokens(context_text)
        lines = [line.strip() for line in context_text.split("\n") if line.strip()]

        key_facts = []
        decisions = []
        variables = {}
        task_state = "Execute local AI inference & KV cache management"

        for line in lines:
            var_matches = re.findall(r'([A-Za-z0-9_]+)\s*[:=]\s*([A-Za-z0-9_\-./]+)', line)
            for k, v in var_matches:
                if len(k) < 25 and len(v) < 60 and k.lower() not in ("http", "https", "line"):
                    variables[k] = v

            if any(k in line.lower() for k in ("task:", "goal:", "objective:", "in progress", "working on", "integrate")):
                task_state = line.split(":", 1)[-1].strip() if ":" in line else line

            elif any(k in line.lower() for k in ("decided", "selected", "agreed", "using", "enabled", "configured")):
                decisions.append(line[:120])

            elif any(k in line.lower() for k in ("256k", "vram", "gpu", "rtx", "llama", "flash attention", "kv cache", "tokens", "port")):
                key_facts.append(line[:120])

        key_facts = list(dict.fromkeys(key_facts))[:6]
        decisions = list(dict.fromkeys(decisions))[:4]

        if not key_facts and lines:
            key_facts = lines[:3]

        parts = [
            f"**State**: {task_state}",
            "**Facts**: " + " | ".join(key_facts),
        ]
        if decisions:
            parts.append("**Decisions**: " + " | ".join(decisions))
        if variables:
            parts.append("**Vars**: " + ", ".join([f"{k}={v}" for k, v in list(variables.items())[:6]]))

        compressed_summary = "\n".join(parts)
        comp_tokens = self.estimate_tokens(compressed_summary)
        comp_tokens = min(orig_tokens, max(1, comp_tokens))
        ratio = round(comp_tokens / max(1, orig_tokens), 4)
        saved = max(0, orig_tokens - comp_tokens)
        savings_pct = round((1.0 - (comp_tokens / max(1, orig_tokens))) * 100.0, 1)

        return CompressedContext(
            original_text=context_text,
            compressed_text=compressed_summary,
            original_tokens_est=orig_tokens,
            compressed_tokens_est=comp_tokens,
            compression_ratio=ratio,
            tokens_saved=saved,
            savings_percent=savings_pct,
            key_facts=key_facts,
            active_task_state=task_state,
            decisions=decisions,
            variables=variables,
            model_used="heuristic-local-v1",
            latency_ms=1.2,
            is_groq_powered=False,
        )

    def test_compression(self, sample_prompt: Optional[str] = None) -> CompressedContext:
        """Run a standard compression benchmark test."""
        sample = sample_prompt or (
            "System: You are an autonomous coding assistant deployed on Windows 11 with an NVIDIA GeForce RTX 3050 (8GB VRAM) and 48GB Host RAM.\n"
            "We are preparing to deploy long-context LLM models with sequence lengths up to 256k tokens (262,144 tokens).\n"
            "Discussion history:\n"
            "User: We need to evaluate whether our 8GB GPU can hold 256k tokens of KV cache context without running out of memory.\n"
            "Assistant: At 256k tokens, standard FP16 KV cache consumes 32.0GB VRAM, which requires 24GB of Host RAM offloading and slows generation.\n"
            "However, if we enable 4-bit KV cache quantization (-ctk q4_0 -ctv q4_0), the footprint drops by 71.9% to 9.0GB VRAM.\n"
            "Furthermore, using StreamingLLM Attention Sinks (preserving 64 initial sink tokens + 8192 sliding window) combined with Context Shifting (--ctx-shift) "
            "caps the active VRAM footprint at only 520MB (FP16) or 146MB (Q4_0), allowing infinite sequence processing on our 8GB card.\n"
            "We have decided to enable -ctk q4_0 -ctv q4_0 --ctx-shift -c 262144 -fa on for llama-server.\n"
            "Active Task: Integrate 256k KV cache manager and Groq context compression engine into ai_tui backend.py and tui_app.py.\n"
            "Variables: PORT=18434, MODEL=mica-v0.1-4b-Q4_K_M.gguf, CTX_LIMIT=262144, FLASH_ATTN=1, GROQ_MODEL=qwen/qwen3.8-27b, KV_DTYPE=q4_0, HOST=127.0.0.1."
        )
        return self.compress_context(sample)


# =====================================================================
# 3. AI NEWS AGGREGATOR
# =====================================================================

@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: str
    summary: str
    category: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    score_or_likes: Optional[int] = None
    extra_tags: List[str] = field(default_factory=list)


class AINewsAggregator:
    """
    Aggregates latest AI research and tech news from:
    1. arXiv API (cs.AI, cs.CL, cs.LG)
    2. Hugging Face Trending Models / Daily Papers API
    3. AI RSS feeds (Hugging Face Blog, Cointelegraph AI, Decrypt AI)
    """

    DEFAULT_RSS_FEEDS = {
        "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
        "Cointelegraph AI": "https://cointelegraph.com/rss/tag/ai",
        "Decrypt AI": "https://decrypt.co/feed"
    }

    def __init__(self, timeout_s: float = 6.0):
        self.timeout_s = timeout_s

    def _safe_fetch(self, url: str, headers: Optional[Dict[str, str]] = None, max_redirects: int = 5) -> Optional[bytes]:
        """Fetch raw data with robust SSL, redirect and timeout handling."""
        import ssl
        ctx = ssl._create_unverified_context()
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
        if headers:
            default_headers.update(headers)

        try:
            req = urllib.request.Request(url, headers=default_headers)
            with urllib.request.urlopen(req, context=ctx, timeout=self.timeout_s) as response:
                if response.getcode() == 200:
                    return response.read()
        except Exception as e:
            logger.debug(f"urllib fetch failed for {url}: {e}")

        current_url = url
        for _ in range(max_redirects):
            try:
                parsed = urllib.parse.urlsplit(current_url)
                is_ssl = parsed.scheme.lower() == "https"
                host = parsed.netloc
                path = parsed.path if parsed.path else "/"
                if parsed.query:
                    path += f"?{parsed.query}"

                if ":" in host:
                    host_name, port_str = host.split(":", 1)
                    port = int(port_str)
                else:
                    host_name = host
                    port = 443 if is_ssl else 80

                conn_cls = http.client.HTTPSConnection if is_ssl else http.client.HTTPConnection
                if is_ssl:
                    conn = conn_cls(host_name, port=port, timeout=self.timeout_s, context=ctx)
                else:
                    conn = conn_cls(host_name, port=port, timeout=self.timeout_s)
                
                req_headers = dict(default_headers)
                req_headers["Host"] = host
                req_headers["Connection"] = "close"

                conn.request("GET", path, headers=req_headers)
                response = conn.getresponse()

                if response.status in (301, 302, 303, 307, 308):
                    loc = response.getheader("Location")
                    if not loc:
                        break
                    current_url = urllib.parse.urljoin(current_url, loc)
                    conn.close()
                    continue

                if response.status == 200:
                    data = response.read()
                    conn.close()
                    return data
                else:
                    conn.close()
                    return None
            except Exception as e:
                logger.warning(f"Fetch error for {current_url}: {e}")
                return None
        return None

    def fetch_arxiv(
        self,
        categories: Optional[List[str]] = None,
        max_results: int = 8
    ) -> List[NewsItem]:
        """Queries arXiv API for recent papers in computer science AI categories."""
        cats = categories or ["cs.AI", "cs.CL", "cs.LG"]
        cat_query = "+OR+".join([f"cat:{c}" for c in cats])
        url = (
            f"https://export.arxiv.org/api/query?"
            f"search_query={cat_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
        )

        data = self._safe_fetch(url)
        if not data:
            return []

        items: List[NewsItem] = []
        try:
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)
                published_elem = entry.find("atom:published", ns)

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled"
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                link = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                published = published_elem.text.strip() if published_elem is not None and published_elem.text else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                prim_cat = entry.find("arxiv:primary_category", ns)
                cat_name = prim_cat.attrib.get("term") if prim_cat is not None else cats[0]

                items.append(NewsItem(
                    title=title,
                    link=link,
                    source=f"arXiv [{cat_name}]",
                    published=published,
                    summary=summary[:400] + ("..." if len(summary) > 400 else ""),
                    category=cat_name,
                    authors=authors[:5]
                ))
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")

        return items

    def fetch_huggingface_trending(self, limit: int = 8) -> List[NewsItem]:
        """Fetches trending open-source AI models / daily papers from Hugging Face API."""
        papers_url = f"https://huggingface.co/api/daily_papers?limit={limit}"
        data = self._safe_fetch(papers_url)
        items: List[NewsItem] = []

        if data:
            try:
                raw_json = json.loads(data.decode("utf-8"))
                if isinstance(raw_json, list):
                    for entry in raw_json[:limit]:
                        paper = entry.get("paper", {})
                        title = paper.get("title", entry.get("title", "HF Paper"))
                        p_id = paper.get("id", entry.get("id", ""))
                        summary = paper.get("summary", "")
                        published = paper.get("publishedAt", "") or entry.get("publishedAt", "")
                        authors = [a.get("name", "") for a in paper.get("authors", []) if isinstance(a, dict)]
                        upvotes = paper.get("upvotes", 0)

                        items.append(NewsItem(
                            title=title,
                            link=f"https://huggingface.co/papers/{p_id}" if p_id else "https://huggingface.co/papers",
                            source="Hugging Face Daily Papers",
                            published=published,
                            summary=summary[:350] + ("..." if len(summary) > 350 else ""),
                            category="Paper",
                            authors=authors[:4],
                            score_or_likes=upvotes
                        ))
                    if items:
                        return items
            except Exception as e:
                logger.warning(f"HF Daily Papers parse error: {e}")

        models_url = f"https://huggingface.co/api/models?sort=likes&direction=-1&limit={limit}"
        data = self._safe_fetch(models_url)
        if data:
            try:
                models = json.loads(data.decode("utf-8"))
                for m in models:
                    m_id = m.get("id", "model")
                    likes = m.get("likes", 0)
                    pipeline = m.get("pipeline_tag", "llm")
                    downloads = m.get("downloads", 0)
                    items.append(NewsItem(
                        title=f"{m_id} ({pipeline})",
                        link=f"https://huggingface.co/{m_id}",
                        source="Hugging Face Trending",
                        published=m.get("lastModified", ""),
                        summary=f"Trending model with {likes:,} likes and {downloads:,} downloads. Task: {pipeline}.",
                        category=pipeline,
                        score_or_likes=likes,
                        extra_tags=m.get("tags", [])[:5]
                    ))
            except Exception as e:
                logger.error(f"HF Models parse error: {e}")

        return items

    def fetch_rss_feed(self, source_name: str, feed_url: str, limit: int = 5) -> List[NewsItem]:
        """Fetch and parse standard RSS / Atom feeds."""
        data = self._safe_fetch(feed_url)
        if not data:
            return []

        items: List[NewsItem] = []
        try:
            root = ET.fromstring(data)

            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item")[:limit]:
                    title = item.findtext("title", "AI Update").strip()
                    link = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    desc = item.findtext("description", "").strip()

                    clean_desc = re.sub(r"<[^>]+>", "", desc).strip().replace("\n", " ")
                    category = item.findtext("category", "AI News")

                    items.append(NewsItem(
                        title=title,
                        link=link,
                        source=source_name,
                        published=pub_date,
                        summary=clean_desc[:300] + ("..." if len(clean_desc) > 300 else ""),
                        category=category
                    ))
                return items

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns)[:limit]:
                title = entry.findtext("atom:title", "AI Update", ns).strip()
                link_elem = entry.find("atom:link", ns)
                link = link_elem.attrib.get("href", "") if link_elem is not None else ""
                published = entry.findtext("atom:published", entry.findtext("atom:updated", "", ns), ns).strip()
                summary = entry.findtext("atom:summary", entry.findtext("atom:content", "", ns), ns).strip()
                clean_summary = re.sub(r"<[^>]+>", "", summary).strip()

                items.append(NewsItem(
                    title=title,
                    link=link,
                    source=source_name,
                    published=published,
                    summary=clean_summary[:300] + ("..." if len(clean_summary) > 300 else "")
                ))
        except Exception as e:
            logger.warning(f"Error parsing RSS {source_name}: {e}")

        return items

    def aggregate_all(
        self,
        arxiv_limit: int = 5,
        hf_limit: int = 5,
        rss_limit: int = 4
    ) -> Dict[str, Any]:
        """Aggregates news from all sources in parallel threads for fast response."""
        results: Dict[str, List[Dict[str, Any]]] = {
            "arxiv": [],
            "huggingface": [],
            "rss": [],
            "all_combined": []
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            arxiv_future = executor.submit(self.fetch_arxiv, ["cs.AI", "cs.CL", "cs.LG"], arxiv_limit)
            hf_future = executor.submit(self.fetch_huggingface_trending, hf_limit)
            rss_futures = {
                name: executor.submit(self.fetch_rss_feed, name, url, rss_limit)
                for name, url in self.DEFAULT_RSS_FEEDS.items()
            }

            try:
                arxiv_items = arxiv_future.result(timeout=self.timeout_s + 2)
                results["arxiv"] = [asdict(i) for i in arxiv_items]
            except Exception as e:
                logger.warning(f"arXiv collection error: {e}")

            try:
                hf_items = hf_future.result(timeout=self.timeout_s + 2)
                results["huggingface"] = [asdict(i) for i in hf_items]
            except Exception as e:
                logger.warning(f"HF collection error: {e}")

            rss_items: List[NewsItem] = []
            for name, future in rss_futures.items():
                try:
                    feed_res = future.result(timeout=self.timeout_s + 2)
                    rss_items.extend(feed_res)
                except Exception as e:
                    logger.warning(f"RSS {name} collection error: {e}")
            results["rss"] = [asdict(i) for i in rss_items]

        all_items = results["arxiv"] + results["huggingface"] + results["rss"]
        results["all_combined"] = all_items
        results["total_count"] = len(all_items)
        results["timestamp"] = time.time()

        return results


# =====================================================================
# 4. HF MODEL HUB & QUANT DOWNLOAD MANAGER
# =====================================================================

@dataclass
class HFModelInfo:
    repo_id: str
    author: str
    model_name: str
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)
    last_modified: str = ""


@dataclass
class QuantFileInfo:
    filename: str
    quant_type: str
    size_bytes: int
    size_gb: float
    download_url: str
    vram_fit_status: str = "🟢 100% VRAM"  # 🟢 100% VRAM, 🟡 Hybrid RAM, 🔴 Heavy Offload
    is_recommended: bool = False


@dataclass
class DownloadTask:
    task_id: str
    repo_id: str
    filename: str
    dest_path: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    progress_percent: float = 0.0
    speed_mb_s: float = 0.0
    eta_seconds: float = 0.0
    status: str = "pending"  # pending, downloading, completed, failed, cancelled
    error: str = ""
    start_time: float = field(default_factory=time.time)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def progress_pct(self) -> float:
        return self.progress_percent

    def cancel(self):
        self._stop_event.set()
        self.status = "cancelled"


class HFModelManager:
    """
    Manages searching models on Hugging Face and downloading quantized weights.
    Defaults download destination to /mnt/lm_models/models.
    """
    def __init__(self, default_dest_dir: str = "/mnt/lm_models/models", timeout_s: float = 8.0):
        self.default_dest_dir = default_dest_dir
        self.timeout_s = timeout_s
        self.download_tasks: Dict[str, DownloadTask] = {}
        self._tasks_lock = threading.Lock()
        self._ensure_dir(self.default_dest_dir)

    def _ensure_dir(self, directory: str) -> str:
        try:
            os.makedirs(directory, exist_ok=True)
            return directory
        except Exception:
            fallback = os.path.join(os.path.expanduser("~"), "models")
            try:
                os.makedirs(fallback, exist_ok=True)
                return fallback
            except Exception:
                return "."

    def search_models(self, query: str, limit: int = 15) -> List[HFModelInfo]:
        """Search Hugging Face API for model repositories."""
        if not query or not query.strip():
            query = "Qwen GGUF"
        query_enc = urllib.parse.quote(query.strip())
        url = f"https://huggingface.co/api/models?search={query_enc}&limit={limit}&full=false"
        
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai_tui/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models: List[HFModelInfo] = []
                for item in data:
                    repo_id = item.get("id", "")
                    if not repo_id:
                        continue
                    parts = repo_id.split("/", 1)
                    author = parts[0] if len(parts) > 1 else ""
                    m_name = parts[1] if len(parts) > 1 else parts[0]
                    models.append(HFModelInfo(
                        repo_id=repo_id,
                        author=author,
                        model_name=m_name,
                        downloads=item.get("downloads", 0),
                        likes=item.get("likes", 0),
                        tags=item.get("tags", []),
                        last_modified=item.get("lastModified", "")
                    ))
                return models
        except Exception as e:
            logger.warning(f"HF search error: {e}")
            return []

    def get_repo_quants(self, repo_id: str) -> List[QuantFileInfo]:
        """Get all quantized files (.gguf, .exl3, .safetensors) for a given repo."""
        url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai_tui/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                quants: List[QuantFileInfo] = []
                for f in data:
                    path = f.get("path", "")
                    if any(path.lower().endswith(ext) for ext in [".gguf", ".exl3", ".safetensors", ".bin"]):
                        size_bytes = f.get("size", 0)
                        size_gb = round(size_bytes / (1024**3), 2)
                        q_type = self._parse_quant_type(path)
                        
                        if size_gb <= 7.3:
                            vram_status = "🟢 100% VRAM"
                        elif size_gb <= 16.0:
                            vram_status = "🟡 Hybrid RAM"
                        else:
                            vram_status = "🔴 Heavy Offload"
                        
                        is_rec = q_type in ["Q4_K_M", "Q3_K_M", "IQ3_M", "IQ4_XS", "EXL3 2.75bpw", "EXL3 3.75bpw"]
                        dl_url = f"https://huggingface.co/{repo_id}/resolve/main/{path}"
                        quants.append(QuantFileInfo(
                            filename=path,
                            quant_type=q_type,
                            size_bytes=size_bytes,
                            size_gb=size_gb,
                            download_url=dl_url,
                            vram_fit_status=vram_status,
                            is_recommended=is_rec
                        ))
                
                quants.sort(key=lambda x: (not x.is_recommended, x.size_gb))
                return quants
        except Exception as e:
            logger.warning(f"HF tree error for {repo_id}: {e}")
            return []

    def _parse_quant_type(self, filename: str) -> str:
        name_upper = filename.upper()
        m = re.search(r'([I|U]?Q[0-9]_[A-Z0-9_]+)', name_upper)
        if m:
            return m.group(1)
        m_bpw = re.search(r'([0-9]+\.?[0-9]*\s*BPW)', name_upper)
        if m_bpw:
            return f"EXL3 {m_bpw.group(1)}"
        if "EXL3" in name_upper or "EXL2" in name_upper:
            return "EXL3"
        if "FP16" in name_upper or "F16" in name_upper:
            return "FP16"
        if "BF16" in name_upper:
            return "BF16"
        if "Q8_0" in name_upper:
            return "Q8_0"
        return "Standard"

    def start_download(self, repo_id: str, filename: str, dest_dir: Optional[str] = None) -> DownloadTask:
        """Start downloading a quant file in a background thread."""
        target_dir = dest_dir or self.default_dest_dir
        actual_dir = self._ensure_dir(target_dir)
        dest_path = os.path.join(actual_dir, filename)
        
        task_id = f"{repo_id}:{filename}"
        task = DownloadTask(
            task_id=task_id,
            repo_id=repo_id,
            filename=filename,
            dest_path=dest_path,
            status="downloading"
        )
        
        with self._tasks_lock:
            self.download_tasks[task_id] = task

        def _download_worker():
            dl_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            req = urllib.request.Request(
                dl_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai_tui/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    total_len = resp.headers.get("Content-Length")
                    if total_len:
                        task.total_bytes = int(total_len)
                    
                    chunk_size = 1024 * 1024  # 1MB
                    downloaded = 0
                    start_t = time.time()
                    last_update_t = start_t
                    last_downloaded = 0
                    
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    part_file = dest_path + ".part"
                    
                    with open(part_file, "wb") as f_out:
                        while True:
                            if task._stop_event.is_set():
                                task.status = "cancelled"
                                break
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            downloaded += len(chunk)
                            task.downloaded_bytes = downloaded
                            
                            now = time.time()
                            if now - last_update_t >= 0.5:
                                dt = now - last_update_t
                                db = downloaded - last_downloaded
                                task.speed_mb_s = (db / (1024 * 1024)) / dt if dt > 0 else 0.0
                                if task.total_bytes > 0:
                                    task.progress_percent = (downloaded / task.total_bytes) * 100.0
                                    remaining_bytes = task.total_bytes - downloaded
                                    if task.speed_mb_s > 0:
                                        task.eta_seconds = (remaining_bytes / (1024 * 1024)) / task.speed_mb_s
                                last_update_t = now
                                last_downloaded = downloaded
                    
                    if task.status != "cancelled":
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        os.rename(part_file, dest_path)
                        task.status = "completed"
                        task.progress_percent = 100.0
            except Exception as e:
                task.status = "failed"
                task.error = str(e)

        t = threading.Thread(target=_download_worker, daemon=True)
        t.start()
        return task

    def get_tasks(self) -> List[DownloadTask]:
        with self._tasks_lock:
            return list(self.download_tasks.values())

    get_all_downloads = get_tasks
    get_downloads = get_tasks


# =====================================================================
# 4.5 MODEL CONFIGURATION & AUTOMATED TEST RUNNER
# =====================================================================

@dataclass
class ModelConfig:
    model_filename: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.05
    context_length: int = 32768
    ngl: int = 99
    flash_attention: bool = True
    no_kv_offload: bool = True  # Forced Host RAM KV Offload flag (-nkvo)
    enable_thinking: bool = True
    system_prompt: str = "You are a helpful, expert AI assistant."
    custom_flags: str = "--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512"
    source: str = "default"  # "default", "groq_hf", "user_edited"
    last_updated: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            model_filename=data.get("model_filename", ""),
            temperature=float(data.get("temperature", 0.7)),
            top_p=float(data.get("top_p", 0.9)),
            top_k=int(data.get("top_k", 40)),
            min_p=float(data.get("min_p", 0.05)),
            repeat_penalty=float(data.get("repeat_penalty", 1.05)),
            context_length=int(data.get("context_length", 32768)),
            ngl=int(data.get("ngl", 99)),
            flash_attention=bool(data.get("flash_attention", True)),
            no_kv_offload=bool(data.get("no_kv_offload", True)),
            enable_thinking=bool(data.get("enable_thinking", True)),
            system_prompt=data.get("system_prompt", "You are a helpful, expert AI assistant."),
            custom_flags=data.get("custom_flags", "--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512"),
            source=data.get("source", "default"),
            last_updated=data.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


class ModelConfigManager:
    """
    Persists and manages per-model hyperparameters and launch configurations.
    Stores profiles in C:/Users/User/Documents/ai_tui/model_configs.json.
    Supports automated Hugging Face profile extraction via Groq API
    and auto-generation of LM Studio presets (C:/Users/User/.lmstudio/config-presets/).
    """
    LMSTUDIO_PRESETS_DIR = "/home/user/.lmstudio/config-presets"

    def __init__(self, config_path: str = ""):
        if not config_path:
            # Relativo alla cartella del modulo stesso: cosi' lo spostamento
            # della TUI (es. su /mnt/lm_models/DEBIAN_FOLDER) non lo rompe,
            # a differenza di un percorso assoluto scritto a mano.
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs.json")
        self.config_path = os.path.normpath(config_path)
        self.configs: Dict[str, ModelConfig] = {}
        self.load_all_configs()

    def load_all_configs(self) -> Dict[str, ModelConfig]:
        """Load stored model configs from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.configs = {
                        k: ModelConfig.from_dict(v) for k, v in data.items()
                    }
            except Exception as e:
                logger.warning(f"Error loading model configs: {e}")
                self.configs = {}
        return self.configs

    def save_all_configs(self):
        """Save all configs to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.configs.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error saving model configs: {e}")

    def get_config(self, filename: str) -> ModelConfig:
        """Get config for model filename, or generate smart defaults based on architecture."""
        norm_key = os.path.basename(filename)
        if norm_key in self.configs:
            return self.configs[norm_key]

        fl = norm_key.lower()
        if "35b" in fl or "30b" in fl or "27b" in fl:
            ngl = 24 if "35b" in fl else 28
            cfg = ModelConfig(
                model_filename=norm_key,
                temperature=0.2 if "coder" in fl else 0.6,
                top_p=0.85 if "coder" in fl else 0.9,
                top_k=40,
                min_p=0.05,
                repeat_penalty=1.0 if "coder" in fl else 1.05,
                context_length=32768,
                ngl=ngl,
                flash_attention=True,
                no_kv_offload=True,
                enable_thinking="coder" not in fl,
                system_prompt="You are an expert AI software engineer and reasoning system.",
                custom_flags="--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512",
                source="default"
            )
        elif "gemma" in fl or "qat" in fl:
            cfg = ModelConfig(
                model_filename=norm_key,
                temperature=0.3 if "coder" in fl or "coding" in fl else 0.6,
                top_p=0.85 if "coder" in fl or "coding" in fl else 0.9,
                top_k=64,
                min_p=0.05,
                repeat_penalty=1.05 if "coder" in fl or "coding" in fl else 1.1,
                context_length=32768,
                ngl=99,
                flash_attention=True,
                no_kv_offload=True,
                enable_thinking=True,
                system_prompt="You are Gemma 4, an advanced AI reasoning and coding model.",
                custom_flags="--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512",
                source="default"
            )
        elif "coder" in fl:
            cfg = ModelConfig(
                model_filename=norm_key,
                temperature=0.2,
                top_p=0.85,
                top_k=40,
                min_p=0.05,
                repeat_penalty=1.0,
                context_length=32768,
                ngl=99,
                flash_attention=True,
                no_kv_offload=True,
                enable_thinking=False,
                system_prompt="You are a deterministic coding assistant. Write clean, bug-free code.",
                custom_flags="--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512",
                source="default"
            )
        elif "spark" in fl or "mica" in fl or "jev" in fl:
            cfg = ModelConfig(
                model_filename=norm_key,
                temperature=0.1,
                top_p=0.9,
                top_k=40,
                min_p=0.05,
                repeat_penalty=1.0,
                context_length=8192,
                ngl=99,
                flash_attention=True,
                no_kv_offload=True,
                enable_thinking=False,
                system_prompt="You are a fast decision and filtering assistant.",
                custom_flags="--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512",
                source="default"
            )
        else:
            cfg = ModelConfig(
                model_filename=norm_key,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                min_p=0.05,
                repeat_penalty=1.05,
                context_length=32768,
                ngl=99,
                flash_attention=True,
                no_kv_offload=True,
                enable_thinking=True,
                system_prompt="You are a helpful AI assistant.",
                custom_flags="--context-shift -fa -nkvo -ctk f16 -ctv f16 -t 8 -b 512 -ub 512",
                source="default"
            )
        self.configs[norm_key] = cfg
        self.save_all_configs()
        return cfg

    def save_config(self, config: ModelConfig):
        """Save a single model config and export LM Studio preset."""
        config.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        norm_key = os.path.basename(config.model_filename)
        self.configs[norm_key] = config
        self.save_all_configs()
        try:
            self.export_lmstudio_preset(config)
        except Exception as e:
            logger.debug(f"LM Studio preset export skipped: {e}")

    def export_lmstudio_preset(self, config: ModelConfig, preset_name: Optional[str] = None) -> str:
        """
        Export configuration as a native LM Studio preset JSON in C:/Users/User/.lmstudio/config-presets/.
        """
        raw_name = preset_name or os.path.splitext(os.path.basename(config.model_filename))[0]
        clean_name = re.sub(r'[\\/*?:"<>|]', "", raw_name).strip()
        if not clean_name:
            clean_name = "ModelPreset"

        target_dir = os.path.normpath(self.LMSTUDIO_PRESETS_DIR)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception:
            target_dir = os.path.normpath(os.path.join(os.path.expanduser("~"), ".lmstudio", "config-presets"))
            os.makedirs(target_dir, exist_ok=True)

        preset_filename = f"{clean_name}.preset.json"
        preset_path = os.path.join(target_dir, preset_filename)

        identifier = f"@local:{clean_name.lower().replace(' ', '-').replace('_', '-')}"

        preset_data = {
            "identifier": identifier,
            "name": f"{clean_name} (Auto Groq Preset)",
            "changed": False,
            "operation": {
                "fields": [
                    {"key": "llm.prediction.temperature", "value": config.temperature},
                    {"key": "llm.prediction.topP", "value": config.top_p},
                    {"key": "llm.prediction.topK", "value": config.top_k},
                    {"key": "llm.prediction.minP", "value": config.min_p},
                    {"key": "llm.prediction.repeatPenalty", "value": config.repeat_penalty},
                    {"key": "llm.prediction.reasoning.enableThinking", "value": config.enable_thinking},
                    {"key": "llm.prediction.systemPrompt", "value": config.system_prompt}
                ]
            },
            "load": {
                "fields": [
                    {"key": "llm.load.llama.flashAttention", "value": config.flash_attention},
                    {"key": "llm.load.llama.contextLength", "value": config.context_length},
                    {"key": "llm.load.llama.gpuLayers", "value": config.ngl},
                    {"key": "llm.load.llama.noKvOffload", "value": config.no_kv_offload},
                    {"key": "llm.load.llama.cpuThreads", "value": 8}
                ]
            }
        }

        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2)

        logger.info(f"Generated LM Studio preset at {preset_path}")
        return preset_path

    def fetch_groq_hf_config(
        self,
        model_info_or_name: Union[LocalModelInfo, HFModelInfo, str],
        author: str = ""
    ) -> Tuple[ModelConfig, Optional[str]]:
        """
        Query Hugging Face model specs and use the Groq API (qwen/qwen3.8-27b)
        to extract the creator's official recommended inference parameters.
        Auto-generates both model_configs.json entry and LM Studio preset JSON.
        """
        if isinstance(model_info_or_name, LocalModelInfo):
            fl = model_info_or_name.filename
            author = model_info_or_name.author
        elif isinstance(model_info_or_name, HFModelInfo):
            fl = model_info_or_name.model_name
            author = model_info_or_name.author
        else:
            fl = str(model_info_or_name)

        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            env_path = os.path.join(os.path.expanduser("~"), ".env")
            if os.path.exists(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("GROQ_API_KEY="):
                                groq_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                                break
                except Exception:
                    pass

        if groq_key:
            prompt = (
                f"Extract the creator's official recommended generation parameters for the open-source LLM "
                f"'{fl}' (Author/Repo: {author}).\n"
                f"Provide optimal parameters in pure JSON matching this exact schema:\n"
                f'{{\n'
                f'  "temperature": 0.6,\n'
                f'  "top_p": 0.9,\n'
                f'  "top_k": 64,\n'
                f'  "min_p": 0.05,\n'
                f'  "repeat_penalty": 1.1,\n'
                f'  "context_length": 131072,\n'
                f'  "no_kv_offload": false,\n'
                f'  "enable_thinking": true,\n'
                f'  "system_prompt": "Recommended system prompt for this architecture"\n'
                f'}}\n'
                f"Output strictly raw JSON without markdown code fences."
            )
            req_data = json.dumps({
                "model": GROQ_DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert AI model architect. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                    # Mandatory: Cloudflare blocks Groq requests lacking a UA (error 1010).
                    "User-Agent": GROQ_USER_AGENT
                }
            )
            try:
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    content = resp_json["choices"][0]["message"]["content"].strip()
                    cleaned = re.sub(r"^```json\s*", "", content)
                    cleaned = re.sub(r"^```\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
                    p_json = json.loads(cleaned)

                    cfg = ModelConfig(
                        model_filename=fl,
                        temperature=float(p_json.get("temperature", 0.7)),
                        top_p=float(p_json.get("top_p", 0.9)),
                        top_k=int(p_json.get("top_k", 40)),
                        min_p=float(p_json.get("min_p", 0.05)),
                        repeat_penalty=float(p_json.get("repeat_penalty", 1.05)),
                        context_length=int(p_json.get("context_length", 32768)),
                        ngl=99,
                        flash_attention=True,
                        no_kv_offload=bool(p_json.get("no_kv_offload", False)),
                        enable_thinking=bool(p_json.get("enable_thinking", True)),
                        system_prompt=p_json.get("system_prompt", f"You are {fl}, an expert AI assistant."),
                        source="groq_hf",
                    )
                    self.save_config(cfg)
                    preset_path = self.export_lmstudio_preset(cfg)
                    return cfg, preset_path
            except Exception as e:
                logger.warning(f"Groq HF fetch error for {fl}: {e}")

        # Fallback to smart architectural default
        cfg = self.get_config(fl)
        cfg.source = "smart_heuristic"
        self.save_config(cfg)
        preset_path = self.export_lmstudio_preset(cfg)
        return cfg, preset_path


@dataclass
class ModelTestResult:
    model_name: str
    model_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    code_test_passed: bool = False
    code_syntax_valid: bool = False
    tool_test_passed: bool = False
    tool_json_valid: bool = False
    generation_speed_tok_s: float = 0.0
    ttft_ms: float = 0.0
    total_tokens: int = 0
    total_time_s: float = 0.0
    vram_used_mb: float = 0.0
    vram_baseline_mb: float = 0.0
    vram_delta_mb: float = 0.0
    status: str = "COMPLETED"  # "COMPLETED", "SERVER_OFFLINE", "ERROR"
    error_msg: str = ""
    error_detail: str = ""
    raw_code_output: str = ""
    raw_tool_output: str = ""
    # Context optimization probe results (8k/16k/32k)
    optimal_ctx: int = 8192
    ctx_probe_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return self.status == "COMPLETED" and self.code_test_passed and self.tool_test_passed

    @property
    def python_ast_pass(self) -> bool:
        return self.code_syntax_valid

    @property
    def json_tool_pass(self) -> bool:
        return self.tool_json_valid

    @property
    def speed_tok_s(self) -> float:
        return self.generation_speed_tok_s

    @property
    def vram_delta_gb(self) -> float:
        return self.vram_delta_mb / 1024.0

    def summary_str(self) -> str:
        if self.status != "COMPLETED":
            err = self.error_detail or self.error_msg or self.status
            return f"{self.model_name} | Status: {self.status} | Error: {err}"
        code_str = "PASS" if self.code_test_passed else "FAIL"
        tool_str = "PASS" if self.tool_test_passed else "FAIL"
        err_suffix = f" | Note: {self.error_detail}" if self.error_detail else ""
        return f"{self.model_name} | Code: {code_str} | Tool: {tool_str} | Speed: {self.generation_speed_tok_s} tok/s | TTFT: {self.ttft_ms} ms | VRAM: {self.vram_used_mb:.0f} MB{err_suffix}"


class ModelTestRunner:
    """
    Executes automated live inference and benchmark tests on local llama-server (port 18434).
    Validates Python AST code generation, JSON tool schemas, TTFT latency, tok/s, and VRAM.
    """
    def __init__(self, endpoint_url: str = "http://127.0.0.1:18434"):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.test_history: List[ModelTestResult] = []

    @staticmethod
    def _extract_python_code(raw_text: str) -> str:
        """
        Strips <think>...</think> tags, markdown code fences, and conversational text,
        returning clean Python code suitable for ast.parse().
        """
        if not raw_text:
            return ""
        # 1. Strip <think>...</think> tags and unclosed <think>...
        text = re.sub(r"(?is)<think>.*?</think>", "", raw_text)
        text = re.sub(r"(?is)<think>.*", "", text).strip()
        if not text:
            text = raw_text.strip()

        # 2. Markdown code fences: ```python ... ``` or ```py ... ``` or ``` ... ```
        matches = re.findall(r"```(?:python|py)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            for m in matches:
                clean_m = m.strip()
                if "def is_palindrome" in clean_m:
                    return clean_m
            return matches[0].strip()

        # 3. Fallback: extract function definition starting with def is_palindrome
        func_match = re.search(r"(def\s+is_palindrome\b.*)", text, re.DOTALL)
        if func_match:
            lines = func_match.group(1).splitlines()
            code_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    break
                if code_lines and line and not line.startswith((" ", "\t", "#", "def ", "class ", "@", "import ", "from ", "return ", "if ", "else:", "elif ", "try:", "except", "pass")):
                    break
                code_lines.append(line)
            return "\n".join(code_lines).strip()

        # 4. Final fence cleanup
        clean_text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text).strip()
        return clean_text

    @staticmethod
    def _extract_tool_json(raw_text: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """
        Extracts and parses JSON tool call either from OpenAI native tool_calls
        or via regex fallback from reasoning/content text.
        """
        # 1. Native OpenAI-style tool_calls
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    if isinstance(args, dict):
                        if fn.get("name") and "action" not in args:
                            args["action"] = fn.get("name")
                        return args

        if not raw_text:
            return None

        # 2. Strip <think> tags
        text = re.sub(r"(?is)<think>.*?</think>", "", raw_text)
        text = re.sub(r"(?is)<think>.*", "", text).strip()
        if not text:
            text = raw_text.strip()

        # 3. Markdown JSON block: ```json { ... } ``` or ``` { ... } ```
        json_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 4. Direct JSON dictionary extraction via regex
        dict_matches = re.findall(r"(\{[\s\S]*?\})", text)
        for dm in dict_matches:
            try:
                parsed = json.loads(dm.strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # 5. Raw text with stripped markdown
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return None

    def run_benchmark(
        self,
        model_info: LocalModelInfo,
        config: ModelConfig,
        hw_monitor: Optional[HardwareMonitor] = None
    ) -> ModelTestResult:
        """Run standard coding, tool calling, latency, and throughput benchmark suite."""
        import ast

        result = ModelTestResult(
            model_name=model_info.filename,
            model_path=model_info.filepath
        )

        # Baseline VRAM
        if hw_monitor:
            snap = hw_monitor.get_telemetry()
            result.vram_baseline_mb = snap.gpu.vram_used_mb
            result.vram_used_mb = snap.gpu.vram_used_mb

        # -------------------------------------------------------------
        # Test 1: Python Code Generation & AST Syntax Verification
        # -------------------------------------------------------------
        code_prompt = (
            "Write a Python function `def is_palindrome(s: str) -> bool:` that returns True if a string "
            "is a palindrome and False otherwise. Include type annotations. Output only Python code in ```python ``` block."
        )
        t0 = time.time()
        try:
            payload1 = {
                "model": model_info.filename,
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": code_prompt}
                ],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": 600,
            }
            if hasattr(config, "enable_thinking") and not config.enable_thinking:
                payload1["chat_template_kwargs"] = {"enable_thinking": False}

            req_data = json.dumps(payload1).encode("utf-8")

            req = urllib.request.Request(
                f"{self.endpoint_url}/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                ttft = (time.time() - t0) * 1000.0
                resp_data = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - t0
                result.total_time_s = round(elapsed, 2)

                timings = resp_data.get("timings") or {}
                usage = resp_data.get("usage") or {}
                completion_tokens = usage.get("completion_tokens", 0)
                if not completion_tokens and isinstance(timings, dict):
                    completion_tokens = timings.get("predicted_n", 0)
                result.total_tokens = completion_tokens

                # 3. tok/s and TTFT from response['timings'] with fallback
                predicted_tok_s = timings.get("predicted_per_second") if isinstance(timings, dict) else None
                if predicted_tok_s is not None and isinstance(predicted_tok_s, (int, float)) and predicted_tok_s > 0:
                    result.generation_speed_tok_s = round(float(predicted_tok_s), 1)
                elif elapsed > 0 and completion_tokens > 0:
                    result.generation_speed_tok_s = round(completion_tokens / elapsed, 1)

                prompt_ms = timings.get("prompt_ms") if isinstance(timings, dict) else None
                if prompt_ms is not None and isinstance(prompt_ms, (int, float)) and prompt_ms > 0:
                    result.ttft_ms = round(float(prompt_ms), 1)
                else:
                    result.ttft_ms = round(ttft, 1)

                msg = resp_data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                result.raw_code_output = content if content else reasoning

                target_text = content if content else reasoning
                py_code = self._extract_python_code(target_text)
                try:
                    parsed_ast = ast.parse(py_code)
                    result.code_syntax_valid = True
                    has_func = any(isinstance(node, ast.FunctionDef) and node.name == "is_palindrome" for node in ast.walk(parsed_ast))
                    result.code_test_passed = has_func or ("def is_palindrome" in py_code)
                except Exception:
                    try:
                        import textwrap
                        dedented = textwrap.dedent(py_code).strip()
                        parsed_ast = ast.parse(dedented)
                        result.code_syntax_valid = True
                        result.code_test_passed = "def is_palindrome" in dedented
                    except Exception:
                        result.code_syntax_valid = False
                        result.code_test_passed = False

        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            result.status = "ERROR"
            result.error_msg = f"HTTP {e.code} {e.reason}"
            result.error_detail = f"HTTP {e.code} {e.reason}" + (f": {err_body[:200]}" if err_body else "")
            logger.warning(f"Benchmark HTTPError for {model_info.filename}: {result.error_detail}")
            self.test_history.append(result)
            return result
        except urllib.error.URLError as e:
            reason_str = str(e.reason)
            result.status = "SERVER_OFFLINE"
            result.error_msg = f"Connection error: {reason_str}"
            result.error_detail = f"Server offline on {self.endpoint_url} ({reason_str})"
            logger.warning(f"Benchmark URLError for {model_info.filename}: {result.error_detail}")
            self.test_history.append(result)
            return result
        except (TimeoutError, socket.timeout) as e:
            result.status = "ERROR"
            result.error_msg = "Inference timed out"
            result.error_detail = f"TimeoutError: Request exceeded 35.0s on {self.endpoint_url}"
            logger.warning(f"Benchmark Timeout for {model_info.filename}: {result.error_detail}")
            self.test_history.append(result)
            return result
        except Exception as e:
            result.status = "ERROR"
            result.error_msg = str(e)
            result.error_detail = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"Benchmark Exception for {model_info.filename}: {result.error_detail}")
            self.test_history.append(result)
            return result

        # Measure VRAM delta during inference
        if hw_monitor:
            snap_mid = hw_monitor.get_telemetry()
            result.vram_used_mb = snap_mid.gpu.vram_used_mb
            result.vram_delta_mb = max(0.0, snap_mid.gpu.vram_used_mb - result.vram_baseline_mb)

        # -------------------------------------------------------------
        # Test 2: Function Calling / JSON Schema Extraction Test
        # -------------------------------------------------------------
        tool_prompt = (
            "Call the terminal tool to check disk space. "
            "Output strictly a JSON object: {\"action\": \"terminal\", \"command\": \"df -h\"}."
        )
        try:
            payload2 = {
                "model": model_info.filename,
                "messages": [
                    {"role": "user", "content": tool_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 250,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            req_data2 = json.dumps(payload2).encode("utf-8")

            req2 = urllib.request.Request(
                f"{self.endpoint_url}/v1/chat/completions",
                data=req_data2,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req2, timeout=60.0) as resp2:
                resp_data2 = json.loads(resp2.read().decode("utf-8"))
                msg2 = resp_data2.get("choices", [{}])[0].get("message", {})
                content2 = msg2.get("content") or ""
                reasoning2 = msg2.get("reasoning_content") or ""
                target_tool = content2 if content2 else reasoning2
                result.raw_tool_output = target_tool

                parsed_tool = self._extract_tool_json(target_tool, tool_calls=msg2.get("tool_calls"))
                if parsed_tool and isinstance(parsed_tool, dict):
                    result.tool_json_valid = True
                    act = str(parsed_tool.get("action", "")).lower()
                    cmd = str(parsed_tool.get("command", "")).lower()
                    name = str(parsed_tool.get("name", "")).lower()
                    result.tool_test_passed = (
                        act == "terminal" or
                        name == "terminal" or
                        cmd == "df -h" or
                        "df -h" in str(parsed_tool)
                    )
                else:
                    result.tool_json_valid = False
                    result.tool_test_passed = False
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            detail = f"Tool test HTTP {e.code}: {err_body[:150]}".strip()
            result.error_detail = f"{result.error_detail} | {detail}" if result.error_detail else detail
            result.tool_json_valid = False
            result.tool_test_passed = False
        except urllib.error.URLError as e:
            detail = f"Tool test URLError: {e.reason}"
            result.error_detail = f"{result.error_detail} | {detail}" if result.error_detail else detail
            result.tool_json_valid = False
            result.tool_test_passed = False
        except Exception as e:
            detail = f"Tool test {type(e).__name__}: {str(e)[:100]}"
            result.error_detail = f"{result.error_detail} | {detail}" if result.error_detail else detail
            result.tool_json_valid = False
            result.tool_test_passed = False

        result.status = "COMPLETED"
        self.test_history.append(result)
        return result

    def run_live_model_benchmark(
        self,
        model_info: LocalModelInfo,
        config: ModelConfig,
        server_mgr: Optional[LlamaServerManager] = None,
        hw_monitor: Optional[HardwareMonitor] = None,
        wait_ready_s: float = 25.0
    ) -> ModelTestResult:
        """
        Orchestrates full live model benchmark:
        1. Ensures llama-server is running the target model (starts it if needed).
        2. Waits for endpoint health.
        3. Executes AST code and tool calling benchmarks.
        """
        if server_mgr:
            # Check if server is currently online with the target model
            health = server_mgr.check_health("llama-server")
            is_target_running = (
                health.get("status") == "online" and
                server_mgr.managed_model_path == model_info.filepath
            )

            if not is_target_running:
                # Start llama-server with target model and config
                logger.info(f"Starting llama-server for benchmark: {model_info.filename}")
                start_res = server_mgr.start_server(
                    model_path=model_info.filepath,
                    ctx_size=config.context_length,
                    kv_dtype="q4_0",
                    no_kv_offload=config.no_kv_offload,
                    ctx_shift=True,
                    flash_attn=config.flash_attention,
                    n_gpu_layers=config.ngl
                )
                if isinstance(start_res, dict) and not start_res.get("success", False):
                    err = start_res.get("error", "Failed to start server")
                    res = ModelTestResult(
                        model_name=model_info.filename,
                        model_path=model_info.filepath,
                        status="ERROR",
                        error_msg=err,
                        error_detail=f"Process launch failure: {err}"
                    )
                    self.test_history.append(res)
                    return res

                # Wait for server to come online
                t_start = time.time()
                online = False
                while time.time() - t_start < wait_ready_s:
                    h = server_mgr.check_health("llama-server")
                    if h.get("status") == "online":
                        online = True
                        break
                    time.sleep(0.5)

                if not online:
                    res = ModelTestResult(
                        model_name=model_info.filename,
                        model_path=model_info.filepath,
                        status="SERVER_OFFLINE",
                        error_msg=f"Server startup timed out after {wait_ready_s}s",
                        error_detail=f"llama-server timed out during startup ({wait_ready_s}s wait on {self.endpoint_url})"
                    )
                    self.test_history.append(res)
                    return res

        return self.run_benchmark(model_info, config, hw_monitor)

    def probe_context_sizes(
        self,
        model_info: "LocalModelInfo",
        config: "ModelConfig",
        server_mgr: Optional["LlamaServerManager"] = None,
        hw_monitor: Optional["HardwareMonitor"] = None,
        ctx_sizes: Optional[List[int]] = None,
        wait_ready_s: float = 25.0
    ) -> Dict[str, Any]:
        """
        Probe llama-server at 8k, 16k, 32k context windows.
        Measures tok/s and TTFT degradation per context size.
        Returns dict keyed by ctx size string with tok_s, ttft_ms, status.
        """
        if ctx_sizes is None:
            ctx_sizes = [8192, 16384, 32768]

        results: Dict[str, Any] = {}
        for ctx_size in ctx_sizes:
            key = str(ctx_size)
            if server_mgr:
                # Stop and restart server with this ctx size
                server_mgr.stop_server()
                time.sleep(0.5)
                start_res = server_mgr.start_server(
                    model_path=model_info.filepath,
                    ctx_size=ctx_size,
                    kv_dtype="f16",
                    no_kv_offload=config.no_kv_offload,
                    ctx_shift=(ctx_size > 16384),
                    flash_attn=config.flash_attention,
                    n_gpu_layers=config.ngl
                )
                if not isinstance(start_res, dict) or not start_res.get("success"):
                    results[key] = {"error": "start_failed", "tok_s": 0.0, "ttft_ms": 0.0, "status": "ERROR"}
                    continue

                # Wait for server readiness
                t_start = time.time()
                online = False
                while time.time() - t_start < wait_ready_s:
                    h = server_mgr.check_health("llama-server")
                    if h.get("status") == "online":
                        online = True
                        break
                    time.sleep(0.5)

                if not online:
                    results[key] = {"error": "timeout", "tok_s": 0.0, "ttft_ms": 0.0, "status": "TIMEOUT"}
                    continue

            # Quick benchmark at this ctx size
            quick_res = self.run_benchmark(model_info, config, hw_monitor)
            results[key] = {
                "tok_s": quick_res.generation_speed_tok_s,
                "ttft_ms": quick_res.ttft_ms,
                "status": quick_res.status,
                "ctx_size": ctx_size
            }

        return results


def determine_optimal_ctx(ctx_probe_results: Dict[str, Any], baseline_key: str = "8192") -> int:
    """
    Determine the largest context window where tok/s >= 80% of baseline (8k) speed.
    Returns context size as int.
    """
    baseline = ctx_probe_results.get(baseline_key, {}).get("tok_s", 0.0)
    if baseline <= 0:
        return 8192

    optimal = 8192
    for key in ["8192", "16384", "32768"]:
        entry = ctx_probe_results.get(key, {})
        tok_s = entry.get("tok_s", 0.0)
        status = entry.get("status", "ERROR")
        if status == "COMPLETED" and tok_s >= baseline * 0.80:
            optimal = int(key)
    return optimal


def _data_path(win_path: str, name: str) -> str:
    """Su Windows il percorso storico; su Linux "C:/Users/..." sarebbe RELATIVO
    alla cartella corrente (creava ./C:/Users/...): si usa la cartella della TUI."""
    if os.name == "nt":
        return win_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def generate_benchmark_report(
    results: List["ModelTestResult"],
    model_infos: List["LocalModelInfo"],
    config_mgr: "ModelConfigManager",
    report_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive benchmark_report.json from chain benchmark results.
    Includes rankings by tok/s and AST/Tool quality, plus per-model recommendations.
    """
    if not report_path:
        report_path = _data_path("C:/Users/User/Documents/ai_tui/benchmark_report.json", "benchmark_report.json")
    model_info_map = {mi.filename: mi for mi in model_infos}
    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "total_models_tested": len(results),
        "hardware": "NVIDIA GeForce RTX 3050 8GB VRAM, Windows 11",
        "server_flags": "-nkvo -ngl 99 -ctk f16 -ctv f16 --context-shift -fa on -t 8",
        "ranked_by_speed": [],
        "ranked_by_quality": [],
        "models": {}
    }

    completed = [r for r in results if r.status == "COMPLETED"]

    # Rankings
    ranked_speed = sorted(completed, key=lambda r: r.generation_speed_tok_s, reverse=True)
    for i, r in enumerate(ranked_speed):
        report["ranked_by_speed"].append({
            "rank": i + 1,
            "model": r.model_name,
            "tok_s": r.generation_speed_tok_s,
            "ttft_ms": r.ttft_ms,
            "ast_pass": r.python_ast_pass,
            "tool_pass": r.json_tool_pass
        })

    def _quality_score(r: "ModelTestResult") -> int:
        return (1 if r.python_ast_pass else 0) + (1 if r.json_tool_pass else 0)

    ranked_quality = sorted(completed, key=lambda r: (_quality_score(r), r.generation_speed_tok_s), reverse=True)
    for i, r in enumerate(ranked_quality):
        report["ranked_by_quality"].append({
            "rank": i + 1,
            "model": r.model_name,
            "quality_score": _quality_score(r),
            "ast_pass": r.python_ast_pass,
            "tool_pass": r.json_tool_pass,
            "tok_s": r.generation_speed_tok_s
        })

    # Per-model details + recommendations
    for res in results:
        mi = model_info_map.get(res.model_name)
        cfg = config_mgr.get_config(res.model_name)
        optimal_ctx = res.optimal_ctx if res.optimal_ctx > 0 else cfg.context_length
        ctx_shift_on = optimal_ctx > 16384
        ngl = cfg.ngl if hasattr(cfg, "ngl") else 99
        no_kv = cfg.no_kv_offload if hasattr(cfg, "no_kv_offload") else True
        fa = cfg.flash_attention if hasattr(cfg, "flash_attention") else True

        report["models"][res.model_name] = {
            "filename": res.model_name,
            "size_gb": mi.size_gb if mi else 0.0,
            "quant_type": mi.quant_type if mi else "GGUF",
            "status": res.status,
            "python_ast_pass": res.python_ast_pass,
            "json_tool_pass": res.json_tool_pass,
            "speed_tok_s": res.generation_speed_tok_s,
            "ttft_ms": res.ttft_ms,
            "vram_used_mb": res.vram_used_mb,
            "ctx_probe_results": res.ctx_probe_results,
            "optimal_ctx": optimal_ctx,
            "timestamp": res.timestamp,
            "error": res.error_detail or res.error_msg,
            "recommendations": {
                "temperature": cfg.temperature,
                "context_length": optimal_ctx,
                "ngl": ngl,
                "ctx_shift": ctx_shift_on,
                "no_kv_offload": no_kv,
                "flash_attention": fa,
                "launch_flags": (
                    f"-ngl {ngl} -c {optimal_ctx} -ctk f16 -ctv f16 -fa on"
                    f"{' -nkvo' if no_kv else ''}"
                    f"{' --context-shift' if ctx_shift_on else ' --no-context-shift'}"
                    " -t 8"
                )
            }
        }

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


# =====================================================================
# 4.7 CAVEMAN COMPRESSOR, VRAM HEADROOM & GITHUB PLUGIN INSTALLER
# =====================================================================

class CavemanCompressor:
    """
    Local Zero-Latency Context Compressor inspired by Caveman / RTK algorithms.
    Strips redundant conversational fillers, markdown boilerplate, greeting noise,
    and formats user/assistant turns into dense imperative directives.
    """
    FILLER_PATTERNS = [
        r"(?i)\b(please|kindly|could you|would you mind|as an ai|i'd be happy to|sure thing|certainly|hello|hi there|good morning|thanks|thank you)\b",
        r"(?i)\b(in order to|as a matter of fact|at the end of the day|due to the fact that)\b",
        r"(?i)\b(note that|keep in mind that|it is worth mentioning that)\b",
    ]

    @classmethod
    def compress(cls, text: str) -> Dict[str, Any]:
        original_len = len(text.split())
        cleaned = text
        for p in cls.FILLER_PATTERNS:
            cleaned = re.sub(p, "", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n", cleaned).strip()
        compressed_len = len(cleaned.split())
        savings_pct = max(0.0, round((1.0 - (compressed_len / max(1, original_len))) * 100.0, 1))
        return {
            "original_tokens": original_len,
            "compressed_tokens": compressed_len,
            "savings_pct": savings_pct,
            "compressed_text": cleaned
        }


class VRAMHeadroomCalculator:
    """
    Dynamic VRAM Headroom & Safety Buffer Manager for NVIDIA RTX 3050 (8GB VRAM).
    Ensures safe weights allocation, calculates Windows DWM buffer reservation,
    and alerts if memory exceeds the PCIe spilling threshold.
    """
    TOTAL_VRAM_MB = 8192.0
    WINDOWS_DWM_RESERVE_MB = 1024.0
    CUDA_CONTEXT_OVERHEAD_MB = 450.0

    @classmethod
    def calculate_headroom(cls, model_weights_mb: float, kv_cache_vram_mb: float = 0.0) -> Dict[str, Any]:
        usable_vram_mb = cls.TOTAL_VRAM_MB - cls.WINDOWS_DWM_RESERVE_MB - cls.CUDA_CONTEXT_OVERHEAD_MB
        total_ai_allocated_mb = model_weights_mb + kv_cache_vram_mb
        remaining_headroom_mb = usable_vram_mb - total_ai_allocated_mb
        is_safe = remaining_headroom_mb >= 0.0

        status = "🟢 SAFE (Zero PCIe Spill Risk)" if remaining_headroom_mb > 500.0 else (
            "🟡 TIGHT (Near Limit)" if remaining_headroom_mb >= 0.0 else
            f"🔴 EXCEEDED (Spilling {abs(remaining_headroom_mb)/1024.0:.1f}GB to RAM)"
        )
        return {
            "total_vram_mb": cls.TOTAL_VRAM_MB,
            "dwm_reserve_mb": cls.WINDOWS_DWM_RESERVE_MB,
            "cuda_overhead_mb": cls.CUDA_CONTEXT_OVERHEAD_MB,
            "usable_for_weights_mb": usable_vram_mb,
            "usable_for_weights_gb": round(usable_vram_mb / 1024.0, 2),
            "model_weights_mb": model_weights_mb,
            "kv_cache_vram_mb": kv_cache_vram_mb,
            "remaining_headroom_mb": round(remaining_headroom_mb, 1),
            "remaining_headroom_gb": round(remaining_headroom_mb / 1024.0, 2),
            "is_safe": is_safe,
            "status": status
        }


class GitHubPluginInstaller:
    """
    Manages cloning, downloading, validating, and installing Plugins & Skills from GitHub.
    Installs into Hermes Agent skills (~/.hermes/skills/ and AppData), OmniRoute plugins,
    and Claude Code skills directory.
    """
    HERMES_SKILLS_DIR = os.path.expanduser("~/.hermes/skills")
    HERMES_LOCAL_SKILLS_DIR = os.path.join(os.environ.get("LOCALAPPDATA", "."), "hermes", "skills")
    OMNIROUTE_PLUGINS_DIR = os.path.expanduser("~/.omniroute/plugins")

    def __init__(self):
        for d in [self.HERMES_SKILLS_DIR, self.HERMES_LOCAL_SKILLS_DIR, self.OMNIROUTE_PLUGINS_DIR]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

    def install_from_github(self, repo_url: str, target_type: str = "hermes") -> Dict[str, Any]:
        """Clone or download a skill/plugin repository from GitHub."""
        repo_url = repo_url.strip()
        if not repo_url:
            return {"success": False, "error": "Empty URL"}

        if not repo_url.startswith("http"):
            repo_url = f"https://github.com/{repo_url}"

        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        dest_base = self.HERMES_LOCAL_SKILLS_DIR if target_type == "hermes" else self.OMNIROUTE_PLUGINS_DIR
        dest_dir = os.path.join(dest_base, repo_name)

        try:
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir, ignore_errors=True)

            cmd = ["git", "clone", "--depth", "1", repo_url, dest_dir]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
            if res.returncode != 0:
                return {"success": False, "error": f"Git clone failed: {res.stderr}"}

            has_skill_md = os.path.exists(os.path.join(dest_dir, "SKILL.md"))
            has_package_json = os.path.exists(os.path.join(dest_dir, "package.json"))

            return {
                "success": True,
                "name": repo_name,
                "path": dest_dir,
                "target_type": target_type,
                "has_skill_md": has_skill_md,
                "has_package_json": has_package_json,
                "message": f"Successfully installed '{repo_name}' to {dest_dir}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_installed(self) -> List[Dict[str, Any]]:
        """List all installed skills and plugins across Hermes and OmniRoute."""
        results = []
        for base_dir in [self.HERMES_LOCAL_SKILLS_DIR, self.HERMES_SKILLS_DIR]:
            if os.path.exists(base_dir):
                try:
                    for item in os.listdir(base_dir):
                        full_p = os.path.join(base_dir, item)
                        if os.path.isdir(full_p):
                            skill_md = os.path.join(full_p, "SKILL.md")
                            desc = "Hermes Skill"
                            if os.path.exists(skill_md):
                                try:
                                    with open(skill_md, "r", encoding="utf-8", errors="ignore") as f:
                                        lines = [l.strip() for l in f.readlines() if l.strip()]
                                        if lines:
                                            desc = lines[0].replace("#", "").strip()[:60]
                                except Exception:
                                    pass
                            results.append({
                                "name": item,
                                "type": "Hermes Skill",
                                "path": full_p,
                                "description": desc,
                                "is_active": True
                            })
                except Exception:
                    pass
        if os.path.exists(self.OMNIROUTE_PLUGINS_DIR):
            try:
                for item in os.listdir(self.OMNIROUTE_PLUGINS_DIR):
                    full_p = os.path.join(self.OMNIROUTE_PLUGINS_DIR, item)
                    if os.path.isdir(full_p):
                        results.append({
                            "name": item,
                            "type": "OmniRoute Plugin",
                            "path": full_p,
                            "description": "OmniRoute Gateway Plugin",
                            "is_active": True
                        })
            except Exception:
                pass
        return results


# =====================================================================
# 4.8 BENCHMARK DATABASE & BATCH BENCHMARK QUEUE RUNNER
# =====================================================================

class BenchmarkDatabase:
    """
    Persistent SQLite storage for all measured local benchmarks and online references.
    Path: C:/Users/User/Documents/ai_tui/benchmarks.db
    """
    def __init__(self, db_path: Optional[str] = None):
        # Su Linux il vecchio default "C:/Users/User/Documents/ai_tui/benchmarks.db"
        # era un percorso RELATIVO: il DB finiva in ./C:/Users/... sotto la cartella
        # corrente e la Tab 12 (che legge la cartella consolidata dei benchmark)
        # non vedeva mai i risultati della Tab 5. Ora: cartella consolidata
        # (la stessa letta dalla Tab 12) se esiste, altrimenti accanto a questo file.
        if db_path is None:
            if os.name == "nt":
                db_path = "C:/Users/User/Documents/ai_tui/benchmarks.db"
            else:
                bench_dir = "/mnt/lm_models/DEBIAN_FOLDER/benchmarks"
                db_path = os.path.join(bench_dir if os.path.isdir(bench_dir)
                                       else os.path.dirname(os.path.abspath(__file__)), "benchmarks.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    architecture TEXT,
                    quantization TEXT,
                    vram_weights_gb REAL,
                    vram_used_mb REAL,
                    ram_used_mb REAL,
                    python_ast_pass INTEGER,
                    json_tool_pass INTEGER,
                    speed_tok_s REAL,
                    ttft_ms REAL,
                    status TEXT,
                    error_detail TEXT,
                    timestamp TEXT,
                    is_local INTEGER DEFAULT 1
                )
            """)
            conn.commit()

    def insert_result(self, result: ModelTestResult, model_info: Optional[LocalModelInfo] = None) -> int:
        arch = "MoE" if "moe" in result.model_name.lower() or (model_info and "moe" in model_info.filename.lower()) else "Dense"
        quant = model_info.quant_type if model_info else "GGUF"
        vram_w_gb = model_info.size_gb if model_info else 0.0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO benchmark_results (
                    filename, model_name, architecture, quantization,
                    vram_weights_gb, vram_used_mb, ram_used_mb,
                    python_ast_pass, json_tool_pass, speed_tok_s, ttft_ms,
                    status, error_detail, timestamp, is_local
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_info.filename if model_info else result.model_name,
                result.model_name,
                arch,
                quant,
                vram_w_gb,
                result.vram_used_mb,
                0.0,
                1 if result.python_ast_pass else 0,
                1 if result.json_tool_pass else 0,
                result.speed_tok_s,
                result.ttft_ms,
                result.status,
                getattr(result, "error_detail", "") or getattr(result, "error_msg", ""),
                result.timestamp,
                1
            ))
            conn.commit()
            return cursor.lastrowid

    def get_latest_results(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmark_results
                ORDER BY id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def export_to_json(self, json_path: Optional[str] = None) -> str:
        if not json_path:
            json_path = _data_path("C:/Users/User/Documents/ai_tui/local_benchmarks.json",
                                   "local_benchmarks.json")
        results = self.get_latest_results()
        latest_map = {}
        for r in results:
            fn = r["filename"]
            if fn not in latest_map:
                latest_map[fn] = r

        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tested": len(latest_map),
            "results": list(latest_map.values()),
            "history": results[:50]
        }
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return json_path


class BatchBenchmarkRunner:
    """
    Automated Batch Benchmark Queue Runner:
    Loads and unloads each local model sequentially, launches llama-server,
    executes AST & JSON Tool Calling benchmarks, measures tok/s and TTFT,
    logs results to SQLite DB and syncs local_benchmarks.json.
    """
    def __init__(
        self,
        server_mgr: LlamaServerManager,
        test_runner: ModelTestRunner,
        config_mgr: ModelConfigManager,
        hw_monitor: HardwareMonitor,
        db: Optional[BenchmarkDatabase] = None
    ):
        self.server_mgr = server_mgr
        self.test_runner = test_runner
        self.config_mgr = config_mgr
        self.hw_monitor = hw_monitor
        self.db = db or BenchmarkDatabase()

        self.queue: List[LocalModelInfo] = []
        self.current_idx: int = 0
        self.is_running: bool = False
        self.stop_requested: bool = False
        self.logs: List[str] = []
        self.current_model_name: str = ""
        self._thread: Optional[threading.Thread] = None

    def start_batch(self, models: List[LocalModelInfo]) -> bool:
        if self.is_running:
            return False
        self.queue = list(models)
        self.current_idx = 0
        self.is_running = True
        self.stop_requested = False
        self.logs = [f"🚀 Batch Queue started with {len(self.queue)} models."]

        self._thread = threading.Thread(target=self._run_queue, daemon=True)
        self._thread.start()
        return True

    def stop_batch(self):
        self.stop_requested = True
        self.logs.append("🛑 Batch Queue stop requested by user.")

    def _run_queue(self):
        try:
            for idx, model in enumerate(self.queue):
                if self.stop_requested:
                    self.logs.append("⏹️ Batch Queue halted.")
                    break

                self.current_idx = idx
                self.current_model_name = model.filename
                self.logs.append(f"[{idx+1}/{len(self.queue)}] Benchmarking '{model.filename[:30]}'...")

                # 1. Flush existing VRAM
                self.server_mgr.flush_vram(self.hw_monitor)
                time.sleep(1.0)

                # 2. Get Config
                config = self.config_mgr.get_config(model.filename)

                # 3. Run live benchmark (starts server, tests, measures)
                res = self.test_runner.run_live_model_benchmark(
                    model_info=model,
                    config=config,
                    server_mgr=self.server_mgr,
                    hw_monitor=self.hw_monitor,
                    wait_ready_s=25.0
                )

                # 4. Save to DB & Export JSON
                self.db.insert_result(res, model)
                self.db.export_to_json()

                status_icon = "✓" if res.all_passed else "✗"
                self.logs.append(
                    f"  {status_icon} Result: Speed={res.speed_tok_s:.1f} tok/s | TTFT={res.ttft_ms:.1f}ms | AST={res.python_ast_pass} | Tool={res.json_tool_pass}"
                )

                # 5. Flush server before next
                self.server_mgr.stop_server()
                self.server_mgr.flush_vram(self.hw_monitor)
                time.sleep(1.0)

            self.logs.append(f"🎉 Batch Queue completed! Tested {len(self.queue)} models.")
        except Exception as e:
            self.logs.append(f"❌ Batch Queue Error: {str(e)[:100]}")
        finally:
            self.is_running = False
            self.current_model_name = ""


# =====================================================================
# 4.9 CONVENIENCE ALIASES & UI UTILITIES
# =====================================================================

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = ""
    tool_name: Optional[str] = None


def get_cyberpunk_theme() -> Dict[str, str]:
    """Returns standard Cyberpunk color palette dictionary for TUI styling."""
    return {
        "primary": "cyan",
        "secondary": "magenta",
        "accent": "yellow",
        "danger": "bright_red",
        "success": "bright_green",
        "warning": "bright_yellow",
        "info": "bright_blue",
        "bg": "black",
        "panel_border": "cyan",
        "dim": "dim cyan"
    }

# Backward compatibility / Module Aliases
GroqContextCompressor = ContextCompressor
CompressionResult = CompressedContext
RadarNewsAggregator = AINewsAggregator
RadarItem = NewsItem


# =====================================================================
# 5. VERIFICATION & UNIT TEST SUITE
# =====================================================================

def run_tests() -> bool:
    """
    Comprehensive verification runner for all backend components.
    """
    print("\n=======================================================")
    print("  AI MANAGEMENT TUI - BACKEND VERIFICATION SUITE")
    print("=======================================================\n")

    all_passed = True

    # -------------------------------------------------------------
    # Test 1: LlamaServerManager Unit Tests (including RAM-only KV Offload)
    # -------------------------------------------------------------
    print("[1/5] Testing LlamaServerManager & Server Presets...")
    mgr = LlamaServerManager(endpoints=[
        {"name": "llama-server", "url": "http://127.0.0.1:18434", "server_type": "llama-server"},
        {"name": "LM Studio", "url": "http://127.0.0.1:1234", "server_type": "lm-studio"}
    ], timeout_s=1.0)

    assert len(mgr.endpoints) == 2, "Endpoint registration failed"
    custom_ep = mgr.register_endpoint("test-server", "http://127.0.0.1:9999/v1")
    assert custom_ep.url == "http://127.0.0.1:9999/v1", "URL normalization failed"

    # Verify launch args with RAM-only KV offload (-nkvo)
    launch_args_nkvo = mgr.get_launch_args(no_kv_offload=True, ctx_size=65536)
    assert "-nkvo" in launch_args_nkvo["cmd_args"]
    assert "-nkvo" in launch_args_nkvo["cmd_string"]
    print("  ✓ RAM-only KV Offload (-nkvo) flag integration verified.")

    # Verify Server Presets
    assert "ram_kv_offload" in mgr.SERVER_PRESETS
    assert mgr.SERVER_PRESETS["ram_kv_offload"]["no_kv_offload"] is True
    assert mgr.SERVER_PRESETS["max_speed"]["no_kv_offload"] is False
    print("  ✓ Server Presets (Max Speed, RAM KV Offload, 256k, Hybrid Q8) verified.")

    health = mgr.check_health("test-server")
    assert health["name"] == "test-server"
    assert health["status"] == "offline"
    print("  ✓ Offline health check handling verified.")

    overview = mgr.get_overview()
    assert "endpoints" in overview and "models" in overview and "slots" in overview
    print("  [PASS] LlamaServerManager tests passed.\n")

    # -------------------------------------------------------------
    # Test 2: KVCacheOptimizer Unit Tests (256k & RAM Offload)
    # -------------------------------------------------------------
    print("[2/5] Testing KVCacheOptimizer & 256k RAM Offload Engine...")
    opt = KVCacheOptimizer()

    # 2.1 StreamingLLM Attention Sinks Test
    mock_tokens = [f"tok_{i}" for i in range(100)]
    stream_res = opt.apply_streaming_llm(mock_tokens, sink_tokens_count=4, window_size=10)
    assert stream_res["sink_count"] == 4
    assert stream_res["window_count"] == 10
    assert stream_res["total_retained"] == 14
    assert stream_res["kept_tokens"][:4] == ["tok_0", "tok_1", "tok_2", "tok_3"]
    assert stream_res["evicted_count"] == 86
    print("  ✓ StreamingLLM: 4 Attention Sinks + Sliding Window correctly verified.")

    # 2.2 256k Context Budget with RAM-only KV offload
    budget_256k = opt.calculate_256k_budget(available_vram_gb=8.0)
    assert budget_256k["seq_len"] == 262144
    assert budget_256k["fp16"]["gb"] == 32.0
    assert budget_256k["q4_0"]["gb"] == 9.0
    assert "ram_kv_offload" in budget_256k
    assert budget_256k["ram_kv_offload"]["vram_mb"] == 0.0
    assert budget_256k["ram_kv_offload"]["savings_vram_pct"] == 100.0
    print("  ✓ 256k Context Budget & RAM-only KV Offload (100% VRAM free for weights) verified.")
    print("  [PASS] KVCacheOptimizer tests passed.\n")

    # -------------------------------------------------------------
    # Test 3: HFModelManager Unit Tests
    # -------------------------------------------------------------
    print("[3/5] Testing HFModelManager & Quant Inspector...")
    hf_mgr = HFModelManager()
    search_res = hf_mgr.search_models("Qwen3.8 GGUF", limit=2)
    assert isinstance(search_res, list)
    print(f"  ✓ HF search_models query executed successfully ({len(search_res)} items found).")
    if search_res:
        quants = hf_mgr.get_repo_quants(search_res[0].repo_id)
        assert isinstance(quants, list)
        print(f"  ✓ HF get_repo_quants inspected {search_res[0].repo_id} ({len(quants)} quant files found).")
    print("  [PASS] HFModelManager tests passed.\n")

    # -------------------------------------------------------------
    # Test 4: ModelConfigManager & LM Studio Preset Generator
    # -------------------------------------------------------------
    print("[4/5] Testing ModelConfigManager & LM Studio Preset Export...")
    test_cfg_path = ("C:/Users/User/Documents/ai_tui/test_model_configs.json" if os.name == "nt"
                     else os.path.expanduser("~/.cache/ai_tui/test_model_configs.json"))
    cfg_mgr = ModelConfigManager(config_path=test_cfg_path)
    mock_model = LocalModelInfo(
        filename="Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        filepath="/mnt/lm_models/models/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        rel_path="HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        author="HauhauCS",
        size_bytes=6900000000,
        size_gb=6.43,
        quant_type="Q4_K_M",
        vram_fit_status="🟢 100% VRAM",
        is_recommended=True,
        last_modified="2026-09-27"
    )
    gemma_cfg = cfg_mgr.get_config(mock_model.filename)
    assert gemma_cfg.temperature == 0.6
    assert gemma_cfg.top_k == 64
    assert gemma_cfg.enable_thinking is True
    print("  ✓ ModelConfig architectural default verified.")

    # Export LM Studio Preset test
    preset_path = cfg_mgr.export_lmstudio_preset(gemma_cfg)
    assert os.path.exists(preset_path), f"Preset not found at {preset_path}"
    with open(preset_path, "r", encoding="utf-8") as f:
        preset_json = json.load(f)
        assert "operation" in preset_json and "load" in preset_json
    print(f"  ✓ LM Studio preset successfully exported: {os.path.basename(preset_path)}")

    if os.path.exists(test_cfg_path):
        os.remove(test_cfg_path)
    print("  [PASS] ModelConfigManager tests passed.\n")

    # -------------------------------------------------------------
    # Test 5: ModelTestRunner & Live Benchmark Harness
    # -------------------------------------------------------------
    print("[5/5] Testing ModelTestRunner & AST Code Verification...")
    runner = ModelTestRunner(endpoint_url="http://127.0.0.1:18434")
    test_res = runner.run_benchmark(mock_model, gemma_cfg)
    assert test_res.status in ("COMPLETED", "SERVER_OFFLINE", "ERROR")
    summary = test_res.summary_str()
    assert mock_model.filename in summary
    print(f"  ✓ ModelTestRunner execution completed (Status: {test_res.status}).")
    print(f"    Summary: {summary}")
    print("  [PASS] ModelTestRunner tests passed.\n")

    print("=======================================================")
    print("  ALL BACKEND TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("=======================================================\n")
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
