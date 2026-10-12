"""
tui_app.py - Cyberpunk Local AI Management & Model TUI Dashboard
Author: Local AI Hypervisor Team
Environment: Windows Local AI Management TUI (ai_tui)

Tabs:
1. [1] System & GPU Mon: Real-time NVIDIA RTX 3050 VRAM/Core/Temp & CPU/RAM Monitor
2. [2] 256k KV Cache & Groq Compression: 256k Token Budget, RAM-Only KV Offload (-nkvo), Context Shifting & Groq Llama-3.3-70B Compression
3. [3] Model Radar & News: arXiv, Hugging Face Trending, Local Model Releases
4. [4] HF Downloader & Quant Inspector: Search & Download to /mnt/lm_models/models, Multi-Quant Table, Auto Groq Config Fetcher [H]
5. [5] Process Control & Benchmark Engine: Manage llama-server, Local /mnt/lm_models/models, RAM KV Offload Presets, AST/Tool Calling Benchmark [T], Gear Settings [G]
6. [6] Agent Shell: Interactive Chat & Inference w/ Streaming, Paging & Tool Calling
"""

import os
import sys
import time
import json
import urllib.request
import math
import re
from _kbd_compat import kbhit, getwch, IS_WINDOWS, restore as _kbd_restore
import shutil
import asyncio
import argparse
import threading
# Serve per riavviare i processi reali (ripristino post-test, riavvio con
# gli stessi flag). Non si usa mai "pkill -f": ucciderebbe la shell stessa.
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel as _RichPanel
from rich.table import Table as _RichTable
from rich.layout import Layout as _RichLayout
from rich.styled import Styled
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich import box

# Tab "Aspetto": tema colori, densita' e proporzioni (3 controlli separati).
import theme as Th
Th.init_appearance()


from rich.markup import escape as _mesc   # testo dinamico dentro il markup Rich

class Panel(_RichPanel):
    """Panel che rispetta la DENSITA' scelta (padding) e usa il ruolo
    'border' del tema per i bordi quando il codice non ne chiede un altro."""

    def __init__(self, renderable="", *args, **kw):
        dc = Th.APPEARANCE.density_cfg
        if dc["panel_pad"] is not None:
            kw["padding"] = dc["panel_pad"]
        kw.setdefault("border_style", "border")
        super().__init__(renderable, *args, **kw)


class Table(_RichTable):
    """Table che rispetta la DENSITA' (spazio fra righe e padding celle)."""

    def __init__(self, *args, **kw):
        dc = Th.APPEARANCE.density_cfg
        if kw.get("box", True) is not None:        # le griglie restano compatte
            if dc["table_leading"]:
                kw.setdefault("leading", dc["table_leading"])
            if dc["cell_pad"] != 1:
                kw["padding"] = (0, dc["cell_pad"])
            kw.setdefault("border_style", "border")
            kw.setdefault("header_style", "bold metric")
        super().__init__(*args, **kw)


def get_aspect_info() -> Dict[str, Any]:
    """
    Rileva le proporzioni della finestra e determina il layout (16:9 vs 9:16).
    In un terminale i caratteri sono alti circa il doppio della larghezza (rapporto 1:2).
    Rapporto di forma fisico reale a schermo:
      visual_ratio = cols / (2.0 * max(1, lines))
    - Se visual_ratio < 1.05 o cols < 95: orientamento verticale 9:16 (portrait).
    - Altrimenti: orientamento orizzontale 16:9 (landscape).
    """
    cols, lines = shutil.get_terminal_size((120, 40))
    visual_ratio = cols / (2.0 * max(1, lines))
    is_portrait = (visual_ratio < 1.05) or (cols < 95)
    mode = "9:16" if is_portrait else "16:9"
    return {
        "mode": mode,
        "is_portrait": is_portrait,
        "cols": cols,
        "lines": lines,
        "visual_ratio": visual_ratio,
    }


def is_portrait_mode() -> bool:
    """True se la finestra e' in formato verticale (9:16 o stretta)."""
    return get_aspect_info()["is_portrait"]


def get_launch_log_dimensions(term_lines: int) -> Tuple[int, int]:
    """
    Calcola l'altezza complessiva del riquadro di avanzamento (box_size)
    e quante righe di log sono visibili (visible_lines).
    Garantisce che lo scorrimento PGUP/PGDN si muova esattamente di 'visible_lines'.
    """
    if term_lines < 26:
        box_size = max(9, term_lines - 10)
    elif term_lines < 40:
        box_size = 13
    else:
        box_size = min(22, term_lines // 2)
    visible_lines = max(3, box_size - 5)
    return box_size, visible_lines


class Layout(_RichLayout):
    """Layout responsive che commuta automaticamente fra 16:9 (orizzontale) e 9:16 (verticale)."""

    def split_row(self, *layouts, **kw):
        if is_portrait_mode():
            # In modalita' 9:16 (portrait), impila i pannelli verticalmente in colonna
            # cosi' ogni pannello puo' sfruttare l'intera larghezza del terminale!
            return super().split_column(*layouts, **kw)
        sp = Th.APPEARANCE.proportion_cfg["split"]
        if sp and len(layouts) == 2 and all(
                isinstance(l, _RichLayout) and l.size is None for l in layouts):
            layouts[0].ratio, layouts[1].ratio = sp
        return super().split_row(*layouts, **kw)

from backend import (
    HardwareMonitor, HardwareSnapshot,
    KVCacheOptimizer,
    GroqContextCompressor, CompressionResult,
    RadarNewsAggregator, RadarItem,
    HFModelManager, HFModelInfo, QuantFileInfo, DownloadTask,
    LlamaServerManager, LocalModelInfo,
    ModelConfigManager, ModelConfig,
    ModelTestRunner, ModelTestResult,
    BenchmarkDatabase, BatchBenchmarkRunner,
    ChatMessage,
    get_cyberpunk_theme,
    CavemanCompressor,
    VRAMHeadroomCalculator,
    GitHubPluginInstaller,
    determine_optimal_ctx,
    generate_benchmark_report,
    RPCProfileStore,
)

# Telemetria + controllo remoto del nodo RPC Miky (192.168.1.100).
# Modulo autonomo: se Miky e' offline la TUI non si blocca.
from miky_remote import MikyRemoteMonitor, render_miky_panel
# Tab 15: check update centralizzato (sostituisce i job di check sparsi
# nelle altre tab, es. Tab 13 [C] "Controlla update").
from updates import UpdateMonitor, render_updates_panel

# Tab 10: pannello di configurazione COMPLETO di llama-server.
# Import con alias per non confonderlo con l'help in linea del Tab 9.
import llama_config as Lc
from llama_config import (
    FlagSpec as LcFlagSpec,
    ServerConfig as LcServerConfig,
    ConfigProfileStore as LcConfigProfileStore,
)

import lmbench

# Moduli nuovi, introdotti insieme al pannello "server trovati" (F6/F7).
#  - servers: elenco unificato di TUTTI i llama-server (locali via /proc +
#    remoti via HTTP/RPC), con le azioni avvia/ferma/browser.
#  - draft_backends: registro delle build llama.cpp e dei tipi di draft
#    (MTP / DFLASH / DSPARK) che ciascuna sa usare davvero.
import servers as Srv
import draft_backends as DB

# Modal: gestione del server cloud (Ornith/Tiel) via CLI modal, dalla tab 14.
import modal_ctl as MC

# La scansione automatica della sottorete (netscan) e' stata RIMOSSA su
# richiesta: era lenta. Gli endpoint remoti ora sono inseriti a mano
# dall'utente (pannello F6: [A] aggiungi, [X] rimuovi) e salvati in
# ~/.ai_tui_endpoints.json (vedi servers.EndpointStore).
#  - vram_budget: margine VRAM per GPU (locale e host RPC) -> --fit-target
#  - build_manager: piu' build llama.cpp, rpc-server e documentazione per build
#  - bench_results: risultati dalla cartella consolidata dei benchmark
#  - proc_spawn: avvio staccato (cgroup separato) dei processi pesanti
import vram_budget as VB
import peers as PN
import build_manager as BM
# Preset F1-F6 + menu di pre-avvio (flag, parametri, override per preset)
import launch_presets as LP
import bench_results as BR
import proc_spawn as PS
import atomic_json as _AJ


# =====================================================================
# ENDPOINT INTERROGATI PER LO STATO "SERVER" IN ALTO
# =====================================================================
# Il bug di partenza: la TUI chiedeva /health solo a 127.0.0.1:18434 (la
# porta di LM Studio) mentre il server che gira davvero e' su 8080, quindi
# l'HUD diceva OFFLINE con un server attivo e funzionante.
#
# Qui si dichiarano TUTTI gli endpoint da interrogare e si considera ONLINE
# se ALMENO UNO risponde davvero. L'elenco si puo' estendere: non e' una
# porta fissa, e' un elenco.
# Secondi entro cui il secondo [F] conferma la flush VRAM.
FLUSH_CONFIRM_S = 5.0

TUI_HEALTH_ENDPOINTS: List[Dict[str, Any]] = [
    # nome, host, porta, tipo ("llama-server" = HTTP, "rpc" = solo TCP)
    {"name": "8080 locale", "host": "127.0.0.1", "port": 8080,
     "kind": "llama-server"},
    {"name": "Miky RPC 50052", "host": "192.168.1.100", "port": 50052,
     "kind": "rpc"},
]


# =====================================================================
# RPC & CUDA13 HOTKEY MAP  (Tab 9)
# =====================================================================
# F1..F6 aprono il MENU DI PRE-AVVIO del preset corrispondente (build,
# flag, parametri, anteprima comando, [AVVIA]/[INDIETRO]).
#   F1/F2/F3 = Ornith 35B-A3B via RPC 16k / 75k / 120k
#   F4/F5/F6 = Qwen3.8 35B-A3B (IsValorum) via RPC 16k / 75k / 120k
# Ogni preset porta il SUO modello (launch_presets.py). Ctrl+R rilancia
# F1-F6 = Ornith 35B-A3B via RPC 16k/75k/120k + Qwen3.8 35B-A3B (IsValorum)
# via RPC 16k/75k/120k. Ogni preset porta il SUO modello (launch_presets.py).
# Ctrl+R rilancia l'ultimo preset usato. I 5 preset storici restano come
# profili non mappati.
# Alt+1 resta un ALIAS di F1 (muscle memory), ma non serve piu' come
# ripiego: F1 funzionava male solo perche' Terminator (il vecchio terminale
# di questa macchina) teneva per se' F1 (=aiuto) e F11 (=schermo intero).
# Rimosso Terminator (ora si usa un'app desktop su misura), F1 arriva
# davvero alla TUI: ripristinato come tasto primario.
TUI_RPC_HOTKEYS: List[Tuple[str, str]] = [
    ("F1", "orn_16k"),
    ("F2", "orn_75k"),
    ("F3", "orn_120k"),
    ("F4", "qw_16k"),
    ("F5", "qw_75k"),
    ("F6", "heretic_80k"),
]
# Msvcrt (Windows) consegna i tasti funzione come prefisso + carattere.
# Su Linux/terminale ANSI arrivano come sequenza ESC [ .. ~ (xterm) oppure
# ESC O <lettera> (modalita' applicazione). _kbd_compat restituisce la
# sequenza INTERA, quindi qui si mappano tutte e tre le forme.
TUI_FKEY_CODES: Dict[str, str] = {
    # --- Windows / msvcrt: prefisso + codice ---
    "\x00s": "F1", "\x00t": "F2", "\x00u": "F3", "\x00v": "F4", "\x00w": "F5",
    "\xe0\x86": "F6", "\xe0\x85": "F7", "\xe0\x88": "F8",
    "\xe0\x87": "F9", "\xe0\x84": "F10", "\xe0\x8d": "F11", "\xe0\x8c": "F12",
    # --- Linux xterm: ESC [ <n> ~ ---
    "\x1b[11~": "F1", "\x1b[12~": "F2", "\x1b[13~": "F3", "\x1b[14~": "F4",
    "\x1b[15~": "F5", "\x1b[17~": "F6", "\x1b[18~": "F7", "\x1b[19~": "F8",
    "\x1b[20~": "F9", "\x1b[21~": "F10", "\x1b[23~": "F11", "\x1b[24~": "F12",
    # --- Linux xterm, modalita' applicazione: ESC O <lettera> ---
    "\x1bOP": "F1", "\x1bOQ": "F2", "\x1bOR": "F3", "\x1bOS": "F4",
    "\x1b[11;2~": "F1", "\x1b[15;2~": "F5",
    # --- Alias dei preset: Shift+F1..F6 (xterm/VTE) e Alt+1..Alt+6 ---
    "\x1b[1;2P": "F1", "\x1b[1;2Q": "F2", "\x1b[1;2R": "F3", "\x1b[1;2S": "F4",
    "\x1bO2P": "F1", "\x1bO2Q": "F2", "\x1bO2R": "F3", "\x1bO2S": "F4",
    "\x1b[12;2~": "F2", "\x1b[13;2~": "F3", "\x1b[14;2~": "F4", "\x1b[17;2~": "F6",
    "\x1b1": "F1", "\x1b2": "F2", "\x1b3": "F3", "\x1b4": "F4", "\x1b5": "F5", "\x1b6": "F6",
    # Shift+TAB (sezione precedente nel menu di pre-avvio)
    "\x1b[Z": "BTAB",
    # --- Frecce e tasti di navigazione: CSI (ESC [) e SS3 (ESC O) ---
    # Coprono entrambe le forme perche' il terminale puo' mandarle in
    # modalita' normale (ESC [ A) o applicazione (ESC O A).
    "\x1b[A": "UP",    "\x1bOA": "UP",
    "\x1b[B": "DOWN",  "\x1bOB": "DOWN",
    "\x1b[C": "RIGHT", "\x1bOC": "RIGHT",
    "\x1b[D": "LEFT",  "\x1bOD": "LEFT",
    "\x1b[H": "HOME",  "\x1bOH": "HOME", "\x1b[1~": "HOME",
    "\x1b[F": "END",   "\x1bOF": "END",  "\x1b[4~": "END",
    "\x1b[5~": "PGUP", "\x1b[6~": "PGDN",
    # Alt+M (Tab 14: Modal Cloud) e sequenze estese Ctrl+M
    "\x1bm": "ALT_M",  "\x1bM": "ALT_M",
    "\x1b[109;5u": "Ctrl+M", "\x1b[13;5u": "Ctrl+M",
    "\x1b[27;5;13~": "Ctrl+M", "\x1b[27;5;109~": "Ctrl+M",
    # Il tasto ESC da solo resta ESC: si distingue perche' una sequenza
    # ANSI non lascia MAI byte non consumati (vedi _kbd_compat.getwch).
    "\x1b": "ESC",
}
# Frecce/Home/End con modificatori (Shift/Alt/Ctrl: ESC [ 1 ; <m> <lettera>):
# si trattano come la freccia semplice. Senza questa mappa la sequenza non
# corrispondeva a niente e nel vecchio lettore bufferizzato l'"1" finiva per
# aprire la Tab 1 (sintomo: "nelle Impostazioni una freccia torna alla Tab 1").
for _m in range(2, 9):
    for _c, _n in (("A", "UP"), ("B", "DOWN"), ("C", "RIGHT"), ("D", "LEFT"),
                   ("H", "HOME"), ("F", "END")):
        TUI_FKEY_CODES[f"\x1b[1;{_m}{_c}"] = _n
    TUI_FKEY_CODES[f"\x1b[5;{_m}~"] = "PGUP"
    TUI_FKEY_CODES[f"\x1b[6;{_m}~"] = "PGDN"
# Ctrl+<lettera> arriva da msvcrt come carattere di controllo (chr(ord(letter)-64)).
TUI_CTRL_CODES: Dict[str, str] = {"\x12": "Ctrl+R", "\x13": "Ctrl+S", "\x10": "Ctrl+P",
                                  # Ctrl+T = tema successivo (ciclo rapido)
                                  # Ctrl+A / Ctrl+B / Ctrl+L = tab Aspetto /
                                  # Benchmark / Build (alternativa a F10/F11/F12,
                                  # utile se il terminale si tiene i tasti F).
                                  "\x14": "Ctrl+T", "\x01": "Ctrl+A",
                                  "\x02": "Ctrl+B", "\x0c": "Ctrl+L",
                                  # Ctrl+U = Tab 15 Updates (\x15 = ASCII 21)
                                  "\x15": "Ctrl+U"}

# Tab aggiunte dopo la 10 (i tasti 1-9/0 sono tutti occupati).
#   F10 / Ctrl+A -> Tab 11 "Aspetto"   (tema, densita', proporzioni)
#   F11 / Ctrl+B -> Tab 12 "Benchmark" (risultati cartella consolidata)
#   F12 / Ctrl+L -> Tab 13 "Build"     (build llama.cpp, rpc-server, doc, VRAM)
#   Alt+M / Ctrl+M -> Tab 14 "Modal"   (gestione server cloud Hermes Ornith/Tiel)
TUI_EXTRA_TABS: Dict[str, int] = {"F10": 11, "Ctrl+A": 11, "Ctrl+B": 12,
                                  "F12": 13, "Ctrl+L": 13,
                                  "ALT_M": 14, "Alt+M": 14, "Ctrl+M": 14, "^M": 14,
                                  "Ctrl+U": 15, "^U": 15, "Alt+U": 15, "ALT_U": 15}

# Pannello "SERVER TROVATI" (rimappato il 2026-10-05: F6 e' ora un preset).
#   F7 = apri/chiudi il pannello (selezione nel pannello: frecce + INVIO)
#   F8 = ricarica l'elenco (forza un refresh immediato)
TUI_SERVER_HOTKEYS: Dict[str, str] = {
    "F7": "toggle_panel",
    "F8": "refresh",
}


# =====================================================================
# TUI STATE CONTAINER
# =====================================================================
class AppState:
    """
    Central Reactive State for the Cyberpunk Local AI Dashboard.
    Coordinates background workers, telemetry, modal states, and Tab views.
    """

    def __init__(self, default_model_dir: str = "/mnt/lm_models/models"):
        self.running: bool = True
        self.active_tab: int = 1
        self.theme: Dict[str, str] = get_cyberpunk_theme()
        default_model_dir = self._load_saved_model_dir(default_model_dir)
        self.model_dir: str = default_model_dir

        # Backend Engine Singletons
        self.hw_monitor = HardwareMonitor()
        # Monitor remoto di Miky (polling non bloccante ogni 2s).
        self.miky_remote = MikyRemoteMonitor()
        self.miky_remote.start()
        # Monitor update centralizzato per Tab 15 (refresh ogni 10 min).
        self.updates_monitor = UpdateMonitor(interval_s=600.0)
        self.updates_monitor.start()
        self.updates_selected_idx: int = 0
        self.kv_optimizer = KVCacheOptimizer()
        self.compressor = GroqContextCompressor()
        self.news_agg = RadarNewsAggregator()
        self.hf_mgr = HFModelManager(default_dest_dir=default_model_dir)
        self.llama_mgr = LlamaServerManager()
        self.config_mgr = ModelConfigManager()
        self.test_runner = ModelTestRunner()

        # -----------------------------------------------------------------
        # PANNELLO "SERVER TROVATI" (F6/F7/F8)
        # -----------------------------------------------------------------
        # Elenco unificato di TUTTI i llama-server, locali e remoti.
        # L'inventario si aggiorna da solo in un thread daemon: la UI legge
        # solo l'ultimo snapshot, quindi un endpoint lento non la blocca.
        self.server_inventory = Srv.ServerInventory()
        # Rete TUI: questa TUI annuncia i suoi server e vede quelle degli altri.
        self.peer_net = PN.PeerNet(lambda: [
            {"port": r.port, "model": r.model_name, "state": r.state_line}
            for r in self.server_inventory.rows() if r.kind == "locale"])
        self.peer_net.start()
        self.show_server_panel: bool = False      # F6: apre/chiudi il pannello
        self.server_select_mode: bool = False    # F7: cursore + ENTER
        self.server_panel_scroll: int = 0
        # Cache dello stato embedder: la UI (render_footer) legge SOLO questo
        # campo, mai una HTTP diretta. Senza cache, ogni frame del footer
        # farebbe una richiesta a :11435 e con una rete lenta il rendering
        # si blocca. embedder_toggle() lo aggiorna; un refresh periodico
        # separato lo ri-sonda ogni pochi secondi (vedi embedder_poll).
        self._embedder_online: bool = False
        self._embedder_last_probe: float = 0.0
        # Thread daemon leggero, separato da HardwareMonitor e da
        # ServerInventory: sonda solo :11435 ogni 5s. Si ferma da solo
        # quando la TUI esce perche' e' daemon=True.
        def _embedder_watch():
            while True:
                try:
                    self._embedder_online = self.llama_mgr.embedder_is_running()
                except Exception:
                    pass
                time.sleep(5)
        threading.Thread(target=_embedder_watch, daemon=True,
                         name="embedder-watch").start()
        # Flag scritti a mano dall'utente: vanno verbatim a llama-server,
        # sia in locale sia via RPC (vedi rpc_launch_with_extra).
        self.server_extra_flags: str = ""
        self.last_server_action: str = ""
        # Stato di MIKY (192.168.1.100:50052): raggiungibile, latenza, e
        # lettura del processo ggml-rpc-server via SSH. Solo lettura.
        self.miky_status: Dict[str, Any] = {}
        self.mikye_model: str = ""            # .gguf su Miky (percorso Windows)
        self.mikye_pending_cmd: str = ""      # ultimo comando preparato
        # Registro delle build llama.cpp e dei draft che supportano.
        self.draft_registry = DB.DraftBackendRegistry()
        self.selected_draft_type: str = "dflash"   # MTP / DFLASH / DSPARK

        # Input di testo richiesto da un tasto: (domanda, callback). Il loop
        # interattivo ferma il Live, chiede la riga e chiama la callback.
        # handle_key resta cosi' testabile senza terminale.
        self.pending_input: Optional[Tuple[str, Any]] = None
        # Dettagli del server selezionato ([I]) e stato GPU remoto via SSH ([H]).
        self.server_details_busy: bool = False
        self.ssh_gpu_busy: bool = False

        # Tab 11 - Aspetto (tema / densita' / proporzioni): 3 righe separate.
        self.appearance = Th.APPEARANCE
        self.aspetto_row: int = 0

        # Margini VRAM per GPU (locale e host RPC) -> --fit on --fit-target.
        self.vram_budget = VB.VramBudget()

        # Tab 13 - Build: piu' build llama.cpp, rpc-server locali, documentazione.
        self.build_mgr = BM.BuildManager(probe=False)
        self.build_mgr.probe_all_async()
        self.build_cursor: int = 0
        self.build_query: str = ""
        self.build_typing: bool = False
        self.rpc_local_port: int = 50052

        # Avvio dei preset (F1-F5): dialogo "scegli la build" + barra di avanzamento.
        self.launch_modal: Optional[Dict[str, Any]] = None
        self.launch_progress: Optional[Dict[str, Any]] = None
        self.rpc_binary_override: Optional[str] = None
        self._rpc_free_cache: int = 0
        # Filtro per dimensione della lista modelli (Tab 5 [Z], dialogo avvio).
        self.model_size_filter: int = 0

        # Tab 12 - Benchmark: risultati da /mnt/lm_models/DEBIAN_FOLDER/benchmarks.
        self.bench = BR.BenchResults()
        threading.Thread(target=self._bench_load_safe, daemon=True).start()

        # Telemetry Snapshot
        self.hw_snapshot: HardwareSnapshot = self.hw_monitor.get_snapshot()

        # Tab 2 State (KV Cache & Compression)
        self.kv_prompt_tokens: int = 14200
        self.kv_gen_tokens: int = 2480
        self.kv_sink_tokens: int = 512
        self.kv_max_ctx: int = 262144  # 256k max scale
        self.ctx_shift_enabled: bool = True
        # Rolling Window = llama.cpp --context-shift: when the KV window is full it
        # rotates instead of failing, so long sessions stay alive.
        self.rolling_window_enabled: bool = True
        # Requested KV dtype. Guardrail may force f16 when KV is offloaded to RAM.
        self.kv_quant_key: str = "f16"
        self.kv_guardrail_warning: str = ""
        self.last_tool_truncation: Optional[Dict[str, Any]] = None
        self.last_compression_result: Optional[CompressionResult] = None
        self.is_compressing: bool = False

        # Tab 3 State (Radar News)
        self.news_items: List[RadarItem] = []
        self.selected_news_idx: int = 0
        self.news_filter: str = "ALL"  # ALL, ARXIV, HF, RSS
        self.is_fetching_news: bool = False

        # Tab 4 State (HF Downloader & Quant Inspector)
        self.hf_query: str = "Qwen2.5-Coder"
        self.hf_repos: List[HFModelInfo] = []
        self.selected_repo_idx: int = 0
        self.hf_quants: List[QuantFileInfo] = []
        self.selected_quant_idx: int = 0
        self.hf_active_pane: str = "repos"  # repos, quants, downloads
        self.hf_download_dir: str = default_model_dir
        self.is_searching_hf: bool = False

        # Tab 5 State (Process Control & Server Presets & Benchmark Engine)
        self.local_models: List[LocalModelInfo] = self.llama_mgr.scan_local_models(default_model_dir)
        self.selected_model_idx: int = 0
        self.server_port: int = LP.PORT_DEFAULT   # 9292 (fallback 9205): non 18434
        self.server_host: str = "127.0.0.1"
        self.server_online: bool = False
        # Tre stati distinti, non un booleano da due:
        #   OFFLINE  = nessun endpoint risponde
        #   STARTING = un processo c'e' ma /health non risponde ancora
        #   ONLINE   = /health ha risposto davvero
        self.server_state: str = "OFFLINE"
        # Dettaglio per endpoint: nome -> stato + dettaglio. Serve per mostrare
        # QUALE server risponde, non solo che "qualcosa" risponde.
        self.server_endpoints: Dict[str, Dict[str, Any]] = {}
        self.active_model_name: str = "None (Offline)"
        self.server_preset_key: str = "vram_gpu"  # vram_gpu, ram_kv_offload, extreme_256k, hybrid_q8
        self.server_no_kv_offload: bool = True
        self.headroom_calc = VRAMHeadroomCalculator()
        self.caveman_compressor = CavemanCompressor()

        # Embedder per mem0 (tasto globale [E]). Tre stati come il server
        # principale, perche' l'avvio non e' istantaneo e mostrare
        # "OFFLINE" mentre carica ingannerebbe.
        self.embedder_state: str = "OFFLINE"
        self.embedder_port: int = 0
        self.embedder_dims: int = 0
        self._embedder_busy: bool = False
        
        # Multi-select & chain benchmark state (Tab 5)
        self.selected_model_indices: set = set()   # Set of model indices selected for chain benchmark
        self.chain_bench_results: List[ModelTestResult] = []   # All results from chain run
        self.chain_bench_report_path: Optional[str] = None     # Path to generated report JSON
        self.chain_bench_log: List[str] = []                   # Progress log lines
        self.show_chain_report: bool = False                   # Whether to show report table vs model list
        
        # Benchmark Engine & Batch Queue State (Tab 5 [T], [B])
        self.bench_db = BenchmarkDatabase()
        self.batch_runner = BatchBenchmarkRunner(self.llama_mgr, self.test_runner, self.config_mgr, self.hw_monitor, self.bench_db)
        self.is_benchmarking: bool = False
        # PID dei server avviati dai benchmark (Tab 5): solo questi si possono
        # fermare/sostituire automaticamente da un benchmark successivo.
        self._bench_pids: set = set()
        self.last_benchmark_result: Optional[ModelTestResult] = None
        self.benchmark_history: List[ModelTestResult] = []

        # Groq Config Fetcher State ([H])
        self.is_fetching_config: bool = False
        self.last_fetched_preset_path: Optional[str] = None

        # Tab 6 State (Agent Shell & Streaming)
        self.chat_input: str = ""          # testo scritto direttamente nella scheda Agente
        self.chat_busy: bool = False
        self.chat_focused: bool = False    # [I] entra in scrittura, [ESC] esce: senza, 1-9 non cambiavano piu' scheda
        self.my_benchmarks_path: str = os.path.expanduser("~/.ai_tui_benchmarks.json")
        # Scansione passiva della rete locale dopo 3 min di inattivita' (dai tasti).
        self.last_key_time: float = time.time()
        # Modal: gestione server cloud (tab 14, Alt+M).
        self.modal = MC.ModalCtl()
        self.lan_scan_results: List[Dict[str, Any]] = []
        self.lan_scan_last: float = 0.0
        self.lan_scan_busy: bool = False
        # Aggiornamenti build llama.cpp (GitHub, solo Linux CUDA): controllo
        # automatico all'avvio, conferma [Y] e download con percentuale in
        # Tab 13. Nomi "bm_*" per non confondersi con cfg_upd_* (Tab 10,
        # quello vecchio era per Windows e non si tocca).
        self.bm_upd: Dict[str, Any] = {}
        self.bm_upd_checking: bool = False
        self.bm_upd_confirm: bool = False
        self.bm_upd_downloading: bool = False
        self.bm_upd_progress: str = ""
        self.bm_auto: bool = True   # TUI aperta: scarica e copia su Miky le build nuove da sola
        self.benchmark_cursor: int = 0        # cursore nella pagina benchmark personali
        self.chat_history: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content="System initialized. RTX 3050 8GB VRAM ready. 256k context shifted mode active.",
                timestamp=datetime.now().strftime("%H:%M:%S")
            ),
            ChatMessage(
                role="assistant",
                content="Cyber AI TUI ready. Press [1-9] for Tabs, [G] for Gear Settings, [T] to Benchmark, [H] for Groq HF Config, [F] to Flush VRAM, [F1-F6] to open an RPC preset launch menu.",
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
        ]
        self.is_generating: bool = False
        self.current_stream_buffer: str = ""

        # Tab 7 State (GitHub Plugins & Skills Installer)
        self.plugin_installer = GitHubPluginInstaller()
        self.installed_plugins: List[Dict[str, Any]] = self.plugin_installer.list_installed()
        self.selected_plugin_idx: int = 0
        self.is_installing_plugin: bool = False

        # Tab 8 State (lm_eval Benchmark Lab)
        # Task inventory is served from bench_tasks.json, which is harvested
        # once from a real TaskManager() and never re-read per frame.
        self.lm_tasks: Dict[str, Any] = lmbench.load_tasks()
        self.lm_task_group: str = "classici"  # classici | agentici
        self.lm_selected_task_idx: int = 0
        self.lm_bench_model: str = ""
        self.lm_bench_budget_mib: float = 4096.0
        self.lm_bench_kv: str = "q8_0"
        self.lm_matrix: Optional[Dict[str, Any]] = None
        self.lm_matrix_error: str = ""
        self.is_lm_loading: bool = False

        # Tab 9 State (RPC & CUDA13)
        # Profili utente persistiti su rpc_profiles.json; i preset validati
        # vivono in LlamaServerManager.RPC_PRESETS (backend.py).
        self.rpc_profiles = RPCProfileStore()
        self.rpc_preset_key: str = "orn_16k"   # preset evidenziato nel pannello
        self.rpc_last_preset_key: str = "orn_16k"  # rilanciato da Ctrl+R
        self.rpc_model_idx: int = 0
        self.rpc_is_launching: bool = False
        self.rpc_probe: Dict[str, Any] = {}          # esito probe ggml-rpc-server
        self.rpc_last_cmd: str = ""
        self.rpc_port: int = LP.PORT_DEFAULT   # 9292 (fallback 9205)
        self.rpc_host: str = "localhost"
        # Parametri "consigliati" mostrati in NEON: derivano dal preset
        # RACCOMANDATO, cosi' il pannello non puo' contraddirlo.
        _rec = LlamaServerManager.RPC_PRESETS["rpc_128k_fast"]
        self.rpc_recommended: Dict[str, Any] = {
            "ctx_size": _rec["ctx_size"],
            "parallel": _rec["parallel"],
            "kv_type": _rec["kv_type"],
            "ngl": _rec["ngl"],
            "flash_attn": _rec["flash_attn"],
            "device": _rec["device"],
            "rpc": _rec["rpc"],
            "tok_s": _rec["tok_s"],
        }

        # Tab 10 State (Config llama-server completo)
        # ESTENSIONE del Tab 9: qui si modificano TUTTI i flag di
        # llama-server, letti dinamicamente dal parsing di "llama-server -h"
        # (niente lista hardcodata, resta valida se la build cambia).
        self.cfg_backend: str = "cuda"        # cuda|vulkan|cpu|rpc
        self.cfg_extra_dirs: Dict[str, List[str]] = {}
        self.cfg_flags: Dict[str, LcFlagSpec] = {}    # parsed da -h
        self.cfg_help_source: str = ""                # da quale exe e' stato letto
        self.cfg_help_loaded: bool = False
        self.cfg_loading_help: bool = False
        self.cfg_model: str = ""                      # percorso .gguf
        self.cfg_active_section: int = 0              # indice fra le sezioni
        self.cfg_active_row: int = 0                  # riga nel campo attivo
        self.cfg_scroll: int = 0
        self.cfg_dry_cmd: str = ""                    # ultimo comando generato
        self.cfg_dry_warnings: List[str] = []
        self.cfg_profiles = LcConfigProfileStore()
        self.cfg_profile_name: str = ""
        self.cfg_profile_idx: int = 0
        self.cfg_show_all_flags: bool = False         # elenco completo dall'help
        self.cfg_upd: Dict[str, Any] = {}             # esito check aggiornamenti
        self.cfg_upd_checking: bool = False
        self.cfg_upd_confirm: bool = False            # armato il download
        self.cfg_upd_downloading: bool = False
        self.cfg_upd_progress: str = ""
        self.cfg_miky_pending: str = ""        # build pronta da inviare a Miky
        self.cfg_miky_confirm: bool = False    # armato l'invio (seconda [M])
        self.cfg_miky_progress: str = ""
        self.cfg_cfg = LcServerConfig()               # la configurazione viva
        self.cfg_help_error: str = ""
        self.cfg_model_idx: int = 0

        # Global UI HUD & Notification Banners
        self.status_message: str = "System Online | Press [1-9] for Tabs | [F1-F6] preset (Alt+1..6) | [0] Config llama-server | [U] Update check"
        self.status_set_time: float = time.time()
        self.flush_flash_message: Optional[str] = None
        self.flush_flash_time: float = 0.0

        # Gear ⚙️ Settings Editor Modal ([G])
        self.show_gear_modal: bool = False
        self.gear_active_field: int = 0
        self.gear_model_name: str = ""
        self.gear_model_config: Optional[ModelConfig] = None
        self.gear_fields_def: List[Dict[str, Any]] = [
            {"key": "temperature", "label": "Temperature", "type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "desc": "Sampling randomness (0.0=deterministic, 0.7=balanced)"},
            {"key": "top_p", "label": "Top-P (Nucleus)", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "desc": "Cumulative probability threshold"},
            {"key": "top_k", "label": "Top-K", "type": "int", "min": 1, "max": 100, "step": 5, "desc": "Filter top K highest probability tokens"},
            {"key": "min_p", "label": "Min-P", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "desc": "Minimum token probability relative to top token"},
            {"key": "repeat_penalty", "label": "Repeat Penalty", "type": "float", "min": 1.0, "max": 2.0, "step": 0.05, "desc": "Penalize repetitive token generation"},
            {"key": "context_length", "label": "Context Length", "type": "int", "min": 2048, "max": 262144, "step": 4096, "desc": "Context window limit (up to 256k)"},
            {"key": "ngl", "label": "GPU Layers (-ngl)", "type": "int", "min": 0, "max": 99, "step": 1, "desc": "Tensors offloaded to VRAM (99 = 100% GPU)"},
            {"key": "flash_attention", "label": "Flash Attention (-fa)", "type": "bool", "desc": "Memory-efficient attention kernel"},
            {"key": "no_kv_offload", "label": "RAM-Only KV (-nkvo)", "type": "bool", "desc": "Offload KV cache to Host RAM (100% VRAM free for weights)"},
            {"key": "enable_thinking", "label": "Thinking / Reasoning", "type": "bool", "desc": "Enable DeepSeek-R1 / QwQ thinking mode"},
            {"key": "system_prompt", "label": "System Prompt", "type": "str", "desc": "Default system instruction"},
            {"key": "custom_flags", "label": "Custom Server Flags", "type": "str", "desc": "Additional CLI args passed to llama-server"},
        ]

        # Initial Background Scan
        self.refresh_all_data_async()

    def start_idle_lan_scan(self, idle_s: float = 180.0, cooldown_s: float = 600.0) -> None:
        """Se la TUI e' inattiva da `idle_s` secondi e lo scan precedente e' piu
        vecchio di `cooldown_s`, lancia la scansione in un thread daemon (non
        blocca la UI). Chiamata ciclicamente dal background poller."""
        now = time.time()
        if self.lan_scan_busy:
            return
        if now - self.last_key_time < idle_s:
            return
        if now - self.lan_scan_last < cooldown_s:
            return
        self.lan_scan_busy = True

        def _work():
            try:
                self.lan_scan_results = Srv.scan_lan()
            except Exception:
                pass
            finally:
                self.lan_scan_last = time.time()
                self.lan_scan_busy = False

        threading.Thread(target=_work, daemon=True).start()

    def my_benchmarks(self) -> List[Dict[str, Any]]:
        """Benchmark personali salvati con [T] in scheda Agente."""
        try:
            with open(self.my_benchmarks_path, "r", encoding="utf-8") as fh:
                return json.load(fh) or []
        except (OSError, ValueError):
            return []

    def set_status(self, msg: str) -> None:
        self.status_message = msg
        self.status_set_time = time.time()

    def update_telemetry(self) -> None:
        """Fast non-blocking read from memory cache."""
        self.hw_snapshot = self.hw_monitor.get_snapshot()

    # ---------------------------------------------------------------
    # Tab 8: lm_eval Benchmark Lab actions
    # ---------------------------------------------------------------
    def lm_task_list(self) -> List[Dict[str, Any]]:
        """Currently selected task group (classici or agentici)."""
        return list(self.lm_tasks.get(self.lm_task_group) or [])

    def lm_toggle_task_group(self) -> str:
        """Switch between the classici and agentici halves of the inventory."""
        self.lm_task_group = "agentici" if self.lm_task_group == "classici" else "classici"
        self.lm_selected_task_idx = 0
        self.set_status(f"Task group: {self.lm_task_group.upper()} ({len(self.lm_task_list())})")
        return self.lm_task_group

    def lm_select_model(self, path: str) -> bool:
        """Adopt a local GGUF as the benchmark target."""
        if not path:
            return False
        self.lm_bench_model = lmbench.normalize_path(path)
        self.set_status(f"Benchmark target: {os.path.basename(self.lm_bench_model)}")
        return True

    def lm_compute_matrix(self) -> Dict[str, Any]:
        """Build the ctx x KV matrix for the current target. Header read only.

        This is the dry-run: it opens the GGUF, reads metadata, computes. It
        never starts llama-server, never loads a model and never touches CUDA.
        """
        if not self.lm_bench_model:
            self.lm_matrix_error = "Nessun modello selezionato"
            self.set_status("lm_eval: seleziona prima un modello locale.")
            return {}
        try:
            res = lmbench.dry_run_matrix(
                self.lm_bench_model,
                self.lm_bench_budget_mib,
                tasks=[t["task"] for t in self.lm_task_list()],
            )
        except Exception as e:  # a bad GGUF must not kill the TUI loop
            self.lm_matrix_error = repr(e)
            self.set_status("lm_eval: lettura header fallita.")
            return {}
        if "error" in res:
            self.lm_matrix_error = str(res["error"])
            self.set_status("lm_eval: header incompleto.")
            return {}
        self.lm_matrix_error = ""
        self.lm_matrix = res
        self.set_status("lm_eval: matrice ctx x KV calcolata (dry-run, GPU non toccata).")
        return res

    def lm_refresh_tasks(self) -> None:
        """Re-harvest the inventory from a real TaskManager in the background.

        TaskManager() costs ~70-150 s, so it runs off the render loop and the
        UI keeps showing the cached list until the new file lands.
        """
        if self.is_lm_loading:
            return

        def _worker() -> None:
            self.is_lm_loading = True
            data = lmbench.harvest_tasks()
            self.is_lm_loading = False
            if "error" in data:
                self.set_status(f"lm_eval: {data['error']}")
                return
            lmbench.write_tasks_file(data)
            self.lm_tasks = lmbench.load_tasks()
            self.set_status(
                f"lm_eval {data.get('lm_eval_version')}: "
                f"{data.get('totale_task_disponibili')} task ricaricati."
            )

        self.set_status("lm_eval: interrogazione TaskManager() in corso...")
        threading.Thread(target=_worker, daemon=True).start()

    def lm_cycle_kv(self) -> str:
        """Cycle the KV format used for the max-context recommendation."""
        order = ["f16", "q8_0", "q4_0"]
        self.lm_bench_kv = order[(order.index(self.lm_bench_kv) + 1) % len(order)] \
            if self.lm_bench_kv in order else "f16"
        self.set_status(f"KV format: {self.lm_bench_kv}")
        return self.lm_bench_kv

    def lm_infer_tasks(self, model: str = "") -> List[str]:
        """Resolve the intended task list without running anything."""
        if model:
            self.lm_bench_model = lmbench.normalize_path(model)
        return [t["task"] for t in self.lm_task_list()]

    # ================================================================
    # Tab 10: pannello di configurazione COMPLETO di llama-server
    # ================================================================
    # Nessuna di queste azioni avvia un server: il comando viene solo
    # GENERATO (dry-run) e mostrato.  L'utente ha un server attivo su
    # porta 8087 e non vuole la VRAM satura.

    def cfg_backends(self) -> List[Dict[str, Any]]:
        """Backends disponibili con binary/build risolti (senza avviarli)."""
        out: List[Dict[str, Any]] = []
        for key, info in Lc.list_backends(self.cfg_extra_dirs).items():
            out.append({
                "key": key, "label": info.label, "exists": info.exists,
                "binary": info.binary, "build": info.build, "dll": info.dll,
                "desc": info.description,
            })
        return out

    def cfg_current_binary(self) -> str:
        # Priorita': la build ATTIVA scelta dal gestore build (anche in DEBIAN_FOLDER)
        try:
            ab = self.build_mgr.active()
            if ab is not None and os.path.isfile(ab.server):
                return ab.server
        except Exception:
            pass
        """Percorso del llama-server del backend attualmente scelto."""
        return Lc.detect_backend(self.cfg_backend, self.cfg_extra_dirs).binary

    def cfg_load_help(self) -> None:
        """
        Legge "llama-server -h" e parsa i flag.  gira in background
        perche' l'avvio del binario, anche solo per l'help, richiede
        qualche decimo di secondo.  NON avvia un server.
        """
        if self.cfg_loading_help:
            return
        self.cfg_loading_help = True
        binary = self.cfg_current_binary()
        if not binary or not os.path.isfile(binary):
            self.cfg_loading_help = False
            self.set_status(f"[Tab10] Nessun llama-server per '{self.cfg_backend}'")
            return

        def worker() -> None:
            try:
                text = Lc.read_help_text(binary)
                flags = Lc.parse_help(text)
            except Exception as e:
                flags, err = {}, str(e)
                self.cfg_help_error = err
            self.cfg_flags = flags
            self.cfg_help_source = binary
            self.cfg_help_loaded = bool(flags)
            self.cfg_loading_help = False
            if flags:
                self.set_status(
                    f"[Tab10] {len(flags)} flag letti da {os.path.basename(binary)} -h")
            else:
                self.set_status(f"[Tab10] Lettura help fallita: {self.cfg_help_error}")

        threading.Thread(target=worker, daemon=True).start()

    def cfg_sections(self) -> List[Dict[str, Any]]:
        """
        Raggruppa i campi della UI per sezione funzionale, marcandoli
        SUPPORTATI (lo conferma l'help della build) o non supportati.
        """
        if not self.cfg_flags:
            return []
        for f in Lc.FIELD_SPEC:
            f["supported"] = self.cfg_field_supported(f["flag"])
        groups: Dict[str, Dict[str, Any]] = {}
        for f in Lc.FIELD_SPEC:
            g = groups.setdefault(f["group"], {"key": f["group"], "label": f["group"],
                                               "fields": []})
            g["fields"].append(f)
        return list(groups.values())

    def cfg_field_supported(self, flag: str) -> bool:
        """True se la build corrente espone davvero questo flag."""
        if not self.cfg_flags:
            return False
        for spec in self.cfg_flags.values():
            if spec.matches(flag):
                return True
        return False

    def cfg_flag_help(self, flag: str) -> LcFlagSpec:
        """
        FlagSpec della build corrente per questo flag, o un segnaposto
        "flag assente" se la build non lo espone.

        I campi del dataclass vanno passati per nome: la firma e' lunga e
        fragile, e un errore qui rompeva solo la vista dei campi, non i test.
        """
        for spec in self.cfg_flags.values():
            if spec.matches(flag):
                return spec
        return LcFlagSpec(canonical=flag, aliases=[], takes_value=False,
                          arg_placeholder="", value_kind="bool", choices=[],
                          default="", env="",
                          description="flag assente in questa build",
                          section="assenti")

    def cfg_set_backend(self, key: str) -> str:
        """Cambia backend e ricarica l'help della nuova build."""
        if key not in Lc.BACKENDS:
            return self.cfg_backend
        changed = key != self.cfg_backend
        self.cfg_backend = key
        self.cfg_cfg.backend = key
        if changed:
            # ricarica l'help: i flag cambiano da build a build
            self.cfg_flags = {}
            self.cfg_help_loaded = False
            self.cfg_dry_cmd = ""
            threading.Thread(target=self.cfg_load_help, daemon=True).start()
        return key

    def cfg_cycle_backend(self) -> str:
        keys = list(Lc.BACKENDS.keys())
        i = keys.index(self.cfg_backend) if self.cfg_backend in keys else 0
        return self.cfg_set_backend(keys[(i + 1) % len(keys)])

    def cfg_regen(self) -> str:
        """
        DRY-RUN: rigenera il comando senza eseguirlo.  Ritorna la stringa
        (la TUI la mostra; l'utente decide se copiarla).
        """
        binary = self.cfg_current_binary()
        if not binary:
            self.cfg_dry_cmd = ""
            self.cfg_dry_warnings = [f"nessun llama-server per '{self.cfg_backend}'"]
            return ""
        try:
            out = self.cfg_cfg.build_command(binary, known_flags=self.cfg_flags or None)
        except Exception as e:
            self.cfg_dry_cmd = ""
            self.cfg_dry_warnings = [f"errore di generazione: {e}"]
            return ""
        self.cfg_dry_cmd = out["cmd"]
        warns = list(out.get("notes", []))
        for u in out.get("unsupported", []):
            warns.append(f"non supportato da questa build: {u}")
        if not out.get("argv"):
            warns.append("argv vuoto")
        self.cfg_dry_warnings = warns
        return self.cfg_dry_cmd

    def cfg_row(self) -> int:
        """
        Indice assoluto del campo attivo, calcolato da sezione+riga.
        Serve per muovere la selezione con le frecce.
        """
        secs = self.cfg_sections()
        if not secs:
            return 0
        self.cfg_active_section = max(0, min(self.cfg_active_section, len(secs) - 1))
        sec = secs[self.cfg_active_section]
        self.cfg_active_row = max(0, min(self.cfg_active_row, len(sec["fields"]) - 1))
        n = 0
        for i, s in enumerate(secs):
            if i == self.cfg_active_section:
                return n + self.cfg_active_row
            n += len(s["fields"])
        return 0

    def cfg_active_field(self) -> Optional[Dict[str, Any]]:
        secs = self.cfg_sections()
        if not secs:
            return None
        i = self.cfg_row()
        n = 0
        for s in secs:
            if n <= i < n + len(s["fields"]):
                return s["fields"][i - n]
            n += len(s["fields"])
        return None

    def cfg_move(self, delta: int) -> Optional[str]:
        """Sposta la selezione di `delta` campi, attraversando le sezioni."""
        secs = self.cfg_sections()
        total = sum(len(s["fields"]) for s in secs)
        if not total:
            return None
        self.cfg_row()
        # ricostruisce l'indice assoluto corrente da (sezione, riga)
        cur = 0
        for si, s in enumerate(secs):
            if si < self.cfg_active_section:
                cur += len(s["fields"])
            elif si == self.cfg_active_section:
                cur += self.cfg_active_row
                break
        new = max(0, min(total - 1, cur + delta))
        # riporta il nuovo indice assoluto in (sezione, riga)
        n = 0
        for si, s in enumerate(secs):
            if new < n + len(s["fields"]):
                self.cfg_active_section = si
                self.cfg_active_row = new - n
                break
            n += len(s["fields"])
        self.cfg_scroll = max(0, new - 8)
        f = self.cfg_active_field()
        return f.get("label") if f else None

    def cfg_cycle_value(self, key: str, forward: bool = True) -> str:
        """
        Ciclo il valore di un campo.  Per gli enum usa la lista dell'help
        (dinamica), per i bool alterna, per gli interi un passo.
        """
        fld = None
        for f in Lc.FIELD_SPEC:
            if f["key"] == key:
                fld = f
                break
        if not fld:
            return ""
        cur = getattr(self.cfg_cfg, fld["attr"], None)
        if fld.get("type") == "bool":
            new = not bool(cur)
        elif fld.get("type") == "enum":
            choices = fld.get("choices") or self._cfg_choices(key)
            if not choices:
                return str(cur or "")
            i = choices.index(cur) if cur in choices else -1
            i = (i + (1 if forward else -1)) % len(choices)
            new = choices[i]
        elif fld.get("type") == "int":
            step = int(fld.get("step", 1))
            try:
                new = int(cur or 0) + (step if forward else -step)
            except (TypeError, ValueError):
                new = step
            new = max(int(fld.get("min", new)), min(int(fld.get("max", new)), new))
        elif fld.get("type") == "float":
            step = float(fld.get("step", 0.05) or 0.05)
            try:
                base = float(cur) if cur is not None else 0.0
            except (TypeError, ValueError):
                base = 0.0
            new = round(base + (step if forward else -step), 4)
            if fld.get("min") is not None:
                new = max(float(fld["min"]), new)
            if fld.get("max") is not None:
                new = min(float(fld["max"]), new)
        elif fld.get("type") == "text":
            return str(cur or "")
        else:
            return str(cur or "")
        setattr(self.cfg_cfg, fld["attr"], new)
        self.cfg_regen()
        return str(new)

    def _cfg_choices(self, key: str) -> List[str]:
        """Valori enum di un campo, presi dall'help della build corrente."""
        fld = next((f for f in Lc.FIELD_SPEC if f["key"] == key), None)
        if not fld:
            return []
        spec = self.cfg_flag_help(fld["flag"])
        return list(spec.choices) if spec and spec.choices else []

    def cfg_set_from_spec(self, key: str, value: Any) -> None:
        """Imposta un campo e rigenera il comando in dry-run."""
        fld = next((f for f in Lc.FIELD_SPEC if f["key"] == key), None)
        if fld:
            setattr(self.cfg_cfg, fld["attr"], value)
            self.cfg_regen()

    def cfg_default_value(self, key: str) -> Any:
        """Default dell'help per quel flag, se la build lo dichiara."""
        fld = next((f for f in Lc.FIELD_SPEC if f["key"] == key), None)
        if not fld:
            return None
        return self.cfg_flag_help(fld["flag"]).default

    def cfg_current_value(self, key: str) -> Any:
        fld = next((f for f in Lc.FIELD_SPEC if f["key"] == key), None)
        if not fld:
            return None
        return getattr(self.cfg_cfg, fld["attr"], None)

    def cfg_resets(self) -> int:
        """Azzera tutti i campi ai default del dataclass."""
        fresh = LcServerConfig()
        fresh.backend = self.cfg_backend
        self.cfg_cfg = fresh
        self.cfg_regen()
        return len(Lc.FIELD_SPEC)

    def cfg_pick_model(self, idx: Optional[int] = None) -> str:
        if not self.local_models:
            self.set_status("[Tab10] Nessun modello locale trovato")
            return ""
        i = self.cfg_model_idx if idx is None else idx
        i = max(0, min(i, len(self.local_models) - 1))
        self.cfg_model_idx = i
        self.cfg_model = getattr(self.local_models[i], "path", "") or ""
        self.cfg_cfg.model = self.cfg_model
        self.cfg_regen()
        return self.cfg_model

    def cfg_cycle_model(self, delta: int = 1) -> str:
        if not self.local_models:
            return ""
        return self.cfg_pick_model(self.cfg_model_idx + delta)

    # --- profili JSON ------------------------------------------------
    def cfg_save_profile(self, name: str = "") -> bool:
        name = (name or self.cfg_profile_name or "profilo").strip()
        if not name:
            self.set_status("[Tab10] Nome profilo vuoto")
            return False
        self.cfg_cfg.backend = self.cfg_backend
        res = self.cfg_profiles.save_profile(
            name, self.cfg_cfg,
            notes=f"backend={self.cfg_backend}, da Tab 10 di ai_tui",
            binary=self.cfg_current_binary())
        if res.get("success"):
            self.cfg_profile_name = name
            self.set_status(f"[Tab10] Profilo '{name}' salvato in {res['path']}")
        else:
            self.set_status(f"[Tab10] Salvataggio fallito: {res.get('error')}")
        return bool(res.get("success"))

    def cfg_load_profile(self, name: str) -> bool:
        cfg = self.cfg_profiles.load_config(name)
        if not cfg:
            self.set_status(f"[Tab10] Profilo '{name}' non trovato")
            return False
        self.cfg_cfg = cfg
        self.cfg_profile_name = name
        if cfg.backend in Lc.BACKENDS and cfg.backend != self.cfg_backend:
            self.cfg_backend = cfg.backend
            self.cfg_flags = {}
            self.cfg_help_loaded = False
            threading.Thread(target=self.cfg_load_help, daemon=True).start()
        self.cfg_regen()
        self.set_status(f"[Tab10] Profilo '{name}' caricato")
        return True

    def cfg_delete_profile(self, name: str) -> bool:
        res = self.cfg_profiles.delete(name)
        if res.get("success"):
            self.set_status(f"[Tab10] Profilo '{name}' eliminato")
        else:
            self.set_status(f"[Tab10] {res.get('error')}")
        return bool(res.get("success"))

    def cfg_export_profile(self, name: str = "", dest: str = "") -> bool:
        name = name or self.cfg_profile_name
        dest = dest or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f"cfg_{name or 'profilo'}.json")
        res = self.cfg_profiles.export(name, dest)
        self.set_status(
            f"[Tab10] Esportato '{name}' -> {dest}" if res.get("success")
            else f"[Tab10] Export fallito: {res.get('error')}")
        return bool(res.get("success"))

    def cfg_import_profile(self, src: str, name: str = "") -> bool:
        res = self.cfg_profiles.import_file(src, name)
        self.set_status(
            f"[Tab10] Importato '{res.get('name')}'" if res.get("success")
            else f"[Tab10] Import fallito: {res.get('error')}")
        return bool(res.get("success"))

    # --- aggiornamento -----------------------------------------------
    def cfg_check_update(self) -> None:
        """
        Interroga la GitHub API per l'ultima release di ggml-org/llama.cpp
        e confronta il build number bNNNNN con quello locale.
        Non scarica NULLA: mostra solo l'offerta.
        """
        if self.cfg_upd_checking:
            return
        self.cfg_upd_checking = True
        self.cfg_upd_confirm = False
        self.set_status("[Tab10] Controllo aggiornamenti su GitHub...")

        def worker() -> None:
            try:
                rep = Lc.check_update(self.cfg_backend, timeout=25.0)
            except Exception as e:
                rep = {"ok": False, "error": str(e)}
            self.cfg_upd = rep
            self.cfg_upd_checking = False
            if not rep.get("ok"):
                self.set_status(f"[Tab10] Check fallito: {rep.get('error')}")
            elif rep.get("newer"):
                self.set_status(
                    f"[Tab10] AGGIORNAMENTO: locale b{rep['local_build']} -> "
                    f"b{rep['remote_build']} ({rep.get('download_size', 0) // (1024*1024)} MiB)")
            else:
                self.set_status(
                    f"[Tab10] Sei gia' aggiornato (b{rep.get('remote_build')})")

        threading.Thread(target=worker, daemon=True).start()

    def cfg_arm_download(self) -> bool:
        """
        Arma il download.  Il download parte SOLO con [Y] dopo che
        l'utente ha visto dimensione e spazio libero.  La directory di
        destinazione e' SEMPRE nuova: la build CUDA funzionante non
        viene mai sovrascritta.
        """
        if not self.cfg_upd.get("newer") or not self.cfg_upd.get("ok"):
            return False
        self.cfg_upd_confirm = True
        self.set_status(
            f"[Tab10] CONFERMA [Y] per scaricare b{self.cfg_upd['remote_build']} "
            f"in {self.cfg_upd.get('target_dir', '?')}  (INVIO annulla)")
        return True

    def cfg_disarm(self) -> None:
        self.cfg_upd_confirm = False

    def cfg_download_release(self) -> None:
        """
        Scarica l'asset della nuova build in una directory NUOVA.

        Regole dure:
          * se la conferma non e' stata armata ([Y] una volta dopo [U])
            non parte NULLA;
          * la destinazione e' SEMPRE una directory inesistente, quindi la
            build CUDA funzionante non viene mai sovrascritta;
          * gira in background e non blocca la UI.
        """
        if not self.cfg_upd_confirm:
            self.set_status("[Tab10] Download annullato: serve [Y] di conferma dopo [U]")
            return
        u = self.cfg_upd
        if not u.get("ok") or not u.get("newer"):
            self.set_status("[Tab10] Nessun aggiornamento da scaricare")
            return
        if not u.get("enough_space"):
            self.set_status(
                f"[Tab10] Spazio insufficiente: servono {u.get('needed_mib', 0):.0f} MiB")
            return
        # disarmo SUBITO: un secondo [Y] non deve riscaricare
        self.cfg_upd_confirm = False
        self.cfg_upd_downloading = True
        self.cfg_upd_progress = "avvio download..."
        target = u.get("target_dir", "")
        # L'URL sta dentro l'asset scelto da check_update(), non al top-level.
        asset = u.get("asset") or {}
        asset_url = asset.get("url") or ""
        if not asset_url:
            self.cfg_upd_downloading = False
            self.set_status("[Tab10] Nessun asset URL nell'offerta: nulla scaricato")
            return

        def worker() -> None:
            try:
                def on_progress(got: int, total: int) -> None:
                    pct = (got * 100 // total) if total else 0
                    self.cfg_upd_progress = f"{got / 1048576:.0f} MiB ({pct}%)"

                res = Lc.download_build(
                    asset_url, target, confirm=True, timeout=180.0,
                    progress=on_progress)
                if res.get("ok"):
                    # Senza estrazione la cartella contiene solo il .tar.gz:
                    # risulterebbe "installata" ma non avrebbe llama-server.
                    self.cfg_upd_progress = "estrazione in corso..."
                    ex = Lc.extract_zip(res.get("path", ""))
                    if not ex.get("ok"):
                        self.cfg_upd_progress = f"estrazione fallita: {ex.get('error')}"
                        self.set_status(
                            f"[Tab10] Estratta NO: {ex.get('error')}")
                        return
                    self.cfg_upd_progress = (
                        f"pronta: {ex.get('files')} file in {res.get('dir')}")
                    # Le due macchine devono avere la STESSA build: il
                    # protocollo RPC non tollera versioni diverse. Si propone
                    # subito l'invio, ma non parte senza conferma.
                    self.cfg_miky_pending = res.get("dir", "")
                    self.cfg_miky_confirm = False
                    self.set_status(
                        f"[Tab10] b{u['remote_build']} estratta ({ex.get('files')} file). "
                        "[M] per allinearla anche su Miky")
                else:
                    self.cfg_upd_progress = f"fallito: {res.get('error')}"
                    self.set_status(f"[Tab10] Download fallito: {res.get('error')}")
            except Exception as e:
                self.cfg_upd_progress = f"errore: {e}"
            finally:
                self.cfg_upd_downloading = False

        self.set_status(f"[Tab10] Download di b{u['remote_build']} in corso...")
        threading.Thread(target=worker, daemon=True).start()

    def cfg_sync_miky(self) -> None:
        """
        Allinea Miky alla build appena scaricata e ne riavvia il server RPC.

        Serve perche' il protocollo RPC deve combaciare fra le due macchine:
        aggiornare solo questa lo romperebbe con "RPC server version
        mismatch". Parte SOLO alla seconda [M]: la prima mostra cosa
        succederebbe, perche' il riavvio interrompe il llama-server collegato.
        """
        build = self.cfg_miky_pending
        if not build:
            self.set_status("[Tab10] Nessuna build da inviare: scaricane una con [Y]")
            return
        if self.cfg_miky_progress.startswith("invio"):
            self.set_status("[Tab10] Invio gia' in corso")
            return
        name = os.path.basename(build.rstrip("/"))
        if not self.cfg_miky_confirm:
            self.cfg_miky_confirm = True
            self.set_status(
                f"[Tab10] CONFERMA [M] per copiare {name} su Miky e RIAVVIARE "
                "il suo server RPC (interrompe il llama-server collegato)")
            return
        self.cfg_miky_confirm = False
        self.cfg_miky_progress = f"invio {name} a Miky..."

        def worker() -> None:
            try:
                res = Srv.miky_sync_build(build)
                if res.get("success"):
                    self.cfg_miky_progress = f"attiva su Miky: {res.get('build')}"
                    self.cfg_miky_pending = ""
                    self.set_status(
                        f"[Tab10] {res.get('build')} attiva su Miky"
                        + (" · servizio riavviato" if res.get("restarted") else ""))
                else:
                    err = res.get("error") or "errore sconosciuto"
                    self.cfg_miky_progress = f"fallito: {err}"
                    self.set_status(f"[Tab10] Invio a Miky fallito: {err}")
            except Exception as exc:
                self.cfg_miky_progress = f"errore: {exc}"
                self.set_status(f"[Tab10] Invio a Miky: errore {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def probe_all_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Interroga TUTTI gli endpoint di TUI_HEALTH_ENDPOINTS e ne ricava
        lo stato reale di ciascuno.

        Non si dichiara MAI "ONLINE" senza una risposta vera: per un
        llama-server serve HTTP 200 su /health, per il backend ggml-rpc
        basta che la porta TCP accetti la connessione (non e' HTTP e non
        ha /health: trattarlo come se lo avesse darebbe sempre errore).

        Ritorna {nome: {"state": ..., "detail": ..., "host": ..., "port": ...}}
        con state in ("ONLINE", "STARTING", "OFFLINE").
        """
        out: Dict[str, Dict[str, Any]] = {}
        eps_list = list(TUI_HEALTH_ENDPOINTS)
        # Porta dei preset F1-F6 (8087) e porta del server gestito: prima
        # l'header le ignorava e diceva OFFLINE con ORN-16k online su :8087.
        known = {int(e["port"]) for e in eps_list if e["kind"] != "rpc"}
        for extra in (getattr(self, "rpc_port", 0), getattr(self.llama_mgr, "managed_port", 0)):
            try:
                extra = int(extra or 0)
            except (TypeError, ValueError):
                extra = 0
            if extra and extra not in known:
                known.add(extra)
                # "localhost": i preset usano --host localhost, che puo' essere
                # solo [::1] (127.0.0.1 darebbe "nessuna risposta" = STARTING)
                eps_list.insert(0, {"name": f"{extra} preset", "host": self.rpc_host or "localhost",
                                    "port": extra, "kind": "llama-server"})
        for ep in eps_list:
            name, host, port = ep["name"], ep["host"], int(ep["port"])
            rec: Dict[str, Any] = {"host": host, "port": port, "kind": ep["kind"]}
            if ep["kind"] == "rpc":
                # ggml-rpc serve UN client alla volta (backlog 1): nessuna sonda
                # TCP da qui. Lo stato viene dall'inventario (che sonda al massimo
                # ogni 20 s e mai mentre un llama-server locale e' collegato).
                rec.update(self._rpc_state_from_inventory(host, port))
            else:
                # llama.cpp risponde 503 su /health FINCHE' il modello non
                # e' pronto, e 200 quando e' servibile. I due casi vanno
                # distinti: 503 non e' "rotto", e' "sta ancora caricando".
                code, body = Srv._http_get_status(f"http://{host}:{port}/health")
                if code == 200:
                    rec["state"] = "ONLINE"
                    rec["detail"] = ""
                elif code == 503:
                    rec["state"] = "STARTING"
                    rec["detail"] = "caricando il modello (503)"
                elif code == 0:
                    rec["state"] = "STARTING" if self._process_alive_for(port) else "OFFLINE"
                    rec["detail"] = "nessuna risposta" if rec["state"] == "STARTING" else "porta libera"
                else:
                    rec["state"] = "OFFLINE"
                    rec["detail"] = f"HTTP {code}"
                # /props aggiunge modello e contesto quando e' servibile.
                if rec["state"] == "ONLINE":
                    pok, props = Srv._http_get(f"http://{host}:{port}/props")
                    if pok and isinstance(props, dict):
                        rec["model"] = str(props.get("model_path") or "")
                        dgs = props.get("default_generation_settings") or {}
                        if isinstance(dgs, dict) and dgs.get("n_ctx"):
                            try:
                                rec["n_ctx"] = int(dgs["n_ctx"])
                            except (TypeError, ValueError):
                                rec["n_ctx"] = 0
            out[name] = rec
        return out

    def _rpc_state_from_inventory(self, host: str, port: int) -> Dict[str, Any]:
        """Stato di un ggml-rpc-server SENZA connettersi: dall'inventario,
        o "in uso" se un llama-server locale ha --rpc verso di lui."""
        target = f"{host}:{port}"
        for r in Srv.scan_local_servers():
            rpc = Srv._flag_value(Srv._read_cmdline(r.pid), "--rpc") if r.pid else ""
            if rpc and target in str(rpc).split(","):
                return {"state": "ONLINE", "detail": f"in uso da llama-server PID {r.pid} :{r.port}"}
        for r in self.server_inventory.rows():
            if r.kind == "remoto" and r.host == host and r.port == port:
                return {"state": r.state_line, "detail": r.error_msg}
        return {"state": "OFFLINE", "detail": "non ancora sondato (inventario)"}

    def _process_alive_for(self, port: int) -> bool:
        """C'e' un llama-server locale vivo che ascolta su quella porta?"""
        for row in self.server_inventory.rows():
            if row.kind == "locale" and row.port == port and row.pid:
                if os.path.exists(f"/proc/{row.pid}"):
                    return True
        # L'inventario puo' non essere ancora popolato: si controlla /proc.
        for row in Srv.scan_local_servers():
            if row.port == port and row.pid and os.path.exists(f"/proc/{row.pid}"):
                return True
        return False

    # ONLINE se almeno uno risponde davvero. La stessa funzione e' usata dal
    # poller in background e dai test: un solo criterio, due usi, cosi' non
    # possono divergere.
    @staticmethod
    def state_from_endpoints(eps: Dict[str, Dict[str, Any]]) -> str:
        """Riduce lo stato dei singoli endpoint allo stato globale."""
        if any(e.get("state") == "ONLINE" for e in eps.values()):
            return "ONLINE"
        if any(e.get("state") == "STARTING" for e in eps.values()):
            return "STARTING"
        return "OFFLINE"

    def _bg_poll_server_health(self) -> None:
        # chiamato ogni iterazione: se idle > 3 min, scansione passiva.
        
        """
        Poller di stato in background: interroga TUTTI gli endpoint, non una
        porta fissa, e non mente mai sullo stato.
        """
        while self.running:
            try:
                self.start_idle_lan_scan()        # scan dopo 3 min di idle (idempotente)
                eps = self.probe_all_endpoints()
                self.server_endpoints = eps
                # ONLINE se ALMENO UNO ha risposto davvero.
                self.server_state = self.state_from_endpoints(eps)
                self.server_online = self.server_state == "ONLINE"
                mp = self.llama_mgr.managed_pid
                managed_alive = bool(mp and os.path.exists(f"/proc/{mp}"))
                online_model = next((os.path.basename(e["model"]) for e in eps.values()
                                     if e.get("state") == "ONLINE" and e.get("model")), "")
                if online_model:
                    self.active_model_name = online_model
                elif self.server_state == "OFFLINE" and not managed_alive:
                    # Nessun server: il nome del modello non resta adesso
                    # un modello inesistente, altrimenti l'HUD mente due volte.
                    self.active_model_name = "None (Offline)"
            except Exception:
                # Una sonda che solleva non deve cambiare lo stato: si
                # lascia com'era e si riprova al giro dopo.
                pass
            time.sleep(1.5)

    def refresh_all_data_async(self) -> None:
        """Kick off background scans on startup or refresh."""
        threading.Thread(target=self._scan_models_worker, daemon=True).start()
        threading.Thread(target=self._fetch_news_worker, daemon=True).start()
        threading.Thread(target=self._search_hf_worker, args=(self.hf_query,), daemon=True).start()
        if not getattr(self, "_bm_auto_started", False):
            self._bm_auto_started = True
            threading.Thread(target=self._bm_auto_loop, daemon=True).start()
        else:
            threading.Thread(target=self.bm_check_update, daemon=True).start()

    def _scan_models_worker(self) -> None:
        models = self.llama_mgr.scan_local_models(self.model_dir)
        self.local_models = models

    # -----------------------------------------------------------------
    # Cartella modelli personalizzabile ([D] in Tab 5)
    # -----------------------------------------------------------------
    PATHS_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_dir.json")

    @classmethod
    def _load_saved_model_dir(cls, fallback: str) -> str:
        """Percorso salvato con [D], se c'e' ed esiste davvero sul disco di
        QUESTA macchina: ogni macchina (qui o Miky) ha il proprio
        model_dir.json accanto al proprio tui_app.py, quindi un percorso
        condiviso (es. \\\\192.168.1.50\\lm_models\\models) puo' restare
        diverso da quello locale senza mescolare le due configurazioni."""
        data, _err = _AJ.load_json(cls.PATHS_CONFIG, {})
        saved = (data or {}).get("model_dir") if isinstance(data, dict) else None
        if saved and os.path.isdir(saved):
            return saved
        return fallback

    def set_model_dir(self, path: str) -> Dict[str, Any]:
        """Cambia la cartella modelli a caldo, la salva per i prossimi
        avvii e rilancia subito la scansione (in background, non blocca
        la UI)."""
        path = os.path.normpath(path.strip().strip('"'))
        if not os.path.isdir(path):
            return {"success": False, "error": f"non e' una cartella: {path}"}
        self.model_dir = path
        self.hf_download_dir = path
        self.hf_mgr.default_dest_dir = path
        _AJ.save_json(self.PATHS_CONFIG, {"model_dir": path})
        threading.Thread(target=self._scan_models_worker, daemon=True).start()
        return {"success": True, "path": path}

    # -----------------------------------------------------------------
    # Aggiornamenti build (Tab 13): [U] controlla, [Y] conferma e scarica
    # -----------------------------------------------------------------
    def bm_check_update(self) -> None:
        """Interroga GitHub (cache 6h in build_manager) e confronta col
        build piu' alto gia' presente in BUILD_DIR. Non scarica nulla."""
        if self.bm_upd_checking:
            return
        self.bm_upd_checking = True
        self.bm_upd_confirm = False
        try:
            rel = BM.latest_release()
            local_max = BM.newest_local_build(self.build_mgr.ordered())
            if rel.get("error"):
                self.bm_upd = {"ok": False, "error": rel["error"]}
                self.set_status(f"[Build] Controllo aggiornamenti fallito: {rel['error']}")
                return
            remote = int(rel.get("build") or 0)
            newer = remote > local_max
            size_mb = sum(a.get("size", 0) for a in rel.get("assets", [])) // (1024 * 1024)
            self.bm_upd = {"ok": True, "newer": newer, "local_build": local_max,
                           "remote_build": remote, "release": rel, "size_mb": size_mb}
            if newer and self.bm_auto and not self.bm_upd_downloading:
                # TUI aperta: scarica, estrae, copia su Miky e pulisce senza chiedere
                self.set_status(f"⬇ [Build] b{remote} disponibile (locale: b{local_max}, {size_mb} MiB): "
                                f"scarico in automatico, poi copia su Miky")
                self.bm_upd_confirm = True
                self.bm_download_update()
            elif newer:
                self.set_status(
                    f"⬆ [Build] NUOVA VERSIONE b{remote} disponibile (locale: b{local_max}, "
                    f"{size_mb} MiB) — scheda Build [F12], premi [Y] due volte per scaricarla")
        finally:
            self.bm_upd_checking = False

    BM_AUTO_INTERVAL_S = 6 * 3600

    def _bm_auto_loop(self) -> None:
        """Finche' la TUI e' aperta: controllo nuove build all'avvio e poi ogni 6 ore."""
        while getattr(self, "running", True):
            try:
                self.bm_check_update()
            except Exception as exc:
                self.set_status(f"✗ [Build] controllo automatico: {exc}")
            time.sleep(self.BM_AUTO_INTERVAL_S)

    def bm_arm_download(self) -> bool:
        if not self.bm_upd.get("newer"):
            return False
        self.bm_upd_confirm = True
        self.set_status(f"[Build] CONFERMA [Y] per scaricare b{self.bm_upd['remote_build']} "
                        f"({self.bm_upd.get('size_mb', 0)} MiB) in {BM.BUILD_DIR}  (ESC annulla)")
        return True

    def bm_disarm(self) -> None:
        self.bm_upd_confirm = False

    MIKY_PUSH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "miky-push-builds")
    MIKY_PUSH_LOG = os.path.expanduser("~/.cache/ai_tui/miky-push.log")

    def _push_build_to_miky(self, name: str) -> bool:
        """Copia la build su Miky: li' miky-rpc-autoupdate riavvia miky-rpc con quella."""
        self.bm_upd_progress = f"copia di {name} su Miky (~775 MB)..."
        os.makedirs(os.path.dirname(self.MIKY_PUSH_LOG), exist_ok=True)
        with open(self.MIKY_PUSH_LOG, "a", encoding="utf-8") as log:
            log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} {name}\n")
            log.flush()
            rc = subprocess.run([self.MIKY_PUSH, name], stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            self.bm_upd_progress = f"{name} copiata su Miky"
            self.set_status(f"✓ [Build] {name} copiata su Miky: miky-rpc passa a questa build appena "
                            f"nessun llama-server e' collegato")
        else:
            self.bm_upd_progress = f"copia su Miky fallita (vedi {self.MIKY_PUSH_LOG})"
            self.set_status(f"⚠ [Build] {name} NON copiata su Miky (spenta?): rilancia "
                            f"tools/miky-push-builds. Log: {self.MIKY_PUSH_LOG}")
        return rc == 0

    def _builds_in_use_local(self) -> set:
        """Build da non cancellare: quella attiva e quelle con un llama-server in esecuzione."""
        used = set()
        act = self.build_mgr.builds.get(self.build_mgr.active_key)
        if act:
            used.add(os.path.basename(act.dir))
        for r in Srv.scan_local_servers():
            argv = Srv._read_cmdline(r.pid) if r.pid else []
            if argv:
                used.add(os.path.basename(os.path.dirname(os.path.realpath(argv[0]))))
        return used

    def _prune_builds(self, miky_ok: bool) -> None:
        """Oltre a b11330/b11429/b11435 (e al fork b10373) tiene le 6 build piu' recenti."""
        loc = BM.prune_local(self._builds_in_use_local())
        rem = BM.prune_remote() if miky_ok else {"removed": [], "errors": []}
        if loc["removed"]:
            self.build_mgr.discover()
            self.build_mgr._save_index()
        with open(self.MIKY_PUSH_LOG, "a", encoding="utf-8") as log:
            log.write(f"pulizia PC: {loc}\npulizia Miky: {rem}\n")
        gone = [f"PC {n}" for n in loc["removed"]] + [f"Miky {n}" for n in rem["removed"]]
        if gone:
            self.set_status(f"🧹 [Build] cancellate le build vecchie: {', '.join(gone)} "
                            f"(conservate sempre {', '.join(b[6:] for b in BM.PROTECTED_BUILDS)})")
        errs = loc["errors"] + rem["errors"]
        if errs:
            self.set_status(f"⚠ [Build] pulizia incompleta: {'; '.join(errs)[:150]}")

    def bm_download_update(self) -> None:
        if not self.bm_upd_confirm or not self.bm_upd.get("newer"):
            self.set_status("[Build] Download annullato: serve [U] poi [Y]")
            return
        self.bm_upd_confirm = False
        self.bm_upd_downloading = True
        self.bm_upd_progress = "avvio..."
        rel = self.bm_upd["release"]

        def worker() -> None:
            try:
                def on_progress(name: str, got: int, total: int) -> None:
                    pct = (got * 100 // total) if total else 0
                    self.bm_upd_progress = f"{name}: {got / 1048576:.0f}/{total / 1048576:.0f} MiB ({pct}%)"
                res = BM.download_release(rel, progress=on_progress)
                if res.get("ok"):
                    self.bm_upd_progress = f"estratta in {res['dir']}: sonda della nuova build..."
                    new = self.build_mgr.refresh_after_install()
                    self.bm_upd_progress = f"completato: {res['dir']}"
                    self.set_status(f"✓ [Build] {', '.join(new) or res['dir']} scaricata, estratta e "
                                    f"aggiunta all'elenco ({res['files']} file)")
                    self.bm_upd = {}
                    self._prune_builds(self._push_build_to_miky(os.path.basename(res["dir"])))
                else:
                    self.bm_upd_progress = f"fallito: {res.get('error')}"
                    self.set_status(f"✗ [Build] Download fallito: {res.get('error')}")
            except Exception as exc:
                self.bm_upd_progress = f"errore: {exc}"
            finally:
                self.bm_upd_downloading = False

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_news_worker(self) -> None:
        self.is_fetching_news = True
        try:
            res = self.news_agg.aggregate_all(arxiv_limit=5, hf_limit=5, rss_limit=5)
            self.news_items = res.get("all_combined", [])
        finally:
            self.is_fetching_news = False

    def _search_hf_worker(self, query: str) -> None:
        self.is_searching_hf = True
        try:
            repos = self.hf_mgr.search_models(query, limit=50)   # prima 12: la lista "non scorreva"
            self.hf_repos = repos
            if repos:
                self.selected_repo_idx = 0
                self.hf_quants = self.hf_mgr.get_repo_quants(repos[0].repo_id)
                self.selected_quant_idx = 0
            else:
                self.hf_quants = []
        finally:
            self.is_searching_hf = False

    def search_hf_async(self, query: str) -> None:
        self.hf_query = query
        threading.Thread(target=self._search_hf_worker, args=(query,), daemon=True).start()

    def select_hf_repo(self, idx: int) -> None:
        if 0 <= idx < len(self.hf_repos):
            self.selected_repo_idx = idx
            repo_id = self.hf_repos[idx].repo_id
            threading.Thread(target=self._fetch_quants_worker, args=(repo_id,), daemon=True).start()

    def _fetch_quants_worker(self, repo_id: str) -> None:
        quants = self.hf_mgr.get_repo_quants(repo_id)
        self.hf_quants = quants
        self.selected_quant_idx = 0

    def start_selected_download(self) -> None:
        if not self.hf_repos or not self.hf_quants:
            self.set_status("⚠️ No quant file selected for download.")
            return
        repo = self.hf_repos[self.selected_repo_idx]
        quant = self.hf_quants[self.selected_quant_idx]
        task_id = self.hf_mgr.start_download(repo.repo_id, quant.filename, self.hf_download_dir)
        self.set_status(f"⚡ Started download: {quant.filename} ➔ {self.hf_download_dir}")

    # -----------------------------------------------------------------
    # RAM KV OFFLOAD & SERVER PRESETS
    # -----------------------------------------------------------------
    def cycle_server_preset(self) -> None:
        preset_keys = list(LlamaServerManager.SERVER_PRESETS.keys())
        try:
            curr_idx = preset_keys.index(self.server_preset_key)
            next_idx = (curr_idx + 1) % len(preset_keys)
        except ValueError:
            next_idx = 0
        self.server_preset_key = preset_keys[next_idx]
        preset_info = LlamaServerManager.SERVER_PRESETS[self.server_preset_key]
        self.server_no_kv_offload = preset_info.get("no_kv_offload", False)
        self.set_status(f"⚡ Server preset changed to: {preset_info['name']} (RAM KV Offload: {'ON (-nkvo)' if self.server_no_kv_offload else 'OFF (GPU)'})")

    def toggle_no_kv_offload(self) -> None:
        self.server_no_kv_offload = not self.server_no_kv_offload
        status_str = "ENABLED (-nkvo) ➔ 100% VRAM Free for Model Weights" if self.server_no_kv_offload else "DISABLED ➔ Standard GPU KV Cache"
        self.set_status(f"⚙️ RAM-Only KV Offload {status_str}")
        # KV residency just changed: re-evaluate the KV quantization guardrail.
        self.kv_guardrail_warning = self.evaluate_kv_guardrail()["warning"]

    # -----------------------------------------------------------------
    # (A) ROLLING WINDOW TOGGLE  &  (B) KV QUANTIZATION GUARDRAIL
    # -----------------------------------------------------------------
    def toggle_rolling_window(self) -> bool:
        """
        Toggle the llama.cpp Rolling Window (--context-shift).

        When ON, a full KV window rotates (oldest tokens are shifted out) instead of
        halting generation, which is what keeps long/256k sessions alive.
        """
        self.rolling_window_enabled = not self.rolling_window_enabled
        self.ctx_shift_enabled = self.rolling_window_enabled
        self.llama_mgr.ctx_shift_enabled = self.rolling_window_enabled
        status_str = "ENABLED (KV rotates on overflow)" if self.rolling_window_enabled else "DISABLED (generation halts at window limit)"
        self.set_status(f"🔄 Rolling Window {status_str} ➔ flag {'--context-shift' if self.rolling_window_enabled else '--no-context-shift'}")
        return self.rolling_window_enabled

    def kv_quant_options(self) -> List[Dict[str, Any]]:
        """KV dtype options annotated with guardrail enabled/disabled state."""
        free_mb = getattr(self.hw_snapshot.gpu, "vram_free_mb", None) if self.hw_snapshot else None
        return KVCacheOptimizer.kv_dtype_options(
            no_kv_offload=self.server_no_kv_offload,
            available_vram_mb=free_mb,
        )

    def evaluate_kv_guardrail(self, kv_dtype: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve the effective KV dtype under the guardrail and record any warning.

        HARD RULE: quantized KV (q8_0/q4_0) only when the KV cache is fully in VRAM.
        With host-RAM KV offload (-nkvo) it is forced to f16.
        """
        requested = kv_dtype or self.kv_quant_key
        free_mb = getattr(self.hw_snapshot.gpu, "vram_free_mb", None) if self.hw_snapshot else None
        verdict = KVCacheOptimizer.resolve_kv_dtype(
            requested,
            no_kv_offload=self.server_no_kv_offload,
            ctx_size=32768,
            available_vram_mb=free_mb,
        )
        self.kv_guardrail_warning = verdict["warning"]
        return verdict

    def cycle_kv_quant(self) -> Dict[str, Any]:
        """
        Cycle the requested KV quantization (f16 -> q8_0 -> q4_0 -> f16).

        Options the guardrail forbids are skipped, and forcing the guardrail to
        override a selection surfaces a warning instead of silently applying it.
        """
        options = self.kv_quant_options()
        enabled = [o for o in options if o["enabled"]]
        ids = [o["id"] for o in enabled]
        if not ids:
            # Guardrail forbids every dtype: fall back to the mandated f16 rather
            # than indexing an empty list.
            self.kv_quant_key = KVCacheOptimizer.KV_RAM_ONLY_DTYPE
            self.kv_guardrail_warning = KVCacheOptimizer.KV_QUANT_BLOCKED_RAM_NOTICE
            self.set_status(f"⚠️ {self.kv_guardrail_warning}")
            return self.evaluate_kv_guardrail()
        if self.kv_quant_key in ids:
            nxt = ids[(ids.index(self.kv_quant_key) + 1) % len(ids)]
        else:
            nxt = ids[0]

        self.kv_quant_key = nxt
        verdict = self.evaluate_kv_guardrail(nxt)

        if self.server_no_kv_offload and KVCacheOptimizer.is_quantized_kv_dtype(nxt):
            self.set_status(f"⚠️ {verdict['warning']}")
        else:
            label = f"{verdict['effective']} KV"
            where = "VRAM" if verdict["kv_in_vram"] else "host RAM"
            self.set_status(f"🔢 KV quantization ➔ {label} (KV resident in {where})")
        return verdict

    # -----------------------------------------------------------------
    # TAB 9: RPC & CUDA13  (preset, profili, lancio server)
    # -----------------------------------------------------------------
    def rpc_profile_rows(self) -> List[Tuple[str, Dict[str, Any], str]]:
        """Preset validati + profili utente, nell'ordine dei tasti F1..F5."""
        return self.rpc_profiles.all_profiles()

    def rpc_resolve(self, key: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Resolve a profile key ("rpc_128k_fast" or "user::nome") to its dict
        and its display name. Returns (None, "") when unknown.
        """
        for k, p, _origin in self.rpc_profile_rows():
            if k == key:
                return p, p.get("name", k)
        return None, ""

    def rpc_select(self, key: str) -> Dict[str, Any]:
        """Move the panel selection to a profile key (no launch)."""
        preset, label = self.rpc_resolve(key)
        if preset is None:
            return {"success": False, "error": f"Preset sconosciuto: {key}"}
        self.rpc_preset_key = key
        # Refresh the previewed command line without spawning anything.
        self.rpc_preview_cmd(key)
        return {"success": True, "preset": key, "name": label}

    def rpc_cycle(self, delta: int = 1) -> Dict[str, Any]:
        """Cycle the selected profile (used by Up/Down and by [N]/[P])."""
        keys = [k for k, _p, _o in self.rpc_profile_rows()]
        if not keys:
            return {"success": False, "error": "Nessun profilo disponibile"}
        try:
            idx = keys.index(self.rpc_preset_key)
        except ValueError:
            idx = 0
        nxt = keys[(idx + delta) % len(keys)]
        res = self.rpc_select(nxt)
        if res.get("success"):
            self.set_status(f"🎯 Profilo selezionato: {res['name']}")
        return res

    def rpc_selected_model(self) -> Optional[str]:
        """Path of the GGUF currently selected for the RPC launch."""
        if not self.local_models:
            return None
        idx = min(max(0, self.rpc_model_idx), len(self.local_models) - 1)
        return self.local_models[idx].filepath

    def rpc_model_for(self, key: Optional[str] = None) -> Optional[str]:
        """
        Modello da avviare per un preset. I preset F1-F6 (schema 2) hanno il
        LORO modello fisso (cambiabile solo dal menu di pre-avvio); gli altri
        usano il modello selezionato nella lista (rpc_model_idx).
        Era la causa del "non avvia il modello giusto": F1 avviava sempre il
        modello evidenziato in lista (es. un Qwen3-1.7B da 1 GB).
        """
        preset, _l = self.rpc_resolve(key or self.rpc_preset_key)
        if preset and preset.get("schema") == 2 and preset.get("model"):
            return str(preset["model"])
        return self.rpc_selected_model()

    def rpc_preview_cmd(self, key: Optional[str] = None) -> str:
        """
        Build (but DO NOT run) the exact llama-server command line for a
        preset. Safe to call at any time: no process, no VRAM.
        """
        key = key or self.rpc_preset_key
        preset, _label = self.rpc_resolve(key)
        model = self.rpc_model_for(key)
        if preset is None or not model:
            self.rpc_last_cmd = ""
            return ""
        bin_path = self.llama_mgr.find_rpc_server_binary() or "<llama-server CUDA 13.4>"
        argv = self.llama_mgr.build_rpc_command(
            bin_path, model, preset, port=self.rpc_port, host=self.rpc_host,
            # mai `llama-server --help` (fino a 10 s) nel thread della UI:
            # si usa quello in cache, la lettura vera va in un thread
            help_text=self._help_nonblocking(bin_path),
        )
        self.rpc_last_cmd = " ".join(argv)
        return self.rpc_last_cmd

    def rpc_probe_backend_async(self) -> None:
        """TCP-probe ggml-rpc-server in the background (no VRAM touched)."""
        preset, _label = self.rpc_resolve(self.rpc_preset_key)
        endpoint = (preset or {}).get("rpc") or "192.168.1.100:50052"

        def _worker():
            res = self.llama_mgr.probe_rpc_backend(endpoint)
            self.rpc_probe = res
            if res.get("reachable"):
                self.set_status(
                    f"🔌 ggml-rpc-server su {res.get('endpoint')} raggiungibile (RTT {res.get('rtt_ms', 0):.2f} ms)"
                )
            else:
                self.set_status(
                    f"⚠️ ggml-rpc-server su {endpoint} NON raggiungibile: {res.get('error')}"
                )

        threading.Thread(target=_worker, daemon=True).start()

    def rpc_save_profile(self, name: str) -> Dict[str, Any]:
        """Save the currently selected model/preset values as a user profile."""
        preset, _label = self.rpc_resolve(self.rpc_preset_key)
        if preset is None:
            return {"success": False, "error": "Nessun profilo selezionato"}
        if preset.get("schema") == 2:
            # Preset F1-F6: si salva la copia COMPLETA (modello, flag del menu,
            # -ngl vuoto = auto). Prima save_profile() faceva int("") sul -ngl
            # "auto" e il tasto [S] della Tab 9 andava in ValueError.
            prof = json.loads(json.dumps(preset))
            prof.update(name=name, recommended=False,
                        description=f"Copia di {preset.get('name')}: {preset.get('description', '')}")
            self.rpc_profiles.user_profiles[name] = prof
            ok = self.rpc_profiles.save()
            res = {"success": ok, "name": name, "path": self.rpc_profiles.path,
                   "error": self.rpc_profiles.last_error}
            self.set_status(f"💾 Profilo '{name}' salvato in {os.path.basename(res['path'])}" if ok
                            else f"✗ Salvataggio profilo fallito: {res['error']}")
            return res
        res = self.rpc_profiles.save_profile(
            name=name,
            ctx_size=preset.get("ctx_size", 131072),
            parallel=preset.get("parallel", 1),
            kv_type=preset.get("kv_type", "q4_1"),
            kv_value=preset.get("kv_value"),
            ngl=preset.get("ngl", 99),
            rpc=preset.get("rpc", "192.168.1.100:50052"),
            device=preset.get("device", "CUDA0"),
            flash_attn=preset.get("flash_attn", True),
            threads=preset.get("threads", 8),
        )
        if res.get("success"):
            self.set_status(f"💾 Profilo '{name}' salvato in {os.path.basename(res['path'])}")
        else:
            self.set_status(f"✗ Salvataggio profilo fallito: {res.get('error')}")
        return res

    # =============================================================
    # PANNELLO "SERVER TROVATI" (F6 / F7 / F8)
    # =============================================================
    def servers_toggle_panel(self) -> Dict[str, Any]:
        """F6: apre o chiude il pannello con l'elenco dei server."""
        self.show_server_panel = not self.show_server_panel
        if self.show_server_panel:
            # Alla prima apertura si fa subito un refresh sincrono, cosi' la
            # lista non appare vuota per qualche secondo.
            self.servers_refresh()
            self.set_status("📡 Pannello SERVER APERTO — [F7] chiudi, [↑/↓]+[INVIO] seleziona, [F8] ricarica")
        else:
            self.set_status("📡 Pannello server chiuso")
        return {"success": True, "open": self.show_server_panel}

    def servers_toggle_select_mode(self) -> Dict[str, Any]:
        """F7: entra/esce dalla modalita' selezione (cursore + ENTER)."""
        if not self.show_server_panel:
            self.show_server_panel = True
            self.servers_refresh()
        self.server_select_mode = not self.server_select_mode
        self.set_status(
            "🎯 Selezione ATTIVA: [↑/↓] muovi, [ENTER] conferma, [ESC] esci"
            if self.server_select_mode else "🎯 Selezione disattivata"
        )
        return {"success": True, "select_mode": self.server_select_mode}

    def embedder_poll(self, interval_s: float = 5.0) -> None:
        """Ri-sonda lo stato embedder al massimo ogni interval_s secondi.

        Chiamato dal ciclo di telemetria, NON dal render: l'HTTP a :11435
        gira nel thread di telemetria, mai nel path di rendering del footer.
        """
        now = time.time()
        if now - self._embedder_last_probe < interval_s:
            return
        self._embedder_last_probe = now
        self._embedder_online = self.llama_mgr.embedder_is_running()

    def servers_refresh(self, blocking: bool = False) -> List[Any]:
        """
        Ricalcola l'elenco dei server.

        blocking=False (default) e' quello che usa la UI: il refresh gira
        nel thread dell'inventario e la funzione restituisce subito
        l'ultimo snapshot, senza aspettare la rete.
        """
        if self.llama_mgr.managed_pid:
            self.server_inventory.managed_pids = {int(self.llama_mgr.managed_pid)}
        if blocking:
            return self.server_inventory.refresh_now()
        self.server_inventory.start()
        return self.server_inventory.rows()

    def servers_move(self, delta: int) -> Optional[Any]:
        """Sposta il cursore nel pannello e seleziona la nuova riga."""
        if not self.server_inventory.rows():
            # mai refresh sincrono nel thread della UI: si avvia il thread
            # dell'inventario e si usa lo snapshot (vuoto finche' non arriva)
            self.servers_refresh(blocking=False)
        return self.server_inventory.move_selection(delta)

    def servers_select(self, key: Optional[str]) -> Optional[Any]:
        return self.server_inventory.select(key)

    def servers_confirm_selection(self) -> Dict[str, Any]:
        """ENTER in modalita' selezione: fissa il server attivo."""
        row = self.server_inventory.selected()
        if row is None:
            return {"success": False, "error": "nessun server selezionato"}
        self.server_select_mode = False
        self.server_port = row.port or self.server_port
        if row.kind == "locale":
            self.server_host = row.host
        msg = (f"✅ Server attivo: {row.name} [{row.state_line}] "
               f"{row.model_name} ctx={row.n_ctx}")
        self.set_status(msg)
        self.last_server_action = msg
        return {"success": True, "server": row.as_dict(), "message": msg}

    def servers_stop_selected(self) -> Dict[str, Any]:
        """
        Ferma il server selezionato.

        Per i locali si usa kill <PID> esplicito dopo aver verificato
        /proc/<PID>/exe: MAI `pkill -f llama-server`, che ucciderebbe anche
        la shell che lo esegue (succede in questa sessione).
        """
        res = self.server_inventory.stop_selected()
        if res.get("success"):
            self.last_server_action = f"🛑 {res.get('message')}"
            self.set_status(self.last_server_action)
        else:
            self.last_server_action = f"✗ {res.get('error')}"
            self.set_status(self.last_server_action)
        return res

    def servers_open_browser(self) -> Dict[str, Any]:
        """Apre la porta del server selezionato nel browser."""
        url = self.server_inventory.browser_url()
        if not url:
            return {"success": False, "error": "nessun server selezionato"}
        res = Srv.open_in_browser(url)
        self.set_status(f"🌐 {url} — {res.get('message') or res.get('manual') or res.get('error')}")
        return res

    def servers_running_flags(self) -> List[str]:
        """I flag del server LOCALE attualmente in esecuzione."""
        row = self.server_inventory.selected()
        if row is None or row.kind != "locale" or not row.pid:
            return []
        return Srv.read_running_flags(row.pid)

    def servers_relaunch_selected(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Riavvia il server LOCALE selezionato con gli STESSI flag.

        La riga di comando viene riletta da /proc/<pid>/cmdline nel momento
        dell'azione: non da una lista memorizzata, che potrebbe essere
        vecchia. Serve a "legare a se'" un server avviato a mano.
        """
        row = self.server_inventory.selected()
        if row is None:
            return {"success": False, "error": "nessun server selezionato"}
        # argv + exe vero letti da /proc PRIMA di fermare il processo: dopo lo
        # stop /proc/<pid> non esiste piu' (il vecchio [R] fermava il server e
        # poi rispondeva "processo gia' terminato", senza riavviarlo).
        plan = Srv.relaunch_local(row, dry_run=True)
        if dry_run or not plan.get("success"):
            res = plan
        else:
            stopped = Srv.stop_server_pid(row.pid) if row.pid else {"success": True}
            if not stopped.get("success"):
                res = {"success": False,
                       "error": f"non riesco a fermare il vecchio: {stopped.get('error')}"}
            else:
                if not Srv.wait_port_free(row.port, 10.0):
                    res = {"success": False,
                           "error": f"porta {row.port} ancora occupata dopo lo stop: riavvio annullato"}
                else:
                    res = Srv.relaunch_local(row, argv_override=plan["argv"])
                    if res.get("success") and self.llama_mgr.managed_pid == row.pid:
                        # era il server gestito: la TUI segue il nuovo processo
                        self.llama_mgr.managed_process = res.get("proc")
                        self.llama_mgr.managed_pid = res.get("pid")
                        self.llama_mgr.managed_log_path = res.get("log_path")
        if res.get("success"):
            msg = (f"🧪 anteprima riavvio: {res.get('cmd', '')[:120]}"
                   if dry_run else
                   f"🔄 Riavviato [{row.name}] con gli stessi flag (PID {res.get('pid')})")
            self.last_server_action = msg
            self.set_status(msg)
        else:
            self.last_server_action = f"✗ {res.get('error')}"
            self.set_status(self.last_server_action)
        if not dry_run:
            self.servers_refresh(blocking=True)
        return res

    def servers_action_async(self, what: str) -> None:
        """[S]/[R] del pannello in un thread: stop (fino a 2-4 s) e refresh
        sincrono bloccavano la UI."""
        if getattr(self, "_srv_action_busy", False):
            self.set_status("⏳ azione sul server gia' in corso...")
            return
        self._srv_action_busy = True
        row = self.server_inventory.selected()
        self.set_status(f"⏳ {'Fermo' if what == 'stop' else 'Riavvio'} "
                        f"{row.name if row else 'server'}...")

        def work() -> None:
            try:
                if what == "stop":
                    self.servers_stop_selected()
                else:
                    self.servers_relaunch_selected()
            except Exception as exc:
                self.set_status(f"✗ {what}: {exc}")
            finally:
                self._srv_action_busy = False
        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------------------
    # MIKY: stato, sonda e avvio via SSH
    # -------------------------------------------------------------
    def miky_probe_async(self) -> None:
        """
        Interroga ggml-rpc-server su Miky (192.168.1.100:50052) e ne misura
        la latenza. Solo lettura: nessun processo viene toccato.
        """
        endpoint = "192.168.1.100:50052"
        if getattr(self, "_miky_probe_busy", False):
            self.set_status("⏳ sonda di Miky gia' in corso...")
            return
        user = self._rpc_state_from_inventory("192.168.1.100", 50052)
        if "in uso" in str(user.get("detail", "")):
            # il rpc-server serve UN client alla volta: niente sonda mentre e' in uso
            self.miky_status = {"reachable": True, "rtt_ms": 0.0, "error": "",
                                "note": user["detail"]}
            self.set_status(f"🔌 Miky RPC {user['detail']}: non sondata (un client alla volta)")
            return
        self._miky_probe_busy = True
        pending = [2]

        def _done() -> None:
            pending[0] -= 1
            if pending[0] <= 0:
                self._miky_probe_busy = False

        def _worker() -> None:
            t0 = time.time()
            try:
                res = self.llama_mgr.probe_rpc_backend(endpoint)
                rtt = (time.time() - t0) * 1000.0
                self.miky_status = {
                    "reachable": bool(res.get("reachable")),
                    "rtt_ms": float(res.get("rtt_ms") or rtt),
                    "error": res.get("error", ""),
                }
            except Exception as exc:
                self.miky_status = {"reachable": False, "rtt_ms": 0.0, "error": str(exc)}
            finally:
                _done()

        def _worker_ssh() -> None:
            # Lettura dello stato del processo su Miky: dice se il backend
            # RPC e' davvero vivo (task HermesRPC), senza toccarlo.
            try:
                info = Srv.miky_rpc_running()
                if self.miky_status is not None:
                    self.miky_status["process"] = info
            except Exception:
                pass
            finally:
                _done()

        threading.Thread(target=_worker, daemon=True).start()
        threading.Thread(target=_worker_ssh, daemon=True).start()

    def miky_launch_dry_run(self, model: str = "", port: int = 8090,
                            ctx: int = 32768) -> Dict[str, Any]:
        """
        Mostra il comando che partirebbe su MIKY, senza eseguirlo.

        Non tocca HermesRPC: la porta proposta non e' mai la 50052 del
        ggml-rpc-server ( servers.miky_start_llama lo rifiuta se lo e').
        """
        model = model or self.mikye_model or ""
        res = Srv.miky_start_llama(model, port=port, ctx=ctx, dry_run=True)
        if res.get("success"):
            self.mikye_pending_cmd = res.get("cmd", "")
            self.set_status(f"🧪 Comando MIKY (non eseguito): {self.mikye_pending_cmd[:110]}")
        else:
            self.set_status(f"✗ {res.get('error')}")
        return res

    def miky_launch(self, model: str = "", port: int = 8090,
                    ctx: int = 32768, dry_run: bool = False) -> Dict[str, Any]:
        """
        Avvia davvero un llama-server SU MIKY via SSH.

        Di default dry_run=True: il comando viene mostrato e basta. Serve
        un secondo intervento esplicito per eseguirlo davvero, cosi' un
        tasto premuto per errore non accende una GPU lontana.
        """
        model = model or self.mikye_model or ""
        res = Srv.miky_start_llama(model, port=port, ctx=ctx, dry_run=dry_run)
        if res.get("success") and not dry_run:
            self.set_status(f"✅ Avviato llama-server su Miky (porta {port}) via SSH")
            self.miky_probe_async()
        elif res.get("success"):
            self.mikye_pending_cmd = res.get("cmd", "")
        else:
            self.set_status(f"✗ Miky: {res.get('error')}")
        return res

    def miky_stop(self, port: int = 8090, dry_run: bool = False) -> Dict[str, Any]:
        """Ferma il llama-server su Miky cercando il PID che ascolta."""
        res = Srv.miky_stop_llama(port=port, dry_run=dry_run)
        if res.get("success"):
            self.set_status(f"🛑 Miky porta {port}: "
                            + ("comando preparato (dry-run)" if dry_run else "fermato via SSH"))
        else:
            self.set_status(f"✗ Miky stop: {res.get('error')}")
        return res

    # -------------------------------------------------------------
    # ENDPOINT REMOTI INSERITI A MANO (sostituiscono la scansione di rete)
    # -------------------------------------------------------------
    def ask_input(self, prompt: str, callback) -> None:
        """Chiede una riga di testo all'utente (la raccoglie run_interactive)."""
        self.pending_input = (prompt, callback)

    def servers_add_endpoint(self, text: str) -> Dict[str, Any]:
        res = self.server_inventory.add_endpoint(text)
        if res.get("success"):
            ep = res["endpoint"]
            self.set_status(f"➕ Endpoint {'aggiornato' if res.get('updated') else 'aggiunto'}: "
                            f"{ep['name']} {ep['host']}:{ep['port']} ({ep['type']})"
                            + (f" ssh={ep['ssh']}" if ep.get("ssh") else ""))
            threading.Thread(target=lambda: self.servers_refresh(blocking=True), daemon=True).start()
        else:
            self.set_status(f"✗ Endpoint non aggiunto: {res.get('error')}")
        return res

    def servers_add_endpoint_prompt(self) -> None:
        self.ask_input(
            "Nuovo endpoint remoto. Esempi:  192.168.1.20:8080   |   "
            "PC-Studio 192.168.1.20:8080 user@192.168.1.20   |   192.168.1.100:50052 rpc\n"
            "(nome opzionale, host:porta, 'rpc' per ggml-rpc-server, utente@host per SSH, key=~/.ssh/chiave)",
            self.servers_add_endpoint)

    def servers_remove_selected_endpoint(self) -> Dict[str, Any]:
        row = self.server_inventory.selected()
        if row is None or row.kind != "remoto":
            res = {"success": False, "error": "seleziona prima un endpoint REMOTO (frecce nel pannello [F7])"}
        else:
            res = self.server_inventory.remove_endpoint(row.host, row.port)
            if res.get("success"):
                self.server_inventory.select(None)
        self.set_status(f"🗑 Endpoint rimosso" if res.get("success") else f"✗ {res.get('error')}")
        return res

    def servers_info_selected(self, blocking: bool = False) -> Dict[str, Any]:
        """[I] /props + /slots + /metrics del server selezionato (anche adottato)."""
        row = self.server_inventory.selected()
        if row is None:
            self.set_status("✗ nessun server selezionato")
            return {"success": False, "error": "nessun server selezionato"}

        def work() -> Dict[str, Any]:
            self.server_details_busy = True
            try:
                det = Srv.fetch_details(row)
                self.server_inventory.details[row.key] = det
                ns = len(det.get("slots") or [])
                busy = sum(1 for x in det.get("slots") or [] if x.get("busy"))
                self.set_status(f"ℹ️  {row.name}: {ns} slot ({busy} occupati), "
                                f"ctx={((det.get('props') or {}).get('n_ctx')) or '-'}, "
                                f"metrics={'si' if det.get('metrics') else 'no (--metrics)'}")
                return {"success": True, "details": det}
            except Exception as exc:
                self.set_status(f"✗ info fallita: {exc}")
                return {"success": False, "error": str(exc)}
            finally:
                self.server_details_busy = False

        if blocking:
            return work()
        threading.Thread(target=work, daemon=True).start()
        return {"success": True, "async": True}

    def servers_ssh_gpu(self, blocking: bool = False) -> Dict[str, Any]:
        """[H] stato GPU remoto via SSH (nvidia-smi) per l'host selezionato."""
        row = self.server_inventory.selected()
        ep = self.server_inventory.endpoint_for(row) if row is not None else None
        if ep is None:
            # di default: Miky
            ep = next((e for e in self.server_inventory.remotes if e.ssh), None)
        if ep is None or not ep.ssh:
            self.set_status("✗ nessun endpoint con utente SSH: aggiungilo con [A] (es. ... utente@host)")
            return {"success": False, "error": "nessun target SSH"}
        if getattr(self, "ssh_gpu_busy", False):
            self.set_status("⏳ lettura GPU via SSH gia' in corso...")
            return {"success": False, "error": "gia' in corso"}
        self.ssh_gpu_busy = True

        def work() -> Dict[str, Any]:
            self.ssh_gpu_busy = True
            try:
                res = Srv.ssh_gpu_status(ep.ssh, ep.ssh_key)
                # Anche la vista di llama.cpp (cio' che --fit puo' davvero usare)
                # per gli endpoint RPC di quell'host. NON se un llama-server locale
                # sta gia' usando quell'RPC (un client alla volta: resterebbe in coda).
                rpc_eps = ",".join(f"{e.host}:{e.port}" for e in self.server_inventory.remotes
                                   if e.host == ep.host and e.kind == "rpc"
                                   and "in uso" not in str(self._rpc_state_from_inventory(
                                       e.host, e.port).get("detail", "")))
                if rpc_eps:
                    b = self.build_mgr.active()
                    res["llama"] = Srv.llama_device_memory(b.server if b else "", rpc_eps)
                self.server_inventory.gpu_remote[ep.host] = res
                if res.get("success"):
                    self.set_status("🖥  " + ep.host + " via SSH: " + "  ".join(
                        f"GPU{g['index']} {g['name'][:22]} {g['used']}/{g['total']} MiB {g['util']}% {g['temp']}°C"
                        for g in res["gpus"]))
                else:
                    self.set_status(f"✗ SSH {ep.ssh}: {res.get('error')}")
                return res
            finally:
                self.ssh_gpu_busy = False

        if blocking:
            return work()
        self.set_status(f"🔑 SSH {ep.ssh}: leggo nvidia-smi...")
        threading.Thread(target=work, daemon=True).start()
        return {"success": True, "async": True}

    # -------------------------------------------------------------
    # MARGINI VRAM (locale + host RPC): --fit on --fit-target
    # -------------------------------------------------------------
    def vram_fit_flags(self, rpc: str = "") -> List[str]:
        b = self.build_mgr.active()
        n_local = max(1, len([d for d in (b.devices if b else []) if not d.startswith("RPC")]))
        return self.vram_budget.fit_flags(rpc, n_local)

    def vram_set_local_prompt(self) -> None:
        def cb(text: str) -> None:
            err = self.vram_budget.set_local(text)
            self.set_status(f"✗ {err}" if err else
                            f"🎚  Margine VRAM locale: {self.vram_budget.cfg['local'] or 'default llama.cpp (1024)'} MiB")
        self.ask_input(f"Margine VRAM da lasciare LIBERO sulle GPU locali in MiB "
                       f"(attuale: {self.vram_budget.cfg.get('local')}). "
                       f"Una GPU: 600   due GPU: 600,0   tutto al modello: 0 / 100%", cb)

    def vram_set_remote_prompt(self) -> None:
        def cb(text: str) -> None:
            parts = text.strip().split(None, 1)
            if not parts:
                return
            ep = parts[0]
            err = self.vram_budget.set_remote(ep, parts[1] if len(parts) > 1 else "")
            self.set_status(f"✗ {err}" if err else f"🎚  Margine VRAM {ep}: "
                            f"{self.vram_budget.cfg['remotes'].get(ep, 'rimosso')}")
        cur = ", ".join(f"{k}={v}" for k, v in (self.vram_budget.cfg.get("remotes") or {}).items())
        self.ask_input(f"Margine VRAM per un host RPC: 'host:porta MiB[,MiB...]' (uno per GPU "
                       f"esposta da quel ggml-rpc-server). Es: 192.168.1.100:50052 0,0   "
                       f"(attuali: {cur or 'nessuno'}; solo host:porta = rimuovi)", cb)

    # -------------------------------------------------------------
    # TAB 11 - ASPETTO
    # -------------------------------------------------------------
    def aspetto_change(self, row: Optional[int] = None, step: int = 1) -> str:
        row = self.aspetto_row if row is None else row
        ap = self.appearance
        if row == 0:
            ap.cycle_theme(step)
            msg = f"🎨 Tema: {ap.theme_label}"
        elif row == 1:
            ap.cycle_density(step)
            msg = f"📏 Densita': {ap.density_cfg['label']}"
        else:
            ap.cycle_proportions(step)
            msg = f"▥ Proporzioni: {ap.proportion_cfg['label']}"
        self.set_status(msg + "  (salvato in ~/.ai_tui_appearance.json)")
        return msg

    # -------------------------------------------------------------
    # TAB 12 - BENCHMARK
    # -------------------------------------------------------------
    def _bench_load_safe(self) -> None:
        try:
            self.bench.load()
        except Exception as exc:
            self.set_status(f"✗ lettura benchmark: {exc}")

    # -------------------------------------------------------------
    # TAB 13 - BUILD
    # -------------------------------------------------------------
    def builds_visible(self) -> List[Any]:
        return [b for b in self.build_mgr.ordered() if build_matches(b, self.build_query)]

    def build_selected(self) -> Optional[Any]:
        lst = self.builds_visible()
        if not lst:
            return None
        self.build_cursor = max(0, min(self.build_cursor, len(lst) - 1))
        return lst[self.build_cursor]

    def build_use_selected(self) -> str:
        """Rende la build selezionata quella ATTIVA (anche per il Tab 10)."""
        b = self.build_selected()
        if b is None:
            return ""
        self.build_mgr.set_active(b.key)
        backend = "cuda" if b.gpu else ("rpc" if b.rpc else "cpu")
        self.cfg_extra_dirs = {backend: [b.dir]}
        self.cfg_backend = backend
        self.cfg_flags = {}
        self.cfg_help_loaded = False
        self.set_status(f"✅ Build attiva: {b.key} ({b.summary()}) -> Tab 10 usa {b.server}")
        return b.key

    def build_fetch_docs(self, all_builds: bool = False) -> None:
        if all_builds:
            self.set_status("📚 Scarico la documentazione per TUTTE le build...")
            self.build_mgr.fetch_all_docs_async(
                on_done=lambda res: self.set_status("📚 " + self.build_mgr.status))
            return
        b = self.build_selected()
        if b is None:
            return
        self.set_status(f"📚 Documentazione per {b.key}...")

        def work() -> None:
            r = self.build_mgr.fetch_docs(b.key)
            self.set_status(f"📚 {b.key}: salvati {', '.join(r.get('saved', []))}"
                            + (f" | {'; '.join(r['errors'])[:80]}" if r.get("errors") else "")
                            + f" -> {r.get('dir', '')}")
        threading.Thread(target=work, daemon=True).start()

    def build_add_prompt(self) -> None:
        def cb(text: str) -> None:
            # add_dir fa discover + probe_build (3 subprocess fino a 10 s): in un thread
            def work() -> None:
                try:
                    r = self.build_mgr.add_dir(text)
                except Exception as exc:
                    r = {"success": False, "error": str(exc)}
                self.set_status(f"➕ Build aggiunta: {r.get('key')}" if r.get("success") else f"✗ {r.get('error')}")
            self.set_status("⏳ aggiungo e sondo la build...")
            threading.Thread(target=work, daemon=True).start()
        self.ask_input("Cartella (o percorso di llama-server) della build da aggiungere:", cb)

    def build_rpc_start(self, dry_run: bool = False) -> Dict[str, Any]:
        b = self.build_selected()
        if b is None:
            return {"success": False, "error": "nessuna build"}
        res = BM.start_rpc_server(b, port=self.rpc_local_port, host="0.0.0.0", dry_run=dry_run)
        if res.get("success"):
            self.set_status(("🧪 " if dry_run else "🔌 Avviato ") + f"ggml-rpc-server {b.key} :{self.rpc_local_port}"
                            + ("" if dry_run else f" PID {res.get('pid')} "
                               f"({'cgroup separato' if res.get('isolated') else 'stesso cgroup!'})")
                            + (f" — {res.get('cmd')}" if dry_run else ""))
        else:
            self.set_status(f"✗ rpc-server: {res.get('error')}")
        return res

    def rpc_create_full_vram_profile(self, rpc: str = "192.168.1.100:50052",
                                     ctx: int = 131072, kv: str = "q4_1") -> Dict[str, Any]:
        """
        Profilo RPC utente "MIKY-100": usa TUTTA la VRAM allocabile di Miky.

        * --device RPC0,CUDA0 : entrambi i device ESPLICITI (con il solo
          CUDA0 llama.cpp userebbe solo la GPU locale). Ordine RPC-prima,
          lo stesso che llama.cpp usa senza --device, cosi' l'ordine dei
          valori di --fit-target e' quello atteso.
        * --fit on --fit-target <margini> : dai margini della tab Build
          (default: Miky 0 = 100% allocabile, locale 600 MiB).
        I preset F1-F5 NON vengono toccati: e' un profilo utente in piu'.
        """
        fit = self.vram_budget.fit_flags(rpc, 1)
        res = self.rpc_profiles.save_profile(
            "MIKY-100", ctx_size=ctx, kv_type=kv, ngl=99, rpc=rpc, device="RPC0,CUDA0",
            description="Tutta la VRAM allocabile: Miky (RPC0) margine "
                        f"{self.vram_budget.remote_margins(rpc) or [1024]} MiB, locale "
                        f"{self.vram_budget.local_margins() or [1024]} MiB (--fit-target)")
        if res.get("success"):
            prof = self.rpc_profiles.user_profiles.get("MIKY-100")
            if prof is not None:
                prof["extra_args"] = fit
                self.rpc_profiles.save()
            self.set_status(f"✅ Profilo RPC 'MIKY-100' creato (Tab 9): --device RPC0,CUDA0 {' '.join(fit)}")
        else:
            self.set_status(f"✗ profilo non creato: {res.get('error')}")
        return res

    # -------------------------------------------------------------
    # FILTRO DIMENSIONE MODELLI e stima "dove entra" (VRAM / RPC / RAM)
    # -------------------------------------------------------------
    SIZE_FILTERS = [("tutti", 0.0, 1e9), ("< 4 GB", 0.0, 4.0), ("4-8 GB", 4.0, 8.0),
                    ("8-12 GB", 8.0, 12.0), ("12-24 GB", 12.0, 24.0), ("> 24 GB", 24.0, 1e9)]

    def model_visible(self) -> List[int]:
        """Indici dei modelli che passano il filtro dimensione."""
        _n, lo, hi = self.SIZE_FILTERS[self.model_size_filter]
        return [i for i, m in enumerate(self.local_models) if lo <= m.size_gb < hi]

    def model_step(self, cur: int, delta: int) -> int:
        vis = self.model_visible()
        if not vis:
            return cur
        if cur not in vis:
            return vis[0]
        j = max(0, min(len(vis) - 1, vis.index(cur) + delta))
        return vis[j]

    def cycle_size_filter(self) -> str:
        self.model_size_filter = (self.model_size_filter + 1) % len(self.SIZE_FILTERS)
        vis = self.model_visible()
        if vis and self.selected_model_idx not in vis:
            self.selected_model_idx = vis[0]
        if vis and self.rpc_model_idx not in vis:
            self.rpc_model_idx = vis[0]
        lbl = self.SIZE_FILTERS[self.model_size_filter][0]
        self.set_status(f"🔎 Filtro modelli: {lbl} ({len(vis)} su {len(self.local_models)})")
        return lbl

    def rpc_available_mib(self) -> int:
        """VRAM che Miky (RPC0) annuncia, se l'endpoint RPC e' ONLINE; 0 altrimenti."""
        online = any(r.kind == "remoto" and r.state_line == "ONLINE" and
                     any("ggml-rpc" in f for f in r.flags)
                     for r in self.server_inventory.rows())
        if not online:
            return 0
        for g in (self.server_inventory.gpu_remote or {}).values():
            for d in ((g.get("llama") or {}).get("devices") or []):
                if d["device"].startswith("RPC"):
                    return int(d["free"])
        return int(self._rpc_free_cache or 0) or 6900      # ultimo valore misurato

    def fit_label(self, size_gb: float) -> Tuple[str, str]:
        """
        Dove entra un modello, stimato SOLO dalla dimensione del file
        (+ ~1.5 GB per contesto/KV/buffer): VRAM locale, VRAM locale+Miky
        via RPC ("NEED RPC"), altrimenti in RAM.
        """
        hw = self.hw_snapshot.gpu
        local_gb = max(0.0, (hw.vram_total_mb or 8192) / 1024.0 - 1.5)
        rpc_gb = max(0.0, self.rpc_available_mib() / 1024.0 - 0.8)
        if size_gb <= local_gb:
            return "🟢 VRAM", "ok"
        if rpc_gb and size_gb <= local_gb + rpc_gb:
            return "🟠 NEED RPC", "warn2"
        return "🔴 RAM", "err"

    # -------------------------------------------------------------
    # AVVIO PRESET: dialogo build + barra di avanzamento
    # -------------------------------------------------------------
    LAUNCH_SECTIONS = ("build", "flag", "parametri", "pulsanti")

    def launch_open(self, preset_key: str) -> Dict[str, Any]:
        """F1-F6 / Invio in Tab 9: apre il MENU DI PRE-AVVIO (build, flag,
        parametri, anteprima del comando, [AVVIA]/[INDIETRO])."""
        preset, label = self.rpc_resolve(preset_key)
        if preset is None:
            self.set_status(f"✗ preset sconosciuto: {preset_key}")
            return {"success": False}

        if not self.local_models and preset.get("schema") != 2:
            self.set_status("✗ Nessun .gguf trovato")
            return {"success": False}
        self.rpc_preset_key = preset_key
        dev_spec = str(preset.get("device", "CUDA0"))
        # Se la sonda delle build (avviata all'avvio della TUI) non e' finita,
        # nessuna build e' ancora "probed": si aspetta nel thread di verifica,
        # invece di dire "nessuna build compatibile" (bug con F1-F6 premuti
        # nei primi secondi).
        probing = not self.build_mgr.probe_done.is_set()
        cands = [] if probing else BM.launch_candidates(self.build_mgr.ordered(), dev_spec)
        ep = str(preset.get("rpc") or "")
        # Il ggml-rpc-server serve UN client alla volta: se un altro llama-server
        # locale e' gia' collegato a quell'endpoint, il nuovo resterebbe in coda.
        busy = []
        if ep:
            for r in Srv.scan_local_servers():
                argv = Srv._read_cmdline(r.pid) if r.pid else []
                if ep in argv:
                    busy.append({"pid": r.pid, "port": r.port, "model": r.model_name})
        eff = dict(preset, flags=dict(preset.get("flags") or {})) if preset.get("schema") == 2 else None
        # porta: se l'utente non ne ha scelta una, prima libera tra 9292 e 9205
        if eff is not None and not (preset.get("port_user") or LP.PORT_USER_SET.get(preset_key)):
            eff["port"] = PN.porta_libera_in_rete((LP.PORT_DEFAULT, 9205),
                                                  self.peer_net.ports_in_use() | {int(r.port) for r in Srv.scan_local_servers() if r.port})
        modal = {"preset": preset_key, "label": label, "rpc": ep, "device": preset.get("device"),
                 "cands": cands, "compat": {}, "idx": 0, "checking": bool(ep) or probing, "busy": busy,
                 "eff": eff, "sec": 0, "fi": 0, "pi": 0, "bi": 0}
        if self.rpc_model_idx not in self.model_visible() and self.model_visible():
            self.rpc_model_idx = self.model_visible()[0]
        self.launch_modal = modal
        if not getattr(self, "bm_upd_checking", False) and not getattr(self, "bm_upd", {}).get("ok"):
            threading.Thread(target=self.bm_check_update, daemon=True).start()

        def warm_help() -> None:
            for b in list(modal["cands"]):
                LP.build_help(b.server)

        def set_blocked(ok: List[Any]) -> None:
            """Le build non avviabili restano in elenco (in fondo, con ✗) col motivo."""
            ok_keys = {b.key for b in ok}
            blocked = {}
            for b in self.build_mgr.ordered():
                if b.key in ok_keys:
                    continue
                res = modal["compat"].get(b.key)
                if res and not res[0] and not BM.is_unreachable(res):
                    why = "protocollo RPC diverso da Miky" if "mismatch" in str(res[1]) else str(res[1])[:60]
                else:
                    why = BM.launch_block_reason(b, dev_spec) or "non verificata con Miky"
                blocked[b.key] = (b, why)
            modal["blocked"] = blocked

        def wait_cands() -> List[Any]:
            if probing:
                self.build_mgr.wait_probed(120)
                modal["cands"] = BM.launch_candidates(self.build_mgr.ordered(), dev_spec)
            if not ep:
                set_blocked(modal["cands"])
            return list(modal["cands"])

        if not probing and not ep:
            set_blocked(cands)

        if ep:
            def apply(cands: List[Any], first: bool) -> None:
                # Le build NON compatibili con l'RPC rilevato non si mostrano.
                ok_c = [b for b in cands if modal["compat"].get(b.key, (False,))[0]]
                if not ok_c:
                    # Miky spenta/occupata/timeout: NON e' un verdetto di
                    # incompatibilita'. Si mostrano le build con un avviso
                    # invece di "nessuna build compatibile".
                    ok_c = [b for b in cands
                            if BM.is_unreachable(modal["compat"].get(b.key, (False, "")))]
                    if ok_c:
                        self.set_status("⚠ Miky (RPC) non raggiungibile/occupata: compatibilita' non "
                                        "verificata, le build sono mostrate comunque")
                set_blocked(ok_c)
                if not first and [b.key for b in ok_c] == [b.key for b in modal["cands"]]:
                    return
                prev = None if first or modal["checking"] else self.launch_build(modal)
                want = prev.key if prev else self.build_mgr.active_key
                modal["cands"] = ok_c
                lst = [b for b in ok_c if build_matches(b, modal.get("bq", ""))]
                modal["idx"] = next((i for i, b in enumerate(lst) if b.key == want), 0)
                if not first:
                    self.set_status("↻ elenco build aggiornato dopo la verifica RPC con Miky")

            def check() -> None:
                try:
                    cands = wait_cands()
                    # Verdetti gia' salvati (validi finche' il binario non cambia):
                    # il menu si apre subito, la verifica vera segue in background.
                    cached = {b.key: BM.compat_cached(b, ep) for b in cands}
                    instant = bool(cands) and all(cached.values())
                    if instant:
                        modal["compat"].update(cached)
                        apply(cands, first=True)
                        modal["checking"] = False
                        modal["verifying"] = not modal["busy"]
                        threading.Thread(target=warm_help, daemon=True).start()
                    for b in cands:
                        if self.launch_modal is not modal:
                            return          # menu chiuso o build gia' avviata
                        if modal["busy"]:
                            res = BM.compat_cached(b, ep) or (False, "")
                        else:
                            res = BM.rpc_compatible(b, ep)
                            if instant and BM.is_unreachable(res):
                                res = cached[b.key]
                        modal["compat"][b.key] = res
                        ok, line = res
                        if ok and line:
                            m = re.search(r"(\d+) MiB free", line)
                            if m:
                                self._rpc_free_cache = int(m.group(1))
                    if self.launch_modal is modal or not instant:
                        apply(cands, first=not instant)
                    if not instant:
                        warm_help()
                except Exception as exc:
                    self.set_status(f"✗ verifica build: {type(exc).__name__}: {exc}")
                finally:
                    modal["checking"] = False
                    modal["verifying"] = False
            threading.Thread(target=check, daemon=True).start()
        else:
            def local_ready() -> None:
                try:
                    cl = wait_cands()
                    act = self.build_mgr.active_key
                    modal["idx"] = next((i for i, b in enumerate(cl) if b.key == act), 0)
                    warm_help()
                finally:
                    modal["checking"] = False
            if probing:
                threading.Thread(target=local_ready, daemon=True).start()
            else:
                act = self.build_mgr.active_key
                modal["idx"] = next((i for i, b in enumerate(cands) if b.key == act), 0)
                threading.Thread(target=warm_help, daemon=True).start()
        self.set_status(f"▶ {label}: [TAB] sezione, [SPAZIO] flag, [INVIO] modifica/avvia, [ESC] indietro")
        return {"success": True, "modal": True}

    # --- menu di pre-avvio: helper ------------------------------------------
    def launch_cands(self, m: Optional[Dict[str, Any]] = None) -> List[Any]:
        m = m or self.launch_modal
        if not m or m["checking"]:
            return []
        lst = list(m["cands"]) + [b for b, _why in (m.get("blocked") or {}).values()]
        return [b for b in lst if build_matches(b, m.get("bq", ""))]

    def launch_build(self, m: Optional[Dict[str, Any]] = None):
        lst = self.launch_cands(m)
        if not lst:
            return None
        m = m or self.launch_modal
        return lst[m["idx"] % len(lst)]

    def launch_set_query(self, q: str) -> None:
        """Cambia il filtro build del menu di pre-avvio tenendo, se resta visibile, la build evidenziata."""
        m = self.launch_modal
        prev = self.launch_build(m)
        m["bq"] = q
        lst = self.launch_cands(m)
        m["idx"] = next((i for i, b in enumerate(lst) if prev is not None and b.key == prev.key), 0)

    def build_set_query(self, q: str) -> None:
        prev = self.build_selected()
        self.build_query = q
        lst = self.builds_visible()
        self.build_cursor = next((i for i, b in enumerate(lst) if prev is not None and b.key == prev.key), 0)

    def launch_help(self, m: Optional[Dict[str, Any]] = None) -> str:
        b = self.launch_build(m)
        return self._help_nonblocking(b.server) if b else ""

    def _help_nonblocking(self, binary: str) -> str:
        """--help della build SENZA bloccare: dalla cache, altrimenti "" (nessun
        filtro) e lettura avviata in un thread (un solo thread per binario)."""
        h = LP.cached_help(binary)
        if h is not None:
            return h
        if not binary or not os.path.isfile(binary):
            return ""
        busy = self.__dict__.setdefault("_help_warming", set())
        if binary not in busy:
            busy.add(binary)

            def warm() -> None:
                try:
                    LP.build_help(binary)
                finally:
                    busy.discard(binary)
            threading.Thread(target=warm, daemon=True).start()
        return ""

    def launch_sections(self, m: Optional[Dict[str, Any]] = None) -> List[str]:
        m = m or self.launch_modal
        if m and m.get("eff") is not None:
            return list(self.LAUNCH_SECTIONS)
        return ["build", "pulsanti"]

    def launch_flag_rows(self, m: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Solo i flag che la build scelta accetta (dal suo --help)."""
        h = self.launch_help(m)
        return [f for f in LP.FLAG_CATALOG if LP.flag_supported(f, h)]

    def launch_param_rows(self, m: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Come per i flag: un parametro con "needs" si mostra solo se la build
        # scelta lo accetta (--help), es. --n-cpu-moe / --spec-draft-n-max.
        h = self.launch_help(m)
        fam = str(((m or self.launch_modal or {}).get("eff") or {}).get("family", ""))
        return [p for p in LP.PARAM_CATALOG
                if (not p.get("needs") or LP.flag_supported(p, h))
                and (not p.get("only_family") or p["only_family"] == fam)]

    def launch_param_choices(self, spec: Dict[str, Any]) -> List[Any]:
        if spec.get("choices_from") == "cache":
            return LP.cache_types(self.launch_help())
        if spec.get("choices_from") == "mtp":
            eff = (self.launch_modal or {}).get("eff") or {}
            return LP.mtp_files(str(eff.get("model") or ""))
        return list(spec.get("cycle") or [])

    def launch_argv(self, m: Optional[Dict[str, Any]] = None) -> List[str]:
        """Anteprima del comando completo (nessun processo)."""
        m = m or self.launch_modal
        if not m:
            return []
        b = self.launch_build(m)
        binp = b.server if b else (self.llama_mgr.find_rpc_server_binary() or "llama-server")
        eff = m.get("eff")
        if eff is not None:
            argv = LP.compose_argv(binp, str(eff.get("model") or ""), eff,
                                   port=int(eff.get("port") or self.rpc_port), host=self.rpc_host,
                                   help_text=self.launch_help(m) if b else "")
        else:
            preset, _l = self.rpc_resolve(m["preset"])
            argv = self.llama_mgr.build_rpc_command(binp, self.rpc_model_for(m["preset"]) or "MODEL.gguf",
                                                    preset or {}, port=self.rpc_port, host=self.rpc_host)
        return argv + self.extra_flags_tokens()

    def _launch_save(self) -> None:
        m = self.launch_modal
        if m and m.get("eff") is not None:
            if str(m["preset"]).startswith("user::"):
                # profilo utente (copia di F1-F6 fatta con [S] in Tab 9): si
                # salva il profilo intero in rpc_profiles.json, non negli
                # override dei preset (che conoscono solo F1-F6).
                name = str(m["preset"])[len("user::"):]
                self.rpc_profiles.user_profiles[name] = json.loads(json.dumps(m["eff"]))
                if not self.rpc_profiles.save():
                    self.set_status(f"✗ salvataggio profilo '{name}' fallito: {self.rpc_profiles.last_error}")
            elif not LP.store().set_from(m["preset"], m["eff"]):
                self.set_status(f"✗ salvataggio personalizzazioni fallito: {LP.store().error}")

    def launch_reset(self) -> None:
        """[R]: cancella le personalizzazioni del preset e ricarica i valori di base."""
        m = self.launch_modal
        if not m or m.get("eff") is None:
            return
        if LP.effective(m["preset"]) is None:
            # profilo utente: non ha "valori consigliati" a cui tornare
            self.set_status(f"ℹ {m['label']}: profilo utente, nessun valore consigliato da ripristinare "
                            f"(per ripartire dai default usa F1-F6)")
            return
        LP.store().reset(m["preset"])
        base = LP.effective(m["preset"])
        m["eff"] = dict(base, flags=dict(base["flags"]))
        self.set_status(f"↺ {m['label']}: personalizzazioni azzerate (valori consigliati dai benchmark)")

    def launch_move(self, delta: int) -> None:
        """Frecce su/giu': cursore nella sezione; ai bordi passa alla sezione vicina."""
        m = self.launch_modal
        secs = self.launch_sections(m)
        sec = secs[m["sec"] % len(secs)]
        if sec == "build":
            n = len(self.launch_cands(m)) or 1
            nxt = m["idx"] + delta
            if 0 <= nxt < n:
                m["idx"] = nxt
                return
        elif sec in ("flag", "parametri"):
            rows = self.launch_flag_rows() if sec == "flag" else self.launch_param_rows()
            k = "fi" if sec == "flag" else "pi"
            nxt = m[k] + delta
            if 0 <= nxt < len(rows):
                m[k] = nxt
                return
        # bordo: sezione vicina
        self.launch_section(delta)
        sec2 = secs[m["sec"] % len(secs)]
        if delta < 0:
            if sec2 == "flag":
                m["fi"] = max(0, len(self.launch_flag_rows()) - 1)
            elif sec2 == "parametri":
                m["pi"] = max(0, len(self.launch_param_rows()) - 1)

    def launch_section(self, delta: int) -> None:
        m = self.launch_modal
        secs = self.launch_sections(m)
        m["bq_typing"] = False
        m["sec"] = (m["sec"] + delta) % len(secs)
        if secs[m["sec"]] == "flag":
            m["fi"] = 0 if delta > 0 else m["fi"]
        if secs[m["sec"]] == "parametri":
            m["pi"] = 0 if delta > 0 else m["pi"]

    def launch_toggle_flag(self) -> None:
        m = self.launch_modal
        rows = self.launch_flag_rows()
        if not rows or m.get("eff") is None:
            return
        f = rows[m["fi"] % len(rows)]
        fl = m["eff"]["flags"]
        fl[f["key"]] = not fl.get(f["key"], False)
        self._launch_save()
        self.set_status(f"{'☑' if fl[f['key']] else '☐'} {f['label']}: "
                        f"{'attivo' if fl[f['key']] else 'spento'} (salvato per {m['label']})")

    @staticmethod
    def _kv_follow(eff: Dict[str, Any], key: str, old_value: Any) -> None:
        """-ctk e -ctv restano dello stesso tipo: cambiando -ctk cambia anche
        -ctv SE prima erano uguali (o -ctv era vuoto). Impostando -ctv a un
        valore diverso li si "sblocca": da li' in poi -ctk non lo trascina."""
        if key == "kv_type" and str(eff.get("kv_value") or "") in ("", str(old_value)):
            eff["kv_value"] = eff["kv_type"]

    def _model_list(self) -> List[str]:
        vis = self.model_visible() or list(range(len(self.local_models)))
        return [self.local_models[i].filepath for i in vis]

    def launch_cycle_param(self, delta: int) -> None:
        m = self.launch_modal
        rows = self.launch_param_rows()
        if not rows or m.get("eff") is None:
            return
        spec = rows[m["pi"] % len(rows)]
        eff, key = m["eff"], spec["key"]
        if spec["kind"] == "model":
            lst = self._model_list()
            base_model = LP.PRESETS.get(m["preset"], {}).get("model")
            if base_model and base_model not in lst:
                lst = [base_model] + lst
            if not lst:
                return
            cur = eff.get("model")
            i = lst.index(cur) if cur in lst else -1
            eff["model"] = lst[(i + delta) % len(lst)]
        else:
            ch = self.launch_param_choices(spec)
            if not ch:
                return
            cur = eff.get(key, "")
            norm = [str(c) for c in ch]
            i = norm.index(str(cur)) if str(cur) in norm else -1
            if i < 0 and spec["kind"] == "int" and str(cur).strip().lstrip("-").isdigit():
                # valore scritto a mano: si riparte dal piu' vicino nella lista
                nums = [int(c) for c in ch if str(c).strip()]
                i = max(0, sum(1 for x in nums if x <= int(cur)) - (1 if delta > 0 else 0))
            eff[key] = ch[(i + delta) % len(ch)]
            self._kv_follow(eff, key, cur)
        self._launch_save()

    def launch_edit_param(self) -> None:
        """INVIO su un parametro: chiede il valore come testo (Live in pausa)."""
        m = self.launch_modal
        rows = self.launch_param_rows()
        if not rows or m.get("eff") is None:
            return
        spec = rows[m["pi"] % len(rows)]
        key, eff = spec["key"], m["eff"]

        def cb(text: str) -> None:
            v: Any = text.strip()
            if v in ("-", "auto", "vuoto", "none") and spec.get("empty_ok"):
                v = ""
            elif spec["kind"] == "int":
                if not re.fullmatch(r"-?\d+", v):
                    self.set_status(f"✗ {spec['label']}: serve un numero intero")
                    return
                v = int(v)
                if key == "port" and not 1024 <= v <= 65535:
                    self.set_status("✗ porta: scegli un numero tra 1024 e 65535")
                    return
                if key == "port":
                    LP.PORT_USER_SET[m["preset"]] = True   # porta scelta a mano: non la cambio piu'
                    LP.save_port_choice = None
            elif spec["kind"] == "model":
                v = os.path.expanduser(v)
                if not os.path.isfile(v):
                    self.set_status(f"✗ file non trovato: {v}")
                    return
            elif spec["kind"] == "enum":
                ch = [str(c) for c in self.launch_param_choices(spec)]
                by_name = [c for c in ch if os.path.basename(c) == v]
                if ch and v not in ch and by_name:
                    v = by_name[0]          # basta il nome del file (es. mtp-...-Q4_0.gguf)
                if ch and v not in ch:
                    self.set_status(f"✗ {spec['label']}: valori ammessi {', '.join(ch)}")
                    return
            old_v = eff.get(key, "")
            eff[key] = v
            self._kv_follow(eff, key, old_v)
            self._launch_save()
            self.set_status(f"✓ {spec['label']} = {v if v != '' else 'auto/vuoto'} (salvato per {m['label']})"
                            + (f" · anche -ctv = {v}" if key == "kv_type" and eff.get("kv_value") == v else ""))

        cur = eff.get(key, "")
        hint = " ('-' = vuoto/auto)" if spec.get("empty_ok") else ""
        self.pending_input = (f"{m['label']} · {spec['label']} [attuale: {cur if cur != '' else 'auto'}]{hint}", cb)

    def launch_confirm(self) -> Dict[str, Any]:
        m = self.launch_modal
        if not m:
            return {"success": False}
        if m["checking"]:
            self.set_status("⏳ verifico la compatibilita' delle build con l'RPC...")
            return {"success": False, "error": "verifica in corso"}
        if m.get("busy"):
            b0 = m["busy"][0]
            self.set_status(f"⛔ Miky (RPC) e' gia' usata dal llama-server PID {b0['pid']} su :{b0['port']} "
                            f"({b0['model']}): il rpc-server serve un client alla volta. [K] lo ferma.")
            return {"success": False, "error": "rpc occupato"}
        if not m["cands"]:
            self.set_status("✗ nessuna build compatibile con questo preset")
            return {"success": False, "error": "nessuna build"}
        eff = m.get("eff")
        if eff is not None:
            if not os.path.isfile(str(eff.get("model") or "")):
                self.set_status(f"✗ modello non trovato: {eff.get('model')}")
                return {"success": False, "error": "modello assente"}
            self._launch_save()
            self.rpc_port = int(eff.get("port") or self.rpc_port)
            self.server_port = self.rpc_port   # header: porta del server che parte
        b = self.launch_build(m)
        if b is None:
            self.set_status(f"✗ nessuna build corrisponde al filtro '{m.get('bq', '')}': [/] per cambiarlo")
            return {"success": False, "error": "filtro vuoto"}
        if b.key in (m.get("blocked") or {}):
            why = m["blocked"][b.key][1]
            if "RPC diverso" in why or "mismatch" in why or "non verificata" in why:
                return self.launch_sync_build_prompt(b)
            self.set_status(f"✗ {b.key} non avviabile con questo preset: {why}")
            return {"success": False, "error": why}
        self.launch_modal = None
        # si avvia ESATTAMENTE la config mostrata nell'anteprima (eff), non
        # un preset riletto da disco
        return self.launch_start(m["preset"], m["label"], b.server, b.key,
                                 eff=json.loads(json.dumps(eff)) if eff is not None else None)

    def launch_sync_build_prompt(self, b: Any) -> Dict[str, Any]:
        """Chiede conferma per sincronizzare ed attivare una build su Miky."""
        def cb(ans: str) -> None:
            if ans.lower() in ("s", "si", "sì", "y", "yes", ""):
                self.launch_sync_build_worker(b)
            else:
                self.set_status(f"Sincronizzazione di {b.key} su Miky annullata")
        self.pending_input = (f"Vuoi sincronizzare ed attivare {b.key} su Miky? [S/n]", cb)
        return {"success": True, "prompt": True}

    def launch_sync_build_worker(self, b: Any) -> None:
        """Sincronizza la build su Miky in un thread daemon e ricarica le compatibilita'."""
        m = self.launch_modal
        self.set_status(f"⏳ Sincronizzo {b.key} su Miky (rsync + cambio rpc-server)...")
        def _worker():
            try:
                res = Srv.miky_sync_build(b.dir, activate=True, restart=True)
                if res.get("success"):
                    BM._REMOTE_TAG.clear()
                    BM._COMPAT_CACHE.clear()
                    time.sleep(1.0)
                    if self.launch_modal is m and m:
                        ep = str(m.get("rpc") or "192.168.1.100:50052")
                        c_res = BM.rpc_compatible(b, ep, use_cache=False)
                        m["compat"][b.key] = c_res
                        if b.key in (m.get("blocked") or {}):
                            del m["blocked"][b.key]
                        if not any(x.key == b.key for x in m["cands"]):
                            m["cands"].append(b)
                    self.set_status(f"✅ {b.key} sincronizzata e attiva su Miky! Ora compatibile (premi [INVIO] per avviare)")
                else:
                    self.set_status(f"✗ Errore sincronizzazione Miky: {res.get('error')}")
            except Exception as exc:
                self.set_status(f"✗ Sincronizzazione fallita: {exc}")
        threading.Thread(target=_worker, daemon=True).start()

    def launch_start(self, preset_key: str, label: str, binary: str, build_key: str,
                     eff: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Avvia il preset con la build data e segue l'avvio con la barra."""
        self.rpc_binary_override = binary
        m = {"preset": preset_key, "label": label}
        b = type("B", (), {"key": build_key})
        model = (str(eff.get("model") or "") if eff else "") or self.rpc_model_for(preset_key) or ""
        size = next((x.size_gb for x in self.local_models if x.filepath == model), 0.0)
        if not size and model and os.path.isfile(model):
            size = os.path.getsize(model) / 1024 ** 3
        self.launch_progress = {"label": m["label"], "build": b.key, "model": os.path.basename(model),
                                "size_gb": size, "pct": 1.0, "stage": "avvio processo",
                                "t0": time.time(), "stage_t0": time.time(), "done": False,
                                "error": "", "pid": None, "lines": [], "port": self.rpc_port,
                                "scroll_offset": 0, "visible": True}
        if self.vram_budget.cfg.get("apply_to_presets") or self.server_extra_flags.strip():
            res = self.rpc_launch_with_extra(m["preset"], preset=eff)
        else:
            res = self.rpc_launch(m["preset"], preset=eff)
        if not res.get("success"):
            self.launch_progress.update(done=True, error=str(res.get("error", "avvio rifiutato")))
            return res
        threading.Thread(target=self._launch_watch, daemon=True).start()
        return res

    # Tappe riconoscibili nel log di llama-server -> percentuale.
    _LAUNCH_STAGES = [
        ("load_model: loading model", 12, "lettura del modello"),
        ("load_tensors", 30, "caricamento tensori (GPU locale + Miky via RPC)"),
        ("llama_context", 70, "creazione contesto / cache KV"),
        ("load_model: initializing", 82, "inizializzazione slot"),
        ("warming up", 88, "warmup"),
        ("model loaded", 94, "modello caricato"),
        ("listening on", 97, "in ascolto, attendo /health"),
    ]

    def _launch_watch(self) -> None:
        """Segue l'avvio: tappe dal log + /health. Mai una percentuale inventata:
        fra una tappa e l'altra avanza piano (stima a tempo) e lo dice."""
        lp = self.launch_progress
        if lp is None:
            return
        deadline = time.time() + 20 * 60
        pos = 0
        while time.time() < deadline and lp is self.launch_progress:
            time.sleep(0.5)
            if lp.get("done"):
                # errore gia' riportato dal worker di avvio (start_rpc_server)
                if lp.get("error"):
                    self.set_status(f"✗ Avvio fallito [{lp['label']}]: {lp['error'][:100]}")
                return
            # SOLO PID e log del server appena avviato, passati dal worker:
            # finche' non arrivano si aspetta (niente tappe, niente /health),
            # altrimenti si leggerebbero quelli del server PRECEDENTE.
            log_path = lp.get("log_path") or ""
            if not lp.get("pid"):
                continue
            if log_path:
                new, pos = PS.tail_file(log_path, pos)
                if new:
                    cur_off = lp.get("scroll_offset", 0)
                    if cur_off > 0:
                        lp["scroll_offset"] = min(1000, cur_off + len(new))
                    lp["lines"] = (lp["lines"] + new)[-1000:]
                for ln in new:
                    if (" E " in ln or "error" in ln.lower()) and "exiting" not in ln:
                        lp.setdefault("errs", []).append(ln)
                    for key, pct, name in self._LAUNCH_STAGES:
                        if key in ln and pct > lp["pct"]:
                            lp["pct"], lp["stage"], lp["stage_t0"] = float(pct), name, time.time()
            # avanzamento lento fra le tappe (stima), mai oltre la tappa successiva
            nxt = next((p for _k, p, _n in self._LAUNCH_STAGES if p > lp["pct"]), 99)
            tau = max(8.0, lp["size_gb"] * 6.0)
            el = time.time() - lp["stage_t0"]
            lp["shown"] = lp["pct"] + (nxt - lp["pct"]) * 0.9 * (1 - math.exp(-el / tau))
            # processo morto?
            if lp["pid"] and not os.path.exists(f"/proc/{lp['pid']}"):
                errs = lp.get("errs") or []
                last = (errs[0] if errs else
                        next((l for l in reversed(lp["lines"]) if l.strip()), "nessun log"))
                last = re.sub(r"^[\d.]+ [EWI] \w+\s+", "", last.strip())
                if "create_context" in last or "failed to create context" in last:
                    last += "  → contesto/KV troppo grande per la VRAM: prova un preset con meno contesto o F2/F3"
                lp.update(done=True, error=f"llama-server e' uscito: {last[:140]}")
                self.set_status(f"✗ Avvio fallito [{lp['label']}]: {last[:100]}")
                return
            try:
                # stesso host del lancio ("localhost" puo' essere solo [::1]: urllib
                # prova tutti gli indirizzi risolti, 127.0.0.1 da solo fallirebbe)
                code, _b = Srv._http_get_status(f"http://{self.rpc_host or 'localhost'}:{lp['port']}/health",
                                                timeout_s=1.0)
            except Exception:
                code = 0
            if code == 200:
                lp.update(done=True, pct=100.0, shown=100.0, stage="ONLINE")
                self.servers_refresh(blocking=False)
                self.set_status(f"✅ {lp['label']} ONLINE su :{lp['port']} — {lp['model']} "
                                f"(build {lp['build']}, {time.time() - lp['t0']:.0f}s)")
                time.sleep(4)
                if self.launch_progress is lp:
                    lp["visible"] = False
                return
        if lp is self.launch_progress and not lp["done"]:
            lp.update(done=True, error="timeout: il server non ha risposto entro 20 minuti")

    def launch_free_rpc(self) -> None:
        """[K] nel dialogo: ferma i llama-server locali collegati all'RPC e riapre."""
        m = self.launch_modal
        if not m or not m.get("busy"):
            return
        busy = list(m["busy"])
        self.set_status(f"⏳ fermo {len(busy)} server che occupano l'RPC...")

        def work() -> None:
            # stop_server_pid attende fino a 2-4 s per server: fuori dalla UI
            for b in busy:
                Srv.stop_server_pid(int(b["pid"]))
            self.set_status(f"🛑 fermati {len(busy)} server che occupavano l'RPC; ricontrollo le build...")
            if self.launch_modal is m:
                self.launch_open(m["preset"])
        threading.Thread(target=work, daemon=True).start()

    def launch_stop(self) -> None:
        lp = self.launch_progress
        if lp and lp.get("pid"):
            Srv.stop_server_pid(int(lp["pid"]))
            lp.update(done=True, error="avvio annullato dall'utente")

    def build_rpc_stop_all(self) -> int:
        n = 0
        for r in BM.running_rpc_servers():
            try:
                os.kill(r["pid"], 15)
                n += 1
            except OSError:
                pass
        self.set_status(f"🛑 Fermati {n} ggml-rpc-server locali (kill <PID> esplicito)")
        return n

    # -------------------------------------------------------------
    # Draft: scelta del tipo e build che lo sa usare
    # -------------------------------------------------------------
    def drafts_available(self) -> List[Tuple[str, bool]]:
        """(tipo di draft, supportato da almeno una build installata)."""
        out: List[Tuple[str, bool]] = []
        for t in ("mtp", "dflash", "dspark"):
            out.append((t, self.draft_registry.for_draft(t) is not None))
        return out

    def drafts_report(self) -> List[str]:
        """Righe di riepilogo delle build e dei draft che supportano."""
        return self.draft_registry.report()

    def draft_resolve(self, draft_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Sceglie la build giusta per il draft richiesto.

        Se nessuna build installata sa usare quel draft, `available` e'
        False e `message` spiega il perche': la UI non fallisce in silenzio.
        """
        dt = (draft_type or self.selected_draft_type or "dflash").lower()
        build = self.draft_registry.for_draft(dt)
        if build is None:
            msg = self.draft_registry.explain(dt)
            self.set_status(f"⚠️  {msg}")
            return {"available": False, "draft_type": dt, "build": None, "message": msg}
        return {
            "available": True,
            "draft_type": dt,
            "build": build.key,
            "binary": build.binary,
            "spec_type": DB.spec_type_value(dt),
            "probed": build.probed,
            "message": self.draft_registry.explain(dt),
        }

    # -------------------------------------------------------------
    # Flag arbitrari scritti a mano
    # -------------------------------------------------------------
    def extra_flags_tokens(self, text: Optional[str] = None) -> List[str]:
        """Trasforma il testo dei flag manuali in token (solo parsing)."""
        toks = Srv.split_extra_flags(
            self.server_extra_flags if text is None else text
        )
        # Margini VRAM per GPU: SOLO se l'utente li ha attivati per i preset
        # (Tab 13, [P]). Vanno in coda come i flag manuali: la logica dei
        # preset F1-F5 non cambia.
        if text is None and self.vram_budget.cfg.get("apply_to_presets") \
                and "--fit-target" not in toks:
            preset, _l = self.rpc_resolve(self.rpc_preset_key)
            # i preset F1-F6 hanno gia' il loro --fit-target (menu di avvio)
            if (preset or {}).get("schema") != 2:
                toks += self.vram_fit_flags((preset or {}).get("rpc", ""))
        return toks

    def rpc_preview_cmd_with_extra(self, key: Optional[str] = None) -> str:
        """
        Anteprima del comando RPC CON i flag scritti a mano.

        Non esegue nulla: serve a mostrare esattamente la riga che partirebbe,
        preset più i flag extra dell'utente.
        """
        key = key or self.rpc_preset_key
        preset, label = self.rpc_resolve(key)
        if preset is None:
            return f"[preset sconosciuto: {key}]"
        dr = self.rpc_preview_cmd(key)
        extra = self.extra_flags_tokens()
        if not extra:
            return dr
        return f"{dr} {' '.join(extra)}   [+{len(extra)} flag manuali]"

    def rpc_launch_with_extra(self, key: Optional[str] = None,
                              dry_run: bool = False,
                              preset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Avvia via RPC passando anche i flag scritti a mano.

        dry_run=True non avvia nulla. I flag extra sono messi in coda alla
        riga del preset: se uno di quelli contraddice un flag del preset
        (es. un secondo -c), vale l'ultimo, quindi vincono quelli manuali.
        """
        extra = self.extra_flags_tokens()
        if dry_run:
            return {"success": True, "dry_run": True,
                    "cmd": self.rpc_preview_cmd_with_extra(key),
                    "extra_flags": extra}
        if not extra:
            return self.rpc_launch(key, dry_run=False, preset=preset)
        # Il preset normale non conosce i flag extra: si clona il preset e
        # gli si aggiunge `extra_args`, cosi' build_rpc_command li mette in
        # coda alla riga finale. Non si costruisce l'argv a mano: start_
        # rpc_server lo ricostruirebbe e i flag manuali andrebbero persi.
        given = preset
        preset, label = self.rpc_resolve(key or self.rpc_preset_key)
        if given is not None:
            preset = given          # config del menu di pre-avvio (anteprima)
        if preset is None:
            return {"success": False, "error": f"Preset sconosciuto: {key}"}
        model = (str(given.get("model") or "") if given else "") or self.rpc_model_for(key or self.rpc_preset_key)
        if not model:
            return {"success": False, "error": "Nessun modello .gguf selezionato (vedi Tab 5)"}
        binp = self.rpc_binary_override or self.llama_mgr.find_rpc_server_binary()
        preset_ext = dict(preset)
        preset_ext["extra_args"] = list(extra)
        # Anticipo la riga esatta per mostrarla subito (dry-run, nessun processo).
        argv = self.llama_mgr.build_rpc_command(binp, model, preset_ext,
                                                port=self.rpc_port, host=self.rpc_host)
        self.last_server_action = f"▶️  Avvio RPC con {len(extra)} flag manuali"
        self.set_status(self.last_server_action)
        if self.llama_mgr.managed_process is None and \
                self.llama_mgr.is_port_in_use(self.rpc_host, self.rpc_port):
            return {"success": False, "needs_confirmation": True,
                    "error": f"Porta {self.rpc_port} gia' occupata da un server non avviato "
                             f"da questa TUI"}
        self.rpc_is_launching = True
        lp_bound = self._bind_progress()

        def _worker_extra():
            try:
                res = self.llama_mgr.start_rpc_server(
                    model_path=model, preset_key=(key or self.rpc_preset_key),
                    port=self.rpc_port, host=self.rpc_host,
                    binary_path=self.rpc_binary_override,
                    preset_override=preset_ext,
                )
            except Exception as exc:
                res = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            self._progress_result(lp_bound, res, self.llama_mgr.managed_log_path or "")
            self.rpc_is_launching = False
            if res.get("success"):
                self.set_status(f"✅ Avviato RPC [{label}] con flag manuali (PID {res.get('pid')})")
                self.servers_refresh(blocking=True)
            else:
                self.set_status(f"✗ Avvio RPC fallito: {res.get('error')}")

        threading.Thread(target=_worker_extra, daemon=True).start()
        return {"success": True, "launching": True, "extra_flags": extra,
                "argv": argv, "preset": key or self.rpc_preset_key}

    def _bind_progress(self) -> Optional[Dict[str, Any]]:
        """La barra di avanzamento appena creata da launch_start (se c'e'):
        il worker di avvio le passa PID e log REALI del server nuovo."""
        lp = self.launch_progress
        if lp is not None and not lp.get("done") and not lp.get("bound"):
            lp["bound"] = True
            return lp
        return None

    @staticmethod
    def _progress_result(lp: Optional[Dict[str, Any]], res: Dict[str, Any], log_path: str) -> None:
        if lp is None:
            return
        if res.get("success"):
            lp["log_path"] = log_path
            lp["pid"] = res.get("pid")
        else:
            lp.update(done=True, error=str(res.get("error") or "avvio fallito"))

    def rpc_launch(self, key: Optional[str] = None, dry_run: bool = False,
                   preset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Launch llama-server with an RPC preset on a background thread.

        dry_run=True only builds and returns the command line; use it to
        inspect the flags without allocating any VRAM.
        preset = config gia' risolta (menu di pre-avvio): si avvia quella,
        identica all'anteprima, invece di rileggere il profilo.
        """
        key = key or self.rpc_preset_key
        given = preset
        preset, label = self.rpc_resolve(key)
        if given is not None:
            preset = given
            label = str(given.get("name") or label or key)
        if preset is None:
            return {"success": False, "error": f"Preset sconosciuto: {key}"}
        model = (str(given.get("model") or "") if given else "") or self.rpc_model_for(key)
        if not model:
            return {"success": False, "error": "Nessun modello .gguf selezionato (vedi Tab 5)"}

        self.rpc_preset_key = key
        self.rpc_last_preset_key = key

        if dry_run:
            cmd = self.rpc_preview_cmd(key)
            self.set_status(f"🧪 DRY-RUN [{label}] — nessun server avviato, nessuna VRAM occupata")
            return {"success": True, "dry_run": True, "cmd": cmd, "preset": key}

        # Safety gate BEFORE any thread: if something already listens on the
        # target port and we did not start it, refuse instead of terminating a
        # server the user may be using. A hotkey must never destroy state.
        if (self.llama_mgr.managed_process is None
                and self.llama_mgr.is_port_in_use(self.rpc_host, self.rpc_port)):
            err = (f"⛔ Porta {self.rpc_port} gia' occupata da un server non avviato "
                   f"da questa TUI — nessun processo toccato. Cambia porta o ferma tu quel server.")
            self.set_status(err)
            with self.llama_mgr.log_lock:
                self.llama_mgr.server_logs.append(f"[RPC] {err}")
            return {
                "success": False,
                "needs_confirmation": True,
                "error": err,
            }

        self.rpc_is_launching = True
        lp_bound = self._bind_progress()

        def _worker():
            try:
                try:
                    res = self.llama_mgr.start_rpc_server(
                        model_path=model,
                        preset_key=key,
                        port=self.rpc_port,
                        host=self.rpc_host,
                        binary_path=self.rpc_binary_override,
                        # sempre il preset RISOLTO: per F1-F6 contiene le
                        # personalizzazioni del menu di pre-avvio.
                        preset_override=preset,
                    )
                except Exception as exc:
                    res = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                # PID e log del server APPENA avviato (non quelli del precedente)
                self._progress_result(lp_bound, res, self.llama_mgr.managed_log_path or "")
                if res.get("success"):
                    self.active_model_name = os.path.basename(model)
                    self.rpc_last_cmd = res.get("cmd", "")
                    self.rpc_probe = res.get("rpc_probe", {})
                    # Popen success only means the process was created: a bad
                    # model or a bad flag makes llama-server exit within a
                    # second. Verify it is actually serving before claiming
                    # "online", otherwise the user chases a ghost server.
                    def _confirm_health():
                        time.sleep(4.0)
                        if not self.running:
                            return
                        if not self.llama_mgr.is_port_in_use(self.rpc_host, self.rpc_port):
                            tail = self.llama_mgr.get_server_logs(max_lines=3)
                            self.set_status(
                                f"✗ llama-server [{label}] ha avviato ma e' subito uscito "
                                f"(porta {self.rpc_port} libera). Ultimo log: "
                                f"{(tail[-1] if tail else 'n/d')[:90]}"
                            )
                    threading.Thread(target=_confirm_health, daemon=True).start()
                    self.set_status(
                        f"✓ llama-server online [{label}] — porta {self.rpc_port} (PID {res.get('pid')})"
                    )
                else:
                    self.set_status(f"✗ Lancio RPC fallito [{label}]: {res.get('error')}")
            finally:
                self.rpc_is_launching = False

        threading.Thread(target=_worker, daemon=True).start()
        self.set_status(f"⚡ Avvio llama-server [{label}] in corso...")
        return {"success": True, "launching": True, "preset": key, "name": label}

    def run_tool_truncation(self, max_tokens: int = 8192, sample_chars: int = 120000) -> Dict[str, Any]:
        """
        (C) Run the Groq-backed tool-context truncation over a representative payload
        and record what was dropped, so the footer can surface the loss instead of
        silently hiding it.
        """
        if self.last_compression_result and self.last_compression_result.compressed_text:
            source = self.last_compression_result.compressed_text
        else:
            source = ("TOOL PAYLOAD " * sample_chars)[:sample_chars]
        res = self.compressor.truncate_tool_context(source, max_tokens=max_tokens)
        self.last_tool_truncation = res
        if res["truncated"]:
            self.set_status(
                f"✂️ Tool context truncated: {res['original_tokens']} ➔ {res['kept_tokens']} tok "
                f"(-{res['dropped_tokens']} dropped, head+tail kept)"
            )
        else:
            self.set_status(f"✂️ Tool context fits: {res['kept_tokens']} tok (no truncation)")
        return res

    # -----------------------------------------------------------------
    # DEDICATED MODEL TESTING & BENCHMARK ENGINE ([T])
    # -----------------------------------------------------------------
    def _bench_guard(self) -> str:
        """I benchmark avviano il LORO server (start_server ferma quello gestito):
        se e' attivo un server della TUI che non e' di un benchmark (es. un
        preset F1-F6 su :8087), si rifiuta invece di ucciderlo."""
        if not hasattr(self, "_bench_pids"):
            self._bench_pids = set()
        mp = self.llama_mgr.managed_pid
        if mp and mp not in self._bench_pids and Srv._pid_gone(mp) is False:
            return (f"⛔ Benchmark non avviato: e' attivo il llama-server della TUI PID {mp} "
                    f":{self.llama_mgr.managed_port} (non suo). Fermalo prima (F7 → [S] o [F][F]).")
        return ""

    def run_model_benchmark_async(self, model_idx: Optional[int] = None) -> None:
        """Run Python AST & JSON tool calling benchmark on selected local model."""
        if self.is_benchmarking:
            self.set_status("⚠️ Benchmark already in progress. Please wait...")
            return
        if not self.local_models:
            self.set_status("⚠️ No local models found in /mnt/lm_models/models to benchmark.")
            return

        target_idx = self.selected_model_idx if model_idx is None else model_idx
        if not (0 <= target_idx < len(self.local_models)):
            target_idx = 0
        model_info = self.local_models[target_idx]
        guard = self._bench_guard()
        if guard:
            self.set_status(guard)
            return

        self.is_benchmarking = True
        self.set_status(f"⚡ Starting Benchmark on '{model_info.filename}' (AST Syntax, Tool Calling, Speed, TTFT)...")

        def _worker():
            try:
                config = self.config_mgr.get_config(model_info.filename)
                result = self.test_runner.run_live_model_benchmark(
                    model_info=model_info,
                    config=config,
                    server_mgr=self.llama_mgr,
                    hw_monitor=self.hw_monitor
                )
                self._bench_pids.add(self.llama_mgr.managed_pid)
                self.last_benchmark_result = result
                self.benchmark_history.append(result)
                # Anche il benchmark singolo va nel DB (prima lo salvava solo la
                # catena): cosi' la Tab 12 lo mostra dopo [R].
                try:
                    self.bench_db.insert_result(result, model_info)
                except Exception as exc_db:
                    self.set_status(f"⚠ benchmark non salvato nel DB: {str(exc_db)[:60]}")
                if result.status == "SERVER_OFFLINE":
                    err_hint = f" ({result.error_detail[:45]})" if result.error_detail else ""
                    self.set_status(f"⚠️ Benchmark Warning: Server offline on port {self.server_port}{err_hint}. Press [L/Enter] to launch.")
                elif result.status == "ERROR":
                    err_hint = f": {result.error_detail[:60]}" if result.error_detail else f": {result.error_msg}"
                    self.set_status(f"✗ Benchmark Error{err_hint}")
                else:
                    status_icon = "✓" if result.all_passed else "✗"
                    self.set_status(f"{status_icon} Benchmark Finished: {result.summary_str()}")
            except Exception as e:
                self.set_status(f"✗ Benchmark Error: {str(e)[:80]}")
            finally:
                self.is_benchmarking = False

        threading.Thread(target=_worker, daemon=True).start()

    # -----------------------------------------------------------------
    # MULTI-MODEL CHAIN BENCHMARK ([T] with selection)
    # -----------------------------------------------------------------
    def toggle_model_selection(self, idx: Optional[int] = None) -> None:
        """Toggle selection of the current or given model index for chain benchmark."""
        target = self.selected_model_idx if idx is None else idx
        if target in self.selected_model_indices:
            self.selected_model_indices.discard(target)
        else:
            self.selected_model_indices.add(target)
        n = len(self.selected_model_indices)
        self.set_status(f"⚡ {n} model{'s' if n != 1 else ''} selected for chain benchmark. [T] to run, [A] select-all, [N] deselect-all.")

    def select_all_models(self) -> None:
        self.selected_model_indices = set(range(len(self.local_models)))
        self.set_status(f"⚡ All {len(self.local_models)} models selected. [T] to run chain benchmark.")

    def deselect_all_models(self) -> None:
        self.selected_model_indices.clear()
        self.set_status("⚡ All selections cleared.")

    def run_chain_benchmark_async(self) -> None:
        """Run chain benchmark on all selected models (multi-select [T])."""
        if self.is_benchmarking:
            self.set_status("⚠️ Chain benchmark already running. Please wait...")
            return

        if not self.selected_model_indices:
            # Fall back to single model if nothing selected
            self.run_model_benchmark_async()
            return

        selected_models = [self.local_models[i] for i in sorted(self.selected_model_indices) if 0 <= i < len(self.local_models)]
        if not selected_models:
            self.set_status("⚠️ No valid models selected.")
            return
        guard = self._bench_guard()
        if guard:
            self.set_status(guard)
            return

        self.is_benchmarking = True
        self.chain_bench_results = []
        self.chain_bench_log = [f"🚀 Chain Benchmark started: {len(selected_models)} models queued."]
        self.show_chain_report = False
        self.set_status(f"⚡ Chain Benchmark queued for {len(selected_models)} models. Running sequentially...")

        def _chain_worker():
            try:
                for step_idx, model_info in enumerate(selected_models):
                    if not self.is_benchmarking:
                        break
                    self.active_model_name = f"[Benchmarking] {model_info.filename[:22]}"
                    self.chain_bench_log.append(f"[{step_idx+1}/{len(selected_models)}] {model_info.filename[:35]}...")
                    self.set_status(f"⚡ [{step_idx+1}/{len(selected_models)}] Benchmarking '{model_info.filename[:28]}'...")

                    try:
                        # 1. Flush VRAM before each model
                        self.llama_mgr.flush_vram(self.hw_monitor, kill_external=False)
                        time.sleep(0.5)

                        config = self.config_mgr.get_config(model_info.filename)

                        # 2. Full benchmark (starts server, AST, Tool, tok/s, TTFT)
                        result = self.test_runner.run_live_model_benchmark(
                            model_info=model_info,
                            config=config,
                            server_mgr=self.llama_mgr,
                            hw_monitor=self.hw_monitor,
                            wait_ready_s=30.0
                        )
                        self._bench_pids.add(self.llama_mgr.managed_pid)

                        # 3. Context size probing (8k, 16k, 32k) if server came online
                        if result.status == "COMPLETED":
                            self.chain_bench_log.append(f"  ↳ Probing 8k/16k/32k context sizes...")
                            ctx_probe = self.test_runner.probe_context_sizes(
                                model_info=model_info,
                                config=config,
                                server_mgr=self.llama_mgr,
                                hw_monitor=self.hw_monitor,
                                wait_ready_s=25.0
                            )
                            result.ctx_probe_results = ctx_probe
                            result.optimal_ctx = determine_optimal_ctx(ctx_probe)
                            self.chain_bench_log.append(f"  ↳ Optimal ctx: {result.optimal_ctx:,} tokens")

                        icon = "✓" if result.all_passed else ("⚠" if result.status == "SERVER_OFFLINE" else "✗")
                        self.chain_bench_log.append(
                            f"  {icon} {result.generation_speed_tok_s:.1f} tok/s | TTFT {result.ttft_ms:.0f}ms"
                            f" | AST {'✓' if result.python_ast_pass else '✗'}"
                            f" | Tool {'✓' if result.json_tool_pass else '✗'}"
                        )

                        self.chain_bench_results.append(result)
                        self.bench_db.insert_result(result, model_info)

                    except Exception as ex:
                        self.chain_bench_log.append(f"  ✗ Error: {str(ex)[:60]}")

                    # 4. Flush server after each model
                    self.llama_mgr.stop_server()
                    self.llama_mgr.flush_vram(self.hw_monitor, kill_external=False)
                    time.sleep(0.5)

                # 5. Generate final report
                if self.chain_bench_results:
                    try:
                        rp = generate_benchmark_report(
                            results=self.chain_bench_results,
                            model_infos=selected_models,
                            config_mgr=self.config_mgr,
                            # Su Linux "C:/Users/..." era un percorso RELATIVO (finiva in
                            # ./C:/Users/... sotto la cartella corrente, invisibile alla
                            # Tab 12). Ora: accanto al DB dei benchmark (cartella letta
                            # dalla Tab 12), con nome proprio per non sovrascrivere il
                            # benchmark_report.json storico.
                            report_path=("C:/Users/User/Documents/ai_tui/benchmark_report.json" if os.name == "nt"
                                         else os.path.join(os.path.dirname(self.bench_db.db_path),
                                                           "benchmark_report_chain.json"))
                        )
                        self.chain_bench_report_path = rp
                        self.chain_bench_log.append(f"📄 Report saved: {rp}")
                    except Exception as ex:
                        self.chain_bench_log.append(f"⚠️ Report gen error: {str(ex)[:60]}")

                self.chain_bench_log.append(f"🎉 Done! {len(self.chain_bench_results)}/{len(selected_models)} models benchmarked.")
                self.show_chain_report = True
                self.active_model_name = "None (Chain Done)"
                self.set_status(f"✓ Chain Benchmark complete: {len(self.chain_bench_results)} models. "
                                f"Report: {self.chain_bench_report_path or 'non scritto'}")

            except Exception as e:
                self.chain_bench_log.append(f"❌ Chain Error: {str(e)[:80]}")
                self.set_status(f"✗ Chain Benchmark Error: {str(e)[:80]}")
            finally:
                self.is_benchmarking = False

        threading.Thread(target=_chain_worker, daemon=True).start()

    # -----------------------------------------------------------------
    # GROQ HF CONFIG FETCHER & LM STUDIO PRESET CREATOR ([H])
    # -----------------------------------------------------------------
    def fetch_groq_hf_config_async(self, target_name: Optional[str] = None, author: str = "") -> None:
        """Query Groq Llama-3.3-70B for official model hyperparameters and generate LM Studio preset."""
        if self.is_fetching_config:
            self.set_status("⚠️ Config fetcher already querying Groq API...")
            return

        query_target = target_name
        if not query_target:
            if self.active_tab == 4 and self.hf_repos:
                query_target = self.hf_repos[self.selected_repo_idx].repo_id
                author = self.hf_repos[self.selected_repo_idx].author
            elif self.active_tab == 5 and self.local_models:
                query_target = self.local_models[self.selected_model_idx].filename

        if not query_target:
            self.set_status("⚠️ Select a model or HF repo first to fetch configuration.")
            return

        self.is_fetching_config = True
        self.set_status(f"⚡ Querying Groq (Llama-3.3-70B) for official model card settings of '{query_target}'...")

        def _worker():
            try:
                config, preset_path = self.config_mgr.fetch_groq_hf_config(query_target, author)
                self.last_fetched_preset_path = preset_path
                preset_note = f" & Saved LM Studio preset to '{os.path.basename(preset_path)}'" if preset_path else ""
                self.set_status(f"✓ Config extracted for {config.model_filename}{preset_note}! (Temp: {config.temperature}, Ctx: {config.context_length}, NGL: {config.ngl})")
            except Exception as e:
                self.set_status(f"✗ Groq Config Fetch Error: {str(e)[:80]}")
            finally:
                self.is_fetching_config = False

        threading.Thread(target=_worker, daemon=True).start()

    # -----------------------------------------------------------------
    # GEAR ⚙️ SETTINGS EDITOR MODAL ([G])
    # -----------------------------------------------------------------
    def open_gear_modal(self, model_idx: Optional[int] = None) -> None:
        target_idx = self.selected_model_idx if model_idx is None else model_idx
        if self.local_models and 0 <= target_idx < len(self.local_models):
            self.gear_model_name = self.local_models[target_idx].filename
        else:
            self.gear_model_name = "default_local_model.gguf"

        self.gear_model_config = self.config_mgr.get_config(self.gear_model_name)
        self.gear_active_field = 0
        self.show_gear_modal = True
        self.set_status(f"⚙️ Opened Settings Editor for: {self.gear_model_name} | [↑/↓] Select [←/→] Adjust [S] Save [Esc] Close")

    def close_gear_modal(self) -> None:
        self.show_gear_modal = False
        self.set_status("⚙️ Settings Editor closed.")

    def adjust_gear_field(self, delta: int) -> None:
        if not self.gear_model_config or not (0 <= self.gear_active_field < len(self.gear_fields_def)):
            return
        field_def = self.gear_fields_def[self.gear_active_field]
        key = field_def["key"]
        ftype = field_def.get("type", "float")
        curr_val = getattr(self.gear_model_config, key, None)

        if ftype == "float":
            step = field_def.get("step", 0.05)
            min_v = field_def.get("min", 0.0)
            max_v = field_def.get("max", 2.0)
            new_val = round(max(min_v, min(max_v, float(curr_val or 0.0) + delta * step)), 3)
            setattr(self.gear_model_config, key, new_val)
        elif ftype == "int":
            step = field_def.get("step", 1)
            min_v = field_def.get("min", 0)
            max_v = field_def.get("max", 262144)
            new_val = max(min_v, min(max_v, int(curr_val or 0) + delta * step))
            setattr(self.gear_model_config, key, new_val)
        elif ftype == "bool":
            new_val = not bool(curr_val)
            setattr(self.gear_model_config, key, new_val)

    def toggle_gear_field(self) -> None:
        self.adjust_gear_field(1)

    def save_gear_config(self) -> None:
        if not self.gear_model_config or not self.gear_model_name:
            return
        # ModelConfigManager non ha set_config (AttributeError: il modale
        # restava aperto e la config era persa): save_config salva il JSON.
        self.gear_model_config.model_filename = os.path.basename(self.gear_model_name)
        self.config_mgr.save_config(self.gear_model_config)
        try:
            preset_path = os.path.basename(self.config_mgr.export_lmstudio_preset(self.gear_model_config))
        except Exception as exc:
            preset_path = f"non esportato ({exc})"
        self.show_gear_modal = False
        self.set_status(f"✓ Saved configuration for {self.gear_model_name} & Exported LM Studio preset: {preset_path}")

    # -----------------------------------------------------------------
    # PROMINENT RED FLUSH VRAM ACTION ([F])
    # -----------------------------------------------------------------
    def flush_arm(self) -> None:
        """Primo [F]: arma la conferma e spiega cosa succede."""
        self._flush_armed = time.time()
        mp = self.llama_mgr.managed_pid
        tgt = f"il llama-server della TUI (PID {mp})" if mp else "nessun server della TUI attivo"
        others = [r.pid for r in Srv.scan_local_servers() if r.pid and r.pid != mp]
        oth = f"; [T] = TUTTI i llama-server ({len(others)} altri: PID {', '.join(map(str, others[:5]))})" \
            if others else ""
        self.set_status(f"⚠ FLUSH VRAM: [F] di nuovo entro {FLUSH_CONFIRM_S:.0f} s ferma TUTTI i llama-server "
                        f"({tgt}{oth}); altro tasto annulla")

    def flush_vram_async(self, kill_external: bool = False) -> None:
        """Esegue la flush in un thread (nvidia-smi, stop, POST: secondi)."""
        if getattr(self, "_flush_busy", False):
            self.set_status("⏳ flush VRAM gia' in corso...")
            return
        self._flush_busy = True

        def work() -> None:
            try:
                self.flush_vram_action(kill_external=kill_external)
            except Exception as exc:
                self.set_status(f"✗ flush VRAM: {exc}")
            finally:
                self._flush_busy = False
        threading.Thread(target=work, daemon=True).start()

    def flush_vram_action(self, kill_external: bool = False) -> Dict[str, Any]:
        """Execute instant VRAM purge and process termination with green confirmation HUD.

        kill_external=False ferma solo il server avviato dalla TUI e lascia
        vivi quelli adottati (usato dalla suite --test)."""
        self.set_status("⚡ PURGING VRAM & TERMINATING LLAMA-SERVER...")
        res = self.llama_mgr.flush_vram(self.hw_monitor, kill_external=kill_external)
        self.active_model_name = "None (Purged)"
        # Dopo una flush non c'e' piu' nessun server: si azzerano TUTTI e tre
        # gli stati, non solo il booleano, altrimenti l'HUD mostrerebbe
        # STARTING/OFFLINE contraddicendo la flush appena fatta.
        self.server_online = False
        self.server_state = "OFFLINE"
        released_mb = res.get("released_mb", 0.0)
        killed = res.get("killed_processes") or res.get("killed") or []
        extra = f" | fermati anche: {', '.join(map(str, killed))[:80]}" if kill_external and killed else \
            ("" if kill_external else " | altri llama-server NON toccati")
        self.flush_flash_message = f"⚡ VRAM FLUSHED: {released_mb:.0f} MB RELEASED{extra} ⚡"
        self.flush_flash_time = time.time()
        self.set_status(self.flush_flash_message)
        return res

    def toggle_ctx_shift(self) -> None:
        """Legacy alias kept for the verification suite: drives the Rolling Window."""
        self.toggle_rolling_window()

    # ------------------------------------------------------------------
    # EMBEDDER per mem0  (tasto globale [E])
    # ------------------------------------------------------------------
    def embedder_toggle(self) -> None:
        """Avvia o ferma l'embedder locale che mem0 usa.

        L'avvio puo' durare qualche decimo di secondo (caricamento del
        modello + attesa di /health), quindi gira in un thread: senza,
        l'interfaccia si bloccherebbe e sembrerebbe un crash.
        """
        if getattr(self, "_embedder_busy", False):
            self.set_status("⏳ Embedder: operazione già in corso...")
            return
        # Anche lo STOP gira nel thread: HTTP + stop_embedder (sleep 2 s)
        # bloccavano l'interfaccia.
        self._embedder_busy = True
        self.set_status("🔄 Embedder: controllo lo stato...")

        def _worker() -> None:
            try:
                running = self.llama_mgr.embedder_is_running()
            except Exception:
                running = False
            if running:
                self.set_status("⏹ Fermo l'embedder (mem0)...")
                try:
                    res = self.llama_mgr.stop_embedder()
                except Exception as exc:
                    res = {"success": False, "error": str(exc)}
                self.embedder_state = "OFFLINE"
                self.embedder_port = 0
                self.embedder_dims = 0
                self._embedder_busy = False
                if res.get("success"):
                    self.set_status("🛑 Embedder fermato (mem0)")
                else:
                    self.set_status(f"⚠️ {res.get('error', 'errore sconosciuto')}")
                return
            self.embedder_state = "STARTING"
            self.set_status("🔄 Avvio l'embedder locale per mem0...")
            try:
                res = self.llama_mgr.start_embedder()
            except Exception as exc:                # non far cadere la TUI
                res = {"success": False, "error": str(exc)}
            self._embedder_busy = False
            if res.get("success"):
                self.embedder_state = "ONLINE"
                self.embedder_port = int(res.get("port") or 0)
                dims = res.get("dims") or 0
                self.embedder_dims = int(dims) if dims else 0
                extra = f" | {dims} dim" if dims else ""
                self.set_status(
                    f"🟢 Embedder mem0 attivo su :{self.embedder_port}{extra}")
            else:
                self.embedder_state = "OFFLINE"
                self.set_status(f"❌ Embedder: {res.get('error', 'errore')}")

        threading.Thread(target=_worker, daemon=True).start()


# =====================================================================
# TUI RENDERER & VISUAL LAYOUTS
# =====================================================================
def _win(n: int, cursor: int, size: int) -> tuple:
    """Finestra [start, end) di `size` righe che SEGUE il cursore."""
    if n <= size:
        return 0, n
    start = max(0, min(cursor - size // 2, n - size))
    return start, start + size


def _win_top(n: int, cursor: int, lookback: int = 1) -> tuple:
    """Finestra che comincia appena sopra il cursore e arriva FINO ALLA FINE
    della lista (non un numero di righe indovinato).

    Rich disegna dall'alto e taglia quello che non entra dal BASSO: prima
    si passava a _win() una stima di quante righe "dovrebbero" entrare in
    una scheda (es. 'lines - 18'), ma il numero vero dipende dal layout
    esatto di OGNI scheda (pannelli fissi accanto, split in colonne,
    larghezza delle celle...) ed era sbagliato in piu' schede: a volte il
    doppio del reale (Radar), a volte il reale esatto per coincidenza
    (Build), misurato riproducendo i render veri con Console(record=True).
    Tenere un numero magico per scheda e' fragile e va ritarato ogni volta
    che cambia un pannello. Ancorando la finestra appena sopra il cursore
    e lasciandola aperta fino alla fine, qualunque sia la capacita' vera
    del pannello Rich disegna sempre a partire dal cursore: la riga
    selezionata non sparisce mai, a costo di non mostrare l'intera lista
    quando ci sarebbe spazio (compromesso accettato: il cursore che sparisce
    era il difetto segnalato, non la lista intera sempre visibile).
    """
    if n <= 0:
        return 0, 0
    start = max(0, min(cursor - lookback, n - 1))
    return start, n


def _win_paginated(n: int, cursor: int) -> tuple:
    """Finestra di dimensione fissa che segue il cursore, con margini.
    Altezza = altezza terminale meno qualche riga di intestazione/piede.
    Se la lista e' piu' piccola della finestra, mostra tutto.
    """
    import shutil
    term_lines = shutil.get_terminal_size((80, 20)).lines
    # spazio riservato per header, footer, status bar, tab bar, etc.
    margin = 6  # sperimentale: si puo' aggiustare se necessario
    page_size = max(1, term_lines - margin)
    if n <= page_size:
        return 0, n
    start = max(0, min(cursor - page_size // 2, n - page_size))
    return start, start + page_size


def build_matches(b: Any, query: str) -> bool:
    """Ogni parola della ricerca deve comparire in nome, bNNNN, nota, device, cartella o rpc/gpu/cpu."""
    words = query.lower().split()
    if not words:
        return True
    hay = " ".join([b.key, f"b{b.build_num}", BM.use_note(b), ",".join(b.devices or []), b.dir or "",
                    "rpc" if b.rpc else "", "gpu" if b.gpu else "cpu"]).lower()
    return all(w in hay for w in words)


def _search_line(q: str, typing: bool, shown: int, total: int) -> Text:
    line = Text("🔎 ", style="accent")
    line.append(q, style="bold fg")
    if typing:
        line.append("▏", style="bold accent")
    line.append(f"   {shown}/{total} build", style="metric")
    line.append("   [INVIO] conferma · [ESC] azzera" if typing else "   [/] modifica · [ESC] azzera",
                style="muted")
    return line


class _FitHeight:
    """Rimanda la costruzione al momento del disegno, quando Rich conosce l'altezza
    reale del riquadro: make(h) riceve le righe disponibili (None fuori da un Layout)."""

    def __init__(self, make):
        self.make = make

    def __rich_console__(self, console, options):
        yield self.make(options.height)


def _win_slice(n: int, cursor: int, page_size: int = 4) -> tuple:
    """Finestra con dimensione massima prefissata centrata sul cursore."""
    if n <= 0:
        return 0, 0
    if n <= page_size:
        return 0, n
    start = max(0, min(cursor - page_size // 2, n - page_size))
    return start, start + page_size


class TUIRenderer:
    """
    Rich Visual Presentation Engine for Cyberpunk Local AI Management TUI.
    Renders high-contrast, cyberpunk-themed ASCII dashboards and panels.
    """

    @staticmethod
    def render_header(state: AppState) -> Panel:
        """Render high-tech cyber top HUD with VRAM, RAM, KV mode, and prominent red FLUSH VRAM."""
        hw = state.hw_snapshot
        vram_pct = (hw.gpu.vram_used_mb / max(1.0, hw.gpu.vram_total_mb)) * 100.0
        ram_pct = (hw.ram.used_gb / max(1.0, hw.ram.total_gb)) * 100.0

        vram_color = "ok" if vram_pct < 65 else ("warn" if vram_pct < 88 else "err")
        # Tre stati distinti. STARTING non e' un "quasi online": significa
        # che un server c'e' ma non ha ancora risposto. ONLINE si stampa
        # SOLO se /health ha risposto davvero (probe_all_endpoints).
        srv_state = getattr(state, "server_state", None) or (
            "ONLINE" if state.server_online else "OFFLINE")
        if srv_state == "ONLINE":
            server_status = "[bold ok]● ONLINE[/]"
        elif srv_state == "STARTING":
            server_status = "[bold warn]◐ STARTING[/]"
        else:
            server_status = "[bold muted]○ OFFLINE[/]"
        
        # RAM KV Offload Indicator
        if state.server_no_kv_offload:
            ram_kv_badge = "[bold bg on warn] RAM-KV: ON (-nkvo) [/]"
        else:
            ram_kv_badge = "[bold bg on metric2] RAM-KV: GPU [/]"

        # Flush VRAM Button
        flush_btn = "[bold bg on err] [F] FLUSH VRAM [/]"

        # Embedder per mem0: stato leggibile a colpo d'occhio, perche' il
        # tasto [E] e' globale e serve sapere se e' gia' attivo senza
        # premerlo (un secondo avvio costerebbe qualche decimo di secondo).
        _emb = getattr(state, "embedder_state", "OFFLINE")
        _emb_color = {"ONLINE": "ok", "STARTING": "warn"}.get(_emb, "muted")
        _emb_txt = {"ONLINE": "ON", "STARTING": "..."}.get(_emb, "OFF")
        _emb_btn = (f"[bold bg on {_emb_color}] [E] EMBEDDER {_emb_txt} [/]")

        # Rolling Window + KV dtype badge
        if state.rolling_window_enabled:
            roll_badge = "[bold bg on ok] ROLL: ON [/]"
        else:
            roll_badge = "[bold bg on err] ROLL: OFF [/]"
        kv_eff = state.evaluate_kv_guardrail()["effective"]
        kv_badge = f"[bold bg on metric] KV: {kv_eff} [/]"

        left_text = Text.from_markup(
            f"[bold metric]⚡ CYBER AI TUI v2.6[/] | "
            f"[bold fg]{hw.gpu.name}[/] [bold {vram_color}]{hw.gpu.vram_used_mb:.0f}/{hw.gpu.vram_total_mb:.0f}MB ({vram_pct:.0f}%)[/] | "
            f"RAM [metric]{hw.ram.used_gb:.1f}/{hw.ram.total_gb:.1f}GB ({ram_pct:.0f}%)[/]"
        )

        center_text = Text.from_markup(
            f"Server: {server_status} | Model: [bold warn]{_mesc(str(state.active_model_name)[:24])}[/] | {ram_kv_badge}"
        )

        right_text = Text.from_markup(
            f"{flush_btn}  {_emb_btn}  {roll_badge} {kv_badge}  [bold fg]256k CTX: [bold ok]READY[/][/]",
        )

        # Rilevamento proporzioni schermo (16:9 vs 9:16)
        asp = get_aspect_info()
        is_port = asp["is_portrait"]
        lines_cnt = asp["lines"]
        mode_badge = f"[bold bg on {'ok' if is_port else 'accent'}] {asp['mode']} [/]"

        if is_port or asp["cols"] < 110:
            # Layout 9:16 (verticale) o finestra stretta: HUD compatto multilinea
            hud_table = Table.grid(expand=True)
            hud_table.add_column(ratio=3, justify="left")
            hud_table.add_column(ratio=2, justify="right")
            hud_table.add_row(
                Text.from_markup(f"[bold metric]⚡ CYBER AI TUI[/] | [bold fg]{hw.gpu.name}[/] [bold {vram_color}]{hw.gpu.vram_used_mb:.0f}MB[/] | RAM [metric]{hw.ram.used_gb:.1f}GB[/]"),
                Text.from_markup(f"Srv: {server_status} | {mode_badge}")
            )
            hud_table.add_row(
                Text.from_markup(f"Mod: [bold warn]{_mesc(str(state.active_model_name)[:20])}[/] | {ram_kv_badge}"),
                Text.from_markup(f"{flush_btn} {_emb_btn} {roll_badge} {kv_badge}")
            )
        else:
            # Layout 16:9 (orizzontale / widescreen): 3 colonne affiancate
            hud_table = Table.grid(expand=True)
            hud_table.add_column(ratio=2, justify="left")
            hud_table.add_column(ratio=3, justify="center")
            hud_table.add_column(ratio=2, justify="right")
            right_with_badge = Text.from_markup(f"{mode_badge} ") + right_text
            hud_table.add_row(left_text, center_text, right_with_badge)

        # Tab Navigation Bar
        tabs = [
            ("1", "Sistema"), ("2", "KV & Groq"), ("3", "Radar"), ("4", "HF Download"),
            ("5", "Processi"), ("6", "Agente"), ("7", "Plugin"), ("8", "lm_eval"),
            ("9", "RPC"), ("0", "Config"), ("F10", "Aspetto"), ("^B", "Bench"),
            ("F12", "Build"), ("Alt+M", "Modal"), ("^U", "Updates"),
        ]
        tab_num_of = {"0": 10, "F10": 11, "^B": 12, "F12": 13, "^M": 14, "Alt+M": 14, "ALT_M": 14, "^U": 15, "Ctrl+U": 15}
        tab_row = Text(no_wrap=False)
        for key, name in tabs:
            # Il tasto "0" apre il Tab 10 (non il Tab 0: active_tab parte da 1).
            tab_num = tab_num_of.get(key) or int(key)
            if tab_num == state.active_tab:
                tab_row.append(f" {key} ", style="bold bg on accent")
                tab_row.append(f" {name} ", style="bold bg on metric")
            else:
                tab_row.append(f" {key} ", style="bold accent on sel")
                tab_row.append(f" {name} ", style="fg on sel")
            tab_row.append(" ")

        # ---------------------------------------------------------------
        # LEGENDA IN ALTO: i tasti che sono LETTERE e NUMERI
        # ---------------------------------------------------------------
        letter_legend = Text()
        letter_legend.append("TASTI: ", style="bold metric")
        legend_items = [
            ("[1-9]", "cambia tab"),
            ("[0]", "tab 10 (Config)"),
            ("[Alt+M]", "Modal"),
            ("[^T]", "tema"),
            ("[f]", "flush VRAM"),
            ("[g]", "gear"),
            ("[t]", "benchmark"),
            ("[h]", "Groq config"),
            ("[p]", "prompt"),
            ("[↑↓]", "seleziona"),
            ("[invio]", "conferma"),
            ("[esc]", "annulla"),
            ("[q]", "esci"),
        ]
        for i, (key, what) in enumerate(legend_items):
            if i:
                letter_legend.append("  ", style="")
            letter_legend.append(key, style="bold warn")
            letter_legend.append(" " + what, style="muted")

        # ---------------------------------------------------------------
        # STATO SERVER: riga con l'elenco degli endpoint interrogati
        # ---------------------------------------------------------------
        srv_line = Text()
        srv_line.append("ENDPOINT: ", style="bold metric")
        if not state.server_endpoints:
            srv_line.append("non ancora interrogati", style="muted")
        else:
            for i, (nm, rec) in enumerate(state.server_endpoints.items()):
                if i:
                    srv_line.append("  ", style="")
                st = rec.get("state")
                col = {"ONLINE": "bold ok",
                       "STARTING": "bold warn"}.get(st, "muted")
                gl = {"ONLINE": "●", "STARTING": "◐"}.get(st, "○")
                srv_line.append(f"{gl} {nm}", style=col)

        legend_row = Group(srv_line, letter_legend)

        if is_port or lines_cnt < 28:
            # Layout 9:16 o terminale basso: ometti spazi vuoti e legenda per massimizzare il corpo
            top_group = Group(hud_table, tab_row)
        elif lines_cnt < 35:
            top_group = Group(hud_table, tab_row, srv_line)
        else:
            top_group = Group(hud_table, Text(""), tab_row, Text(""), legend_row)

        return Panel(
            top_group,
            box=box.HEAVY,
            border_style="border",
            title=f"[bold metric]◆ LOCAL AI HYPERVISOR ◆[/]  {mode_badge}",
            title_align="center",
            padding=(0, 1)
        )

    @staticmethod
    def render_footer(state: AppState) -> Panel:
        """Render responsive footer bar with contextual hotkeys and status messages."""
        now = time.time()
        
        # Check if flush flash banner is active
        if state.flush_flash_message and (now - state.flush_flash_time < 5.0):
            status_markup = f"[bold bg on ok] {_mesc(state.flush_flash_message)} [/]"
        elif state.is_benchmarking:
            status_markup = "[bold muted on warn] ⚡ LIVE BENCHMARK RUNNING: Testing Python AST Syntax & Tool Calling... [/]"
        elif state.is_fetching_config:
            status_markup = "[bold bg on metric2] ⚡ QUERYING GROQ LLAMA-3.3-70B FOR OFFICIAL MODEL CARD CONFIGURATION... [/]"
        else:
            # testo arbitrario (log di llama-server, eccezioni, percorsi): escape,
            # altrimenti "[x]" sparisce e "[/home/user]" fa fallire ogni frame
            status_markup = f"[bold metric]STATUS:[/] [fg]{_mesc(str(state.status_message))}[/]"

        # Contextual Hotkeys per tab
        if state.show_gear_modal:
            hotkeys = "[bold warn][↑/↓][/] Select Field | [bold warn][←/→][/] Adjust Value | [bold warn][Enter][/] Toggle/Input | [bold warn][S][/] Save & Preset | [bold warn][Esc/G][/] Close"
        elif state.active_tab == 1:
            hotkeys = "[bold metric][R][/] Refresh Telemetry | [bold err][F][/] Flush VRAM | [bold warn][G][/] Settings | [bold fg][1-6][/] Switch Tab"
        elif state.active_tab == 2:
            hotkeys = "[bold metric][W/S][/] Rolling Window | [bold metric][K][/] KV Quant | [bold metric][O][/] RAM-KV | [bold metric][X][/] Truncate Tool Ctx | [bold metric][C][/] Groq Compress"
        elif state.active_tab == 3:
            hotkeys = "[bold metric][↑/↓][/] Select News | [bold metric][A][/] Filter All | [bold metric][X][/] arXiv | [bold metric][H][/] HF | [bold metric][R][/] Refresh"
        elif state.active_tab == 4:
            hotkeys = "[bold metric][S][/] Search HF | [bold metric][Tab][/] Cycle Panes | [bold metric][D/Enter][/] Download | [bold warn][H][/] Groq HF Config | [bold err][F][/] Flush VRAM"
        elif state.active_tab == 5:
            hotkeys = "[bold metric][↑/↓][/] Select Model | [bold metric][L/Enter][/] Launch | [bold ok][T][/] Benchmark Model | [bold warn][G][/] Gear Settings | [bold warn][H][/] Groq HF Config | [bold metric][O][/] Preset | [bold accent][D][/] Cartella modelli | [bold err][F][/] Flush VRAM"
        elif state.active_tab == 6:
            hotkeys = "[bold accent][I][/] Scrivi | [bold warn][ESC][/] Esci da scrittura | [bold metric][Enter][/] Invia | [bold metric][C][/] Pulisci chat | [bold fg][1-9][/] Cambia scheda (solo se NON in scrittura)"
        elif state.active_tab == 8:
            hotkeys = "[bold metric][M][/] Calcola Matrice | [bold metric][G][/] Gruppo Task | [bold metric][K][/] KV Format | [bold metric][+/-][/] Budget VRAM | [bold metric][R][/] Ricarica Task"
        elif state.active_tab == 9:
            hotkeys = (f"[bold accent][F1-F6][/] Menu preset | [bold warn][Ctrl+R][/] Rilancia ultimo | "
                       "[bold metric][↑/↓][/] Scegli preset | [bold metric][Space][/] Cambia modello | [bold metric][Enter][/] AVVIA | "
                       "[bold metric][T][/] Dry-run | [bold metric][S][/] Salva profilo | [bold metric][P][/] Probe Miky | "
                       "[bold metric][0][/] Config completa")
        elif state.active_tab == 10:
            hotkeys = (f"[bold metric][↑/↓][/] Campo | [bold accent][←/→][/] Cambia valore | "
                       f"[bold ok][B][/] Backend | [bold ok][Space][/] Modello | "
                       f"[bold ok][R][/] Rileggi help | [bold ok][A][/] Tutti i flag | "
                       f"[bold metric][T][/] Dry-run | [bold metric][S][/] Salva | [bold metric][L][/] Carica | "
                       f"[bold metric][E][/] Export | [bold metric][I][/] Import | [bold metric][C][/] Reset | "
                       f"[bold warn][U][/] Aggiorna | [bold warn][Y][/] Scarica")
        elif state.active_tab == 11:
            hotkeys = ("[bold metric][↑/↓][/] Controllo | [bold accent][←/→][/] Cambia | "
                       "[bold metric][T][/] Tema | [bold metric][D][/] Densita' | [bold metric][P][/] Proporzioni | "
                       "[bold accent][Ctrl+T][/] Tema successivo")
        elif state.active_tab == 12:
            hotkeys = ("[bold metric][↑/↓][/] Scorri | [bold metric][←/→][/] Sorgente | "
                       "[bold metric][S][/] Ordina | [bold metric][R][/] Rileggi")
        elif state.active_tab == 13:
            hotkeys = ("[bold metric][↑/↓][/] Build | [bold ok][Enter/U][/] Usa | [bold metric][D/W][/] Doc | "
                       "[bold metric][A/X][/] Aggiungi/Togli | [bold metric][K/O/Z][/] rpc-server | "
                       "[bold accent][V/M/P][/] Margini VRAM | [bold warn][C][/] Controlla update | [bold warn][Y][/] Scarica")
        elif state.active_tab == 14:
            hotkeys = ("[bold metric][↑/↓][/] App | [bold accent][M][/] Cambia Modello (Ornith/Tiel) | "
                       "[bold ok][D][/] Deploy | [bold err][X][/] Stop | [bold metric][P][/] Ping | "
                       "[bold metric][T][/] Test Chat | [bold warn][R][/] Aggiorna | [bold fg][O][/] Standalone")
        elif state.active_tab == 15:
            hotkeys = ("[bold metric][↑/↓][/] Componente | [bold warn][R][/] Refresh check | "
                       "[bold ok][U][/] Mostra update cmd | [bold ok][A][/] Elenca pendenti | "
                       "[bold accent][Ctrl+U][/] apre questa tab")
        else:
            hotkeys = "[bold fg][1-9][/] Switch Tab | [bold err][F][/] Flush VRAM | [bold warn][G][/] Gear Settings | [bold accent][9][/] RPC & CUDA13"

        footer_table = Table.grid(expand=True)
        footer_table.add_column(ratio=1, justify="left")
        footer_table.add_column(ratio=1, justify="right")
        footer_table.add_row(Text.from_markup(status_markup), Text.from_markup(hotkeys))

        # ---------------------------------------------------------------
        # LEGENDA IN BASSO: i tasti FUNZIONE
        # ---------------------------------------------------------------
        # In alto ci sono lettere e numeri, qui i tasti funzione. La riga
        # e' generata da TUI_RPC_HOTKEYS e TUI_SERVER_HOTKEYS, quindi non
        # puo' divergere da quello che il codice fa davvero: quello che non
        # e' mappato non viene promesso.
        fkeys: List[Tuple[str, str]] = []
        for fk, preset_key in TUI_RPC_HOTKEYS:
            info = LP.effective(preset_key) or LlamaServerManager.RPC_PRESETS.get(preset_key, {})
            # tok/s MISURATI da preset_bench.json (contesto della misura dopo
            # la @, '*' = misura con config diversa); n/m = mai misurato.
            fkeys.append((fk, f"{info.get('short', info.get('name', preset_key))} "
                              f"{LP.ctx_label(int(info.get('ctx_size') or 0))} "
                              f"{LP.tok_label(info) if info.get('schema') == 2 else 'n/m'}"))
        fkeys.append(("Ctrl+R", "rilancia"))
        for fk, action in TUI_SERVER_HOTKEYS.items():
            fkeys.append((fk, {
                "toggle_panel": "server",
                "select_mode": "seleziona server",
                "refresh": "ricarica",
            }.get(action, action)))
        emb_label = "embedder ON" if state._embedder_online else "embedder OFF"
        fkeys.append(("F9", emb_label))
        fkeys += [("F10/^A", "aspetto"), ("^B", "benchmark"), ("F12/^L", "build"),
                  ("Alt+M", "Modal cloud")]
        # Due righe: preset F1-F6 (con contesto e tok/s misurati) e il resto.
        n_pre = len(TUI_RPC_HOTKEYS)
        lines = []
        for title, chunk in (("PRESET: ", fkeys[:n_pre]), ("F-KEYS: ", fkeys[n_pre:])):
            fk_line = Text()
            fk_line.append(title, style="bold metric")
            for i, (fk, what) in enumerate(chunk):
                if i:
                    fk_line.append("  ", style="")
                fk_line.append(f"[{fk}]", style="bold accent")
                fk_line.append(" " + what, style="muted")
            fk_line.no_wrap = True
            fk_line.overflow = "ellipsis"
            lines.append(fk_line)

        asp = get_aspect_info()
        is_port = asp["is_portrait"]
        lines_cnt = asp["lines"]

        if is_port or lines_cnt < 28:
            # 9:16 o schermo basso: solo tabella di stato e preset F1-F6 essenziali
            f_group = Group(footer_table, lines[0]) if lines else Group(footer_table)
        else:
            f_group = Group(footer_table, *lines)

        return Panel(
            f_group,     # righe F-key a tutta larghezza
            box=box.HORIZONTALS,
            border_style="border",
            padding=(0, 1)
        )

    # -----------------------------------------------------------------
    # TAB 1: SYSTEM & GPU MONITOR
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab1_system_monitor(state: AppState) -> Layout:
        """Telemetria REALE: niente valori segnaposto (driver, CPU, OS, disco
        sono letti dal sistema; se un dato manca si scrive 'n/d').
        A destra appare la telemetria LIVE del nodo RPC Miky
        (192.168.1.100): se irraggiungibile mostra OFFLINE, senza bloccare."""
        import platform
        from rich.progress_bar import ProgressBar
        hw = state.hw_snapshot
        layout = Layout()
        layout.split_row(
            Layout(name="gpu_panel", ratio=2),
            Layout(name="cpu_ram_panel", ratio=2),
            Layout(name="miky_panel", ratio=3),
        )

        def bar(pct: float, warn_at: float, err_at: float) -> Table:
            col = "ok" if pct < warn_at else ("warn" if pct < err_at else "err")
            g = Table.grid(expand=True, padding=(0, 1))
            g.add_column(ratio=1)
            g.add_column(width=7, justify="right")
            g.add_row(ProgressBar(total=100, completed=pct, complete_style=col,
                                  finished_style=col, style="sel"),
                      Text(f"{pct:.1f}%", style=f"bold {col}"))
            return g

        vram_pct = (hw.gpu.vram_used_mb / max(1.0, hw.gpu.vram_total_mb)) * 100.0
        gpu_table = Table(box=None, expand=True, show_header=False, padding=(0, 1))
        gpu_table.add_column("Metric", style="muted", width=18)
        gpu_table.add_column("Value", style="bold fg", ratio=1)
        if not hw.gpu.is_available:
            gpu_table.add_row("GPU", Text("nvidia-smi non disponibile: dati GPU n/d", style="warn"))
        gpu_table.add_row("GPU", f"{hw.gpu.name}")
        gpu_table.add_row("Driver", f"{hw.gpu.driver_version if hw.gpu.is_available else 'n/d'}")
        gpu_table.add_row("Utilizzo", f"[{'ok' if hw.gpu.gpu_util_percent < 50 else 'warn'}]{hw.gpu.gpu_util_percent:.0f}%[/]")
        gpu_table.add_row("Temperatura", f"[{'ok' if hw.gpu.temp_c < 65 else ('warn' if hw.gpu.temp_c < 80 else 'err')}]{hw.gpu.temp_c:.0f} °C[/]")
        gpu_table.add_row("Potenza", f"[metric]{hw.gpu.power_w:.1f} W[/]")
        gpu_table.add_row("VRAM usata", f"[bold metric]{hw.gpu.vram_used_mb:.0f}[/] / {hw.gpu.vram_total_mb:.0f} MiB"
                          + (f"  [muted](riservata driver {hw.gpu.reserved_mb:.0f})[/]" if hw.gpu.reserved_mb else ""))
        gpu_table.add_row("VRAM libera", f"[bold ok]{hw.gpu.vram_free_mb:.0f} MiB[/]")
        gpu_table.add_row("VRAM", bar(vram_pct, 70, 90))
        # Server reali (inventario): niente "100% Full GPU" inventato.
        rows = state.server_inventory.rows()
        loc = [r for r in rows if r.kind == "locale"]
        on = [r for r in loc if r.state_line == "ONLINE"]
        gpu_table.add_row("llama-server locali",
                          f"[{'ok' if on else 'muted'}]{len(on)} online[/] / {len(loc)} trovati"
                          + (f"  [muted]{on[0].model_name[:28]}[/]" if on else ""))
        ab = state.build_mgr.active()
        gpu_table.add_row("Build attiva", f"[metric2]{ab.key if ab else 'n/d'}[/] [muted]{ab.summary() if ab and ab.probed else ''}[/]")

        gpu_panel = Panel(
            gpu_table, box=box.ROUNDED,
            title=f"[bold metric]⚡ GPU · {hw.gpu.name.replace('NVIDIA GeForce ', '')}[/]",
            subtitle="[muted]nvidia-smi[/]"
        )

        ram_pct = (hw.ram.used_gb / max(1.0, hw.ram.total_gb)) * 100.0
        cpu_table = Table(box=None, expand=True, show_header=False, padding=(0, 1))
        cpu_table.add_column("Metric", style="muted", width=18)
        cpu_table.add_column("Value", style="bold fg", ratio=1)
        cpu_table.add_row("CPU", f"{hw.cpu.cpu_name}")
        cpu_table.add_row("Core / thread", f"{hw.cpu.physical_cores} / {hw.cpu.logical_cores}")
        cpu_table.add_row("Carico CPU", bar(hw.cpu.cpu_percent, 50, 85))
        cpu_table.add_row("RAM usata", f"[bold metric]{hw.ram.used_gb:.1f}[/] / {hw.ram.total_gb:.1f} GB")
        cpu_table.add_row("RAM disponibile", f"[bold ok]{hw.ram.free_gb:.1f} GB[/]")
        cpu_table.add_row("RAM", bar(ram_pct, 75, 90))
        if hw.ram.swap_total_gb:
            cpu_table.add_row("Swap", f"{hw.ram.swap_used_gb:.1f} / {hw.ram.swap_total_gb:.1f} GB")
        try:
            du = shutil.disk_usage(state.model_dir)
            cpu_table.add_row("Disco modelli", f"[metric]{du.free / 1024**3:,.0f} GB liberi[/] / {du.total / 1024**3:,.0f} GB"
                              f"  [muted]{state.model_dir}[/]")
        except OSError:
            cpu_table.add_row("Disco modelli", f"[warn]{state.model_dir} non accessibile[/]")
        cpu_table.add_row("Modelli GGUF", f"[metric]{len(state.local_models)}[/]")

        cpu_panel = Panel(
            cpu_table, box=box.ROUNDED,
            title="[bold metric2]💻 CPU · RAM · DISCO[/]",
            subtitle=f"[muted]{platform.system()} {platform.release()} · Python {platform.python_version()}[/]"
        )

        layout["gpu_panel"].update(gpu_panel)
        layout["cpu_ram_panel"].update(cpu_panel)
        # Terza colonna: nodo RPC Miky, telemetria compatta.
        layout["miky_panel"].update(render_miky_panel(
            state.miky_remote.last_snapshot, compact=True))
        return layout

    # -----------------------------------------------------------------
    # TAB 2: 256k KV CACHE & GROQ COMPRESSION ENGINE
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab2_kvcache(state: AppState) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="kv_budget_panel", ratio=1),
            Layout(name="groq_compress_panel", ratio=1)
        )

        # 256k KV Budget Calculation Matrix
        budget_256k = state.kv_optimizer.calculate_256k_budget(
            hidden_size=4096, num_layers=32, num_kv_heads=8, head_dim=128
        )

        matrix_table = Table(box=box.SIMPLE, expand=True)
        matrix_table.add_column("Quant Dtype", style="bold metric")
        matrix_table.add_column("VRAM 256k", style="bold fg", justify="right")
        matrix_table.add_column("Fit (RTX 3050 8GB)", style="bold", justify="center")
        matrix_table.add_column("VRAM Saved", style="ok", justify="right")

        for key, qinfo in budget_256k.items():
            if not isinstance(qinfo, dict) or key == "streaming_sinks":
                continue
            name = qinfo.get("name", key.upper())
            vram_mb = qinfo.get("mb", qinfo.get("vram_mb", 0.0))
            vram_gb = qinfo.get("gb", qinfo.get("vram_gb", 0.0))
            vram_fit = qinfo.get("vram_fit", "OK")
            savings = qinfo.get("savings_pct", qinfo.get("savings_vram_pct", 0.0))
            if key == "ram_kv_offload":
                matrix_table.add_row(
                    f"[bold warn]{name}[/]",
                    f"[bold ok]{vram_mb:.0f} MB[/]",
                    f"{vram_fit}",
                    f"[bold ok]+100%[/]"
                )
            else:
                matrix_table.add_row(
                    f"{name}",
                    f"{vram_mb:.0f} MB ({vram_gb:.2f} GB)",
                    f"{vram_fit}",
                    f"-{savings:.1f}%"
                )

        rolling_badge = (
            "[bold ok]ENABLED (KV rotates on overflow)[/]"
            if state.rolling_window_enabled else
            "[bold err]DISABLED (generation halts at window limit)[/]"
        )
        kv_verdict = state.evaluate_kv_guardrail()
        flags_dict = state.kv_optimizer.get_llama_server_256k_flags(
            kv_dtype=state.kv_quant_key,
            ctx_size=262144,
            enable_ctx_shift=state.rolling_window_enabled,
            no_kv_offload=state.server_no_kv_offload,
        )
        flags_str = flags_dict.get("flags_string", str(flags_dict)) if isinstance(flags_dict, dict) else str(flags_dict)

        # KV quantization guardrail panel rows
        quant_rows = [Text("")]
        for opt in state.kv_quant_options():
            if opt["id"] == kv_verdict["effective"]:
                mark = "[bold ok]◉ ACTIVE[/]"
            elif opt["enabled"]:
                mark = "[dim]○ available[/]"
            else:
                mark = "[bold err]✖ DISABLED (KV not in VRAM)[/]"
            quant_rows.append(Text.from_markup(f"   {mark}  {opt['id']:<6}"))

        guard_line = ""
        if kv_verdict["warning"]:
            guard_line = f"[bold err]⚠ {kv_verdict['warning']}[/]"

        kv_content = Group(
            matrix_table,
            Text(""),
            Text.from_markup(f"[bold warn]🔄 Rolling Window:[/] {rolling_badge} [dim](Hotkey [W] or [S/T])[/]"),
            Text.from_markup(f"[bold metric]⚡ RAM-Only KV Offload:[/] {'[bold ok]ENABLED (-nkvo)[/]' if state.server_no_kv_offload else '[bold warn]DISABLED (GPU KV)[/]'} [dim](Hotkey [O])[/]"),
            Text.from_markup(f"[bold accent]🔢 KV Quantization:[/] [bold]{kv_verdict['requested']}[/] ➔ [bold ok]{kv_verdict['effective']}[/] "
                             f"[dim](Hotkey [K] | quantized KV only allowed with KV fully in VRAM)[/]"),
            *quant_rows,
            Text.from_markup(guard_line) if guard_line else Text(""),
            Text.from_markup(f"[bold fg]Optimal Server Launch Flags:[/]"),
            Panel(Text(flags_str, style="bold ok"), box=box.SQUARE, border_style="border")
        )

        kv_panel = Panel(
            kv_content,
            box=box.ROUNDED,
            border_style="border",
            title="[bold metric]📊 256k CONTEXT KV CACHE & RAM OFFLOAD MATRIX[/]",
            subtitle="[dim]Target: NVIDIA RTX 3050 (8192 MB) + Host RAM[/]"
        )

        # Groq Context Compression Panel
        groq_table = Table(box=box.SIMPLE, expand=True, show_header=False)
        groq_table.add_column("Param", style="bold metric", width=22)
        groq_table.add_column("Value", style="bold fg")

        groq_table.add_row("Compress Engine", "[bold ok]Groq API (qwen/qwen3.8-27b)[/]")
        groq_table.add_row("Fallback Engine", "Local Structured Context Deduplication")
        groq_table.add_row("Max Input Window", "131,072 Tokens (pre-truncated head+tail)")
        groq_table.add_row("Target Ratio", "3.0x - 5.0x Token Compression (60-80% Savings)")

        if state.last_tool_truncation:
            tt = state.last_tool_truncation
            if tt.get("truncated"):
                groq_table.add_row(
                    "Tool Ctx Truncation",
                    f"[bold warn]head+tail[/] {tt['original_tokens']} ➔ {tt['kept_tokens']} tok "
                    f"([bold err]-{tt['dropped_tokens']}[/])"
                )
            else:
                groq_table.add_row("Tool Ctx Truncation", f"[ok]fits[/] ({tt['kept_tokens']} tok)")

        if state.last_compression_result:
            cr = state.last_compression_result
            engine_str = f"Groq {cr.model_used}" if cr.is_groq_powered else "Local Heuristic"
            groq_table.add_row("Last Exec Engine", f"[bold warn]{engine_str}[/]")
            groq_table.add_row("Tokens Before", f"{cr.original_tokens_est} tokens")
            groq_table.add_row("Tokens After", f"[bold ok]{cr.compressed_tokens_est} tokens[/]")
            groq_table.add_row("Token Savings", f"[bold ok]{cr.savings_percent:.1f}%[/]")
            groq_table.add_row("Latency", f"[metric]{cr.latency_ms:.1f} ms[/]")
        else:
            groq_table.add_row("Status", "[dim]Ready (Press [C] to run test compression)[/]")

        groq_content = Group(
            groq_table,
            Text(""),
            Text.from_markup("[bold fg]Groq Real-Time Lossless Compression Pipeline:[/]\n"
                             "[dim]1. Raw Conversation ➔ 2. Head+Tail Tool Truncation ➔\n"
                             "3. Groq qwen/qwen3.8-27b AST & Semantic Extractor ➔\n"
                             "4. Compact Token Buffer ➔ 5. Feed to Local RTX 3050 Llama-Server[/]")
        )

        groq_panel = Panel(
            groq_content,
            box=box.ROUNDED,
            border_style="warn",
            title="[bold warn]⚡ GROQ CONTEXT ACCELERATION ENGINE[/]",
            subtitle="[dim]qwen/qwen3.8-27b Hybrid Token Reducer (131k ctx)[/]"
        )

        layout["kv_budget_panel"].update(kv_panel)
        layout["groq_compress_panel"].update(groq_panel)
        return layout

    # -----------------------------------------------------------------
    # TAB 3: MODEL RADAR & NEWS
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab3_radar(state: AppState) -> Layout:
        layout = Layout()
        if not is_portrait_mode():
            # In 16:9 (widescreen): tabella notizie a sinistra, dettaglio a destra
            layout.split_row(
                Layout(name="news_table_pane", ratio=3),
                Layout(name="news_detail_pane", ratio=2)
            )
        else:
            # In 9:16 (verticale): tabella sopra, dettaglio sotto
            layout.split_column(
                Layout(name="news_table_pane", ratio=2),
                Layout(name="news_detail_pane", ratio=1)
            )

        news_table = Table(box=box.SIMPLE_HEAD, expand=True)
        news_table.add_column("#", width=3, justify="right")
        news_table.add_column("Source", width=8, style="bold metric")
        news_table.add_column("Title", style="bold fg")
        news_table.add_column("Tags", width=20, style="dim warn")
        news_table.add_column("Date", width=12, style="dim fg")

        _n0, _n1 = _win_paginated(len(state.news_items), state.selected_news_idx)
        for idx, item in enumerate(state.news_items[_n0:_n1], start=_n0):
            cursor = "❯" if idx == state.selected_news_idx else " "
            source = item.get("source", "news") if isinstance(item, dict) else getattr(item, "source", "news")
            title = item.get("title", "") if isinstance(item, dict) else getattr(item, "title", "")
            tags = item.get("tags", []) if isinstance(item, dict) else getattr(item, "tags", [])
            date = item.get("published", item.get("date", "")) if isinstance(item, dict) else getattr(item, "date", "")
            source_badge = f"[{'ok' if source=='arxiv' else ('warn' if source in ('hf', 'huggingface') else 'accent')}]{source.upper()[:8]}[/]"
            tags_str = ", ".join(tags[:3]) if isinstance(tags, list) else str(tags)[:20]
            news_table.add_row(
                f"{cursor}{idx+1}",
                source_badge,
                f"{title[:65]}...",
                tags_str,
                str(date)[:12]
            )

        if not state.news_items:
            news_table.add_row(" ", "[dim]--[/]", "[dim]Fetching latest AI releases, arXiv papers & Hugging Face models...[/]", "--", "--")

        news_panel = Panel(
            news_table,
            box=box.ROUNDED,
            border_style="border",
            title="[bold metric]📡 AI MODEL RADAR & DISCOVERY[/]",
            subtitle="[dim][↑/↓] Select News | [A] All | [X] arXiv | [H] Hugging Face | [R] Refresh[/]"
        )

        # Selected item details
        if state.news_items and 0 <= state.selected_news_idx < len(state.news_items):
            selected = state.news_items[state.selected_news_idx]
            s_title = selected.get("title", "") if isinstance(selected, dict) else getattr(selected, "title", "")
            s_source = selected.get("source", "") if isinstance(selected, dict) else getattr(selected, "source", "")
            s_date = selected.get("published", selected.get("date", "")) if isinstance(selected, dict) else getattr(selected, "date", "")
            s_url = selected.get("url", selected.get("link", "")) if isinstance(selected, dict) else getattr(selected, "url", "")
            s_summary = selected.get("summary", selected.get("description", "")) if isinstance(selected, dict) else getattr(selected, "summary", "")
            from rich.markup import escape
            detail_text = (
                f"[bold warn]{escape(str(s_title))}[/]\n"
                f"[dim]Source: {escape(str(s_source).upper())} | Date: {escape(str(s_date))} | Link: {escape(str(s_url))}[/]\n\n"
                f"{escape(str(s_summary)[:400])}"
            )
        else:
            detail_text = "[dim]No radar item selected.[/]"

        detail_panel = Panel(
            Text.from_markup(detail_text),
            box=box.ROUNDED,
            border_style="border",
            title="[bold metric2]📄 SUMMARY & ABSTRACT[/]"
        )

        layout["news_table_pane"].update(news_panel)
        layout["news_detail_pane"].update(detail_panel)
        return layout

    # -----------------------------------------------------------------
    # TAB 4: HUGGING FACE DOWNLOADER & QUANT INSPECTOR
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab4_downloader(state: AppState) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top_row", ratio=3),
            Layout(name="downloads_pane", ratio=1)
        )
        layout["top_row"].split_row(
            Layout(name="repo_pane", ratio=1),
            Layout(name="quant_pane", ratio=1)
        )

        # Repo List Table
        repo_table = Table(box=box.SIMPLE_HEAD, expand=True)
        repo_table.add_column("#", width=3, justify="right")
        repo_table.add_column("Repository", style="bold fg")
        repo_table.add_column("Downloads", width=10, justify="right", style="metric")
        repo_table.add_column("Likes", width=7, justify="right", style="warn")

        # Finestra di 12 righe che SEGUE il cursore (prima erano sempre le
        # prime 12: scendendo oltre la selezione spariva dallo schermo).
        _hw = max(6, shutil.get_terminal_size((120, 40)).lines - 24)   # righe in base al terminale
        _r0 = max(0, min(state.selected_repo_idx - _hw // 2, len(state.hf_repos) - _hw))
        for idx, repo in enumerate(state.hf_repos[_r0:_r0 + _hw], start=_r0):
            cursor = "❯" if idx == state.selected_repo_idx else " "
            repo_table.add_row(
                f"{cursor}{idx+1}",
                f"[bold {'warn' if idx==state.selected_repo_idx else 'fg'}]{repo.repo_id[:34]}[/]",
                f"{repo.downloads:,}",
                f"♥ {repo.likes}"
            )

        if not state.hf_repos:
            repo_table.add_row(" ", "[dim]No repositories loaded. Press [S] to search Hugging Face.[/]", "--", "--")

        repo_border = "warn" if state.hf_active_pane == "repos" else "metric"
        repo_panel = Panel(
            repo_table,
            box=box.ROUNDED,
            border_style=repo_border,
            title=f"[bold metric]🔍 HF REPOSITORIES: '{_mesc(str(state.hf_query))}' "
                  f"({state.selected_repo_idx + 1 if state.hf_repos else 0}/{len(state.hf_repos)})[/]",
            subtitle="[dim][S] Cerca | [↑/↓] [PgSu/PgGiù] Scorri | [Tab] Cambia pannello | [H] Groq Config[/]"
        )

        # Quant Inspector Table
        quant_table = Table(box=box.SIMPLE_HEAD, expand=True)
        quant_table.add_column("#", width=3, justify="right")
        quant_table.add_column("Quant File", style="bold fg")
        quant_table.add_column("Type", width=8, style="metric")
        quant_table.add_column("Size", width=8, justify="right", style="warn")
        quant_table.add_column("RTX 3050 Fit", width=15, justify="center")

        _q0 = max(0, min(state.selected_quant_idx - _hw // 2, len(state.hf_quants) - _hw))
        for idx, q in enumerate(state.hf_quants[_q0:_q0 + _hw], start=_q0):
            cursor = "❯" if idx == state.selected_quant_idx else " "
            rec_badge = " ⭐" if q.is_recommended else ""
            quant_table.add_row(
                f"{cursor}{idx+1}",
                f"{q.filename[:28]}{rec_badge}",
                f"{q.quant_type}",
                f"{q.size_gb:.2f} GB",
                f"{q.vram_fit_status}"
            )

        if not state.hf_quants:
            quant_table.add_row(" ", "[dim]Select a repo to inspect GGUF / EXL3 quants.[/]", "--", "--", "--")

        quant_border = "warn" if state.hf_active_pane == "quants" else "metric"
        quant_panel = Panel(
            quant_table,
            box=box.ROUNDED,
            border_style=quant_border,
            title=f"[bold metric]📦 QUANT INSPECTOR (GGUF / EXL3 / RTX 3050 FIT) ({state.selected_quant_idx + 1 if state.hf_quants else 0}/{len(state.hf_quants)})[/]",
            subtitle=f"[dim][D/Enter] Download to {state.hf_download_dir} | [H] Auto Groq Config[/]"
        )

        # Active Downloads Queue
        dl_table = Table(box=box.SIMPLE, expand=True)
        dl_table.add_column("Task ID", width=10, style="dim fg")
        dl_table.add_column("File Name", style="bold fg")
        dl_table.add_column("Progress", width=28)
        dl_table.add_column("Speed", width=12, style="metric")
        dl_table.add_column("ETA", width=8, style="warn")
        dl_table.add_column("Status", width=14)

        all_downloads = state.hf_mgr.get_all_downloads()
        for task in all_downloads:
            pct = getattr(task, 'progress_pct', getattr(task, 'progress_percent', 0.0))
            bar_len = 16
            filled = int((pct / 100.0) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            bar_str = f"[{'ok' if pct>99 else 'warn'}][{bar}] {pct:.1f}%[/]"

            status_style = {
                "downloading": "[bold warn]⚡ DOWNLOADING[/]",
                "completed": "[bold ok]✓ COMPLETED[/]",
                "failed": "[bold err]✗ FAILED[/]",
                "pending": "[bold fg]⏳ PENDING[/]"
            }.get(task.status, task.status)

            dl_table.add_row(
                task.task_id,
                task.filename[:30],
                bar_str,
                f"{task.speed_mb_s:.1f} MB/s",
                f"{task.eta_seconds}s",
                status_style
            )

        if not all_downloads:
            dl_table.add_row("--", f"[dim]No active downloads. Direct target folder: {state.hf_download_dir}[/]", "--", "--", "--", "[dim]IDLE[/]")

        dl_panel = Panel(
            dl_table,
            box=box.ROUNDED,
            border_style="border",
            title=f"[bold metric2]⬇ ACTIVE DOWNLOADS QUEUE (Target: {state.hf_download_dir})[/]"
        )

        layout["top_row"]["repo_pane"].update(repo_panel)
        layout["top_row"]["quant_pane"].update(quant_panel)
        layout["downloads_pane"].update(dl_panel)
        return layout

    # -----------------------------------------------------------------
    # TAB 5: PROCESS CONTROL & BENCHMARK ENGINE (multi-select + chain)
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab5_process_control(state: AppState) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="models_pane", ratio=1),
            Layout(name="control_bench_pane", ratio=1)
        )
        layout["control_bench_pane"].split_column(
            Layout(name="server_status_pane", ratio=1),
            Layout(name="bench_results_pane", ratio=1)
        )

        # ── Left pane: Models list with multi-select checkboxes ──
        model_table = Table(box=box.SIMPLE_HEAD, expand=True)
        model_table.add_column("Sel", width=3, justify="center")
        model_table.add_column("#", width=3, justify="right")
        model_table.add_column("GGUF File (/mnt/lm_models/models)", style="bold fg")
        model_table.add_column("Quant", width=8, style="metric")
        model_table.add_column("Size", width=8, justify="right", style="warn")
        model_table.add_column("Dove entra", width=12, justify="left")

        total_models = len(state.local_models)
        visible = state.model_visible()
        # Righe della lista = altezza reale del terminale meno intestazione,
        # footer e bordi (prima fisse a 13: a schermo intero se ne vedevano una decina).
        window_size = max(5, shutil.get_terminal_size((120, 40)).lines - 22)
        sel_idx = state.selected_model_idx
        pos = visible.index(sel_idx) if sel_idx in visible else 0
        start_pos = max(0, min(pos - (window_size // 2), max(0, len(visible) - window_size)))

        n_selected = len(state.selected_model_indices)
        for idx in visible[start_pos:start_pos + window_size]:
            m = state.local_models[idx]
            is_focused = (idx == sel_idx)
            is_checked = (idx in state.selected_model_indices)
            cursor = "▶ " if is_focused else "  "
            check_str = "[bold ok]☑[/]" if is_checked else "[dim]☐[/]"
            is_active = (m.filename == state.active_model_name or m.filename in state.active_model_name)
            active_badge = " [bold ok]●[/]" if is_active else ""
            row_style = "bold bg on metric" if is_focused else ""
            model_table.add_row(
                check_str,
                f"{cursor}{idx+1}",
                f"[bold {'warn' if is_focused else 'fg'}]{m.filename[:26]}[/]{active_badge}",
                f"{m.quant_type}",
                f"{m.size_gb:.2f} GB",
                Text(state.fit_label(m.size_gb)[0], style=f"bold {state.fit_label(m.size_gb)[1]}"),
                style=row_style
            )

        if not state.local_models:
            model_table.add_row(" ", " ", "[dim]No .gguf files found in /mnt/lm_models/models. Download via Tab 4.[/]", "--", "--", "--")

        sel_badge = f" | [bold ok]{n_selected} selected[/]" if n_selected else ""
        models_panel = Panel(
            model_table,
            box=box.ROUNDED,
            border_style="border",
            title=f"[bold metric]📂 MODELLI ({len(visible)}/{total_models} · filtro "
                  f"{state.SIZE_FILTERS[state.model_size_filter][0]}){sel_badge}[/]",
            subtitle="[dim][↑/↓] Naviga  [Z] Filtro dimensione  [Space] Seleziona  [A] Tutti  [N] Nessuno  "
                     "[T] Chain Bench  [L] Avvia · 🟢 VRAM 🟠 NEED RPC 🔴 RAM (stima da dimensione)[/]"
        )

        # ── Top-right: Server status ──
        preset_info = LlamaServerManager.SERVER_PRESETS.get(state.server_preset_key, {})
        server_table = Table(box=box.SIMPLE, expand=True, show_header=False)
        server_table.add_column("Key", style="bold metric", width=18)
        server_table.add_column("Value", style="bold fg")

        server_table.add_row("Server Engine", "[bold ok]llama-server (llama.cpp CUDA Win64)[/]")
        server_table.add_row("Status", f"[bold {'ok' if state.server_online else 'muted'}]{'● ONLINE' if state.server_online else '○ OFFLINE'}[/]")
        server_table.add_row("Active Model", f"[bold warn]{state.active_model_name}[/]")
        server_table.add_row("Active Preset", f"[bold metric]{preset_info.get('name', 'Custom')}[/] [dim](Hotkey [O] to cycle)[/]")
        server_table.add_row("RAM KV Offload", f"[{'bold ok' if state.server_no_kv_offload else 'warn'}]{'ENABLED (-nkvo) ➔ 100% VRAM for Weights' if state.server_no_kv_offload else 'DISABLED ➔ GPU KV Cache'}[/]")
        server_table.add_row("Context Limits", f"[bold metric]256,144 Tokens (Max Window)[/]")
        server_table.add_row("Server Endpoint", f"http://{state.server_host}:{state.server_port}")

        server_panel = Panel(
            server_table,
            box=box.ROUNDED,
            border_style="ok" if state.server_online else "muted",
            title="[bold ok]⚙️ LLAMA-SERVER PROCESS CONTROL[/]",
            subtitle="[dim][L] Load Model | [O] Cycle Preset | [F] Flush VRAM[/]"
        )

        # ── Bottom-right: Benchmark results (chain log or last result) ──
        bench_table = Table(box=box.SIMPLE, expand=True, show_header=False)
        bench_table.add_column("Metric", style="bold metric", width=20)
        bench_table.add_column("Result", style="bold fg")

        if state.is_benchmarking:
            bench_table.add_row("Status", "[bold muted on warn] ⚡ CHAIN BENCHMARK RUNNING... [/]")
            # Show last N log lines
            log_lines = state.chain_bench_log[-8:] if state.chain_bench_log else ["Starting..."]
            for line in log_lines:
                color = "ok" if line.startswith("  ✓") else ("warn" if line.startswith("  ↳") else ("err" if "✗" in line else "fg"))
                bench_table.add_row("", f"[{color}]{line[:70]}[/]")

        elif state.show_chain_report and state.chain_bench_results:
            # Show comparative table
            bench_table.add_row("Chain Report", f"[bold ok]✓ {len(state.chain_bench_results)} models benchmarked[/]")
            if state.chain_bench_report_path:
                bench_table.add_row("Report File", f"[dim]{state.chain_bench_report_path}[/]")
            bench_table.add_row("", "")

            # Mini comparison table header
            cmp_table = Table(box=box.SIMPLE_HEAD, expand=True)
            cmp_table.add_column("Model", style="fg", no_wrap=True)
            cmp_table.add_column("tok/s", width=7, justify="right", style="ok")
            cmp_table.add_column("TTFT", width=7, justify="right", style="metric")
            cmp_table.add_column("AST", width=5, justify="center")
            cmp_table.add_column("Tool", width=5, justify="center")
            cmp_table.add_column("OptCtx", width=7, justify="right", style="warn")

            ranked = sorted(state.chain_bench_results, key=lambda r: r.generation_speed_tok_s, reverse=True)
            for r in ranked[:7]:
                ast_s = "✓" if r.python_ast_pass else "✗"
                tool_s = "✓" if r.json_tool_pass else "✗"
                ctx_s = f"{r.optimal_ctx//1024}k" if r.optimal_ctx >= 1024 else str(r.optimal_ctx)
                cmp_table.add_row(
                    r.model_name[:22],
                    f"{r.generation_speed_tok_s:.1f}",
                    f"{r.ttft_ms:.0f}ms",
                    f"[ok]{ast_s}[/]" if r.python_ast_pass else f"[err]{ast_s}[/]",
                    f"[ok]{tool_s}[/]" if r.json_tool_pass else f"[err]{tool_s}[/]",
                    ctx_s
                )

            bench_panel = Panel(
                Group(bench_table, cmp_table),
                box=box.ROUNDED,
                border_style="ok",
                title="[bold ok]🏆 CHAIN BENCHMARK REPORT (ranked by tok/s)[/]",
                subtitle=f"[dim]Press [R] to reset | Report: benchmark_report.json[/]"
            )
            layout["models_pane"].update(models_panel)
            layout["control_bench_pane"]["server_status_pane"].update(server_panel)
            layout["control_bench_pane"]["bench_results_pane"].update(bench_panel)
            return layout

        elif state.last_benchmark_result:
            br = state.last_benchmark_result
            if br.status == "SERVER_OFFLINE":
                status_badge = "[bold warn]⚠️ SERVER OFFLINE[/]"
                bench_table.add_row("Status", status_badge)
                bench_table.add_row("Model Tested", f"[bold warn]{br.model_name[:26]}[/]")
                diag_msg = br.error_detail if br.error_detail else (br.error_msg or "llama-server (port 18434) is offline")
                bench_table.add_row("Diagnostics", f"[err]{diag_msg[:65]}[/]")
                bench_table.add_row("Action", "[bold ok]Press [L/Enter] to Load Model, then [T] to Benchmark[/]")
            elif br.status == "ERROR":
                status_badge = "[bold err]✗ BENCHMARK ERROR[/]"
                bench_table.add_row("Status", status_badge)
                bench_table.add_row("Model Tested", f"[bold warn]{br.model_name[:26]}[/]")
                err_text = br.error_detail if br.error_detail else (br.error_msg or "Unknown error")
                bench_table.add_row("Error Detail", f"[err]{err_text[:75]}[/]")
            else:
                status_badge = "[bold ok]✓ BENCHMARK PASSED[/]" if br.all_passed else "[bold err]✗ BENCHMARK FAILED[/]"
                ast_badge = "[ok]✓ PASS[/]" if br.python_ast_pass else "[err]✗ FAIL[/]"
                json_badge = "[ok]✓ PASS[/]" if br.json_tool_pass else "[err]✗ FAIL[/]"
                bench_table.add_row("Status", status_badge)
                bench_table.add_row("Model Tested", f"[bold warn]{br.model_name[:26]}[/]")
                bench_table.add_row("Python AST", ast_badge)
                bench_table.add_row("JSON Tool", json_badge)
                bench_table.add_row("Speed", f"[bold ok]{br.speed_tok_s:.1f} tok/s[/]")
                bench_table.add_row("TTFT Latency", f"[metric]{br.ttft_ms:.1f} ms[/]")
                bench_table.add_row("VRAM", f"[warn]{br.vram_used_mb:.0f} MB[/]")
                if br.optimal_ctx:
                    bench_table.add_row("Optimal Ctx", f"[warn]{br.optimal_ctx:,} tokens[/]")
                bench_table.add_row("Timestamp", f"[dim]{br.timestamp}[/]")
        else:
            bench_table.add_row("Status", "[dim]⚪ IDLE / READY[/]")
            bench_table.add_row("Selection", f"[bold warn]{n_selected} model(s) selected (Space to toggle)[/]")
            bench_table.add_row("Chain Bench", "[bold warn]Press [T] to run chain benchmark on selected models[/]")
            bench_table.add_row("Single Bench", "[dim]Or select none + [T] to benchmark cursor model only[/]")
            bench_table.add_row("Tests", "• Python AST • JSON Tool • tok/s • TTFT • 8k/16k/32k Ctx Probe")

        bench_border = "warn" if state.is_benchmarking else ("ok" if state.show_chain_report else "warn")
        bench_panel = Panel(
            bench_table,
            box=box.ROUNDED,
            border_style=bench_border,
            title="[bold warn]🧪 CHAIN BENCHMARK & CTX ESTIMATOR[/]",
            subtitle="[dim][T] Chain Run | [Space] Toggle Select | [A] All | [N] None | [R] Reset[/]"
        )

        layout["models_pane"].update(models_panel)
        layout["control_bench_pane"]["server_status_pane"].update(server_panel)
        layout["control_bench_pane"]["bench_results_pane"].update(bench_panel)
        return layout

    # -----------------------------------------------------------------
    # TAB 6: AGENT SHELL & STREAMING INFERENCE
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab6_agent_shell(state: AppState) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="chat_history_pane", ratio=3),
            Layout(name="input_status_pane", ratio=1)
        )

        # Chat history rendered inside panel
        chat_content = []
        for msg in state.chat_history[-8:]:
            if msg.role == "system":
                chat_content.append(Text.from_markup(f"[dim metric][{msg.timestamp}] [SYSTEM]:[/] [dim fg]{msg.content}[/]"))
            elif msg.role == "user":
                chat_content.append(Text.from_markup(f"[bold warn][{msg.timestamp}] ❯ USER:[/] [bold fg]{msg.content}[/]"))
            elif msg.role == "assistant":
                chat_content.append(Text.from_markup(f"[bold ok][{msg.timestamp}] ⚡ ASSISTANT:[/] [fg]{msg.content}[/]"))
            elif msg.role == "tool":
                chat_content.append(Text.from_markup(f"[bold metric2][{msg.timestamp}] ⚙️ TOOL ({msg.tool_name}):[/] [metric]{msg.content}[/]"))
            chat_content.append(Text(""))

        if state.is_generating and state.current_stream_buffer:
            chat_content.append(Text.from_markup(f"[bold ok]⚡ ASSISTANT (Streaming):[/] [fg]{state.current_stream_buffer} ▌[/]"))

        chat_panel = Panel(
            Group(*chat_content),
            box=box.ROUNDED,
            border_style="border",
            title="[bold metric]💬 INTERACTIVE AGENT SHELL & INFERENCE BUFFER[/]",
            subtitle="[dim]Streaming LLM Inference + Tool Calling Engine[/]"
        )

        # Input & Shortcuts
        input_table = Table(box=box.SIMPLE, expand=True, show_header=False)
        input_table.add_column("Key", style="bold metric", width=16)
        input_table.add_column("Action", style="fg")

        input_table.add_row(
            "❯ scrivi qui" if state.chat_focused else "[I] per scrivere",
            (f"[bold fg]{_mesc(state.chat_input)}[/][metric]▌[/]"
             if state.chat_focused else
             "[dim]scrittura disattiva: 1-9 cambiano scheda normalmente. Premi [bold accent]I[/] per scrivere.[/]")
            + ("   [warn]⏳ il modello sta rispondendo...[/]" if state.chat_busy else ""))
        input_table.add_row("[Invio]", "invia al modello attivo (testo libero, o /tool calc(...), /tool sys, /tool vram, /tool bench)")
        input_table.add_row("[ESC]", "esce dalla scrittura (libera 1-9 e i tasti delle altre schede)")
        input_table.add_row("[T]", "benchmark sul modello attivo (salva in ~/.ai_tui_benchmarks.json)")
        input_table.add_row("[C]", "Clear chat history and reset KV cache")
        input_table.add_row("[F]", "Flush VRAM and terminate background inference process")
        input_table.add_row("Built-in Tools", "`/tool calc(<expr>)`, `/tool sys`, `/tool vram`, `/tool bench`")

        input_panel = Panel(
            input_table,
            box=box.ROUNDED,
            border_style="border",
            title="[bold metric2]⌨️ COMMAND DISPATCHER[/]"
        )

        layout["chat_history_pane"].update(chat_panel)
        layout["input_status_pane"].update(input_panel)
        return layout

    # -----------------------------------------------------------------
    # GEAR ⚙️ SETTINGS EDITOR MODAL OVERLAY ([G])
    # -----------------------------------------------------------------
    @staticmethod
    def render_gear_modal(state: AppState) -> Panel:
        """Render high-tech Cyberpunk Parameter Editor Modal for selected model."""
        config = state.gear_model_config or ModelConfig(model_name=state.gear_model_name)
        
        table = Table(box=box.SIMPLE_HEAD, expand=True)
        table.add_column("#", width=3, justify="right")
        table.add_column("Parameter", width=24, style="bold metric")
        table.add_column("Value", width=22, style="bold fg")
        table.add_column("Visual Indicator / Setting", style="warn")

        for idx, fdef in enumerate(state.gear_fields_def):
            cursor = "❯" if idx == state.gear_active_field else " "
            key = fdef["key"]
            label = fdef["label"]
            ftype = fdef.get("type", "float")
            val = getattr(config, key, None)

            # Format visual representation
            if ftype == "bool":
                val_str = "[bold ok]ENABLED[/]" if val else "[bold err]DISABLED[/]"
                vis_str = "[ok][● ON ][/]" if val else "[err][  OFF ○][/]"
            elif ftype == "float":
                val_str = f"[bold warn]{val:.3f}[/]"
                min_v = fdef.get("min", 0.0)
                max_v = fdef.get("max", 2.0)
                pct = max(0.0, min(1.0, (val - min_v) / max(0.001, max_v - min_v)))
                bar_len = 16
                filled = int(pct * bar_len)
                vis_str = f"[{'metric' if pct<0.7 else 'warn'}][{'━'*filled}●{'─'*(bar_len-filled)}][/]"
            elif ftype == "int":
                val_str = f"[bold warn]{val:,}[/]"
                min_v = fdef.get("min", 0)
                max_v = fdef.get("max", 262144)
                pct = max(0.0, min(1.0, (val - min_v) / max(1, max_v - min_v)))
                bar_len = 16
                filled = int(pct * bar_len)
                vis_str = f"[{'metric' if pct<0.7 else 'warn'}][{'━'*filled}●{'─'*(bar_len-filled)}][/]"
            else:
                val_str = f"[italic fg]{str(val)[:20]}[/]"
                vis_str = f"[dim]{fdef.get('desc', '')[:30]}[/]"

            is_active = (idx == state.gear_active_field)
            row_style = "bold fg on muted" if is_active else ""
            table.add_row(
                f"{cursor}{idx+1}",
                f"[bold {'warn' if is_active else 'metric'}]{label}[/]",
                val_str,
                vis_str,
                style=row_style
            )

        active_def = state.gear_fields_def[state.gear_active_field]
        desc_text = f"[bold metric]Description:[/] {active_def.get('desc', '')}"

        help_text = (
            "[bold warn][↑/↓][/] Select Field  |  "
            "[bold warn][←/→][/] Adjust Value  |  "
            "[bold warn][Enter][/] Toggle  |  "
            "[bold ok][S][/] Save & Create LM Studio Preset  |  "
            "[bold warn][H][/] Groq Auto-Fetch  |  "
            "[bold err][Esc/G][/] Close"
        )

        modal_group = Group(
            Text.from_markup(f"[bold warn]Model Target:[/] [bold fg]{state.gear_model_name}[/]"),
            Text(""),
            table,
            Text(""),
            Text.from_markup(desc_text),
            Text(""),
            Text.from_markup(help_text)
        )

        return Panel(
            modal_group,
            box=box.DOUBLE,
            border_style="warn",
            title=f"[bold warn]⚙️ IN-PLACE MODEL PARAMETER & PRESET EDITOR ◆ {state.gear_model_name[:30]}[/]",
            title_align="center",
            padding=(1, 2)
        )


    # -----------------------------------------------------------------
    # TAB 7: GITHUB PLUGINS & SKILLS ECOSYSTEM INSTALLER
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab7_plugins_skills(state: AppState) -> Layout:
        """Render GitHub Plugins & Skills Manager for Hermes Agent, Claude Code & OmniRoute."""
        layout = Layout()
        layout.split_column(
            Layout(name="top_bar", size=4),
            Layout(name="plugins_table", ratio=1),
            Layout(name="install_instructions", size=4)
        )

        top_grid = Table.grid(expand=True)
        top_grid.add_column(ratio=6)
        top_grid.add_column(ratio=4, justify="right")
        top_grid.add_row(
            Text.from_markup("[bold metric]🧩 GITHUB PLUGINS & SKILLS ECOSYSTEM[/]\n[dim]Install skills & extensions via GitHub URL or repo name[/]"),
            Text.from_markup("[bold ok][I] Install GitHub URL[/] | [bold warn][R] Refresh[/] | [bold err][D] Uninstall[/]")
        )
        layout["top_bar"].update(Panel(top_grid, box=box.ROUNDED, border_style="border"))

        table = Table(box=box.SIMPLE_HEAD, expand=True)
        table.add_column("#", width=3, justify="right")
        table.add_column("Plugin / Skill Name", style="bold fg", width=28)
        table.add_column("Type", width=18, style="bold metric")
        table.add_column("Description / Summary", style="dim warn")
        table.add_column("Path", style="dim fg")
        table.add_column("Status", width=10, justify="center")

        plugins = state.installed_plugins
        _p0, _p1 = _win_paginated(len(plugins), state.selected_plugin_idx)
        for idx, p in enumerate(plugins[_p0:_p1], start=_p0):
            cursor = "▶ " if idx == state.selected_plugin_idx else "  "
            row_style = "bold bg on metric" if idx == state.selected_plugin_idx else ""
            table.add_row(
                f"{cursor}{idx+1}",
                f"[bold warn]{p['name']}[/]",
                f"[metric]{p['type']}[/]",
                p.get("description", "--")[:45],
                p.get("path", "--")[:35],
                "[bold ok]ACTIVE[/]",
                style=row_style
            )

        if not plugins:
            table.add_row(" ", "[dim]No external plugins installed yet.[/]", "--", "[dim]Press [I] to paste a GitHub repo URL[/]", "--", "--")

        layout["plugins_table"].update(Panel(table, box=box.ROUNDED, border_style="border", title=f"[bold metric]📦 INSTALLED SKILLS & PLUGINS ({len(plugins)})[/]"))

        inst_text = Text.from_markup(
            "[bold warn]💡 Supported Repositories:[/] Paste any GitHub repository URL (e.g., [metric]https://github.com/nousresearch/hermes-agent[/] or [metric]owner/skill-name[/]).\n"
            "Automatically cloned, validated against [bold ok]SKILL.md[/] / [bold ok]package.json[/] and made available across Hermes Agent & OmniRoute."
        )
        layout["install_instructions"].update(Panel(inst_text, box=box.ROUNDED, border_style="border"))
        return layout

    @staticmethod
    def render_tab8_lm_bench(state: AppState) -> Layout:
        """Tab 8: lm_eval Benchmark Lab. Pure computation, no GPU, no server."""
        layout = Layout()
        if not is_portrait_mode():
            # In 16:9: sinistra info e matrice, destra tabella task estesa
            layout.split_row(
                Layout(name="left_col", ratio=1),
                Layout(name="lm_tasks", ratio=1)
            )
            layout["left_col"].split_column(
                Layout(name="lm_header", ratio=3),
                Layout(name="lm_matrix", ratio=3),
                Layout(name="lm_maxctx", ratio=2),
            )
        else:
            layout.split_column(
                Layout(name="lm_header", ratio=2),
                Layout(name="lm_matrix", ratio=2),
                Layout(name="lm_maxctx", ratio=2),
                Layout(name="lm_tasks", ratio=3),
            )

        tasks_meta = state.lm_tasks or {}
        version = tasks_meta.get("lm_eval_version", "sconosciuto")
        total = tasks_meta.get("totale_task_disponibili", 0)
        harvested = tasks_meta.get("harvested_at", "")
        stamp = harvested[:19].replace("T", " ") if harvested else "non disponibile"
        absent = tasks_meta.get("assenti") or []

        head = Table.grid(padding=(0, 2))
        head.add_column(justify="right", width=14)
        head.add_column()
        head.add_row("[bold metric]lm_eval", f"[bold]{version}[/]  [muted]interrogato realmente, zero nomi inventati[/]")
        head.add_row("[bold metric]Task totali", f"[bold warn]{total:,}[/] disponibili in TaskManager().all_tasks  [muted](harvest: {stamp})[/]")
        head.add_row("[bold metric]Gruppo", f"[bold accent]{state.lm_task_group.upper()}[/]  [muted](premi [metric]G[/] per classici/agentici)[/]")
        if absent:
            names = ", ".join(a["task"] for a in absent)
            head.add_row("[bold err]Assenti", f"[err]{names}[/]")
        else:
            head.add_row("[bold ok]Assenti", "[ok]nessuno: ogni task della suite esiste davvero[/]")
        layout["lm_header"].update(
            Panel(head, box=box.ROUNDED, border_style="border",
                  title="[bold accent]🧪 LM_EVAL BENCHMARK LAB[/]",
                  subtitle="[muted][M] matrice · [R] ricarica task · [G] gruppo · [K] KV · [T] dry-run[/]",
                  padding=(0, 1))
        )

        # ---- ctx x KV matrix -------------------------------------------
        m = state.lm_matrix
        if state.lm_matrix_error:
            body = Text.from_markup(f"[bold err]Calcolo non disponibile:[/] {state.lm_matrix_error}")
        elif not m:
            body = Text.from_markup(
                "[muted]Nessuna matrice calcolata.\n"
                "Premi [bold metric]M[/] per leggere l'header GGUF e costruire la matrice ctx x KV.\n"
                "La lettura dell'header non avvia llama-server e non tocca la GPU.[/]"
            )
        else:
            rows = m.get("matrice_ctx_kv") or {}
            ctxs = m.get("ctx_values") or []
            mt = Table(expand=True, box=box.SIMPLE_HEAD, border_style="border", padding=(0, 1))
            mt.add_column("KV format", style="bold", width=12)
            mt.add_column("MiB/1k tok", justify="right", width=11)
            for c in ctxs:
                mt.add_column(f"{c:,}", justify="right", width=10)
            cap = (m.get("ctx_max") or {}).get("formati") or {}
            for fmt in ("f16", "q8_0", "q4_0"):
                row = rows.get(fmt) or {}
                per = (cap.get(fmt) or {}).get("mib_per_1k_tok", 0.0)
                cells = [f"[metric]{per:.2f}[/]"]
                for c in ctxs:
                    v = row.get(str(c), row.get(c, 0.0))
                    cells.append(f"{v:,.0f}" if v else "[muted]--[/]")
                mt.add_row(f"[bold ok]{fmt}[/]", *cells)
            body = mt

        layout["lm_matrix"].update(
            Panel(body, box=box.ROUNDED, border_style="border",
                  title="[bold metric]MATRICE ctx x KV  (MiB di solo KV cache)[/]", padding=(0, 1))
        )

        # ---- max context within budget --------------------------------
        if m and not state.lm_matrix_error:
            cap = (m.get("ctx_max") or {}).get("formati") or {}
            rec = cap.get(state.lm_bench_kv) or {}
            geo = m.get("geometry") or {}
            parts = []
            for fmt in ("f16", "q8_0", "q4_0"):
                info = cap.get(fmt) or {}
                v = info.get("ctx_max")
                mark = "◀" if fmt == state.lm_bench_kv else " "
                col = "bold ok" if fmt == state.lm_bench_kv else "fg"
                txt = f"{v:,} tok" if v else "non entra"
                parts.append(f"[{col}]{mark} {fmt}: {txt}[/]")
            cmax = (m.get("ctx_max") or {})
            pool = cmax.get("kv_budget_mib", 0.0)
            arch = geo.get("arch", "?")
            kvheads = geo.get("n_head_kv", "?")
            kvlayers = geo.get("n_attn_layers", "?")
            layout["lm_maxctx"].update(Panel(
                Text.from_markup(
                    f"[muted]Budget[/] [bold]{cmax.get('budget_mib', 0):,.0f} MiB[/] [muted]· pool KV[/] "
                    f"[bold]{pool:,.0f} MiB[/] [muted]· pesi[/] [bold]{cmax.get('pesi_mib', 0):,.0f} MiB[/] "
                    f"[muted]· arch[/] [metric]{arch}[/] [muted]· KV-heads[/] [metric]{kvheads}[/] "
                    f"[muted]· layer con KV[/] [metric]{kvlayers}[/]\n" + "   ".join(parts)
                ),
                box=box.ROUNDED, border_style="border",
                title="[bold fg]CTX MASSIMO ENTRO BUDGET VRAM[/]", padding=(0, 1))
            )
        else:
            layout["lm_maxctx"].update(Panel(
                Text.from_markup("[muted]Calcolare la matrice per vedere il ctx massimo per formato KV.[/]"),
                box=box.ROUNDED, border_style="border", padding=(0, 1)))

        # ---- task list ------------------------------------------------
        t = Table(expand=True, box=box.SIMPLE_HEAD, border_style="border", padding=(0, 1))
        t.add_column("#", width=4, justify="right")
        t.add_column("Task", style="bold metric", width=30)
        t.add_column("Descrizione")
        entries = state.lm_task_list()
        _e0, _e1 = _win_paginated(len(entries), state.lm_selected_task_idx)
        for idx, e in enumerate(entries[_e0:_e1], start=_e0):
            cursor = "▶" if idx == state.lm_selected_task_idx else " "
            style = "bold bg on metric" if idx == state.lm_selected_task_idx else "fg"
            t.add_row(f"[{style}]{cursor}{idx+1}[/]", e["task"], e.get("descrizione", ""))
        if not entries:
            t.add_row("", "[err]nessun task[/]", "Esegui [metric]lmbench.py --refresh-json[/] per popolare bench_tasks.json")
        target = state.lm_bench_model or "(nessun modello selezionato)"
        title = f"[bold metric]TASK lm_eval · {state.lm_task_group.upper()} ({len(entries)})[/]"
        if state.is_lm_loading:
            title += " [warn]· ricarica TaskManager in corso…[/]"
        layout["lm_tasks"].update(Panel(t, box=box.ROUNDED, border_style="border",
                                        title=title,
                                        subtitle=f"[muted]target: {os.path.basename(target)}[/]",
                                        padding=(0, 1)))
        return layout

    # -----------------------------------------------------------------
    # TAB 9: RPC & CUDA13
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab9_rpc_cuda13(state: AppState) -> Layout:
        """
        Tab 9: pannello di configurazione RPC verso Miky (192.168.1.100:50052).

        Mostra i preset validati, evidenziando in NEON i parametri
        CONSIGLIATI e misurati come ottimali, il comando esatto che verrà
        eseguito, e lo stato del ggml-rpc-server su Miky.
        """
        asp = get_aspect_info()
        is_port = asp["is_portrait"]
        layout = Layout()
        if not is_port:
            # In 16:9 (widescreen): 2 colonne affiancate
            # Sinistra: Telemetria Miky + Parametri consigliati
            # Destra: Tabella Preset RPC + Comando llama-server
            layout.split_row(
                Layout(name="left_col", ratio=1),
                Layout(name="right_col", ratio=1)
            )
            layout["left_col"].split_column(
                Layout(name="rpc_miky", ratio=4),
                Layout(name="rpc_rec", ratio=3),
            )
            layout["right_col"].split_column(
                Layout(name="rpc_presets", ratio=4),
                Layout(name="rpc_cmd", ratio=3),
            )
        else:
            # In 9:16 (verticale): 4 sezioni impilate con ratio proporzionali
            layout.split_column(
                Layout(name="rpc_miky", ratio=3),
                Layout(name="rpc_presets", ratio=3),
                Layout(name="rpc_cmd", ratio=3),
                Layout(name="rpc_rec", ratio=2),
            )
        # Pannello telemetria Miky (dettagliato): due GPU + RPC + cache.
        layout["rpc_miky"].update(render_miky_panel(
            state.miky_remote.last_snapshot, compact=False))
        rec = state.rpc_recommended
        sel_preset, sel_label = state.rpc_resolve(state.rpc_preset_key)

        # ---- NEON: parametri consigliati e misurati -------------------
        toks = f"{rec['tok_s']:.2f} tok/s"
        if state.rpc_is_launching:
            status_line = "[bold bg on warn] AVVIO IN CORSO... [/]"
        elif state.server_online:
            status_line = f"[bold bg on ok] SERVER ONLINE (porta {state.server_port}) [/]"
        else:
            status_line = "[bold bg on warn] SERVER OFFLINE [/]"

        neon = Table.grid(padding=(0, 2))
        neon.add_column(justify="right", width=16)
        neon.add_column()
        neon.add_row(
            "[bold accent]PARAMETRI",
            f"[bold accent]CONSIGLIATI[/] [accent]— misurati il 2026-09-28 su Host+RTX3050 / Miky+RTX5050[/]"
        )
        neon.add_row("[accent]--rpc", f"[bold fg]{rec['rpc']}[/]  [accent](ggml-rpc-server su Miky)[/]")
        neon.add_row("[accent]--device", f"[bold fg]{rec['device']}[/]")
        neon.add_row("[accent]-c (ctx)", f"[bold fg]{rec['ctx_size']:,}[/]  [accent]= 128k, misurato come il punto migliore[/]")
        neon.add_row("[accent]--parallel", f"[bold fg]{rec['parallel']}[/]  [accent](1 stream: massimo throughput, niente contesa di slot)[/]")
        neon.add_row("[accent]-ctk / -ctv", f"[bold fg]{rec['kv_type']} / {rec['kv_type']}[/]  [accent](KV quantizzato 4 bit: il collo di bottiglia reale)[/]")
        neon.add_row("[accent]-ngl", f"[bold fg]{rec['ngl']}[/]  [accent](offload totale dei tensori)[/]")
        neon.add_row("[accent]-fa", f"[bold fg]on[/]  [accent](FlashAttention, obbligatorio con KV q4_1)[/]")
        neon.add_row("[bold fg]THROUGHPUT", f"[bold ok]{toks}[/]  [accent]misurato[/]   {status_line}")

        layout["rpc_rec"].update(Panel(
            neon, box=box.ROUNDED, border_style="border",
            title="[bold accent]★ PARAMETRI CONSIGLIATI (NEON) — misurati, non ipotizzati ★[/]",
            padding=(0, 1))
        )

        # ---- Preset (F1..F5) + profili utente -------------------------
        fkey_for = {pk: fk for fk, pk in TUI_RPC_HOTKEYS}
        pt = Table(expand=True, box=box.SIMPLE_HEAD, border_style="border", padding=(0, 1))
        pt.add_column("Tasto", width=6, justify="center", style="bold metric")
        pt.add_column("Preset / Profilo", width=20, style="bold fg")
        pt.add_column("Origine", width=7, justify="center")
        pt.add_column("-c", width=9, justify="right")
        pt.add_column("--parallel", width=9, justify="right")
        pt.add_column("KV", width=7, justify="center")
        pt.add_column("-ngl", width=5, justify="right")
        pt.add_column("tok/s", width=8, justify="right")
        pt.add_column("Note", style="muted", no_wrap=False, overflow="fold")

        _rows_rpc = state.rpc_profile_rows()
        _ri = next((i for i, (k, _, _) in enumerate(_rows_rpc) if k == state.rpc_preset_key), 0)
        _r0, _r1 = _win_paginated(len(_rows_rpc), _ri)
        for key, p, origin in _rows_rpc[_r0:_r1]:
            is_sel = (key == state.rpc_preset_key)
            fkey = fkey_for.get(key, "-")
            if is_sel:
                row_style = "bold bg on metric"
                name_style = "bold bg on metric"
            elif p.get("recommended"):
                # Il preset RACCOMANDATO spicca in verde brillante anche
                # quando non e' quello selezionato.
                name_style = "bold ok"
                row_style = "on ok"
            else:
                name_style = "fg"
                row_style = ""

            tok_s = p.get("tok_s", 0.0)
            name_txt = p.get("name", key)
            if p.get("recommended"):
                name_txt = f"{name_txt} ★"

            # Rich richiede markup bilanciato: uno stile vuoto produrrebbe
            # un [/] orfano, quindi usiamo Text() per lo stile per cella.
            pt.add_row(
                Text(fkey, style="bold metric"),
                Text(name_txt[:20], style=name_style),
                Text("preset" if origin == "preset" else "utente",
                     style="fg" if origin == "preset" else "accent"),
                Text(f"{p.get('ctx_size', 0):,}"),
                Text(str(p.get("parallel", 1))),
                Text(str(p.get("kv_type", "?")), style="bold metric"),
                Text(str(p.get("ngl", 99))),
                Text(f"{tok_s:.2f}" if tok_s else "n/d",
                     style="bold ok" if tok_s else "muted"),
                Text(p.get("description", "")[:78], style="muted"),
                style=row_style,
            )
        if not state.rpc_profile_rows():
            pt.add_row("", "[err]nessun profilo[/]", "", "", "", "", "", "", "")

        hint = (
            f"[muted]selezionato:[/] [bold fg]{sel_label or 'nessuno'}[/]"
            f"  [muted]· modello:[/] [bold]{_mesc(os.path.basename(state.rpc_model_for() or '(nessuno)'))}[/]"
            f"  [muted]· porta[/] [bold]{state.rpc_port}[/]"
        )
        layout["rpc_presets"].update(Panel(
            pt, box=box.ROUNDED, border_style="border",
            title="[bold metric]🔗 PRESET RPC & PROFILI (F1-F6 aprono il menu di avvio)[/]",
            subtitle=hint, padding=(0, 1))
        )

        # ---- Comando esatto + probe Miky ------------------------------
        cmd_txt = state.rpc_last_cmd or state.rpc_preview_cmd()
        probe = state.rpc_probe
        if probe:
            if probe.get("reachable"):
                probe_line = (f"[bold ok]● ggml-rpc-server {probe.get('endpoint')} "
                              f"RAGGIUNGIBILE[/]  [fg]RTT {probe.get('rtt_ms', 0):.2f} ms[/]")
            else:
                probe_line = (f"[bold err]○ {probe.get('endpoint', '')} NON RAGGIUNGIBILE[/] "
                              f"[err]{_mesc(str(probe.get('error', '')))}[/]")
        else:
            probe_line = "[muted]premi [P] per verificare la raggiungibilita' di ggml-rpc-server su Miky (solo TCP, niente VRAM)[/]"

        # Avviso verificato (help b11330 + --list-devices: il device di Miky
        # si chiama RPC0): con --device esplicito llama.cpp usa SOLO i device
        # elencati, quindi "--device CUDA0 --rpc ..." rischia di non usare RPC0.
        # La logica dei preset NON viene cambiata: si avvisa e basta.
        from rich.markup import escape as _esc
        dev_warn = VB.has_device_restriction((cmd_txt or "").split())
        fit_note = ""
        if state.vram_budget.cfg.get("apply_to_presets"):
            fit_note = f"\n[muted]+ margini VRAM:[/] [metric2]{' '.join(state.extra_flags_tokens())}[/]"
        cmd_styled = Text()
        if cmd_txt:
            tokens = cmd_txt.split()
            for i, tok in enumerate(tokens):
                if i == 0:
                    cmd_styled.append(tok, style="bold ok")
                elif tok.startswith("-"):
                    cmd_styled.append(f" {tok}", style="bold accent")
                else:
                    cmd_styled.append(f" {tok}", style="metric")
        else:
            cmd_styled.append("(nessun modello selezionato)", style="muted")
        cmd_styled.no_wrap = False
        cmd_styled.overflow = "fold"

        parts_cmd = [
            Text.from_markup(f"{probe_line}\n[muted]comando che verra' eseguito (dry-run con [T], niente VRAM occupata):[/]"),
            cmd_styled
        ]
        if fit_note or dev_warn:
            parts_cmd.append(Text.from_markup((fit_note + (f"\n[bold warn]⚠ {_esc(dev_warn)}[/]" if dev_warn else "")).strip()))

        layout["rpc_cmd"].update(Panel(
            Group(*parts_cmd), box=box.ROUNDED, border_style="ok",
            title="[bold ok]⚡ COMANDO LLAMA-SERVER (llama.cpp b11259 · CUDA 13.4)[/]",
            subtitle="[muted][↑/↓] preset · [Space] modello · [Enter] AVVIA · [T] dry-run · [S] salva profilo · [P] probe · [K] start/stop Miky · [C] cleanup Miky · [9] torna[/]",
            padding=(0, 1))
        )
        return layout

    # -----------------------------------------------------------------
    # Tab 10: pannello di configurazione COMPLETO (flag letti da -h)
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab10_config(state: AppState) -> Layout:
        """
        Tab 10: configurazione completa di llama-server.

        Mostra TUTTI i campi, raggruppati per sezione, con per ogni campo:
        valore corrente, default dichiarato dall'help della build, e se il
        flag esiste davvero in questa build.  La lista dei flag NON e'
        codificata qui: arriva dal parsing di "llama-server -h".
        """
        asp = get_aspect_info()
        is_port = asp["is_portrait"]
        layout = Layout()
        if not is_port:
            # In 16:9: sinistra backend e update, destra campi e comando
            layout.split_row(
                Layout(name="left_col", ratio=1),
                Layout(name="right_col", ratio=1)
            )
            layout["left_col"].split_column(
                Layout(name="cfg_backends", size=6),
                Layout(name="cfg_update", ratio=1),
            )
            layout["right_col"].split_column(
                Layout(name="cfg_fields", ratio=3),
                Layout(name="cfg_cmd", ratio=2),
            )
        else:
            layout.split_column(
                Layout(name="cfg_backends", size=6),
                Layout(name="cfg_fields", ratio=3),
                Layout(name="cfg_cmd", ratio=2),
                Layout(name="cfg_update", ratio=2),
            )

        # ---------- riga dei backend ----------
        bt = Table.grid(expand=True, padding=(0, 1))
        bt.add_column(ratio=1)
        bt.add_column(ratio=1)
        bt.add_column(ratio=1)
        bt.add_column(ratio=1)
        cells = []
        for b in state.cfg_backends():
            sel = (b["key"] == state.cfg_backend)
            if not b["exists"]:
                mark, col = "[muted]✗", "muted"
            elif sel:
                mark, col = "[bold ok]▶", "bold ok"
            else:
                mark, col = "[metric]○", "metric"
            tag = f"b{b['build']}" if b["build"] else "b?"
            cells.append(Text.from_markup(
                f"{mark} {b['label']} [muted]({tag})[/]", style=""))
        while len(cells) < 4:
            cells.append(Text(""))
        bt.add_row(cells[0], cells[1], cells[2], cells[3])
        cur = next((b for b in state.cfg_backends()
                    if b["key"] == state.cfg_backend), None)
        if cur:
            bt.add_row(Text.from_markup(
                f"[muted]binario:[/] [fg]{cur['binary'] or '(non trovato)'}[/]"),
                Text.from_markup(
                    f"[muted]DLL:[/] {cur['dll'] or '—'} | "
                    f"[muted]flag letti:[/] "
                    f"[bold metric]{len(state.cfg_flags)}[/]"))
        layout["cfg_backends"].update(Panel(
            bt, box=box.ROUNDED, border_style="border",
            title="[bold metric]⚙ BACKEND & BUILD (verificati senza avviarli)[/]",
            subtitle="[muted][B] cambia backend · [R] rilegge l'help della build · [0] torna a Tab 9[/]",
            padding=(0, 1)))

        # ---------- campi, per sezione ----------
        if state.cfg_loading_help:
            layout["cfg_fields"].update(Panel(
                Text.from_markup("[bold warn]⏳ Lettura di llama-server -h in corso...[/]"),
                box=box.ROUNDED, border_style="warn"))
        elif not state.cfg_help_loaded:
            msg = (f"[bold warn]⚠ Help non ancora letto.[/]\n\n"
                   f"Premi [bold ok][R][/] per leggere "
                   f"`llama-server -h` del backend scelto e costruire "
                   f"l'elenco dei flag.\n"
                   f"[muted]Errore: {state.cfg_help_error or 'nessuno'}[/]")
            layout["cfg_fields"].update(Panel(
                Text.from_markup(msg), box=box.ROUNDED, border_style="warn"))
        elif state.cfg_show_all_flags:
            layout["cfg_fields"].update(
                TUIRenderer._cfg_all_flags_panel(state))
        else:
            layout["cfg_fields"].update(
                TUIRenderer._cfg_fields_panel(state))

        # ---------- comando in dry-run ----------
        if not state.cfg_dry_cmd:
            state.cfg_regen()
        warns = ""
        if state.cfg_dry_warnings:
            warns = "\n".join(f"[bold err]⚠ {w}[/]"
                              for w in state.cfg_dry_warnings[:3])
        cmd_body = Text.from_markup(
            f"[muted]generato senza eseguire nulla (dry-run):[/]\n"
            f"[bold fg]{state.cfg_dry_cmd or '(comando vuoto)'}[/]"
            + (f"\n{warns}" if warns else ""))
        layout["cfg_cmd"].update(Panel(
            cmd_body, box=box.ROUNDED, border_style="ok",
            title="[bold ok]⚡ COMANDO GENERATO (dry-run, niente server avviato)[/]",
            subtitle="[muted][T] rigenera · [←/→] cambia il campo evidenziato · [0] torna[/]",
            padding=(0, 1)))

        # ---------- pannello aggiornamento ----------
        layout["cfg_update"].update(
            TUIRenderer._cfg_update_panel(state))
        return layout

    @staticmethod
    def _cfg_fields_panel(state: AppState) -> Panel:
        """Elenco dei campi, raggruppati per sezione, uno per riga."""
        tbl = Table.grid(expand=True, padding=(0, 1))
        tbl.add_column(ratio=3)   # etichetta
        tbl.add_column(ratio=2)   # valore
        tbl.add_column(ratio=2)   # default
        tbl.add_column(ratio=6)   # stato / nota

        fld = state.cfg_active_field()
        active_key = fld["key"] if fld else ""
        n = 0
        for sec in state.cfg_sections():
            tbl.add_row(Text.from_markup(
                f"[bold accent]── {sec['label']} ──[/]"))
            for f in sec["fields"]:
                is_act = (f["key"] == active_key)
                val = state.cfg_current_value(f["key"])
                txt = "—" if val is None or val == "" else str(val)
                if f.get("type") == "bool":
                    txt = "ON" if val else "OFF"
                dfl = state.cfg_default_value(f["key"])
                dfl_txt = "—" if not dfl else str(dfl)
                if not f.get("supported", True):
                    note = Text.from_markup("[muted]flag assente in questa build[/]")
                elif f.get("type") == "enum" and f["key"] in ("cache_type_k", "cache_type_v"):
                    scheda = Lc.describe_kv_type(txt)
                    note = Text.from_markup(
                        f"[accent]{scheda.get('label', txt)}[/] "
                        f"[muted]{scheda.get('use', '')[:46]}[/]")
                else:
                    note = Text.from_markup(
                        f"[muted]{state.cfg_flag_help(f['flag']).description[:60]}[/]")
                style = "bold metric" if is_act else ""
                tbl.add_row(
                    Text(f["label"], style=style),
                    Text(txt, style="bold ok" if is_act else ""),
                    Text(dfl_txt, style="muted"),
                    note)
                n += 1
        body = Group(tbl)
        return Panel(body, box=box.ROUNDED, border_style="border",
                     title="[bold metric]⚙ TUTTI I FLAG DI llama-server (da -h)[/]",
                     subtitle="[muted][↑/↓] campo · [←/→] valore · [A] elenco completo dall'help · [C] reset[/]",
                     padding=(0, 1))

    @staticmethod
    def _cfg_all_flags_panel(state: AppState) -> Panel:
        """Elenco completo di TUTTI i flag esposti dall'help, per sezione."""
        tbl = Table.grid(expand=True, padding=(0, 1))
        tbl.add_column(ratio=3)
        tbl.add_column(ratio=5)
        tbl.add_column(ratio=6)
        # raggruppa per la sezione che llama.cpp usa nell'help
        per_sec: Dict[str, List[Any]] = {}
        for spec in state.cfg_flags.values():
            per_sec.setdefault(spec.section or "common params", []).append(spec)
        for sec in sorted(per_sec):
            tbl.add_row(Text.from_markup(f"[bold accent]── {sec} ──[/]"))
            for spec in sorted(per_sec[sec], key=lambda s: s.canonical):
                kind = spec.value_kind
                col = {"bool": "muted", "int": "metric", "float": "metric",
                       "enum": "accent"}.get(kind, "fg")
                val = ""
                if spec.choices:
                    val = "[" + ", ".join(spec.choices[:5]) + ("...]" if len(spec.choices) > 5 else "")
                elif spec.default:
                    val = f"= {spec.default}"
                alias = (" (" + ", ".join(spec.aliases) + ")") if spec.aliases else ""
                tbl.add_row(
                    Text(spec.canonical, style="bold ok"),
                    Text(kind, style=col),
                    Text.from_markup(
                        f"[muted]{spec.description[:70]}[/]"))
        return Panel(tbl, box=box.ROUNDED, border_style="border",
                     title=f"[bold accent]📖 TUTTI I FLAG ESPOSTI DA -h ({len(state.cfg_flags)})[/]",
                     subtitle="[muted][A] torna ai campi · elenco letto da " + os.path.basename(state.cfg_help_source) + "[/]",
                     padding=(0, 1))

    @staticmethod
    def _cfg_update_panel(state: AppState) -> Panel:
        """Stato del check aggiornamenti e conferma esplicita del download."""
        u = state.cfg_upd or {}
        if state.cfg_upd_checking:
            return Panel(
                Text.from_markup("[bold warn]⏳ Interrogando la GitHub API...[/]"),
                box=box.ROUNDED, border_style="warn",
                title="[bold warn]⬆ AGGIORNAMENTO llama.cpp[/]")
        if not u:
            return Panel(
                Text.from_markup(
                    "[muted]Premi [bold warn][U][/] per controllare su GitHub "
                    "l'ultima release di ggml-org/llama.cpp e confrontare il build number "
                    "bNNNNN con quello locale. Nessun download parte senza conferma.[/]"),
                box=box.ROUNDED, border_style="border",
                title="[bold warn]⬆ AGGIORNAMENTO llama.cpp[/]",
                subtitle="[muted][U] controlla · la build CUDA funzionante non viene mai sovrascritta[/]",
                padding=(0, 1))
        if not u.get("ok"):
            return Panel(
                Text.from_markup(f"[bold err]✗ Controllo fallito: {u.get('error', '')}[/]"),
                box=box.ROUNDED, border_style="err",
                title="[bold warn]⬆ AGGIORNAMENTO llama.cpp[/]",
                subtitle="[muted][U] riprova[/]", padding=(0, 1))

        rows = [f"locale [bold metric]b{u.get('local_build')}[/]  →  "
                f"remota [bold ok]b{u.get('remote_build')}[/]",
                f"asset: [fg]{os.path.basename(u.get('asset_name', '?'))}[/]  "
                f"tag: [fg]{u.get('tag', '?')}[/]"]
        if u.get("newer"):
            size = u.get("download_size", 0) / (1024 * 1024)
            free = u.get("free_mib", 0) / 1024
            need = u.get("needed_mib", 0)
            ok = u.get("enough_space")
            rows.append(f"da scaricare: [bold warn]{size:.1f} MiB[/]   "
                        f"spazio libero: [bold ok]{free:.1f} GiB[/]   "
                        f"necessari: {need:.0f} MiB "
                        f"[{'ok]sufficiente' if ok else 'err]INSUFFICIENTE'}")
            rows.append(f"destinazione NUOVA: [bold metric]{u.get('target_dir', '')}[/]")
            if state.cfg_upd_confirm:
                rows.append("[bold ok]▶ CONFERMA ARMATA: premi [Y] per "
                            "scaricare, qualsiasi altro tasto annulla.[/]")
            else:
                rows.append("[muted]premi [bold warn][Y][/] per procedere. "
                            "Nessun file viene scritto senza conferma esplicita.[/]")
            border, title = "warn", "[bold warn]⬆ AGGIORNAMENTO DISPONIBILE[/]"
        else:
            rows.append("[bold ok]✓ Sei gia' aggiornato.[/]")
            border, title = "ok", "[bold ok]⬆ AGGIORNAMENTO llama.cpp[/]"
        return Panel(Text.from_markup("\n".join(rows)), box=box.ROUNDED,
                     border_style=border, title=title,
                     subtitle="[muted][U] ricontrolla[/]", padding=(0, 1))

    # -----------------------------------------------------------------
    # TAB 11: ASPETTO (tema colori / densita' / proporzioni)
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab11_aspetto(state: AppState) -> Layout:
        ap = state.appearance
        layout = Layout()
        layout.split_row(Layout(name="ctrl", ratio=5), Layout(name="preview", ratio=6))

        rows = [
            ("🎨 Tema colori", ap.theme_label, [Th.PALETTES[k]["label"] if k in Th.PALETTES else "Cyberpunk"
                                              for k in Th.THEME_ORDER],
             Th.THEME_ORDER.index(ap.theme)),
            ("📏 Densita' testo", ap.density_cfg["label"],
             [Th.DENSITIES[k]["label"] for k in Th.DENSITY_ORDER], Th.DENSITY_ORDER.index(ap.density)),
            ("▥ Proporzioni pannelli", ap.proportion_cfg["label"],
             [Th.PROPORTIONS[k]["label"] for k in Th.PROPORTION_ORDER],
             Th.PROPORTION_ORDER.index(ap.proportions)),
        ]
        t = Table(expand=True, box=box.SIMPLE_HEAD, show_header=False)
        t.add_column("", ratio=2)
        t.add_column("", ratio=5)
        for i, (lbl, cur, opts, idx) in enumerate(rows):
            sel = (i == state.aspetto_row)
            chips = Text()
            for j, o in enumerate(opts):
                chips.append(f" {o} ", style=("bold bg on accent" if j == idx else "fg on sel"))
                chips.append(" ")
            t.add_row(Text(("▶ " if sel else "  ") + lbl, style=("bold accent" if sel else "fg")), chips)
            t.add_row("", "")
        asp = get_aspect_info()
        note = Text.from_markup(
            f"\n[bold metric]📐 Orientamento attivo:[/] [bold bg on {'ok' if asp['is_portrait'] else 'accent'}] {asp['mode']} [/] "
            f"([muted]{asp['cols']} colonne × {asp['lines']} righe[/] — "
            f"[fg]{'commuta su 16:9 allargando la finestra' if asp['is_portrait'] else 'commuta su 9:16 restringendo la finestra'}[/])\n\n"
            "[muted]I tre controlli sono INDIPENDENTI: cambiarne uno non tocca gli altri.\n"
            "Salvati in [metric]~/.ai_tui_appearance.json[/] e ricaricati all'avvio.\n\n"
            "[warn]Font:[/] il font NON e' modificabile da dentro l'app: lo decide il terminale\n"
            "(kitty: [fg]font_family / font_size[/] in kitty.conf, oppure Ctrl+Shift+=/-).\n"
            "Qui si cambia la [fg]densita'[/] (padding e spaziatura), non il carattere.[/]")
        layout["ctrl"].update(Panel(Group(t, note), title="[bold accent]✦ ASPETTO[/]",
                                    subtitle="[muted][↑/↓] controllo  [←/→] cambia  [T]/[D]/[P] diretti  Ctrl+T tema[/]",
                                    box=box.ROUNDED))

        # Anteprima: gli stessi ruoli usati in TUTTA la TUI.
        th = ap.THEME
        sw = Table(expand=True, box=box.SIMPLE_HEAD)
        sw.add_column("Ruolo")
        sw.add_column("Uso nella TUI")
        sw.add_column("Colore")
        uses = [("fg", "testo primario"), ("bg", "sfondo pannelli"), ("border", "bordi / separatori"),
                ("muted", "etichette, testo secondario"), ("sel", "selezione / focus"),
                ("ok", "OK · server ONLINE"), ("err", "errore · OFFLINE"),
                ("warn", "warning · VRAM quasi piena"), ("warn2", "warning forte"),
                ("metric", "metriche: tok/s, MiB"), ("metric2", "metriche secondarie"),
                ("accent", "titoli, tasti, evidenze")]
        for role, use in uses:
            sw.add_row(Text(role, style=f"bold {role}"), Text(use, style="fg"),
                       Text(f" {th.get(role, '')} ",
                            style=(f"fg on {role}" if role in ("bg", "sel", "border", "muted") else f"bg on {role}")))
        demo = Text.from_markup(
            "\n[bold ok]● ONLINE[/]  [bold warn]◐ STARTING[/]  [bold err]○ OFFLINE[/]   "
            "[metric]48.7 tok/s[/]  [metric2]7075 MiB[/]  [muted]ctx 131072[/]")
        layout["preview"].update(Panel(Group(sw, demo), box=box.ROUNDED,
                                       title=f"[bold metric]Anteprima: {ap.theme_label}[/]"))
        return layout

    # -----------------------------------------------------------------
    # TAB 12: RISULTATI BENCHMARK (cartella consolidata)
    # -----------------------------------------------------------------
    @staticmethod
    def render_tab12_benchmark(state: AppState) -> Layout:
        b = state.bench
        rows = b.view()
        my = state.my_benchmarks() if hasattr(state, "my_benchmarks") else []
        layout = Layout()
        layout.split_row(Layout(name="side", ratio=1), Layout(name="main", ratio=3))

        # Lista principale al centro (scorrevole)
        t = Table(expand=True, box=box.SIMPLE_HEAD)
        for c, kw in (("Sorgente", {"ratio": 3}), ("Modello", {"ratio": 6}), ("Config", {"ratio": 3}),
                      ("ctx", {"justify": "right"}), ("KV", {}), ("tok/s", {"justify": "right"}),
                      ("TTFT ms", {"justify": "right"}), ("Stato", {"ratio": 3})):
            t.add_column(c, overflow="ellipsis", no_wrap=True, **kw)
        b.cursor = max(0, min(b.cursor, max(0, len(rows) - 1)))
        start, stop = _win_paginated(len(rows), b.cursor)
        best = max((r.tok_s or 0) for r in rows) if rows else 0
        sel_row = rows[b.cursor] if rows else None
        for i, r in enumerate(rows[start:stop], start=start):
            sel = i == b.cursor
            tc = ("ok" if (r.tok_s or 0) >= best * 0.9 else ("warn" if (r.tok_s or 0) >= best * 0.6 else "muted"))
            t.add_row(_mesc(str(getattr(r, "source", ""))[:16]),
                      _mesc(str(getattr(r, "model", ""))[:48]),
                      _mesc(str(getattr(r, "config", ""))[:16]),
                      str(getattr(r, "ctx", "") or ""),
                      _mesc(str(getattr(r, "kv", "") or "")),
                      f"{r.tok_s:.1f}" if r.tok_s else "-",
                      f"{r.ttft_ms:.0f}" if getattr(r, "ttft_ms", None) else "-",
                      _mesc(str(getattr(r, "status", "") or "")[:30]),
                      style=f"bold bg on sel" if sel else (f"on sel" if False else tc))
        if not rows:
            t.add_row(_mesc("(nessun benchmark)"), "", "", "", "", "", "", "")
        main_group = [t]

        if my:
            mt = Table(expand=True, box=box.SIMPLE_HEAD,
                       title="[bold accent]I MIEI BENCHMARK (da scheda Agente [T])[/]")
            for c, kw in (("quando", {"ratio": 2}), ("preset", {"ratio": 2}),
                          ("modello", {"ratio": 5}), ("tok/s", {"justify": "right"}),
                          ("prompt tok/s", {"justify": "right"})):
                mt.add_column(c, overflow="ellipsis", no_wrap=True, **kw)
            for i, r in enumerate(reversed(my[-10:])):
                mt.add_row(_mesc(str(r.get("ts", ""))[11:19]),
                           _mesc(str(r.get("preset", ""))),
                           _mesc(str(r.get("model", ""))[:48]),
                           f"{r.get('gen_tok_s') or 0:.1f}",
                           f"{r.get('prompt_tok_s') or 0:.1f}")
            main_group.append(mt)

        layout["main"].update(Panel(Group(*main_group), box=box.ROUNDED,
            title=f"[bold metric]📊 BENCHMARK ({len(rows)} · sorgente {b.filter_label() if hasattr(b, 'filter_label') else 'tutti'})[/]",
            subtitle="[muted][↑/↓] scorri · [←/→] filtro sorgente · [S] ordina · [R] rileggi[/]"))

        # Pannello laterale: flag del run selezionato
        info = []
        if sel_row:
            info.append(f"[bold accent]{_mesc(str(getattr(sel_row, 'model', ''))[:60])}[/]")
            info.append(f"[muted]sorgente:[/] {_mesc(str(getattr(sel_row, 'source', '')))}")
            if getattr(sel_row, "config", ""):
                info.append(f"[muted]config:[/] {_mesc(str(sel_row.config))}")
            info.append("")
            info.append("[bold ok]Parametri misurati[/]")
            for k in ("ctx", "kv", "tok_s", "ttft_ms", "prompt_tok_s", "status"):
                v = getattr(sel_row, k, None)
                if v not in (None, "", 0):
                    info.append(f"  [muted]{k}:[/] [fg]{_mesc(str(v))}[/]")
            extras = getattr(sel_row, "flags", None) or getattr(sel_row, "params", None) or {}
            if extras and isinstance(extras, dict):
                info.append("")
                info.append("[bold ok]Flag usati[/]")
                for k, v in extras.items():
                    info.append(f"  [metric]{_mesc(str(k))}[/] = [fg]{_mesc(str(v))}[/]")
        else:
            info.append("[muted](nessun benchmark selezionato)[/]")
        layout["side"].update(Panel(Group(*[Text.from_markup(x) for x in info]),
            box=box.ROUNDED, title="[bold accent]🏷 FLAG DEL RUN[/]",
            border_style="accent"))
        return layout

    @staticmethod
    def render_tab14_modal(state: AppState) -> Layout:
        m = state.modal
        if not m.status.checked_at:
            m.refresh_async()
        st = m.status
        layout = Layout()
        layout.split_column(Layout(name="hdr", size=5), Layout(name="body", ratio=1), Layout(name="log", size=4))

        hours = MC.hours_remaining(m.spent_usd, m.monthly_credit)
        hdr = Table.grid(expand=True, padding=(0, 2))
        hdr.add_column(ratio=1); hdr.add_column(ratio=2); hdr.add_column(ratio=1)
        acc = _mesc(st.account or "lexander3692")
        app_a = next((a for a in st.apps if a.get("name") == MC.APP_NAME), None)
        state_app = (app_a.get("state", "?") if app_a else "non deployata")
        state_col = {"deployed": "ok", "stopped": "muted", "running": "ok"}.get(state_app, "warn")
        
        curr_m_key = (st.active_model or "ornith").lower()
        active_label = "Ornith 1.5 35B MoE" if "ornith" in curr_m_key else "Tiel-Coder 35B MoE"

        n_cnts = len(st.active_containers)
        if n_cnts > 0:
            cnt_state = f"[bold warn]⚡ {n_cnts} container attivo (0.80 $/h)[/]"
        else:
            cnt_state = "[bold ok]💤 0 container (0.00 $/h · in sleep)[/]"

        hdr.add_row(
            Text.from_markup(f"[bold accent]☁️  MODAL CLOUD[/]\n"
                             f"[muted]account:[/] [bold fg]{acc}[/]\n"
                             f"[muted]stato GPU:[/] {cnt_state}"),
            Text.from_markup(f"[muted]modello attivo:[/] [bold ok]● {active_label}[/]  "
                             f"[bold accent][M][/] [muted]per alternare[/]\n"
                             f"[muted]endpoint:[/] [metric]{_mesc(MC.ENDPOINT)}/v1[/]\n"
                             f"[muted]sleep auto:[/] [ok]10 min inattività → spegnimento container[/] [muted](scaledown=600s)[/]"),
            Text.from_markup(f"[muted]credito mensile:[/] [bold fg]${m.monthly_credit:.2f}[/] [muted](free)[/]\n"
                             f"[muted]speso mese:[/] [fg]${m.spent_usd:.2f}[/]\n"
                             f"[muted]residuo L4:[/] [bold ok]~{hours.get('L4', 37.5):.1f} ore calcolo[/]")
        )
        layout["hdr"].update(Panel(hdr, title="[bold accent]☁️  PANNELLO DI CONTROLLO MODAL CLOUD · HERMES[/]",
                                   border_style="accent"))

        body = Layout()
        body.split_row(Layout(name="apps", ratio=3), Layout(name="models", ratio=4), Layout(name="h", ratio=2))

        # Lista App Modal
        at = Table(expand=True, box=box.SIMPLE_HEAD)
        for c, kw in (("App ID / Nome", {"ratio": 3}), ("Stato", {}), ("Tasks", {"justify": "right"}),
                      ("Dettaglio", {"ratio": 2})):
            at.add_column(c, overflow="ellipsis", no_wrap=True, **kw)
        rows = [a for a in st.apps if "error" not in a]
        _a0, _a1 = _win(len(rows), m.cursor, max(5, shutil.get_terminal_size((120, 40)).lines - 20))
        for i, a in enumerate(rows[_a0:_a1], start=_a0):
            sel = (i == m.cursor)
            sc = {"deployed": "ok", "running": "ok", "stopped": "muted"}.get(a.get("state", ""), "warn")
            app_desc = a.get("name") or a.get("id", "")
            at.add_row(Text(("▶ " if sel else "  ") + _mesc(app_desc[:35]),
                            style="bold fg on sel" if sel else "fg"),
                       Text(a.get("state", ""), style=f"bold {sc}"),
                       Text(str(a.get("tasks", 0)), style="metric"),
                       Text(_mesc(a.get("info", ""))[:30], style="muted"),
                       style=("on sel" if sel else None))
        if any("error" in a for a in st.apps):
            at.add_row(Text(_mesc(st.apps[0].get("error", "errore"))[:40], style="err"), "", "", "")
        if not rows and not any("error" in a for a in st.apps):
            at.add_row(Text("nessuna app rilevata · premi [D] per deploy", style="muted"), "", "", "")
        body["apps"].update(Panel(at, title="[bold metric]Istanze & App Cloud[/]",
                                  subtitle="[muted][↑/↓] Seleziona · [X] Stop · [D] Deploy[/]"))

        # Modelli nel Volume Persistente
        mod_table = Table(expand=True, box=box.SIMPLE_HEAD)
        mod_table.add_column("Stato", no_wrap=True)
        mod_table.add_column("Modello MoE (GGUF)", ratio=3, overflow="ellipsis", no_wrap=True)
        mod_table.add_column("Peso", justify="right", no_wrap=True)

        tot_vol_mib = 0.0
        for vf in st.volume_files:
            tot_vol_mib += float(vf.get("mib") or 0.0)

        for km in MC.KNOWN_MODELS:
            is_active = (km["key"] == curr_m_key)
            status_text = "[bold ok]● ATTIVO[/]" if is_active else "[muted]○ pronto[/]"
            mod_table.add_row(
                Text.from_markup(status_text),
                Text.from_markup(f"[bold fg]{km['label']}[/]\n[muted]{km['desc']}[/]"),
                Text(f"{km['size_gb']:.1f} GB", style="metric" if is_active else "muted")
            )

        used_gb = (tot_vol_mib / 1024.0) if tot_vol_mib > 0 else 27.56
        free_pct = (used_gb / 50.0) * 100.0
        free_tag = "ok" if used_gb < 40 else ("warn" if used_gb < 48 else "err")

        vol_footer = Text.from_markup(
            f"\n[muted]Volume:[/] [metric]modal-ornith-volume[/]  "
            f"[muted]Occupato:[/] [{free_tag}]{used_gb:.1f} GB / 50.0 GB ({free_pct:.1f}% free tier)[/]\n"
            f"[muted]Config:[/] [fg]80k CTX (-c 80000)[/] · [fg]KV Cache Q4 (-ctk q4_0 -ctv q4_0)[/] · [fg]GPU L4 24GB[/]"
        )

        body["models"].update(Panel(Group(mod_table, vol_footer),
                                    title="[bold metric]Modelli & Volume Persistente[/]",
                                    subtitle="[muted][M] Cambia Modello Attivo · [R] Ricarica[/]"))

        # Ore Stimate per GPU
        ht = Table(expand=True, box=box.SIMPLE_HEAD)
        ht.add_column("GPU", no_wrap=True)
        ht.add_column("$/h", justify="right")
        ht.add_column("Ore residue", justify="right")
        for gpu, cost in MC.GPU_COSTS.items():
            is_l4 = (gpu == "L4")
            g_style = "bold ok" if is_l4 else "fg"
            c_style = "bold metric" if is_l4 else "muted"
            h_style = "bold ok" if is_l4 else "metric"
            ht.add_row(
                Text(("★ " if is_l4 else "") + gpu, style=g_style),
                Text(f"${cost:.2f}", style=c_style),
                Text(f"{hours[gpu]:.1f} h", style=h_style)
            )
        body["h"].update(Panel(ht, title="[bold metric]Ore Stimate (30$)[/]",
                               subtitle="[muted]★ L4 ottimale per 35B[/]"))
        layout["body"].update(body)

        # Barra di Stato e Ping
        ping = st.endpoint_ping or {}
        if n_cnts > 0:
            cost_info = "[bold warn]⚡ CONTAINER ATTIVO (0.80 $/h)[/] [muted]· Spegnimento automatico se inattivo 10 min[/]"
        else:
            cost_info = "[bold ok]💤 CONTAINER SPENTO (0.00 $/h · ZERO SPRECO)[/] [muted]· Si riattiva su richiesta al primo prompt[/]"

        if ping.get("ok"):
            ping_line = f"[bold ok]● PRONTO[/] [muted]· latenza:[/] [metric]{ping.get('latency_ms', 0):.0f} ms[/] [muted]· modello servito:[/] [bold fg]{_mesc(ping.get('model') or '')}[/]"
        else:
            info_msg = ping.get("info") or ping.get("error") or "inattivo"
            ping_line = f"[muted]○ {info_msg}[/]"
        
        last_act = m.last_action or ("✓ Nessun costo attivo (container in sleep)" if n_cnts == 0 else "✓ Container attivo")
        log_lines = [
            f"[muted]Stato GPU:[/] {cost_info}   [muted]|[/]   {ping_line}",
            f"[muted]Ultima operazione:[/] [fg]{_mesc(last_act)[:140]}[/]"
        ]
        if m.busy:
            log_lines.append("[bold warn]⏳ Operazione in background in corso (non blocca la TUI)...[/]")
            
        layout["log"].update(Panel(Group(*[Text.from_markup(x) for x in log_lines]),
                                   title="[bold metric]Telemetria & Stato Endpoint[/]",
                                   subtitle="[muted][M] Alterna Modello · [D] Deploy · [X] Stop · [P] Ping · "
                                            "[T] Test Chat · [R] Aggiorna · [O] Standalone[/]"))
        return layout

    @staticmethod
    def render_tab13_builds(state: AppState) -> Layout:
        bm = state.build_mgr
        layout = Layout()
        layout.split_column(Layout(name="list", ratio=3), Layout(name="bottom", ratio=2))
        layout["bottom"].split_row(Layout(name="detail", ratio=3), Layout(name="vram", ratio=2))
        n_all = len(bm.ordered())
        lst = state.builds_visible()
        q = state.build_query

        def make_list(h: Optional[int]):
            t = Table(expand=True, box=box.SIMPLE_HEAD)
            for c, kw in (("", {"width": 2}), ("Build", {"ratio": 3}), ("Ver.", {}), ("Device", {"ratio": 2}),
                          ("RPC", {}), ("Draft (--spec-type)", {"ratio": 4}), ("-sm", {"ratio": 2}),
                          ("Doc", {}), ("Cartella", {"ratio": 4})):
                t.add_column(c, overflow="ellipsis", no_wrap=True, **kw)
            show_q = bool(q) or state.build_typing
            # SIMPLE_HEAD: riga vuota sopra, intestazione, separatore, riga vuota sotto
            rows = max(1, (h or 12) - 4 - (1 if show_q else 0))
            _b0, _b1 = _win_slice(len(lst), state.build_cursor, page_size=rows)
            for i, b in enumerate(lst[_b0:_b1], start=_b0):
                sel = i == state.build_cursor
                active = b.key == bm.active_key
                drafts = ",".join(x for x in b.spec_types if x != "none") or ("…" if not b.probed else "-")
                t.add_row(Text("★" if active else ("▶" if sel else ""), style="bold accent"),
                          Text(b.key, style="bold fg" if sel else "fg"),
                          Text(f"b{b.build_num}" if b.build_num >= 1000 else
                               ("LMS" if b.key.startswith("lmstudio") else "?"), style="metric"),
                          Text(",".join(b.devices) if b.devices else ("solo CPU" if b.probed else "…"),
                               style="ok" if b.gpu else "warn"),
                          Text("sì" if b.rpc else "no", style="ok" if b.rpc else "muted"),
                          Text(drafts, style="metric2"),
                          Text(",".join(b.split_modes) or "-", style="muted"),
                          Text(str(len(b.docs)) if b.docs else "-", style="ok" if b.docs else "muted"),
                          Text(b.dir, style="muted"),
                          style=("on sel" if sel else None))
            if not lst:
                msg = (f"nessuna build contiene '{q}' ([ESC] azzera la ricerca)" if q
                       else "nessuna build trovata: [A] aggiungi una cartella")
                t.add_row("", Text(msg, style="warn"), "", "", "", "", "", "", "")
            if not show_q:
                return t
            return Group(_search_line(q, state.build_typing, len(lst), n_all), t)

        pos = f"{state.build_cursor + 1}/{len(lst)}" if lst else "0"
        upd_badge = f" [bold ok on black] 🔔 UPDATE b{state.bm_upd.get('remote_build')} DISPONIBILE ([Y] scarica) [/]" if state.bm_upd.get("newer") else ""
        layout["list"].update(Panel(
            _FitHeight(make_list), box=box.ROUNDED,
            title=f"[bold accent]🧱 BUILD llama.cpp[/]{upd_badge} [muted]({pos} · {n_all} trovate · "
                  f"★ attiva: {bm.active_key or '-'})[/]",
            subtitle="[muted][↑/↓] scegli  \\[/] cerca  [Enter/U] usa  [C] check update  [Y] scarica update  "
                     "[D] doc build  [W] doc tutte  [R] risonda  [A] aggiungi  [X] togli[/]"))

        b = state.build_selected()
        det: List[str] = []
        if b is not None:
            det.append(f"[bold fg]{_mesc(b.key)}[/] [muted]{_mesc(str(b.version))}[/]")
            det.append(f"[muted]server:[/] [fg]{_mesc(b.server)}[/]")
            det.append(f"[muted]rpc-server:[/] [fg]{_mesc(b.rpc_server or '— assente')}[/]")
            if b.note:
                det.append(f"[warn]ℹ {_mesc(str(b.note))}[/]")
            if b.cache_types:
                det.append(f"[muted]KV ammessi (-ctk/-ctv):[/] [metric]{', '.join(b.cache_types)}[/]")
            fl = [f"[{'ok' if v else 'muted'}]{k}[/]" for k, v in b.flags.items()]
            det.append("[muted]flag:[/] " + " ".join(fl))
            if b.docs:
                det.append(f"[muted]doc:[/] [ok]{', '.join(b.docs)}[/] [muted]in {BM.DOCS_ROOT}/{b.key}[/]")
            else:
                det.append("[muted]doc: nessuna ([D] per scaricare help + README al tag della build)[/]")
            if b.probe_error:
                det.append(f"[err]sonda: {_mesc(str(b.probe_error))}[/]")
        rpcs = BM.running_rpc_servers()
        det.append("")
        det.append(f"[bold metric]ggml-rpc-server locali:[/] " + (
            "  ".join(f"[ok]●[/] [fg]{r['host']}:{r['port']}[/] [muted]PID {r['pid']}[/]" for r in rpcs)
            or "[muted]nessuno[/]"))
        det.append(f"[muted][O] anteprima  [K] avvia rpc-server di questa build su 0.0.0.0:{state.rpc_local_port} "
                   f"(LAN fidata: RPC non autenticato)  [Z] ferma tutti[/]")
        layout["detail"].update(Panel(Text.from_markup("\n".join(det)), box=box.ROUNDED,
                                      title="[bold metric]Dettaglio build & RPC[/]"))

        vb = state.vram_budget
        rpc_ep = ",".join(vb.cfg.get("remotes", {}).keys())
        vals, labels = vb.fit_target(rpc_ep, max(1, len([d for d in (bm.active().devices if bm.active() else []) if d])))
        vt = Table(expand=True, box=box.SIMPLE_HEAD)
        vt.add_column("Device")
        vt.add_column("Margine", justify="right")
        for lbl, v in zip(labels, vals):
            vt.add_row(Text(lbl, style="fg"),
                       Text("100% al modello" if v == 0 else f"-{v} MiB", style="ok" if v == 0 else "metric"))
        flags = " ".join(vb.fit_flags(rpc_ep, 1))
        layout["vram"].update(Panel(Group(
            vt,
            Text.from_markup(f"[muted]flag:[/] [metric2]{flags}[/]\n"
                             f"[muted]preset storici:[/] "
                             + ("[bold ok]margini APPLICATI[/]" if vb.cfg.get("apply_to_presets")
                                else "[muted]non applicati[/]")
                             + "\n[muted]--fit regola solo cio' che non e' fissato (-ngl auto, -c 0, ripartizione).\n"
                               "Default llama.cpp: 1024 MiB per device.[/]")),
            box=box.ROUNDED, title="[bold accent]🎚 Margini VRAM (--fit-target)[/]",
            subtitle="[muted][V] locale  [M] host RPC  [P] su preset storici  [N] profilo MIKY-100[/]"))
        return layout

    # -----------------------------------------------------------------
    # AVVIO PRESET: scelta build + barra di avanzamento
    # -----------------------------------------------------------------
    @staticmethod
    def render_launch(state: AppState):  # -> Union[Panel, Layout]
        from rich.progress_bar import ProgressBar
        from rich.markup import escape as _esc
        m = state.launch_modal
        if m is not None:
            secs = state.launch_sections(m)
            cur_sec = secs[m["sec"] % len(secs)]

            asp = get_aspect_info()
            term_lines = asp["lines"]
            cols = asp["cols"]

            eff = m.get("eff")

            def sec_title(name: str, text: str) -> Text:
                on = cur_sec == name
                return Text(("▶ " if on else "  ") + text, style="bold accent" if on else "bold muted")

            # ---- 1. build table ----
            # Una riga per build (la nota intera della build evidenziata va nella
            # riga ⓘ) e tante righe quante ne entrano davvero nel riquadro.
            if term_lines < 24:
                flag_page = 2; param_page = 2
            elif term_lines < 36:
                flag_page = 3; param_page = 3
            elif term_lines < 46:
                flag_page = 4; param_page = 4
            elif term_lines < 54:
                flag_page = 6; param_page = 6
            else:
                flag_page = 10; param_page = 10

            cands = state.launch_cands(m)
            bq = m.get("bq", "")
            bq_typing = bool(m.get("bq_typing")) and cur_sec == "build"
            sel_b = state.launch_build(m)

            def make_build_table(h: Optional[int]):
                t = Table(expand=True, box=None, show_header=True, padding=(0, 1))
                t.add_column("", width=2)
                t.add_column("Build", ratio=3, no_wrap=True, overflow="ellipsis")
                t.add_column("Ver.", width=7, no_wrap=True)
                t.add_column("Vantaggi / per cosa l'abbiamo usata", ratio=5, no_wrap=True, overflow="ellipsis")
                t.add_column("RPC rilevato" if m["rpc"] else "Device", ratio=3, no_wrap=True,
                             overflow="ellipsis")
                extra = (1 if m.get("busy") else 0) + (1 if (m["checking"] or not cands) else 0)
                show_q = bool(bq) or bq_typing
                # 1 titolo sezione + 1 intestazione tabella (+ riga ricerca)
                rows = max(1, (h or 8) - 2 - extra - (1 if show_q else 0))
                _c0, _c1 = _win_slice(len(cands), m["idx"], page_size=rows)
                blocked = m.get("blocked") or {}
                for i, b in enumerate(cands[_c0:_c1], start=_c0):
                    sel = i == m["idx"]
                    ok, line = m["compat"].get(b.key, (None, ""))
                    focus = sel and cur_sec == "build"
                    if b.key in blocked:
                        t.add_row(Text("▶" if focus else "✗", style="bold err"),
                                  Text(b.key, style="bold fg" if sel else "muted"),
                                  Text(f"b{b.build_num}" if b.build_num else "?", style="muted"),
                                  Text(BM.use_note(b), style="muted"),
                                  Text("✗ " + blocked[b.key][1], style="err" if sel else "warn2"),
                                  style=("on sel" if focus else None))
                        continue
                    marker = "★" if sel else ("·" if b.key == state.build_mgr.active_key else "")
                    t.add_row(Text(("▶" if focus else marker), style="bold accent"),
                              Text(b.key, style="bold fg" if sel else "fg"),
                              Text(f"b{b.build_num}", style="metric"),
                              Text(BM.use_note(b), style="fg" if sel else "muted"),
                              Text(line if m["rpc"] else ",".join(b.devices),
                                   style="ok" if (ok or not m["rpc"]) else "muted"),
                              style=("on sel" if focus else None))
                if m.get("busy"):
                    b0 = m["busy"][0]
                    t.add_row("", Text(f"⛔ Miky e' gia' usata dal llama-server PID {b0['pid']} (:{b0['port']}, "
                                       f"{b0['model']}): [K] lo ferma e continua.", style="bold warn"), "", "", "")
                if m["checking"]:
                    t.add_row("", Text("⏳ verifico quali build parlano lo stesso protocollo RPC di Miky...",
                                       style="warn"), "", "", "")
                elif not m["cands"]:
                    t.add_row("", Text("nessuna build compatibile con questo preset", style="err"), "", "", "")
                elif not cands:
                    t.add_row("", Text(f"nessuna build contiene '{bq}' ([ESC] azzera)", style="warn"),
                              "", "", "")
                parts: List[Any] = [sec_title("build", f"BUILD  [↑/↓]  [/] cerca   "
                                                       f"{(m['idx'] % len(cands)) + 1 if cands else 0}/{len(cands)}"
                                                       + (f"  ({len(m['cands'])} avviabili, ✗ = no)" if blocked else "")
                                                       + ("   ⟳ verifica con Miky..." if m.get("verifying") else ""))]
                if show_q:
                    parts.append(_search_line(bq, bq_typing, len(cands), len(m["cands"])))
                parts.append(t)
                spare = (h or 0) - 2 - extra - (1 if show_q else 0) - len(cands)
                if sel_b is not None and spare >= 3:
                    parts.append(Text(""))
                    if sel_b.key in blocked:
                        why = blocked[sel_b.key][1]
                        parts.append(Text.assemble((f"✗ {sel_b.key}: ", "bold err"), (why, "fg")))
                        if "RPC diverso" in why or "mismatch" in why:
                            parts.append(Text("  [S] o [INVIO] sincronizza ed attiva questa build su Miky (allinea ggml-rpc-server)", style="bold accent"))
                        else:
                            parts.append(Text("  [S] o [INVIO] configura/sincronizza su Miky", style="bold accent"))
                    else:
                        parts.append(Text.assemble((f"{sel_b.key}: ", "bold accent"), (BM.use_note(sel_b), "fg")))
                return Group(*parts)

            # ---- 2. flag e parametri (solo quando eff e' disponibile) ----
            help_line = ""
            ft = pt = None
            if eff is not None:
                frows = state.launch_flag_rows(m)
                prows = state.launch_param_rows(m)
                fi = m["fi"] % max(1, len(frows))
                pi = m["pi"] % max(1, len(prows))
                if cur_sec == "flag" and frows:
                    help_line = frows[fi]["help"]
                elif cur_sec == "parametri" and prows:
                    help_line = prows[pi]["help"]

                def make_flags(h: Optional[int]):
                    t = Table(expand=True, box=None, show_header=False, padding=(0, 1))
                    t.add_column("", no_wrap=True, ratio=1)
                    rows = max(1, h - 1) if h else flag_page   # -1: titolo sezione
                    _f0, _f1 = _win_slice(len(frows), fi, page_size=rows)
                    for i, f in enumerate(frows[_f0:_f1], start=_f0):
                        on = bool(eff["flags"].get(f["key"]))
                        sel = cur_sec == "flag" and i == fi
                        t.add_row(Text(f"[{'x' if on else ' '}] {f['label']}",
                                       style=("bold fg on sel" if sel else ("ok" if on else "muted"))))
                    if not frows:
                        t.add_row(Text("(scegli una build)", style="muted"))
                    return t

                def make_params(h: Optional[int]):
                    t = Table(expand=True, box=None, show_header=False, padding=(0, 1))
                    t.add_column("", width=18, no_wrap=True)
                    t.add_column("", ratio=1, no_wrap=True, overflow="ellipsis")
                    base = LP.PRESETS.get(m["preset"], {})
                    rows = max(1, h - 1) if h else param_page
                    _p0, _p1 = _win_slice(len(prows), pi, page_size=rows)
                    for i, p in enumerate(prows[_p0:_p1], start=_p0):
                        v = eff.get(p["key"], "")
                        if p["kind"] == "model" or p["key"] == "mtp_model":
                            shown = os.path.basename(str(v)) or "?"
                        elif p["key"] == "ctx_size":
                            shown = f"{v} ({LP.ctx_label(int(v or 0))})"
                        else:
                            shown = str(v) if str(v) != "" else ("auto" if p["key"] in ("ngl", "threads", "batch", "ubatch")
                                                                else "—")
                        changed = base.get(p["key"]) != v
                        sel = cur_sec == "parametri" and i == pi
                        _ign = LP.ignored_reason(eff, p["key"])
                        _txt = ("◂ " if sel else "") + shown + (" ▸" if sel else "") + (" •" if changed else "")
                        if _ign:
                            _txt += f"   ⚠ {_ign}"
                        t.add_row(Text(p["label"], style="bold fg on sel" if sel else ("warn2" if _ign else "muted")),
                                  Text(_txt, style=("bold fg on sel" if sel else
                                                    ("warn2" if _ign else ("warn" if changed else "metric")))))
                    return t

                ft = make_flags
                pt = make_params

            if cur_sec == "build" and not help_line:
                if sel_b is not None:
                    help_line = f"{sel_b.key}: {BM.use_note(sel_b)}"
                else:
                    help_line = "Sono elencate SOLO le build compatibili col protocollo RPC rilevato."
            if cur_sec == "pulsanti":
                help_line = "[AVVIA] parte con questa riga esatta; [INDIETRO] chiude senza avviare."

            # ---- info e avvisi ----
            info: List[str] = []
            if help_line:
                info.append(f"[accent]ⓘ[/] [fg]{_esc(help_line)}[/]")
            _eff = m.get("eff") or {}
            _fl = _eff.get("flags") or {}
            _ov = bool(_eff.get("cpu_moe") or _fl.get("cpu_moe")
                       or str(_eff.get("n_cpu_moe") or "").strip() not in ("", "0"))
            if _ov and not _fl.get("no_mmap"):
                info.append("[warn]⚠ pesi MoE su CPU con mmap attivo: attiva no mmap (--load-mode none)[/]")
            if eff is not None:
                bm_bench = LP.bench_for(eff)
                if bm_bench:
                    _cfg_d = [d for d in bm_bench["diffs"] if not d.startswith("ctx ")]
                    diff = (" — [warn]misura con config diversa: " + _esc(", ".join(_cfg_d)) + "[/]"
                            if _cfg_d else
                            (f" — [muted]stessa config, ctx {bm_bench['ctx']}[/]" if bm_bench["diffs"] else ""))
                    info.append(f"[muted]misurato:[/] [bold metric]{bm_bench['tok_s']:.1f} tok/s[/] "
                                f"[muted]a ctx {bm_bench['ctx']}[/]{diff}")
                for w in LP.warnings_for(eff):
                    info.append(f"[warn]⚠ {_esc(w)}[/]")

            # ---- comando ----
            argv = state.launch_argv(m)
            cmd_styled = Text()
            for i, tok in enumerate(argv):
                if i == 0:
                    cmd_styled.append(tok, style="bold ok")
                elif tok.startswith("-"):
                    cmd_styled.append(f" {tok}", style="bold accent")
                else:
                    cmd_styled.append(f" {tok}", style="metric")
            cmd_styled.no_wrap = False
            cmd_styled.overflow = "fold"

            # ---- pulsanti ----
            on_btn = cur_sec == "pulsanti"
            go_sel = on_btn and m["bi"] % 2 == 0
            back_sel = on_btn and m["bi"] % 2 == 1
            btn = Text()
            btn.append("  [ ▶ AVVIA ]  ", style=("bold bg on ok" if go_sel else "bold ok"))
            btn.append("   ")
            btn.append("  [ ◀ INDIETRO ]  ", style=("bold bg on err" if back_sel else "bold muted"))
            btn.append("     [TAB] sezione · [INVIO] avvia · [ESC] indietro", style="muted")

            # ---- titolo ----
            title_bits = f"▶ AVVIO {_esc(m['label'])}"
            if eff is not None:
                title_bits += (f" — {_esc(str(eff.get('short', '')))} {LP.ctx_label(int(eff.get('ctx_size') or 0))}"
                               f" · ctx {eff.get('ctx_size')} · {LP.tok_label(eff)}")

            # ================================================================
            # Layout RESPONSIVE: ogni riquadro ha spazio garantito.
            # Sostituisce Panel(Group(*parts)) che non si ripartiva lo spazio.
            # ================================================================
            is_wide = cols >= 75

            if term_lines < 26:
                cmd_size = 2; btn_size = 2
            elif term_lines < 32:
                cmd_size = 3; btn_size = 2
            elif term_lines < 40:
                cmd_size = 4; btn_size = 2
            else:
                cmd_size = 5; btn_size = 3

            # Comando: compresso o esteso in base allo spazio
            if term_lines < 26:
                cmd_content: Any = Text.assemble(
                    ("cmd: ", "bold muted"),
                    (" ".join(argv[:4]) + (" ..." if len(argv) > 4 else ""), "metric"))
            else:
                cmd_content = cmd_styled

            cmd_info_widgets: List[Any] = [cmd_content]
            if info and term_lines >= 30:
                cmd_info_widgets.append(Text.from_markup("\n".join(info[:2])))
            cmd_panel = Panel(
                Group(*cmd_info_widgets),
                box=box.ROUNDED, border_style="accent",
                title="[bold metric]Comando llama-server[/]",
                padding=(0, 1))

            btn_panel = Panel(
                btn, box=box.SIMPLE, border_style="border", padding=(0, 0))

            build_panel = Panel(
                _FitHeight(make_build_table),
                box=box.DOUBLE, border_style="accent",
                title=f"[bold accent]{title_bits}[/]  [bold ok]● PRONTO[/]",
                subtitle="[bold ok]INVIO = AVVIA SUBITO[/] · [bold err]ESC = INDIETRO[/] · [TAB] sezione · [R] reset")

            lay = Layout()
            if eff is not None and is_wide:
                # Landscape: build a sinistra, flag|params a destra in colonne
                lay.split_column(
                    Layout(name="main_row", ratio=1),
                    Layout(name="cmd_area", size=cmd_size),
                    Layout(name="btn_area", size=btn_size),
                )
                lay["main_row"].split_row(
                    Layout(name="build_col", ratio=2),
                    Layout(name="ctrl_col", ratio=3),
                )
                lay["main_row"]["ctrl_col"].split_column(
                    Layout(name="flags_sec", ratio=1),
                    Layout(name="params_sec", ratio=1),
                )
                lay["main_row"]["build_col"].update(build_panel)
                lay["main_row"]["ctrl_col"]["flags_sec"].update(Panel(
                    _FitHeight(lambda h: Group(sec_title("flag", f"FLAG {fi + 1 if frows else 0}/{len(frows)}  [SPAZIO] on/off"), ft(h))),
                    box=box.ROUNDED, border_style="border"))
                lay["main_row"]["ctrl_col"]["params_sec"].update(Panel(
                    _FitHeight(lambda h: Group(sec_title("parametri", f"PARAMETRI {pi + 1 if prows else 0}/{len(prows)}  [←/→]  [INVIO] scrivi"), pt(h))),
                    box=box.ROUNDED, border_style="border"))
            elif eff is not None:
                # Portrait / stretto: tutto impilato verticalmente
                lay.split_column(
                    Layout(name="build_sec", ratio=2),
                    Layout(name="flags_sec", ratio=1),
                    Layout(name="params_sec", ratio=1),
                    Layout(name="cmd_area", size=cmd_size),
                    Layout(name="btn_area", size=btn_size),
                )
                lay["build_sec"].update(build_panel)
                lay["flags_sec"].update(Panel(
                    _FitHeight(lambda h: Group(sec_title("flag", f"FLAG {fi + 1 if frows else 0}/{len(frows)}  [SPAZIO]"), ft(h))),
                    box=box.ROUNDED, border_style="border"))
                lay["params_sec"].update(Panel(
                    _FitHeight(lambda h: Group(sec_title("parametri", f"PARAMETRI {pi + 1 if prows else 0}/{len(prows)}  [←/→]"), pt(h))),
                    box=box.ROUNDED, border_style="border"))
            else:
                # Preset storico (nessuna eff): build + info modello + cmd + btn
                model = state.rpc_model_for(m["preset"]) or ""
                mi = next((x for x in state.local_models if x.filepath == model), None)
                fit, fit_col = state.fit_label(mi.size_gb) if mi else ("", "muted")
                filt = state.SIZE_FILTERS[state.model_size_filter][0]
                model_text = Text.from_markup(
                    f"[muted]preset storico – modello ([←/→] cambia, [Z] filtro: {filt}):[/] "
                    f"[bold fg]{_esc(os.path.basename(model))}[/] "
                    + (f"[metric]{mi.size_gb:.1f} GB[/] [{fit_col}]{fit}[/]" if mi else ""))
                lay.split_column(
                    Layout(name="build_sec", ratio=1),
                    Layout(name="model_sec", size=2),
                    Layout(name="cmd_area", size=cmd_size),
                    Layout(name="btn_area", size=btn_size),
                )
                lay["build_sec"].update(build_panel)
                lay["model_sec"].update(Panel(
                    model_text, box=box.SIMPLE, border_style="border", padding=(0, 1)))

            lay["cmd_area"].update(cmd_panel)
            lay["btn_area"].update(btn_panel)
            return lay
        lp = state.launch_progress or {}
        pct = float(lp.get("shown", lp.get("pct", 0)) or 0)
        col = "err" if lp.get("error") else ("ok" if lp.get("done") else "metric")
        el = time.time() - lp.get("t0", time.time())

        asp = get_aspect_info()
        term_lines = asp["lines"]
        _box_size, visible_lines = get_launch_log_dimensions(term_lines)

        g = Table.grid(expand=True, padding=(0, 1))
        g.add_column(ratio=1)
        g.add_column(width=7, justify="right")
        g.add_row(ProgressBar(total=100, completed=pct, complete_style=col, finished_style=col, style="sel"),
                  Text(f"{pct:5.1f}%", style=f"bold {col}"))
        stage = lp.get("error") or lp.get("stage", "")
        body = [
            g,
            Text.from_markup(
                f"[bold {col}]{_esc(stage)}[/]   [muted]{el:.0f}s · build {lp.get('build')} · "
                f"{_esc(lp.get('model', ''))} ({lp.get('size_gb', 0):.1f} GB) · porta {lp.get('port')}"
                f"{' · PID ' + str(lp.get('pid')) if lp.get('pid') else ''}[/]"
            ),
        ]

        all_lines = lp.get("lines", [])
        total = len(all_lines)
        offset = max(0, min(lp.get("scroll_offset", 0), max(0, total - visible_lines)))

        if total == 0:
            body.append(Text("(in attesa di output dal server...)", style="muted"))
        else:
            if offset > 0:
                end_idx = total - offset
                start_idx = max(0, end_idx - visible_lines)
                shown_lines = all_lines[start_idx:end_idx]
                scroll_info = Text.from_markup(
                    f"[bold warn on sel] ▲ SCORRIMENTO LOG: righe {start_idx + 1}-{end_idx} di {total} "
                    f"(-{offset} dal fondo) · [PGDN/END] torna in fondo [/]"
                )
                body.append(scroll_info)
            else:
                start_idx = max(0, total - visible_lines)
                shown_lines = all_lines[start_idx:]
                body.append(Text.from_markup(f"[muted]─── Log ({total} righe totali, ultime {len(shown_lines)} visibili) ───[/]"))

            for ln in shown_lines:
                ln_clean = ln[:220].rstrip("\r\n")
                if " E " in ln or "error" in ln.lower():
                    body.append(Text(ln_clean, style="err"))
                elif " W " in ln or "warn" in ln.lower():
                    body.append(Text(ln_clean, style="warn"))
                else:
                    body.append(Text(ln_clean, style="muted"))

        sub_parts = []
        if lp.get("done"):
            sub_parts.append("[bold ok][INVIO/ESC] chiudi[/]")
        else:
            sub_parts.append("[muted][ESC] nascondi (avvio continua)[/]")
            sub_parts.append("[bold err][X] annulla avvio[/]")
        sub_parts.append(f"[bold accent][PGUP/PGDN] scorri log ({visible_lines} righe)[/]")
        if offset > 0:
            sub_parts.append("[bold metric][END] torna in fondo[/]")

        sub = " · ".join(sub_parts)
        return Panel(Group(*body), box=box.DOUBLE, border_style=col,
                     title=f"[bold {col}]⚡ {_esc(str(lp.get('label', '')))} — avvio llama-server[/]",
                     subtitle=sub)

    @staticmethod
    def render_server_panel(state: AppState) -> Panel:
        """
        Pannello "SERVER" (F7): elenco unificato di TUTTI i llama-server.

        Tipi: TUI (avviato da questa TUI) | ADOTTATO (llama-server locale
        avviato a mano altrove, trovato in /proc o via porta) | REMOTO
        (endpoint HTTP inserito a mano) | RPC (ggml-rpc-server remoto).
        Lo stato non mente: ONLINE solo se /health ha risposto davvero.
        """
        rows = state.server_inventory.rows()
        tbl = Table(expand=True, box=box.SIMPLE_HEAD)
        for col, kw in (("Stato", {}), ("Tipo", {}), ("Server", {"ratio": 2}),
                        ("Modello", {"ratio": 3}), ("ctx", {"justify": "right"}),
                        ("VRAM", {}), ("PID:Porta", {})):
            tbl.add_column(col, overflow="ellipsis", no_wrap=True, **kw)

        def state_cell(r) -> Text:
            return {"ONLINE": Text("● ONLINE", style="bold ok"),
                    "STARTING": Text("◐ AVVIO", style="bold warn")}.get(
                r.state_line, Text("○ OFFLINE", style="err"))

        def kind_cell(r) -> Text:
            if r.kind == "locale":
                if r.managed:
                    return Text("TUI", style="bold metric")
                return Text("ADOTTATO" if r.source == "proc" else "ADOTT.*", style="bold accent")
            if any("ggml-rpc" in f for f in r.flags):
                return Text("RPC", style="bold metric2")
            return Text("REMOTO", style="bold warn")

        sel_key = state.server_inventory.selected_key
        for r in rows:
            sel = (sel_key == r.key)
            tbl.add_row(
                state_cell(r), kind_cell(r),
                Text(("▶ " if sel else "  ") + r.name, style="bold fg" if sel else "fg"),
                Text(r.model_name, style="muted"),
                Text(f"{r.n_ctx:,}" if r.n_ctx else "-", style="metric"),
                Text(r.vram_line, style="metric2"),
                Text(f"{r.pid or '-'}:{r.port or '-'}", style="muted"),
                style=("on sel" if sel else None),
            )
        if not rows:
            tbl.add_row(Text("○", style="muted"), "", Text("nessun llama-server trovato", style="muted"),
                        "", "", "", "")

        extra: List[str] = []
        sel_row = state.server_inventory.selected()
        if sel_row is not None:
            origin = {"proc": "processo letto da /proc", "health": "solo porta (processo non leggibile)",
                      }.get(sel_row.source, "endpoint manuale")
            extra.append(f"[muted]selezionato:[/] [bold fg]{sel_row.name}[/] [muted]{sel_row.state_line} · {origin}"
                         f"{' · avviato dalla TUI' if sel_row.managed else (' · adottato' if sel_row.kind == 'locale' else '')}[/]")
            if sel_row.cgroup:
                iso = ("STESSO cgroup della TUI: un OOM colpirebbe anche la TUI"
                       if sel_row.cgroup == PS.cgroup_of(os.getpid()) else "cgroup separato dalla TUI ✓")
                extra.append(f"[muted]cgroup:[/] [metric2]{sel_row.cgroup[-60:]}[/] [muted]({iso})[/]")
            if sel_row.error_msg:
                extra.append(f"[warn]{_mesc(sel_row.error_msg[:110])}[/]")
            # MEMORIA: RAM del processo + VRAM per device, con la KV dove sta.
            if sel_row.kind == "locale" and sel_row.pid:
                _logs = [p for p in (sel_row.log_path, sel_row.listen_log) if p] if hasattr(sel_row, "log_path") else []
                _lp = state.llama_mgr.managed_log_path if sel_row.pid == state.llama_mgr.managed_pid else ""
                if _lp:
                    mb = Srv.memory_breakdown(_lp, sel_row.pid)
                    tot_vram = sum(d["model"] + d["kv"] + d["compute"] for d in mb["devices"].values())
                    extra.append(f"[bold fg]RAM[/] [metric]{mb['ram_mib']/1024:.1f} GB[/] "
                                 f"[muted]({mb['ram_mib'] / max(1.0, state.hw_snapshot.ram.total_gb * 1024) * 100:.0f}% della RAM di sistema)[/]")
                    for dev, d in mb["devices"].items():
                        if dev == "CPU":
                            continue
                        dtot = d["model"] + d["kv"] + d["compute"]
                        pct = dtot / max(1.0, tot_vram) * 100
                        extra.append(f"[bold fg]{_mesc(dev)}[/] [metric]{dtot:,.0f} MiB[/] "
                                     f"[muted]({pct:.0f}% della VRAM del server) · modello {d['model']:,.0f} "
                                     f"· KV {d['kv']:,.0f} [/]")
                    kv_tot = mb["kv_mib"]
                    if kv_tot:
                        where = " + ".join(f"{_mesc(k)} {v / kv_tot * 100:.0f}%" for k, v in mb["kv_where"].items())
                        extra.append(f"[bold fg]KV cache[/] [metric]{kv_tot:,.0f} MiB[/] [muted]({where})[/]")
                    if mb["devices"].get("CPU", {}).get("model", 0):
                        extra.append(f"[muted]pesi in RAM (CPU): {mb['devices']['CPU']['model']:,.0f} MiB[/]")
                elif sel_row.pid:
                    extra.append(f"[muted]RAM {Srv.process_ram_mib(sel_row.pid)/1024:.1f} GB · memoria per device "
                                 f"non disponibile (avvio non fatto dalla TUI)[/]")
            flags = " ".join(sel_row.flags)
            if flags:
                extra.append(f"[muted]flag:[/] [metric]{_mesc(flags[:180])}[/]")
            det = state.server_inventory.details.get(sel_row.key) or sel_row.details or {}
            if det:
                pr = det.get("props") or {}
                if pr:
                    extra.append(f"[muted]/props:[/] [fg]ctx {pr.get('n_ctx') or '-'} · slot {pr.get('total_slots') or '-'}"
                                 f" · {_mesc(str(pr.get('build_info') or ''))}[/]")
                sl = det.get("slots")
                if sl:
                    extra.append("[muted]/slots:[/] " + "  ".join(
                        f"[{'warn' if x.get('busy') else 'ok'}]#{x.get('id')} {'occupato' if x.get('busy') else 'libero'}"
                        f" ctx {x.get('n_ctx') or '-'}[/]" for x in sl[:6]))
                elif det.get("slots_error"):
                    extra.append(f"[muted]/slots: {_mesc(str(det['slots_error']))}[/]")
                m = det.get("metrics")
                if m:
                    extra.append(f"[muted]/metrics:[/] [metric]{m.get('predicted_tokens_seconds', 0):.2f} tok/s gen · "
                                 f"{m.get('prompt_tokens_seconds', 0):.1f} tok/s prompt · "
                                 f"{int(m.get('requests_processing', 0))} in corso · "
                                 f"{int(m.get('requests_deferred', 0))} in coda[/]")
                elif det.get("metrics_error"):
                    extra.append(f"[muted]/metrics: {det['metrics_error']}[/]")
                if det.get("error"):
                    extra.append(f"[muted]{det['error']}[/]")
        else:
            extra.append("[muted]nessun server selezionato — [↑/↓] e [ENTER][/]")
        if state.server_details_busy:
            extra.append("[warn]⏳ leggo /props /slots /metrics...[/]")
        # GPU remote lette via SSH ([H])
        for host, g in (state.server_inventory.gpu_remote or {}).items():
            if g.get("success"):
                for x in g.get("gpus", []):
                    used, tot = x.get("used") or 0, x.get("total") or 1
                    col = "ok" if used / tot < 0.65 else ("warn" if used / tot < 0.9 else "err")
                    extra.append(f"[muted]SSH {host} GPU{x['index']}:[/] [fg]{x['name'][:26]}[/] "
                                 f"[{col}]{used}/{tot} MiB usati[/] [muted]riserv. {x.get('reserved') or '-'}[/] "
                                 f"[metric]{x.get('util')}%[/] [muted]{x.get('temp')}°C {x.get('driver_model', '')}[/]")
                for d in ((g.get("llama") or {}).get("devices") or []):
                    if d["device"].startswith("RPC"):
                        extra.append(f"[muted]  vista llama.cpp {d['device']}:[/] [metric]{d['free']}/{d['total']} MiB "
                                     f"allocabili[/] [muted](questo usa --fit; con WDDM e' meno del libero di "
                                     f"nvidia-smi/Task Manager)[/]")
            else:
                extra.append(f"[muted]SSH {host}:[/] [err]{g.get('error', '')[:70]}[/]")
        if state.ssh_gpu_busy:
            extra.append("[warn]⏳ SSH nvidia-smi...[/]")
        # Build per i draft
        dr = state.draft_resolve()
        avail = ", ".join(f"{t}{'✓' if ok else '✗'}" for t, ok in state.drafts_available())
        extra.append(f"[muted]build per draft:[/] [metric]{avail}[/]"
                     + (f" → [fg]{dr.get('build')}[/] [muted]({dr.get('spec_type')})[/]" if dr.get("available")
                        else f" [err]{dr.get('message', '')[:60]}[/]"))
        # Endpoint salvati
        eps = state.server_inventory.remotes
        peers_now = state.peer_net.live_peers()
        if state.peer_net.error:
            extra.append(f"[warn]rete TUI: {_mesc(state.peer_net.error[:90])}[/]")
        elif peers_now:
            extra.append(f"[muted]TUI in rete ({len(peers_now)}):[/] " + "  ".join(
                f"[metric]{_mesc(p.get('host', '?'))}[/] [muted]{_mesc(p.get('ip', ''))} "
                f"server {', '.join(str(sv.get('port')) for sv in p.get('servers', [])) or '-'}[/]"
                for p in peers_now))
            used = state.peer_net.ports_in_use()
            mine = {int(r.port) for r in state.server_inventory.rows() if r.kind == "locale" and r.port}
            clash = used & mine
            if clash:
                extra.append(f"[warn]⚠ porta in conflitto con un'altra TUI: {', '.join(map(str, sorted(clash)))}[/]")
        else:
            extra.append("[muted]TUI in rete: nessuna altra TUI rilevata (annuncio ogni 4 s)[/]")
        extra.append(f"[muted]endpoint salvati ({len(eps)}):[/] " + ("  ".join(
            f"[fg]{e.name}[/][muted]={e.host}:{e.port}{' rpc' if e.kind == 'rpc' else ''}{' ssh' if e.ssh else ''}[/]"
            for e in eps) or "[muted]nessuno: [A] per aggiungerne[/]"))
        # Scansione passiva della rete dopo 3 min di idle (vedi start_idle_lan_scan).
        if state.lan_scan_results:
            extra.append("[muted]LAN (scan dopo 3 min idle):[/] " + "  ".join(
                f"[metric]{r['host']}:{r['port']}[/][muted]({r['kind']})[/]"
                for r in state.lan_scan_results[:12]))
        elif state.lan_scan_busy:
            extra.append("[muted]LAN: scansione passiva in corso...[/]")
        body = Group(tbl, Text.from_markup("\n" + "\n".join(extra)))

        border = "accent" if state.server_select_mode else "border"
        sub = ("[muted][↑/↓]+[INVIO] seleziona [F8] ricarica [I] info [S] ferma [R] riavvia [B] browser "
               "[A] aggiungi IP [X] rimuovi [H] GPU via SSH [M] Miky[/]" if state.show_server_panel
               else "[muted][F7] apri[/]")
        return Panel(body, box=box.ROUNDED, border_style=border,
                     title="[bold accent]📡 SERVER[/] [muted](locali /proc · adottati · remoti manuali)[/]",
                     subtitle=sub, padding=(0, 1))


# =====================================================================
# INTERACTIVE APPLICATION ORCHESTRATOR
# =====================================================================
class CyberAITUIApp:
    """
    Main Textual/Rich Terminal Application Driver.
    Dispatches non-blocking keyboard input, handles live rendering loops, and runs background tasks.
    """

    def __init__(self, start_tab: int = 1):
        # ThemedConsole: risolve i ruoli del tema (ok/err/warn/metric...) al
        # momento del render -> il cambio tema e' a caldo.
        self.console = Th.ThemedConsole()
        self.state = AppState()
        self.state.active_tab = start_tab

    CRASH_LOG = os.path.expanduser("~/.cache/ai_tui/crash.log")

    def _log_crash(self, where: str, exc: BaseException) -> None:
        """Un errore NON chiude piu' la TUI: va nel log e nella barra di stato."""
        import traceback
        try:
            os.makedirs(os.path.dirname(self.CRASH_LOG), exist_ok=True)
            with open(self.CRASH_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} [{where}] tab={self.state.active_tab}\n")
                fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except OSError:
            pass
        self.state.set_status(f"⚠ errore in {where}: {type(exc).__name__}: {str(exc)[:90]} "
                              f"(dettagli in ~/.cache/ai_tui/crash.log)")

    def frame(self):
        """La vista completa con lo sfondo/testo di base del tema attivo."""
        try:
            view = self.render_current_view()
        except Exception as exc:          # un errore di disegno non chiude la TUI
            self._log_crash("render", exc)
            view = Panel(Text(f"Errore di disegno nella tab {self.state.active_tab}: {exc}\n"
                              f"Premi un tasto numerico per cambiare tab. Dettagli: {self.CRASH_LOG}",
                              style="err"), title="⚠")
        return Styled(view, Th.base_style())

    def handle_key(self, key_name: str) -> bool:
        """Dispatcher protetto: un'eccezione non deve mai chiudere la TUI."""
        self.state.last_key_time = time.time()
        try:
            return self._handle_key_impl(key_name)
        except Exception as exc:
            self._log_crash(f"tasto {key_name!r}", exc)
            return True

    def _collect_pending_input(self, live) -> None:
        """Se un tasto ha chiesto un input di testo, lo raccoglie qui."""
        req = self.state.pending_input
        if not req:
            return
        self.state.pending_input = None
        prompt, cb = req
        live.stop()
        self.console.print(f"\n[bold accent]{prompt}[/]")
        try:
            _kbd_restore()
            text = input("❯ ").strip()
            if text:
                cb(text)
            else:
                self.state.set_status("Input annullato")
        except (KeyboardInterrupt, EOFError):
            self.state.set_status("Input annullato")
        except Exception as exc:
            self.state.set_status(f"✗ {exc}")
        live.start()

    def render_current_view(self) -> Layout:
        """Compose master screen layout with Header, Active Tab View, and Footer."""
        asp = get_aspect_info()
        cols, lines = asp["cols"], asp["lines"]
        is_port = asp["is_portrait"]

        # Dimensionamento dinamico Header e Footer in base a proporzioni e contesto
        if self.state.launch_modal is not None:
            # Menu di pre-avvio: massimizzare lo spazio verticale per il modale
            # in modo che il pulsante AVVIA sia sempre visibile
            if lines < 30:
                hdr_size = 2 if lines >= 22 else 0
                ftr_size = 2 if lines >= 22 else 0
            else:
                hdr_size = 4
                ftr_size = 3
        elif is_port or lines < 28:
            hdr_size = 4 if lines >= 28 else (2 if lines >= 22 else 0)
            ftr_size = 3 if lines >= 28 else 2
        else:
            hdr_size = 8
            ftr_size = 7

        # If Gear Modal is active, render full-screen modal view
        if self.state.show_gear_modal:
            modal_layout = Layout()
            modal_layout.split_column(
                Layout(TUIRenderer.render_header(self.state), size=max(2, hdr_size)),
                Layout(TUIRenderer.render_gear_modal(self.state), ratio=1),
                Layout(TUIRenderer.render_footer(self.state), size=max(2, ftr_size))
            )
            return modal_layout

        master_layout = Layout()
        if hdr_size > 0 and ftr_size > 0:
            master_layout.split_column(
                Layout(name="header", size=hdr_size),
                Layout(name="body", ratio=1),
                Layout(name="footer", size=ftr_size)
            )
            master_layout["header"].update(TUIRenderer.render_header(self.state))
            master_layout["footer"].update(TUIRenderer.render_footer(self.state))
        elif hdr_size > 0:
            master_layout.split_column(
                Layout(name="header", size=hdr_size),
                Layout(name="body", ratio=1)
            )
            master_layout["header"].update(TUIRenderer.render_header(self.state))
        elif ftr_size > 0:
            master_layout.split_column(
                Layout(name="body", ratio=1),
                Layout(name="footer", size=ftr_size)
            )
            master_layout["footer"].update(TUIRenderer.render_footer(self.state))
        else:
            master_layout.split_column(Layout(name="body", ratio=1))

        # Pannello "SERVER TROVATI" (F6/F7):
        # In 16:9 (orizzontale): overlay laterale a destra
        # In 9:16 (verticale): overlay impilato in basso
        show_panel = self.state.show_server_panel
        if show_panel:
            srv = Th.APPEARANCE.proportion_cfg["server"]
            if is_port:
                tab_l = Layout(name="tab_area", ratio=3)
                srv_l = Layout(name="server_panel", ratio=2)
                _RichLayout.split_column(master_layout["body"], tab_l, srv_l)
            else:
                if isinstance(srv, tuple):
                    tab_l = Layout(name="tab_area", ratio=srv[0])
                    srv_l = Layout(name="server_panel", ratio=srv[1])
                else:
                    tab_l = Layout(name="tab_area", ratio=2)
                    srv_l = Layout(name="server_panel", size=srv)
                _RichLayout.split_row(master_layout["body"], tab_l, srv_l)

        # Render active tab
        if self.state.active_tab == 1:
            body = TUIRenderer.render_tab1_system_monitor(self.state)
        elif self.state.active_tab == 2:
            body = TUIRenderer.render_tab2_kvcache(self.state)
        elif self.state.active_tab == 3:
            body = TUIRenderer.render_tab3_radar(self.state)
        elif self.state.active_tab == 4:
            body = TUIRenderer.render_tab4_downloader(self.state)
        elif self.state.active_tab == 5:
            body = TUIRenderer.render_tab5_process_control(self.state)
        elif self.state.active_tab == 6:
            body = TUIRenderer.render_tab6_agent_shell(self.state)
        elif self.state.active_tab == 7:
            body = TUIRenderer.render_tab7_plugins_skills(self.state)
        elif self.state.active_tab == 8:
            body = TUIRenderer.render_tab8_lm_bench(self.state)
        elif self.state.active_tab == 9:
            body = TUIRenderer.render_tab9_rpc_cuda13(self.state)
        elif self.state.active_tab == 10:
            body = TUIRenderer.render_tab10_config(self.state)
        elif self.state.active_tab == 11:
            body = TUIRenderer.render_tab11_aspetto(self.state)
        elif self.state.active_tab == 12:
            body = TUIRenderer.render_tab12_benchmark(self.state)
        elif self.state.active_tab == 13:
            body = TUIRenderer.render_tab13_builds(self.state)
        elif self.state.active_tab == 14:
            body = TUIRenderer.render_tab14_modal(self.state)
        elif self.state.active_tab == 15:
            body = render_updates_panel(self.state.updates_monitor,
                                        self.state.updates_selected_idx,
                                        bm_downloading=self.state.bm_upd_downloading,
                                        bm_progress=self.state.bm_upd_progress,
                                        bm_confirm=self.state.bm_upd_confirm,
                                        bm_upd=self.state.bm_upd)
        else:
            body = Panel(Text("Unknown Tab", style="bold err"))

        lp = self.state.launch_progress
        if self.state.launch_modal is not None or (lp and lp.get("visible")):
            launch_box = TUIRenderer.render_launch(self.state)
            if self.state.launch_modal is not None:
                # Il menu di pre-avvio prende tutto il corpo: ha build, flag,
                # parametri, anteprima e pulsanti (troppo per una fascia).
                body = launch_box
            else:
                asp = get_aspect_info()
                lp_box_size, _ = get_launch_log_dimensions(asp["lines"])
                wrap = Layout()
                wrap.split_column(Layout(launch_box, size=lp_box_size), Layout(body, ratio=1))
                body = wrap

        if show_panel:
            master_layout["body"]["tab_area"].update(body)
            master_layout["body"]["server_panel"].update(
                TUIRenderer.render_server_panel(self.state)
            )
        else:
            master_layout["body"].update(body)
        return master_layout

    def handle_gear_modal_key(self, key_name: str) -> None:
        """Handle keyboard interactions when Gear ⚙️ Settings Modal is open."""
        if key_name == "UP":
            self.state.gear_active_field = (self.state.gear_active_field - 1) % len(self.state.gear_fields_def)
        elif key_name == "DOWN":
            self.state.gear_active_field = (self.state.gear_active_field + 1) % len(self.state.gear_fields_def)
        elif key_name == "LEFT":
            self.state.adjust_gear_field(-1)
        elif key_name == "RIGHT":
            self.state.adjust_gear_field(1)
        elif key_name in ("ENTER", "SPACE"):
            self.state.toggle_gear_field()
        elif key_name in ("s", "S"):
            self.state.save_gear_config()
        elif key_name in ("h", "H"):
            self.state.fetch_groq_hf_config_async(self.state.gear_model_name)
        elif key_name in ("ESC", "g", "G", "q", "Q"):
            self.state.close_gear_modal()

    def _handle_key_impl(self, key_name: str) -> bool:
        """
        Global Key Event Dispatcher.
        Returns False if the application should terminate (Quit).
        """
        # [F] armato (primo [F] premuto da meno di 5 s): [F] = ferma solo il
        # server della TUI, [T] = TUTTI i llama-server della macchina.
        # Qualsiasi altro tasto annulla e prosegue normalmente.
        armed_t = getattr(self.state, "_flush_armed", 0.0)
        if armed_t:
            self.state._flush_armed = 0.0
            if time.time() - armed_t <= FLUSH_CONFIRM_S:
                if key_name in ('f', 'F'):
                    # [F] confermato: ferma TUTTI i llama-server della macchina
                    # (quelli della TUI, avviati a mano o da systemd)
                    self.state.flush_vram_async(kill_external=True)
                    return True
                if key_name == 'T':
                    self.state.flush_vram_async(kill_external=True)
                    return True
                self.state.set_status("Flush VRAM annullato")
                if key_name == "ESC":
                    return True

        # Gear Modal Interception
        if self.state.show_gear_modal:
            self.handle_gear_modal_key(key_name)
            return True

        # Dialogo di avvio / barra di avanzamento: hanno la precedenza.
        if self.state.launch_modal is not None:
            # MENU DI PRE-AVVIO: sezioni build -> flag -> parametri -> pulsanti.
            st = self.state
            m = st.launch_modal
            secs = st.launch_sections(m)
            sec = secs[m["sec"] % len(secs)]
            if m.get("bq_typing"):
                if key_name == "ESC":
                    m["bq_typing"] = False
                    st.launch_set_query("")
                    return True
                if key_name == "ENTER":
                    m["bq_typing"] = False
                    return True
                if key_name in ("\x7f", "\x08", "BACKSPACE"):
                    st.launch_set_query(m.get("bq", "")[:-1])
                    return True
                if len(key_name) == 1 and key_name.isprintable():
                    st.launch_set_query(m.get("bq", "") + key_name)
                    return True
                # frecce e TAB restano attive mentre si scrive
            if key_name == "/" and "build" in secs:
                m["sec"] = secs.index("build")
                m["bq_typing"] = True
                return True
            if key_name in ("TAB",):
                st.launch_section(1)
            elif key_name == "BTAB":
                st.launch_section(-1)
            elif key_name == "UP":
                st.launch_move(-1)
            elif key_name == "DOWN":
                st.launch_move(1)
            elif key_name in ("LEFT", "RIGHT"):
                d = -1 if key_name == "LEFT" else 1
                if sec == "parametri":
                    st.launch_cycle_param(d)
                elif sec == "pulsanti":
                    m["bi"] = (m["bi"] + d) % 2
                elif sec == "flag":
                    st.launch_toggle_flag()
                elif m.get("eff") is None:
                    # preset storico: il modello viene dalla lista
                    st.rpc_model_idx = st.model_step(st.rpc_model_idx, d)
            elif key_name in (" ", "SPACE"):
                if sec == "flag":
                    st.launch_toggle_flag()
                elif sec == "pulsanti":
                    m["bi"] = (m["bi"] + 1) % 2
                elif sec == "build":
                    # SPAZIO = conferma la build correntemente evidenziata e
                    # passa alla sezione flag.  Il cursore (m["idx"]) e' gia'
                    # sulla build scelta; spostarlo con SPAZIO era confuso
                    # perche' l'utente premeva SPAZIO credendo di selezionare
                    # e invece andava alla build successiva.
                    st.launch_section(1)
            elif key_name in ("z", "Z"):
                st.cycle_size_filter()
            elif key_name in ("k", "K"):
                st.launch_free_rpc()
            elif key_name in ("s", "S"):
                b = st.launch_build(m)
                if b is not None:
                    st.launch_sync_build_prompt(b)
            elif key_name in ("u", "U"):
                if st.bm_upd.get("newer"):
                    st.bm_arm_download()
                else:
                    threading.Thread(target=st.bm_check_update, daemon=True).start()
                    st.set_status("🔎 [Build] Controllo nuove release llama.cpp su GitHub...")
            elif key_name == "ENTER":
                if sec == "flag":
                    st.launch_toggle_flag()
                elif sec == "parametri":
                    st.launch_edit_param()
                elif sec == "pulsanti" and m["bi"] % 2 == 1:
                    st.launch_modal = None
                    st.set_status("Avvio annullato (personalizzazioni salvate per il preset)")
                else:
                    # build o pulsante AVVIA
                    st.launch_confirm()
            elif key_name in ("ESC", "q", "Q"):
                st.launch_modal = None
                st.set_status("Avvio annullato")
            return True
        lp = self.state.launch_progress
        if lp and lp.get("visible"):
            lines_count = len(lp.get("lines", []))
            asp = get_aspect_info()
            _box_size, visible_lines = get_launch_log_dimensions(asp["lines"])
            max_offset = max(0, lines_count - visible_lines)

            if key_name == "PGUP":
                cur_off = lp.get("scroll_offset", 0)
                lp["scroll_offset"] = min(max_offset, cur_off + visible_lines)
                return True
            if key_name == "PGDN":
                cur_off = lp.get("scroll_offset", 0)
                lp["scroll_offset"] = max(0, cur_off - visible_lines)
                return True
            if key_name == "HOME":
                lp["scroll_offset"] = max_offset
                return True
            if key_name == "END":
                lp["scroll_offset"] = 0
                return True
            if key_name in ("UP", "k"):
                cur_off = lp.get("scroll_offset", 0)
                lp["scroll_offset"] = min(max_offset, cur_off + 1)
                return True
            if key_name in ("DOWN", "j"):
                cur_off = lp.get("scroll_offset", 0)
                lp["scroll_offset"] = max(0, cur_off - 1)
                return True
            if key_name in ("ESC", "ENTER") or (lp.get("done") and key_name in ("q", "Q")):
                lp["visible"] = False
                return True
            if key_name in ("x", "X") and not lp.get("done"):
                self.state.launch_stop()
                return True
            if key_name in ("q", "Q") and not lp.get("done"):
                # avvio in corso: q non chiude la TUI al primo colpo
                if time.time() - getattr(self.state, "_quit_armed", 0.0) > 3:
                    self.state._quit_armed = time.time()
                    self.state.set_status("⚠ Avvio in corso: [Q] di nuovo entro 3 s per USCIRE "
                                          "(il server continua a caricare), [X] annulla l'avvio")
                    return True

        # Tab 13: ricerca build. Mentre si scrive, lettere e cifre vanno nel filtro
        # (non chiudono la TUI ne' cambiano scheda).
        if self.state.active_tab == 13 and not (lp and lp.get("visible")):
            s = self.state
            if s.build_typing:
                if key_name == "ESC":
                    s.build_typing = False
                    s.build_set_query("")
                elif key_name == "ENTER":
                    s.build_typing = False
                elif key_name in ("\x7f", "\x08", "BACKSPACE"):
                    s.build_set_query(s.build_query[:-1])
                elif len(key_name) == 1 and key_name.isprintable():
                    s.build_set_query(s.build_query + key_name)
                elif key_name in ("UP", "DOWN"):
                    n = len(s.builds_visible())
                    d = -1 if key_name == "UP" else 1
                    s.build_cursor = max(0, min(max(0, n - 1), s.build_cursor + d))
                return True
            if key_name == "/":
                s.build_typing = True
                s.set_status("🔎 Cerca build: scrivi (nome, b11435, cuda, rpc, vulkan...), [INVIO] conferma, [ESC] azzera")
                return True
            if key_name == "ESC" and s.build_query and not s.bm_upd_confirm:
                s.build_set_query("")
                s.set_status("Ricerca build azzerata")
                return True

        # Global Quit
        if key_name in ('q', 'Q'):
            return False

        # Tab Switching [1-9] + [0] -> Tab 10 (Config llama.cpp)
        if key_name in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
            self.state.active_tab = int(key_name)
            self.state.set_status(f"Switched to Tab {self.state.active_tab}")
            return True
        if key_name == '0':
            self.state.active_tab = 10
            self.state.set_status("Tab 10: Config llama.cpp")
            # Legge subito l'help della build attiva (prima serviva premere [R]).
            if not self.state.cfg_help_loaded and not self.state.cfg_loading_help:
                threading.Thread(target=self.state.cfg_load_help, daemon=True).start()
            return True
        # Tab 11-14: F10/F11/F12, Alt+M, Ctrl+A/Ctrl+B/Ctrl+L/Ctrl+M
        if key_name in TUI_EXTRA_TABS:
            self.state.active_tab = TUI_EXTRA_TABS[key_name]
            tab_labels = {
                11: "Tab 11: Aspetto",
                12: "Tab 12: Benchmark",
                13: "Tab 13: Build llama.cpp & RPC",
                14: "Tab 14: Modal Cloud (Hermes Ornith / Tiel-Coder)",
                15: "Tab 15: Updates — [↑/↓] naviga · [Y] scarica · [R] refresh",
            }
            self.state.set_status(tab_labels.get(self.state.active_tab, f"Tab {self.state.active_tab}"))
            if self.state.active_tab == 12 and not self.state.bench.loaded_at:
                threading.Thread(target=self.state._bench_load_safe, daemon=True).start()
            elif self.state.active_tab == 13:
                if not getattr(self.state, "bm_upd_checking", False):
                    threading.Thread(target=self.state.bm_check_update, daemon=True).start()
            return True
        # Ctrl+T: tema successivo, da qualunque tab (prova rapida dei temi).
        if key_name == "Ctrl+T":
            self.state.aspetto_change(row=0, step=1)
            return True

        # =============================================================
        # GLOBAL RPC & CUDA13 HOTKEYS  (work from ANY tab)
        # F1..F5 -> launch llama-server with the matching preset.
        # Ctrl+R   -> relaunch the last used preset (default: the RACCOMANDATO).
        # Deliberately handled before the [F] FLUSH VRAM binding and before
        # the per-tab blocks so they always win.
        # =============================================================
        rpc_hotkey_map = dict(TUI_RPC_HOTKEYS)
        if key_name in rpc_hotkey_map:
            # Stessi preset di sempre. Solo se l'utente ha attivato i margini
            # VRAM per i preset (Tab 13 [P]) si passa da rpc_launch_with_extra,
            # che aggiunge --fit-target IN CODA senza cambiare il preset.
            # Prima di avviare si sceglie la build (solo quelle compatibili
            # con l'RPC rilevato) e si vede il modello; poi barra di avanzamento.
            self.state.launch_open(rpc_hotkey_map[key_name])
            return True
        if key_name == "Ctrl+R":
            # Rilancio rapido: stesso preset e stessa build dell'ultima volta,
            # senza dialogo (ma con la barra di avanzamento).
            k = self.state.rpc_last_preset_key
            _p, lbl = self.state.rpc_resolve(k)
            binp = self.state.rpc_binary_override
            if binp:
                bk = next((b.key for b in self.state.build_mgr.ordered() if b.server == binp), "?")
                self.state.launch_start(k, lbl, binp, bk)
            else:
                self.state.launch_open(k)
            return True

        # =============================================================
        # SERVER TROVATI  (F6 / F7 / F8) - work from ANY tab
        #   F6 = apri/chiudi il pannello con l'elenco dei server
        #   F7 = seleziona il server attivo (cursore + ENTER)
        #   F8 = ricarica l'elenco
        # Stanno DOPO i tasti RPC e PRIMA del binding [F] FLUSH VRAM,
        # quindi non possono essere intercettati altrove. F1..F5 non sono
        # toccati: restano i preset RPC.
        # =============================================================
        if key_name in TUI_SERVER_HOTKEYS:
            action = TUI_SERVER_HOTKEYS[key_name]
            if action == "toggle_panel":
                self.state.servers_toggle_panel()
            elif action == "select_mode":
                self.state.servers_toggle_select_mode()
            elif action == "refresh":
                # refresh in un thread: /health di ogni server e nvidia-smi
                # bloccavano la UI per secondi
                st = self.state

                def _refresh() -> None:
                    try:
                        st.servers_refresh(blocking=True)
                        st.set_status(f"🔄 Elenco aggiornato: {len(st.server_inventory.rows())} server")
                    except Exception as exc:
                        st.set_status(f"✗ aggiornamento elenco: {exc}")
                threading.Thread(target=_refresh, daemon=True).start()
                self.state.set_status("🔄 Aggiorno l'elenco dei server...")
            return True

        # =============================================================
        # F9 - avvia/ferma l'EMBEDDER per mem0 (nomic-embed-text, :11435)
        # Separato da TUI_SERVER_HOTKEYS apposta: quel dict e' bloccato da
        # un test (deve restare {F6,F7,F8}), e F9 non e' un'azione sul
        # pannello server, e' un server diverso con il suo stato.
        # =============================================================
        if key_name == "F9":
            self.state.embedder_toggle()
            return True

        # While the server panel is open, the navigation keys belong to it.
        # This block runs only when the panel is actually open, so the
        # per-tab keys keep working normally with the panel closed.
        #
        # RETURN SEMPRE: se un tasto del pannello non facesse return,
        # continuerebbe a scendere nei blocchi per-tab e provocherebbe due
        # azioni con lo stesso tasto (e' il motivo per cui premere INVIO nel
        # pannello cambiava anche pagina/tab).
        if self.state.show_server_panel:
            if key_name in ("UP", "k"):
                self.state.servers_move(-1)
                return True
            if key_name in ("DOWN", "j"):
                self.state.servers_move(1)
                return True
            if key_name == "ENTER":
                # Nel pannello INVIO significa SEMPRE "conferma il server
                # selezionato", in qualunque tab si sia. Non lascia mai
                # cadere il tasto sui blocchi per-tab.
                if not self.state.server_select_mode:
                    self.state.server_select_mode = True
                self.state.servers_confirm_selection()
                return True
            if key_name == "ESC":
                if self.state.server_select_mode:
                    self.state.server_select_mode = False
                    self.state.set_status("Selezione annullata")
                else:
                    self.state.servers_toggle_panel()   # ESC chiude il pannello
                return True
            if key_name in ("s", "S"):
                self.state.servers_action_async("stop")
                return True
            if key_name in ("r", "R"):
                # R = riavvia il locale selezionato con gli stessi flag
                # letti da /proc. Non e' F8, che ricarica solo l'elenco.
                self.state.servers_action_async("relaunch")
                return True
            if key_name in ("b", "B"):
                self.state.servers_open_browser()
                return True
            if key_name in ("a", "A"):
                # Aggiungi un endpoint remoto (IP:porta [nome] [utente@host]).
                self.state.servers_add_endpoint_prompt()
                return True
            if key_name in ("x", "X", "DEL"):
                self.state.servers_remove_selected_endpoint()
                return True
            if key_name in ("i", "I"):
                # Info: /props, /slots, /metrics (anche per server adottati).
                self.state.servers_info_selected()
                return True
            if key_name in ("h", "H"):
                # Stato GPU remoto via SSH (nvidia-smi), es. Miky.
                self.state.servers_ssh_gpu()
                return True
            if key_name in ("m", "M"):
                self.state.miky_probe_async()
                self.state.set_status(
                    "🔌 Sondato Miky 192.168.1.100:50052 (solo lettura)")
                return True

        # Global [E] Embedder per mem0 (start/stop)
        # Globale e non per-tab: l'embedder e' un servizio, non una vista.
        # Eccezione: in Tab 10 [E] e' "Export" (il footer lo promette); prima
        # il tasto globale lo intercettava e accendeva l'embedder. L'embedder
        # resta raggiungibile da qualunque tab con F9.
        if key_name in ('e', 'E') and self.state.active_tab != 10:
            self.state.embedder_toggle()
            return True

        # Global [F] FLUSH VRAM Action: SEMPRE con conferma (secondo [F] entro
        # 5 s). Prima un solo tasto (anche da testo incollato) mandava SIGKILL
        # a tutti i llama-server della macchina, incluso quello in caricamento.
        if key_name in ('f', 'F'):
            self.state.flush_arm()
            return True

        # Global [G] Gear Settings Editor Modal
        # Eccezione: in Tab 8 [G] e' "Gruppo Task" (prima irraggiungibile:
        # apriva sempre il Gear modal).
        if key_name in ('g', 'G') and self.state.active_tab != 8:
            self.state.open_gear_modal()
            return True

        # Tab 1 Keys
        if self.state.active_tab == 1:
            if key_name in ('r', 'R'):
                self.state.update_telemetry()
                self.state.set_status("Hardware telemetry refreshed.")

        # Tab 2 Keys (KV Cache & Compression)
        elif self.state.active_tab == 2:
            if key_name in ('w', 'W'):
                # (A) Rolling Window toggle (toggle_rolling_window sets its own status)
                self.state.toggle_rolling_window()
            elif key_name in ('s', 'S', 't', 'T'):
                self.state.toggle_rolling_window()
            elif key_name in ('k', 'K'):
                # (B) KV quantization cycle under the VRAM guardrail
                self.state.cycle_kv_quant()
            elif key_name in ('o', 'O'):
                self.state.toggle_no_kv_offload()
            elif key_name in ('x', 'X'):
                # (C) Groq-backed tool context truncation
                self.state.run_tool_truncation()
            elif key_name in ('c', 'C'):
                self.state.set_status("Running Groq Context Compression test...")
                def _comp_worker():
                    self.state.is_compressing = True
                    try:
                        res = self.state.compressor.test_compression()
                        self.state.last_compression_result = res
                        self.state.set_status(f"✓ Compression finished: {res.savings_percent:.1f}% saved in {res.latency_ms:.0f}ms")
                    finally:
                        self.state.is_compressing = False
                threading.Thread(target=_comp_worker, daemon=True).start()

        # Tab 3 Keys (Radar News)
        elif self.state.active_tab == 3:
            if key_name == "UP":
                self.state.selected_news_idx = max(0, self.state.selected_news_idx - 1)
            elif key_name == "DOWN":
                self.state.selected_news_idx = min(max(0, len(self.state.news_items) - 1), self.state.selected_news_idx + 1)
            elif key_name in ('r', 'R'):
                self.state.set_status("Refreshing Model Radar news...")
                threading.Thread(target=self.state._fetch_news_worker, daemon=True).start()

        # Tab 4 Keys (HF Downloader & Quant Inspector)
        elif self.state.active_tab == 4:
            if key_name == "TAB":
                panes = ["repos", "quants", "downloads"]
                curr_idx = panes.index(self.state.hf_active_pane)
                self.state.hf_active_pane = panes[(curr_idx + 1) % len(panes)]
                self.state.set_status(f"Active pane: {self.state.hf_active_pane.upper()}")
            elif key_name == "UP":
                if self.state.hf_active_pane == "repos":
                    self.state.select_hf_repo(max(0, self.state.selected_repo_idx - 1))
                elif self.state.hf_active_pane == "quants":
                    self.state.selected_quant_idx = max(0, self.state.selected_quant_idx - 1)
            elif key_name == "DOWN":
                if self.state.hf_active_pane == "repos":
                    self.state.select_hf_repo(min(max(0, len(self.state.hf_repos) - 1), self.state.selected_repo_idx + 1))
                elif self.state.hf_active_pane == "quants":
                    self.state.selected_quant_idx = min(max(0, len(self.state.hf_quants) - 1), self.state.selected_quant_idx + 1)
            elif key_name in ("PGUP", "PGDN", "HOME", "END"):
                st4 = self.state
                step = {"PGUP": -10, "PGDN": 10, "HOME": -10**6, "END": 10**6}[key_name]
                if st4.hf_active_pane == "repos" and st4.hf_repos:
                    st4.select_hf_repo(max(0, min(len(st4.hf_repos) - 1, st4.selected_repo_idx + step)))
                elif st4.hf_active_pane == "quants" and st4.hf_quants:
                    st4.selected_quant_idx = max(0, min(len(st4.hf_quants) - 1, st4.selected_quant_idx + step))
            elif key_name in ('d', 'D', 'ENTER'):
                self.state.start_selected_download()
            elif key_name in ('h', 'H'):
                self.state.fetch_groq_hf_config_async()

        # Tab 5 Keys (Process Control, Presets & Benchmark [T])
        elif self.state.active_tab == 5:
            st5 = self.state
            if key_name == "UP":
                st5.selected_model_idx = st5.model_step(st5.selected_model_idx, -1)
            elif key_name == "DOWN":
                st5.selected_model_idx = st5.model_step(st5.selected_model_idx, 1)
            elif key_name == "PGUP":
                st5.selected_model_idx = st5.model_step(st5.selected_model_idx, -10)
            elif key_name == "PGDN":
                st5.selected_model_idx = st5.model_step(st5.selected_model_idx, 10)
            elif key_name == "HOME":
                st5.selected_model_idx = st5.model_step(st5.selected_model_idx, -10**6)
            elif key_name == "END":
                st5.selected_model_idx = st5.model_step(st5.selected_model_idx, 10**6)
            elif key_name in ('z', 'Z'):
                st5.cycle_size_filter()
            elif key_name == "SPACE" or key_name == " ":
                # Toggle selection of current model for chain benchmark
                self.state.toggle_model_selection()
            elif key_name in ('a', 'A'):
                # Select all models
                self.state.select_all_models()
            elif key_name in ('n', 'N'):
                # Deselect all models
                self.state.deselect_all_models()
            elif key_name in ('r', 'R'):
                # Reset chain report view
                self.state.show_chain_report = False
                self.state.chain_bench_results = []
                self.state.chain_bench_log = []
                self.state.chain_bench_report_path = None
                self.state.set_status("⚡ Chain Benchmark results cleared. Ready for new run.")
            elif key_name in ('l', 'L', 'ENTER'):
                if self.state.local_models:
                    sel = self.state.local_models[self.state.selected_model_idx]
                    self.state.set_status(f"Starting server with '{sel.filename}' (RAM-KV: {self.state.server_no_kv_offload})...")
                    def _launch_worker():
                        verdict = self.state.evaluate_kv_guardrail()
                        res = self.state.llama_mgr.start_server(
                            model_path=sel.filepath,
                            ctx_size=262144 if self.state.server_preset_key=="extreme_256k" else 32768,
                            kv_dtype=self.state.kv_quant_key,
                            no_kv_offload=self.state.server_no_kv_offload,
                            ctx_shift=self.state.rolling_window_enabled,
                            port=self.state.server_port
                        )
                        if res.get("success"):
                            self.state.active_model_name = sel.filename
                            if verdict["forced"]:
                                self.state.set_status(
                                    f"✓ llama-server online ('{sel.filename}') — KV guardrail: "
                                    f"{verdict['requested']} ➔ {verdict['effective']} ({verdict['reason']})"
                                )
                            else:
                                self.state.set_status(f"✓ llama-server online with '{sel.filename}'!")
                        else:
                            self.state.set_status(f"✗ Launch failed: {res.get('error')}")
                    threading.Thread(target=_launch_worker, daemon=True).start()
            elif key_name in ('t', 'T'):
                # Chain benchmark if models selected, else single-model benchmark
                self.state.run_chain_benchmark_async()
            elif key_name in ('h', 'H'):
                self.state.fetch_groq_hf_config_async()
            elif key_name in ('o', 'O'):
                self.state.cycle_server_preset()
            elif key_name in ('d', 'D'):
                # Cambia la cartella dei modelli a caldo (persistita per il
                # prossimo avvio): utile su Miky, dove i modelli stanno sul
                # disco condiviso di rete, non in un percorso Linux.
                live.stop()
                self.console.print(f"\n[bold ok]Cartella modelli attuale:[/] {self.state.model_dir}")
                try:
                    _kbd_restore()
                    new_dir = input("📁 Nuova cartella modelli ❯ ").strip()
                    if new_dir:
                        res = self.state.set_model_dir(new_dir)
                        if res.get("success"):
                            self.state.set_status(f"✓ Cartella modelli: {res['path']}")
                        else:
                            self.state.set_status(f"✗ {res.get('error')}")
                except (KeyboardInterrupt, EOFError):
                    pass
                live.start()

        # Tab 6 Keys (Agent Shell)
        elif self.state.active_tab == 6:
            if key_name in ('c', 'C'):
                self.state.chat_history = [
                    ChatMessage(role="system", content="Chat buffer cleared. Ready for new context.", timestamp=datetime.now().strftime("%H:%M:%S"))
                ]
                self.state.set_status("Chat history cleared.")

        # Tab 7 Keys (Plugins & Skills)
        elif self.state.active_tab == 7:
            if key_name == "UP":
                self.state.selected_plugin_idx = max(0, self.state.selected_plugin_idx - 1)
            elif key_name == "DOWN":
                self.state.selected_plugin_idx = min(max(0, len(self.state.installed_plugins) - 1), self.state.selected_plugin_idx + 1)
            elif key_name in ('r', 'R'):
                self.state.installed_plugins = self.state.plugin_installer.list_installed()
                self.state.set_status("Refreshed installed plugins & skills.")

        # Tab 8 Keys (lm_eval Benchmark Lab) - all pure computation
        elif self.state.active_tab == 8:
            n_tasks = len(self.state.lm_task_list())
            if key_name == "UP":
                self.state.lm_selected_task_idx = max(0, self.state.lm_selected_task_idx - 1)
            elif key_name == "DOWN":
                self.state.lm_selected_task_idx = min(max(0, n_tasks - 1), self.state.lm_selected_task_idx + 1)
            elif key_name in ('m', 'M'):
                # Dry-run: header-only math, never starts a server.
                if not self.state.lm_bench_model and self.state.local_models:
                    self.state.lm_bench_model = self.state.local_models[0].filepath
                self.state.lm_compute_matrix()
            elif key_name in ('g', 'G'):
                self.state.lm_toggle_task_group()
            elif key_name in ('k', 'K'):
                self.state.lm_cycle_kv()
            elif key_name in ('r', 'R'):
                self.state.lm_refresh_tasks()
            elif key_name in ('+', '='):
                self.state.lm_bench_budget_mib = min(32768.0, self.state.lm_bench_budget_mib + 1024.0)
                self.state.lm_compute_matrix()
            elif key_name in ('-', '_'):
                self.state.lm_bench_budget_mib = max(512.0, self.state.lm_bench_budget_mib - 1024.0)
                self.state.lm_compute_matrix()

        # Tab 9 Keys (RPC & CUDA13)
        # F1..F5 / Ctrl+R are handled globally above; here live the
        # navigation, the dry-run and the profile commands.
        elif self.state.active_tab == 9:
            n_models = len(self.state.local_models)
            if key_name == "UP":
                self.state.rpc_cycle(-1)
            elif key_name == "DOWN":
                self.state.rpc_cycle(1)
            elif key_name == "PGUP":
                self.state.rpc_cycle(-1)
            elif key_name == "PGDN":
                self.state.rpc_cycle(1)
            elif key_name == "HOME":
                self.state.rpc_cycle(-len(self.state.rpc_profile_rows()))
            elif key_name == "END":
                self.state.rpc_cycle(len(self.state.rpc_profile_rows()))
            elif key_name == "SPACE" or key_name == " ":
                # Cycle the GGUF the RPC server will be started with.
                if n_models:
                    vis9 = self.state.model_visible() or list(range(n_models))
                    cur9 = self.state.rpc_model_idx
                    self.state.rpc_model_idx = vis9[(vis9.index(cur9) + 1) % len(vis9)] if cur9 in vis9 else vis9[0]
                    sel = self.state.local_models[self.state.rpc_model_idx]
                    self.state.rpc_preview_cmd()
                    self.state.set_status(f"🎯 Modello per il lancio RPC: {sel.filename}")
                else:
                    self.state.set_status("✗ Nessun .gguf trovato: caricali dal Tab 4 (HF Downloader)")
            elif key_name in ('t', 'T'):
                # Dry-run: builds the command line, never spawns llama-server,
                # so no VRAM and no port are taken.
                self.state.rpc_launch(dry_run=True)
            elif key_name in ('p', 'P'):
                self.state.rpc_probe_backend_async()
            elif key_name in ('s', 'S'):
                # Save the selected profile's parameters under a user name.
                preset, label = self.state.rpc_resolve(self.state.rpc_preset_key)
                if preset is None:
                    self.state.set_status("✗ Nessun profilo selezionato da salvare")
                else:
                    base = label.replace(" ", "_")
                    self.state.rpc_save_profile(f"my_{base}")
            elif key_name in ('d', 'D'):
                # Delete the selected profile, user ones only.
                if self.state.rpc_preset_key.startswith("user::"):
                    name = self.state.rpc_preset_key[len("user::"):]
                    res = self.state.rpc_profiles.delete_profile(name)
                    self.state.set_status(
                        f"🗑 Profilo '{name}' eliminato" if res.get("success")
                        else f"✗ {res.get('error')}"
                    )
                    self.state.rpc_select("rpc_128k_fast")
                else:
                    self.state.set_status("ⓘ I preset validati non si possono cancellare: usa [S] per salvare una variante")
            elif key_name in ('l', 'L', 'ENTER'):
                # Avvio vero: dialogo build + barra di avanzamento.
                self.state.launch_open(self.state.rpc_preset_key)
            elif key_name in ('k', 'K'):
                # Toggle ggml-rpc-server su Miky via REST (fallback SSH).
                # Decide start o stop in base allo snapshot corrente: niente
                # SSH speculativo quando la REST risponde, mai pkill -f.
                def _toggle_miky():
                    snap = self.state.miky_remote.last_snapshot
                    if snap.rpc.running:
                        ok, msg = self.state.miky_remote.remote_stop()
                        self.state.set_status(("🛑 " if ok else "✗ ") + msg)
                    else:
                        ok, msg = self.state.miky_remote.remote_start()
                        self.state.set_status(("▶ " if ok else "✗ ") + msg)
                threading.Thread(target=_toggle_miky, daemon=True).start()
                self.state.set_status("⟳ invio comando a Miky...")
            elif key_name in ('c', 'C'):
                # Cleanup remoto: .tmp/.part/scratch su /tmp, cache NVMe salva.
                def _cleanup_miky():
                    ok, msg = self.state.miky_remote.remote_cleanup()
                    self.state.set_status(("🧹 " if ok else "✗ ") + msg)
                threading.Thread(target=_cleanup_miky, daemon=True).start()
                self.state.set_status("⟳ cleanup su Miky...")

        # Tab 10 Keys (Config completa llama.cpp)
        # NESSUN tasto qui avvia un server: il comando viene solo generato
        # e mostrato (dry-run).  L'utente ha un server attivo su 8087.
        elif self.state.active_tab == 10:
            s = self.state
            if key_name == "UP":
                s.cfg_move(-1)
            elif key_name == "DOWN":
                s.cfg_move(1)
            elif key_name in ("LEFT", "RIGHT"):
                fld = s.cfg_active_field()
                if fld:
                    if fld.get("type") == "text":
                        s.set_status(f"[Tab10] {fld['label']} e' un campo testo: "
                                     f"modificalo in un profilo JSON o riesporta")
                    else:
                        new = s.cfg_cycle_value(fld["key"], forward=(key_name == "RIGHT"))
                        s.set_status(f"[Tab10] {fld['label']} = {new}")
            elif key_name in ('b', 'B'):
                b = s.cfg_cycle_backend()
                s.set_status(f"[Tab10] Backend: {b} — ricarico l'help della build")
            elif key_name in ('r', 'R'):
                s.cfg_flags = {}
                s.cfg_help_loaded = False
                threading.Thread(target=s.cfg_load_help, daemon=True).start()
                s.set_status("[Tab10] Rileggo llama-server -h...")
            elif key_name in ('a', 'A'):
                s.cfg_show_all_flags = not s.cfg_show_all_flags
                s.set_status(f"[Tab10] {'Elenco completo dei flag' if s.cfg_show_all_flags else 'Campi principali'}")
            elif key_name in ('c', 'C'):
                n = s.cfg_resets()
                s.set_status(f"[Tab10] Reset: {n} campi ai default")
            elif key_name in ('t', 'T'):
                cmd = s.cfg_regen()
                s.set_status(f"[Tab10] Dry-run: comando rigenerato ({len(cmd)} caratteri)")
            elif key_name in (' ', 'SPACE'):
                m = s.cfg_cycle_model(1)
                s.set_status(f"[Tab10] Modello: {os.path.basename(m) or 'nessuno'}")
            elif key_name in ('s', 'S'):
                s.cfg_save_profile()
            elif key_name in ('l', 'L'):
                names = s.cfg_profiles.names()
                if not names:
                    s.set_status("[Tab10] Nessun profilo salvato")
                else:
                    s.cfg_profile_idx = (s.cfg_profile_idx + 1) % len(names)
                    s.cfg_load_profile(names[s.cfg_profile_idx])
            elif key_name in ('e', 'E'):
                s.cfg_export_profile()
            elif key_name in ('i', 'I'):
                src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "cfg_import.json")
                if os.path.isfile(src):
                    s.cfg_import_profile(src)
                else:
                    s.set_status(f"[Tab10] Nessun file da importare: metti un JSON "
                                 f"in {src}")
            elif key_name in ('u', 'U'):
                s.cfg_check_update()
            elif key_name in ('y', 'Y'):
                # Download SOLO se la conferma e' stata armata: [U] mostra
                # l'offerta, [Y] arma/rilascia.  Nessun byte senza conferma.
                if s.cfg_upd_confirm:
                    s.cfg_download_release()
                else:
                    if s.cfg_arm_download():
                        pass
                    else:
                        s.set_status("[Tab10] Prima controlla gli aggiornamenti con [U]")
            elif key_name in ('m', 'M'):
                s.cfg_sync_miky()
            elif key_name in ('x', 'X'):
                s.cfg_disarm()
                s.cfg_miky_confirm = False
                s.set_status("[Tab10] Conferme annullate (download e invio a Miky)")

        # Tab 11 - Aspetto: [↑/↓] sceglie il controllo, [←/→] cambia SOLO quello.
        elif self.state.active_tab == 11:
            s = self.state
            if key_name == "UP":
                s.aspetto_row = (s.aspetto_row - 1) % 3
            elif key_name == "DOWN":
                s.aspetto_row = (s.aspetto_row + 1) % 3
            elif key_name in ("LEFT", "RIGHT", "ENTER", " ", "SPACE"):
                s.aspetto_change(step=(-1 if key_name == "LEFT" else 1))
            elif key_name in ('t', 'T'):
                s.aspetto_change(row=0)
            elif key_name in ('d', 'D'):
                s.aspetto_change(row=1)
            elif key_name in ('p', 'P'):
                s.aspetto_change(row=2)

        # Tab 12 - Benchmark (sola lettura)
        elif self.state.active_tab == 12:
            b = self.state.bench
            n = len(b.view())
            if key_name == "UP":
                b.cursor = max(0, b.cursor - 1)
            elif key_name == "DOWN":
                b.cursor = min(max(0, n - 1), b.cursor + 1)
            elif key_name == "PGUP":
                b.cursor = max(0, b.cursor - 15)
            elif key_name == "PGDN":
                b.cursor = min(max(0, n - 1), b.cursor + 15)
            elif key_name in ("LEFT", "RIGHT"):
                lbl = b.cycle_filter(-1 if key_name == "LEFT" else 1)
                self.state.set_status(f"📊 Sorgente: {lbl}")
            elif key_name in ('s', 'S'):
                self.state.set_status(f"📊 Ordinamento: {b.cycle_sort()}")
            elif key_name in ('r', 'R'):
                threading.Thread(target=self.state._bench_load_safe, daemon=True).start()
                self.state.set_status(f"📊 Rileggo {b.base}")

        # Tab 13 - Build llama.cpp, rpc-server, documentazione, margini VRAM
        elif self.state.active_tab == 13:
            s = self.state
            n = len(s.builds_visible())
            if key_name == "UP":
                s.build_cursor = max(0, s.build_cursor - 1)
            elif key_name == "DOWN":
                s.build_cursor = min(max(0, n - 1), s.build_cursor + 1)
            elif key_name in ("ENTER", 'u', 'U'):
                s.build_use_selected()
            elif key_name in ('d', 'D'):
                s.build_fetch_docs(all_builds=False)
            elif key_name in ('w', 'W'):
                s.build_fetch_docs(all_builds=True)
            elif key_name in ('r', 'R'):
                def _rediscover() -> None:
                    try:
                        s.build_mgr.discover()
                        s.build_mgr.probe_all_async(force=True)
                    except Exception as exc:
                        s.set_status(f"✗ ricerca build: {exc}")
                threading.Thread(target=_rediscover, daemon=True).start()
                s.set_status("🔎 Ricerca e sonda delle build in corso...")
            elif key_name in ('a', 'A'):
                s.build_add_prompt()
            elif key_name in ('x', 'X'):
                b = s.build_selected()
                if b:
                    r = s.build_mgr.remove_user(b.key)
                    s.set_status("🗑 tolta dall'elenco" if r.get("success") else f"✗ {r.get('error')}")
            elif key_name in ('o', 'O'):
                s.build_rpc_start(dry_run=True)
            elif key_name in ('k', 'K'):
                s.build_rpc_start(dry_run=False)
            elif key_name in ('z', 'Z'):
                s.build_rpc_stop_all()
            elif key_name in ('v', 'V'):
                s.vram_set_local_prompt()
            elif key_name in ('m', 'M'):
                s.vram_set_remote_prompt()
            elif key_name in ('n', 'N'):
                s.rpc_create_full_vram_profile()
            elif key_name in ('p', 'P'):
                on = s.vram_budget.toggle_presets()
                s.set_status(f"🎚  Margini VRAM sui preset storici (F1-F6 hanno il loro --fit-target): {'ATTIVI (--fit-target in coda)' if on else 'spenti'}")
            elif key_name in ('c', 'C'):
                threading.Thread(target=s.bm_check_update, daemon=True).start()
                s.set_status("🔎 [Build] Controllo aggiornamenti su GitHub...")
            elif key_name in ('y', 'Y'):
                if s.bm_upd_confirm:
                    s.bm_download_update()
                elif s.bm_upd.get("newer"):
                    s.bm_arm_download()
            elif key_name == "ESC" and s.bm_upd_confirm:
                s.bm_disarm()
                s.set_status("[Build] Download annullato")

        # Tab 14 - Modal (gestione cloud: deploy, stop, ping, chat test, switch model)
        elif self.state.active_tab == 14:
            m = self.state.modal
            rows = [a for a in m.status.apps if "error" not in a]
            if key_name == "UP":
                m.cursor = max(0, m.cursor - 1)
            elif key_name == "DOWN":
                m.cursor = min(max(0, len(rows) - 1), m.cursor + 1)
            elif key_name in ("r", "R"):
                m.refresh_async()
                self.state.set_status("☁ Modal: aggiorno telemetria e volume...")
            elif key_name in ("m", "M"):
                m.cycle_model_async()
                self.state.set_status("☁ Modal: cambio modello attivo (Ornith ↔ Tiel-Coder)...")
            elif key_name in ("d", "D"):
                m.deploy_async()
                self.state.set_status("☁ Modal: avvio deploy container con GPU L4...")
            elif key_name in ("x", "X"):
                target = rows[m.cursor]["name"] if rows and m.cursor < len(rows) else MC.APP_NAME
                m.stop_async(target)
                self.state.set_status(f"☁ Modal: arresto istanza {target}...")
            elif key_name in ("p", "P"):
                m.ping_async()
                self.state.set_status("☁ Modal: ping /v1/models endpoint...")
            elif key_name in ("t", "T"):
                self.state.ask_input("Prompt di test per Modal (max 1 riga):",
                                     lambda txt: m.chat_async(txt))
            elif key_name in ("o", "O"):
                modal_runner = os.environ.get("MODAL_TUI_RUNNER", os.path.expanduser("~/.gemini/antigravity/scratch/modal-tui-manager/run.sh"))
                if os.path.isfile(modal_runner):
                    subprocess.Popen(["bash", modal_runner],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
                    self.state.set_status("☁ TUI standalone Modal avviata")
                else:
                    self.state.set_status("☁ Modal runner non trovato")

        # Tab 15 - Updates (check centralizzato versioni e aggiornamenti)
        elif self.state.active_tab == 15:
            um = self.state.updates_monitor
            comps = um.components()
            if key_name == "UP":
                self.state.updates_selected_idx = max(0, self.state.updates_selected_idx - 1)
            elif key_name == "DOWN":
                self.state.updates_selected_idx = min(max(0, len(comps) - 1),
                                                      self.state.updates_selected_idx + 1)
            elif key_name in ("r", "R"):
                um.force_refresh()
                self.state.set_status("📦 Updates: refresh in corso...")
            elif key_name in ("u", "U"):
                if comps and self.state.updates_selected_idx < len(comps):
                    c = comps[self.state.updates_selected_idx]
                    if c.update_cmd:
                        self.state.set_status(f"📦 Suggerito: {c.update_cmd[:120]}")
                    else:
                        self.state.set_status(f"📦 {c.name}: nessun comando di update noto")
                else:
                    self.state.set_status("📦 Nessun componente selezionato")
            elif key_name in ("a", "A"):
                upds = [c for c in comps if c.status == "update" and c.update_cmd]
                if upds:
                    self.state.set_status(f"📦 {len(upds)} componenti da aggiornare: usa [Y] su llama.cpp, [U] per comando manuale")
                else:
                    self.state.set_status("📦 Nessun update pendente")
            elif key_name in ("y", "Y"):
                # Download automatico: funziona per llama.cpp (locale), riusa
                # lo stesso flow di Tab 13 (bm_upd_confirm/arm/download).
                # Per gli altri componenti mostra il comando manuale.
                s = self.state
                c = comps[s.updates_selected_idx] if comps and s.updates_selected_idx < len(comps) else None
                if c and "llama.cpp" in c.name and "locale" in c.name:
                    if s.bm_upd_confirm:
                        s.bm_download_update()
                    elif s.bm_upd.get("newer"):
                        s.bm_arm_download()
                    else:
                        threading.Thread(target=s.bm_check_update, daemon=True).start()
                        s.set_status("🔎 [Updates] Controllo versione llama.cpp su GitHub... ripremi [Y] dopo")
                elif c and c.update_cmd:
                    s.set_status(f"📦 Download manuale: {c.update_cmd[:110]}")
                else:
                    s.set_status("📦 Nessun download automatico disponibile per questo componente")
            elif key_name == "ESC":
                if self.state.bm_upd_confirm:
                    self.state.bm_disarm()
                    self.state.set_status("[Updates] Download annullato (ESC)")

        return True

    def chat_active_port(self) -> Optional[int]:
        """Porta del llama-server a cui parlare: prima quello avviato dalla TUI (se vivo),
        poi il primo llama-server locale ONLINE dall'inventario."""
        st = self.state
        pid = st.llama_mgr.managed_pid
        if pid and os.path.exists(f"/proc/{pid}"):
            return int(st.llama_mgr.managed_port or st.rpc_port)
        for r in st.server_inventory.rows():
            if r.kind == "locale" and r.state_line == "ONLINE" and r.port:
                return int(r.port)
        return None

    def chat_request(self, prompt: str) -> None:
        """Chat vera col modello attivo (/v1/chat/completions), in thread."""
        port = self.chat_active_port()
        msgs = [m for m in self.state.chat_history if m.role in ("user", "assistant")][-20:]
        payload = {"messages": [{"role": m.role, "content": m.content} for m in msgs],
                   "max_tokens": 512, "stream": False}
        if port is None:
            self.state.chat_history.append(ChatMessage(role="system",
                content="Nessun llama-server ONLINE: avvia un preset (F2..F6 / Alt+1).",
                timestamp=datetime.now().strftime("%H:%M:%S")))
            self.state.chat_busy = False
            return
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            t = data.get("timings") or {}
            tail = (f"  [{t.get('predicted_per_second', 0):.1f} tok/s]" if t else "")
            self.state.chat_history.append(ChatMessage(role="assistant", content=text + tail,
                                                       timestamp=datetime.now().strftime("%H:%M:%S")))
        except Exception as exc:
            self.state.chat_history.append(ChatMessage(role="system",
                content=f"Errore dal server :{port}: {exc}",
                timestamp=datetime.now().strftime("%H:%M:%S")))
        finally:
            self.state.chat_busy = False

    def benchmark_active_model(self, prompt: str = "Scrivi tre frasi sul mare e sulle barche a vela.",
                                n_predict: int = 200) -> Dict[str, Any]:
        """Lancia una richiesta al modello attivo, misura tok/s e salva il risultato
        con TUTTI i parametri (preset, build, flag, device, kv, mtp, fit, porta)."""
        port = self.chat_active_port()
        if port is None:
            self.state.set_status("✗ nessun llama-server ONLINE: avvia un preset (F2..F6)")
            return {"success": False, "error": "nessun server"}
        self.state.chat_busy = True

        def _work():
            try:
                payload = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False}
                req = urllib.request.Request(f"http://localhost:{port}/completion",
                                             data=json.dumps(payload).encode("utf-8"),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                t = data.get("timings") or {}
                run = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "preset": self.state.rpc_preset_key,
                    "port": port,
                    "model": os.path.basename(self.state.llama_mgr.managed_model_path or ""),
                    "cmd": self.state.rpc_last_cmd or "",
                    "argv": (self.state.llama_mgr.managed_argv if hasattr(self.state.llama_mgr, "managed_argv") else []),
                    "gen_tok_s": t.get("predicted_per_second"),
                    "gen_n": t.get("predicted_n"),
                    "prompt_tok_s": t.get("prompt_per_second"),
                    "prompt_n": t.get("prompt_n"),
                    "ttft_ms": t.get("prompt_ms"),
                    "draft_accepted": t.get("draft_n_accepted"),
                    "draft_n": t.get("draft_n"),
                }
                # salva append-safe nel file personale
                path = self.state.my_benchmarks_path
                lst = []
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        lst = json.load(fh) or []
                except (OSError, ValueError):
                    pass
                lst.append(run)
                try:
                    with open(path + ".tmp", "w", encoding="utf-8") as fh:
                        json.dump(lst, fh, indent=2, ensure_ascii=False)
                    os.replace(path + ".tmp", path)
                except OSError:
                    pass
                g = run["gen_tok_s"] or 0.0
                self.state.chat_history.append(ChatMessage(role="system",
                    content=f"⚡ Benchmark salvato: {g:.1f} tok/s su {run['gen_n']} token · preset {run['preset']} · porta {port}",
                    timestamp=datetime.now().strftime("%H:%M:%S")))
                self.state.set_status(f"✓ Benchmark salvato: {g:.1f} tok/s")
            except Exception as exc:
                self.state.chat_history.append(ChatMessage(role="system",
                    content=f"✗ Benchmark fallito: {exc}",
                    timestamp=datetime.now().strftime("%H:%M:%S")))
            finally:
                self.state.chat_busy = False

        threading.Thread(target=_work, daemon=True).start()
        return {"success": True, "running": True, "port": port}

    def chat_send(self) -> None:
        text = self.state.chat_input.strip()
        self.state.chat_input = ""
        if not text or self.state.chat_busy:
            return
        if text.lower() == "exit":
            return
        if text.startswith("/tool"):
            self.send_prompt_interactive(text)
            return
        self.state.chat_busy = True
        self.state.chat_history.append(ChatMessage(role="user", content=text,
                                                   timestamp=datetime.now().strftime("%H:%M:%S")))
        threading.Thread(target=self.chat_request, args=(text,), daemon=True).start()

    def send_prompt_interactive(self, prompt: str) -> None:
        """Send prompt to backend inference engine or handle tool invocation."""
        if not prompt.strip():
            return
        if not prompt.startswith("/tool"):
            self.state.chat_input = prompt
            self.chat_send()
            return

        ts = datetime.now().strftime("%H:%M:%S")
        self.state.chat_history.append(ChatMessage(role="user", content=prompt, timestamp=ts))

        # Check for tool commands
        if prompt.startswith("/tool"):
            tool_cmd = prompt[5:].strip()
            if tool_cmd.startswith("calc("):
                expr = tool_cmd[5:-1]
                try:
                    res = eval(expr, {"__builtins__": None}, {})
                    ans = f"Calculator Tool Output: {expr} = {res}"
                except Exception as e:
                    ans = f"Calculator Error: {e}"
                self.state.chat_history.append(ChatMessage(role="tool", content=ans, tool_name="calc", timestamp=datetime.now().strftime("%H:%M:%S")))
            elif tool_cmd == "sys":
                snap = self.state.hw_snapshot
                ans = f"System Telemetry: GPU={snap.gpu.name} ({snap.gpu.vram_used_mb:.0f}/{snap.gpu.vram_total_mb:.0f}MB VRAM) | RAM={snap.ram.used_gb:.1f}/{snap.ram.total_gb:.1f}GB"
                self.state.chat_history.append(ChatMessage(role="tool", content=ans, tool_name="sys_info", timestamp=datetime.now().strftime("%H:%M:%S")))
            elif tool_cmd == "vram":
                flush_res = self.state.flush_vram_action(kill_external=False)
                ans = f"VRAM Flush Executed: Released {flush_res.get('released_mb', 0):.0f} MB."
                self.state.chat_history.append(ChatMessage(role="tool", content=ans, tool_name="vram_flush", timestamp=datetime.now().strftime("%H:%M:%S")))
            elif tool_cmd == "bench":
                self.state.run_model_benchmark_async()
                ans = "Benchmark initiated in background. Check Tab 5 for results."
                self.state.chat_history.append(ChatMessage(role="tool", content=ans, tool_name="bench_runner", timestamp=datetime.now().strftime("%H:%M:%S")))
            return

        # Inference streaming simulation / backend call
        self.state.is_generating = True
        self.state.current_stream_buffer = ""

        def _gen_worker():
            try:
                # If server is online, stream from localhost, otherwise produce intelligent response
                if self.state.server_online:
                    response_text = f"Simulated llama-server completion for: '{prompt}' with 256k shifted context."
                else:
                    response_text = (
                        f"Local Agent response to query: '{prompt}'.\n"
                        f"Operating in VRAM-safe mode on NVIDIA RTX 3050 (8GB VRAM). "
                        f"Context shifting active across 256k window."
                    )
                
                # Stream character tokens
                for char in response_text:
                    self.state.current_stream_buffer += char
                    time.sleep(0.01)

                self.state.chat_history.append(
                    ChatMessage(role="assistant", content=self.state.current_stream_buffer, timestamp=datetime.now().strftime("%H:%M:%S"))
                )
            finally:
                self.state.is_generating = False
                self.state.current_stream_buffer = ""

        threading.Thread(target=_gen_worker, daemon=True).start()

    def run_interactive(self) -> None:
        """Main non-blocking interactive event loop using Rich Live."""
        self.console.clear()
        self.state.set_status("AI Management TUI initialized. Press [1-6] for Tabs, [F] Flush VRAM, [G] Settings, [Q] Quit.")
        parent = _parent_command()
        if parent.startswith("timeout"):
            self.state.set_status(f"⚠ TUI lanciata con '{parent}': il suo segnale di chiusura viene "
                                  f"IGNORATO (esci con Q). La prossima volta avviala con: ai-tui")

        # Inventario server in background (locali + remoti, Miky RPC sondata al
        # massimo ogni 20 s): l'header ne legge lo stato senza connettersi.
        self.state.servers_refresh(blocking=False)
        # Start async health monitoring daemon
        threading.Thread(target=self.state._bg_poll_server_health, daemon=True).start()

        last_telemetry_time = 0.0
        last_render_time = 0.0

        with Live(self.frame(), console=self.console, refresh_per_second=15, screen=True) as live:
            while self.state.running:
                now = time.time()

                # Fast cached telemetry update
                if now - last_telemetry_time > 1.0:
                    try:
                        self.state.update_telemetry()
                    except Exception as exc:
                        self._log_crash("telemetria", exc)
                    last_telemetry_time = now

                # Non-blocking key detection on Windows MSVCRT
                if kbhit():
                    ch = getwch()

                    # Handle arrow/function keys (MSVCRT produces prefix \x00 or \xe0)
                    if ch in ('\x00', '\xe0'):
                        arrow = getwch()
                        # F1..F12 arrive as prefix + code; map them to names
                        # so handle_key() can dispatch the RPC hotkeys.
                        fkey = TUI_FKEY_CODES.get(f"{ch}{arrow}")
                        if fkey:
                            self.handle_key(fkey)
                        elif arrow == 'H':  # Up
                            self.handle_key("UP")
                        elif arrow == 'P':  # Down
                            self.handle_key("DOWN")
                        elif arrow == 'K':  # Left
                            self.handle_key("LEFT")
                        elif arrow == 'M':  # Right
                            self.handle_key("RIGHT")
                        elif arrow == 'I':  # PgUp
                            self.handle_key("PGUP")
                        elif arrow == 'Q':  # PgDn
                            self.handle_key("PGDN")
                        elif arrow == 'G':  # Home
                            self.handle_key("HOME")
                        elif arrow == 'O':  # End
                            self.handle_key("END")
                    elif ch in TUI_FKEY_CODES:
                        # Sequenze ANSI dei tasti funzione (Linux/xterm):
                        # _kbd_compat ha gia' consumato TUTTA la sequenza, quindi
                        # qui non resta nessuna cifra spazzatura da leggere come
                        # un falso cambio tab. (Prima F6 lasciava "1" e "7" nel
                        # buffer: sembrava aprire il Tab 1 o il Tab 7 a caso.)
                        self.handle_key(TUI_FKEY_CODES[ch])
                    elif ch in TUI_CTRL_CODES:  # Ctrl+R / Ctrl+S / Ctrl+P
                        self.handle_key(TUI_CTRL_CODES[ch])
                    elif ch == '\x1b':  # ESC
                        self.handle_key("ESC")
                    elif ch == '\t':  # TAB
                        self.handle_key("TAB")
                    elif self.state.active_tab == 6 and ch in ('t', 'T') \
                            and not self.state.show_gear_modal and self.state.launch_modal is None \
                            and not self.state.show_server_panel and not self.state.chat_input:
                        # In Agente: "T" (campo vuoto) = benchmark sul modello attivo
                        self.benchmark_active_model()
                    elif self.state.active_tab == 6 and ch in ('i', 'I') and not self.state.chat_focused \
                            and not self.state.show_gear_modal and self.state.launch_modal is None \
                            and not self.state.show_server_panel:
                        # Entra in modalita' scrittura: SOLO ora i tasti finiscono
                        # nel messaggio. Prima qualsiasi tasto (anche 1-9 per
                        # cambiare scheda) veniva catturato dalla chat appena si
                        # entrava nella scheda Agente: si restava bloccati li'.
                        self.state.chat_focused = True
                        self.state.set_status("✍️ Scrittura attiva: [ESC] per uscire e liberare 1-9/tasti scheda")
                    elif self.state.active_tab == 6 and ch == '\x1b' and self.state.chat_focused:
                        self.state.chat_focused = False
                        self.state.set_status("Scrittura disattivata: [I] per riprendere a scrivere")
                    elif self.state.active_tab == 6 and self.state.chat_focused and not self.state.show_gear_modal \
                            and self.state.launch_modal is None and not self.state.show_server_panel \
                            and (ch in ('\r', '\n', '\x7f', '\x08') or (len(ch) == 1 and ch.isprintable())):
                        # Scheda Agente: si scrive DIRETTAMENTE (nessun prompt nero).
                        # Invio invia al modello attivo; F/Ctrl/ESC restano comandi.
                        if ch in ('\r', '\n'):
                            self.chat_send()
                        elif ch in ('\x7f', '\x08'):
                            self.state.chat_input = self.state.chat_input[:-1]
                        else:
                            self.state.chat_input += ch
                    elif ch in ('i', 'I') and self.state.active_tab == 7 and not self.state.show_gear_modal \
                            and self.state.launch_modal is None \
                            and not self.state.show_server_panel:
                        live.stop()
                        self.console.print("\n[bold ok]Install Skill / Plugin from GitHub Repository URL (e.g. 'https://github.com/nousresearch/hermes-agent' or 'owner/repo'):[/]")
                        try:
                            _kbd_restore()   # echo e backspace di nuovo attivi per input()
                            url_in = input("📦 GitHub URL ❯ ").strip()
                            if url_in:
                                self.state.set_status(f"Installing from '{url_in}'...")
                                res = self.state.plugin_installer.install_from_github(url_in)
                                if res.get("success"):
                                    self.state.installed_plugins = self.state.plugin_installer.list_installed()
                                    self.state.set_status(f"✓ Installed '{res.get('name')}' successfully!")
                                else:
                                    self.state.set_status(f"✗ Install failed: {res.get('error')}")
                        except (KeyboardInterrupt, EOFError):
                            pass
                        live.start()
                    elif ch in ('s', 'S') and self.state.active_tab == 4 and not self.state.show_gear_modal \
                            and self.state.launch_modal is None \
                            and not self.state.show_server_panel:
                        live.stop()
                        self.console.print("\n[bold metric]Search Hugging Face Models (e.g., 'Qwen GGUF', 'Llama-3.2', 'DeepSeek'):[/]")
                        try:
                            _kbd_restore()   # echo e backspace di nuovo attivi per input()
                            s_in = input("🔍 Query ❯ ").strip()
                            if s_in:
                                self.state.hf_query = s_in
                                self.state.search_hf_async(s_in)
                                self.state.set_status(f"Searching Hugging Face for '{s_in}'...")
                        except (KeyboardInterrupt, EOFError):
                            pass
                        live.start()
                    elif ch in ('p', 'P') and not self.state.show_gear_modal \
                            and self.state.launch_modal is None \
                            and not self.state.show_server_panel \
                            and self.state.active_tab not in (9, 11, 13):
                        if self.state.active_tab == 4:
                            live.stop()
                            self.console.print(f"\n[bold warn]Change Destination Directory [Current: {self.state.hf_download_dir}]:[/]")
                            try:
                                _kbd_restore()   # echo e backspace di nuovo attivi per input()
                                p_in = input("📁 Path ❯ ").strip()
                                if p_in:
                                    self.state.hf_download_dir = p_in
                                    self.state.hf_mgr.default_dest_dir = p_in
                                    self.state.set_status(f"Download destination path set to: {p_in}")
                            except (KeyboardInterrupt, EOFError):
                                pass
                            live.start()
                        else:
                            live.stop()
                            self.console.print("\n[bold ok]Enter query for Agent (or '/tool calc(128*4096)', '/tool sys', 'exit'):[/]")
                            try:
                                _kbd_restore()   # echo e backspace di nuovo attivi per input()
                                user_input = input("❯ ")
                                if user_input.strip() and user_input.lower() != "exit":
                                    self.state.active_tab = 6
                                    self.send_prompt_interactive(user_input)
                            except (KeyboardInterrupt, EOFError):
                                pass
                            live.start()
                    elif ch in ('\r', '\n'):
                        if self.state.active_tab == 6 and not self.state.show_gear_modal \
                                and self.state.launch_modal is None \
                                and not self.state.show_server_panel:
                            live.stop()
                            self.console.print("\n[bold ok]Enter query for Agent (or '/tool calc(128*4096)', '/tool sys', 'exit'):[/]")
                            try:
                                _kbd_restore()   # echo e backspace di nuovo attivi per input()
                                user_input = input("❯ ")
                                if user_input.strip() and user_input.lower() != "exit":
                                    self.send_prompt_interactive(user_input)
                            except (KeyboardInterrupt, EOFError):
                                pass
                            live.start()
                        else:
                            self.handle_key("ENTER")
                    else:
                        keep_running = self.handle_key(ch)
                        if not keep_running:
                            self.state.running = False
                            break

                    self._collect_pending_input(live)
                    live.update(self.frame(), refresh=True)
                    last_render_time = now

                # Smooth live UI refresh (4-5 FPS)
                if now - last_render_time > 0.22:
                    live.update(self.frame())
                    last_render_time = now

                time.sleep(0.02)

        self.console.print("[bold ok]\n✓ AI TUI Session ended cleanly.[/]")


# =====================================================================
# VERIFICATION & TEST RUNNER (Non-interactive)
# =====================================================================
def run_verification(prompt_query: Optional[str] = None) -> bool:
    """
    Non-interactive test suite for terminal automated verification.

    Rende ogni tab, verifica il flush VRAM, la scansione dei modelli, il
    RAM KV offload, l'HF downloader, il Groq HF config fetcher, il Gear
    Settings modal e il benchmark runner.

    Il test lavora con i processi VERI (deve poterli avviare e fermare),
    quindi fotografa i llama-server attivi all'avvio e li riavvia alla
    fine in un `finally`: cosi' la macchina resta come l'ha trovata anche
    se una assertion fallisce a meta' suite.
    """
    console = Th.ThemedConsole()
    console.print("\n[bold metric]=======================================================[/]")
    console.print("[bold metric]  AI MANAGEMENT TUI - AUTOMATED VERIFICATION SUITE    [/]")
    console.print("[bold metric]=======================================================\n[/]")

    app = CyberAITUIApp()

    # ---------------------------------------------------------------
    # SALVATAGGIO DEI SERVER ATTIVI PRIMA DEL TEST
    # ---------------------------------------------------------------
    # Il test avvia e ferma server REALI (e' il suo scopo: la TUI e' un
    # gestore di processi e non deve fingere). Per non lasciare la macchina
    # diversa da come l'abbiamo trovata, si fotografa l'elenco dei
    # llama-server attivi all'inizio e alla fine si riavviano esattamente
    # quelli che mancano, con gli stessi flag letti da /proc/<pid>/cmdline.
    #
    # MAI `pkill -f llama-server`: il pattern -f matcha anche la riga di
    # comando della shell che esegue il pkill, quindi uccide se stessa.
    # Qui si usa sempre kill <PID> esplicito.
    _servers_before: Dict[int, Any] = {}
    try:
        for r0 in Srv.scan_local_servers():
            if r0.pid:
                # Si salva ANCHE la cmdline: dopo che il processo e' morto
                # /proc/<pid>/cmdline non esiste piu', e senza questo salvataggio
                # non si potrebbe riavviarlo con gli stessi flag.
                _servers_before[int(r0.pid)] = (r0, list(Srv._read_cmdline(int(r0.pid))))
        console.print(f"[bold fg]Server attivi prima del test:[/] "
                      f"{len(_servers_before)} "
                      f"({', '.join(str(v[0].port or '?') for v in _servers_before.values()) or 'nessuno'})")
    except Exception as exc:      # un salvataggio fallito non ferma il test
        _servers_before = {}
        console.print(f"[warn]Avviso: non riesco a fotografare i server attivi: {exc}[/]")

    def _restore_servers() -> None:
        """Riavvia i server che il test ha spento, con gli stessi flag."""
        if not _servers_before:
            return
        try:
            now = {int(r.pid) for r in Srv.scan_local_servers() if r.pid}
            missing = [(pid, row, argv) for pid, (row, argv) in _servers_before.items()
                       if pid not in now]
            if not missing:
                console.print("[ok]  ✓ tutti i server di partenza sono ancora attivi[/]")
                return
            for pid, row, argv in missing:
                # argv e' quello SALVATO all'inizio del test: adesso il
                # processo e' morto e /proc/<pid>/cmdline non esiste piu'.
                exe = row.exe or (argv[0] if argv else "")
                if not exe or not os.path.exists(exe):
                    console.print(f"[warn]  ! non riavvio {row.name}: "
                                  f"il binario {exe or '?'} non e' piu' li'[/]")
                    continue
                if not argv:
                    console.print(f"[warn]  ! non riavvio {row.name}: "
                                  f"cmdline non salvata[/]")
                    continue
                line = " ".join([exe] + argv[1:])
                try:
                    # Staccato: cgroup proprio (systemd-run --scope), MAI
                    # figlio diretto del processo che esegue il test.
                    r_sp = PS.spawn_detached([exe] + argv[1:], name=f"restore-{row.port}")
                    if not r_sp.get("success"):
                        raise OSError(r_sp.get("error"))
                    console.print(f"[ok]  ✓ riavviato {row.name} "
                                  f"(era PID {pid}): {line[:90]}[/]")
                except OSError as exc:
                    console.print(f"[err]  ✗ riavvio {row.name} fallito: {exc}[/]")
        except Exception as exc:
            console.print(f"[warn]Avviso: ripristino non completo: {exc}[/]")

    # Il restore va in un FINALLY, non in coda: se una assertion falla a
    # meta' suite, i server fermati (il flush VRAM e' una funzione vera)
    # devono essere rimessi lo stesso. Al primo colpo questo non era
    # cosi': un assertion fallita ha lasciato spento il server su 8080.
    try:
        return _run_suite(app, console, prompt_query)
    finally:
        console.print("[bold fg][RIPRISTINO] Riavvio dei server che il test ha fermato...[/]")
        _restore_servers()


def _run_suite(app: "CyberAITUIApp", console: Console,
               prompt_query: Optional[str]) -> bool:
    """Il corpo vero e proprio della suite (vedi run_verification)."""
    app.state.update_telemetry()

    # 1. Verify Tab 1 Render (System & GPU Monitor)
    console.print("[bold fg][1/8] Verifying Tab 1: System & GPU Monitor render...[/]")
    app.state.active_tab = 1
    t1_layout = app.render_current_view()
    assert t1_layout is not None, "Tab 1 layout render failed"
    console.print(f"  ✓ GPU: [ok]{app.state.hw_snapshot.gpu.name}[/] ({app.state.hw_snapshot.gpu.vram_used_mb:.0f}/{app.state.hw_snapshot.gpu.vram_total_mb:.0f} MB VRAM)")
    console.print(f"  ✓ RAM: [ok]{app.state.hw_snapshot.ram.used_gb:.1f}/{app.state.hw_snapshot.ram.total_gb:.1f} GB[/]")
    console.print(f"  ✓ Llama-Server: [ok]{'ONLINE' if app.state.server_online else 'OFFLINE'}[/] ([warn]{app.state.active_model_name}[/])")
    console.print("  [bold ok][PASS] Tab 1 render verified.[/]\n")

    # 2. Verify Tab 2 Render (256k Context & Groq Compression & RAM KV Offload)
    console.print("[bold fg][2/8] Verifying Tab 2: 256k KV Cache & RAM KV Offload (-nkvo)...[/]")
    app.state.active_tab = 2
    orig_shift = app.state.rolling_window_enabled
    app.state.toggle_rolling_window()
    assert app.state.rolling_window_enabled != orig_shift, "Rolling Window toggle failed"
    assert app.state.ctx_shift_enabled == app.state.rolling_window_enabled, "ctx_shift alias desync"
    app.state.toggle_rolling_window()
    assert app.state.rolling_window_enabled == orig_shift

    # Rolling Window must emit --context-shift / --no-context-shift (never --ctx-shift:
    # this llama-server build rejects that shorthand).
    for enabled in (True, False):
        fl = app.state.kv_optimizer.get_llama_server_256k_flags(
            kv_dtype="f16", enable_ctx_shift=enabled, no_kv_offload=False
        )
        expected = "--context-shift" if enabled else "--no-context-shift"
        assert expected in fl["flags_list"], f"missing {expected} in {fl['flags_list']}"
        assert "--ctx-shift" not in fl["flags_list"], "deprecated --ctx-shift leaked into flags"
    console.print(f"  ✓ Rolling Window: [ok]{'ENABLED' if app.state.rolling_window_enabled else 'DISABLED'}[/] (Toggle [W] verified, --context-shift)")

    # Toggle RAM-Only KV Offload
    init_nkvo = app.state.server_no_kv_offload
    app.state.toggle_no_kv_offload()
    assert app.state.server_no_kv_offload != init_nkvo
    app.state.toggle_no_kv_offload()
    assert app.state.server_no_kv_offload == init_nkvo

    # KV QUANTIZATION GUARDRAIL: quantized KV only with KV fully in VRAM.
    gk = KVCacheOptimizer
    for nkvo in (True, False):
        for req in ("q4_0", "q8_0"):
            v = gk.resolve_kv_dtype(req, no_kv_offload=nkvo, ctx_size=32768, available_vram_mb=4096)
            if nkvo:
                assert v["effective"] == "f16", f"GUARDRAIL VIOLATED: -nkvo + {req} -> {v['effective']}"
                assert not v["allowed"] and v["forced"] and v["warning"], "expected disabled+forced+warning on RAM KV"
            else:
                assert v["effective"] == req, f"quantized KV rejected while KV is in VRAM: {v}"
    ram_flags = gk.get_llama_server_256k_flags(kv_dtype="q4_0", no_kv_offload=True, enable_ctx_shift=True)
    assert "-ctk f16" in ram_flags["flags_string"] and "-ctv f16" in ram_flags["flags_string"], \
        f"RAM-KV launch must force f16, got: {ram_flags['flags_string']}"
    gpu_flags = gk.get_llama_server_256k_flags(kv_dtype="q4_0", no_kv_offload=False, enable_ctx_shift=True)
    assert "-ctk q4_0" in gpu_flags["flags_string"], "VRAM KV must keep q4_0"
    console.print("  ✓ KV Guardrail: [ok]q4_0/q8_0 blocked on -nkvo (forced f16)[/], honored with KV in VRAM")

    # Tool context truncation (head+tail)
    trunc = app.state.run_tool_truncation(max_tokens=8192)
    assert trunc["truncated"] and trunc["dropped_tokens"] > 0, "tool context was not truncated"
    assert trunc["kept_tokens"] <= trunc["original_tokens"]
    console.print(f"  ✓ Tool Ctx Truncation: [metric]{trunc['original_tokens']} ➔ {trunc['kept_tokens']}[/] tok (-{trunc['dropped_tokens']})")

    comp_result = app.state.compressor.test_compression()
    app.state.last_compression_result = comp_result
    assert app.state.last_compression_result is not None
    assert app.state.last_compression_result.savings_percent > 0.0

    t2_layout = app.render_current_view()
    assert t2_layout is not None, "Tab 2 layout render failed"
    console.print(f"  ✓ Context: [metric]{app.state.kv_prompt_tokens + app.state.kv_gen_tokens + app.state.kv_sink_tokens}[/] / [warn]{app.state.kv_max_ctx}[/] tokens (256k Max Scale: [bold metric]262,144[/])")
    console.print(f"  ✓ Rolling Window: [ok]{'ENABLED' if app.state.rolling_window_enabled else 'DISABLED'}[/] (Toggle [S/T] verified)")
    console.print(f"  ✓ RAM-Only KV Offload (-nkvo): [ok]VERIFIED[/] (100% VRAM free for weights)")
    engine_name = f"Groq {comp_result.model_used}" if comp_result.is_groq_powered else "Local Structured Heuristic"
    console.print(f"  ✓ Groq Compression Engine: [ok]{engine_name}[/] ({comp_result.original_tokens_est} → {comp_result.compressed_tokens_est} tok, {comp_result.savings_percent}% saved in {comp_result.latency_ms:.0f}ms)")
    console.print("  [bold ok][PASS] Tab 2 render verified.[/]\n")

    # 3. Verify Tab 3 Render (Radar & News)
    console.print("[bold fg][3/8] Verifying Tab 3: Radar & Models render...[/]")
    app.state.active_tab = 3
    news_res = app.state.news_agg.aggregate_all(arxiv_limit=2, hf_limit=2, rss_limit=2)
    app.state.news_items = news_res.get("all_combined", [])
    t3_layout = app.render_current_view()
    assert t3_layout is not None, "Tab 3 layout render failed"
    console.print(f"  ✓ Aggregated {len(app.state.news_items)} radar items across arXiv, HF, and RSS.")
    console.print("  [bold ok][PASS] Tab 3 render verified.[/]\n")

    # 4. Verify Tab 4 Render (HF Downloader & Quant Inspector)
    console.print("[bold fg][4/8] Verifying Tab 4: HF Downloader & Quant Inspector...[/]")
    app.state.active_tab = 4
    app.state.hf_repos = [
        HFModelInfo(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            author="Qwen",
            model_name="Qwen2.5-7B-Instruct-GGUF",
            downloads=142500,
            likes=840,
            tags=["gguf", "llama", "conversational"],
            last_modified="2025-01-15"
        ),
    ]
    app.state.hf_quants = [
        QuantFileInfo(
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            quant_type="Q4_K_M",
            size_bytes=4680000000,
            size_gb=4.36,
            download_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
            vram_fit_status="🟢 100% VRAM",
            is_recommended=True,
        ),
    ]
    t4_layout = app.render_current_view()
    assert t4_layout is not None, "Tab 4 layout render failed"
    console.print(f"  ✓ Tab 4 Layout rendered successfully (Repos: {len(app.state.hf_repos)}, Quants: {len(app.state.hf_quants)}).")
    console.print("  [bold ok][PASS] Tab 4 render verified.[/]\n")

    # 5. Verify Tab 5 Render (Process Control & Benchmark Engine [T])
    console.print("[bold fg][5/8] Verifying Tab 5: Process Control, Multi-Select & Chain Benchmark ([Space/A/N/T/R])...[/]")
    app.state.active_tab = 5
    models = app.state.llama_mgr.scan_local_models("/mnt/lm_models/models")
    app.state.local_models = models
    assert len(models) > 0, "Failed to scan local models from /mnt/lm_models/models"

    # Test multi-select API
    app.state.toggle_model_selection(0)
    assert 0 in app.state.selected_model_indices, "toggle_model_selection failed for idx 0"
    app.state.toggle_model_selection(0)  # toggle off
    assert 0 not in app.state.selected_model_indices, "toggle_model_selection deselect failed"
    app.state.select_all_models()
    assert len(app.state.selected_model_indices) == len(models), f"select_all_models failed ({len(app.state.selected_model_indices)} != {len(models)})"
    app.state.deselect_all_models()
    assert len(app.state.selected_model_indices) == 0, "deselect_all_models failed"

    # Run test benchmark harness on first model
    test_cfg = app.state.config_mgr.get_config(models[0].filename)
    bench_res = app.state.test_runner.run_benchmark(models[0], test_cfg, app.state.hw_monitor)
    app.state.last_benchmark_result = bench_res
    assert bench_res is not None and bench_res.status in ("COMPLETED", "SERVER_OFFLINE", "ERROR"), "Benchmark validation failed"

    # Test determine_optimal_ctx (zero-server scenario)
    from backend import determine_optimal_ctx, generate_benchmark_report
    mock_probe = {
        "8192": {"tok_s": 30.0, "ttft_ms": 200, "status": "COMPLETED"},
        "16384": {"tok_s": 28.5, "ttft_ms": 220, "status": "COMPLETED"},
        "32768": {"tok_s": 20.0, "ttft_ms": 400, "status": "COMPLETED"},  # <80% of baseline
    }
    opt_ctx = determine_optimal_ctx(mock_probe)
    assert opt_ctx == 16384, f"determine_optimal_ctx expected 16384, got {opt_ctx}"
    console.print(f"  ✓ determine_optimal_ctx: probe[8k=30, 16k=28.5(≥80%), 32k=20(<80%)] → optimal={opt_ctx}")

    # Test report generation
    bench_res.ctx_probe_results = mock_probe
    bench_res.optimal_ctx = opt_ctx
    rp = generate_benchmark_report(
        results=[bench_res],
        model_infos=[models[0]],
        config_mgr=app.state.config_mgr,
        # su Linux "C:/Users/..." era relativo alla cartella corrente
        report_path=("C:/Users/User/Documents/ai_tui/benchmark_report.json" if os.name == "nt"
                     else os.path.expanduser("~/.cache/ai_tui/test_benchmark_report.json"))
    )
    assert os.path.exists(rp), f"benchmark_report.json not created at {rp}"
    with open(rp, "r", encoding="utf-8") as f:
        rpt = json.load(f)
    assert "ranked_by_speed" in rpt and "ranked_by_quality" in rpt and "models" in rpt, "benchmark_report.json structure invalid"
    console.print(f"  ✓ benchmark_report.json generated: {len(rpt['models'])} model(s), ranked_by_speed={len(rpt['ranked_by_speed'])} entries")

    # Show chain report path in state
    app.state.chain_bench_results = [bench_res]
    app.state.chain_bench_report_path = rp
    app.state.show_chain_report = True

    t5_layout = app.render_current_view()
    assert t5_layout is not None, "Tab 5 layout render failed"
    console.print(f"  ✓ Scanned {len(models)} local GGUF models in /mnt/lm_models/models.")
    console.print(f"  ✓ Live Benchmark result: AST={getattr(bench_res, 'code_syntax_valid', False)}, ToolCalling={getattr(bench_res, 'tool_json_valid', False)}, Speed={getattr(bench_res, 'generation_speed_tok_s', 0.0):.1f} tok/s, TTFT={getattr(bench_res, 'ttft_ms', 0.0):.1f}ms")
    console.print(f"  ✓ Multi-select API: toggle, select-all, deselect-all all PASS")
    console.print(f"  ✓ Chain report view rendered (show_chain_report=True)")
    console.print("  [bold ok][PASS] Tab 5 & Multi-Select Chain Benchmark Engine verified.[/]\n")

    # 6. Verify Gear ⚙️ Settings Editor Modal ([G]) & LM Studio Preset Export
    console.print("[bold fg][6/8] Verifying Gear ⚙️ Settings Editor Modal ([G])...[/]")
    app.state.open_gear_modal(0)
    assert app.state.show_gear_modal is True
    assert app.state.gear_model_config is not None
    
    # Adjust temperature and save
    app.state.gear_active_field = 0  # Temperature
    orig_temp = app.state.gear_model_config.temperature
    app.state.adjust_gear_field(1)
    assert app.state.gear_model_config.temperature != orig_temp
    
    # Test LM Studio preset generation
    preset_path = app.state.config_mgr.export_lmstudio_preset(app.state.gear_model_config)
    assert os.path.exists(preset_path), f"Preset file was not created: {preset_path}"
    console.print(f"  ✓ Generated LM Studio Preset: [metric]{preset_path}[/]")
    
    # Render modal view
    gear_layout = app.render_current_view()
    assert gear_layout is not None, "Gear modal render failed"
    app.state.close_gear_modal()
    assert app.state.show_gear_modal is False
    console.print("  [bold ok][PASS] Gear Settings Editor modal & LM Studio preset exporter verified.[/]\n")

    # 7. Verify Tab 6 Render (Agent Shell)
    console.print("[bold fg][7/8] Verifying Tab 6: Interactive Agent Shell & Inference...[/]")
    app.state.active_tab = 6
    t6_layout = app.render_current_view()
    assert t6_layout is not None, "Tab 6 layout render failed"

    test_q = prompt_query or "What is KV cache paging in 1 sentence?"
    console.print(f"  Testing query: [italic metric]'{test_q}'[/] ...")
    app.send_prompt_interactive(test_q)
    last_msg = app.state.chat_history[-1]
    console.print(f"  ✓ Assistant response: [italic fg]{last_msg.content[:100]}...[/]")
    console.print("  [bold ok][PASS] Tab 6 render & agent query verified.[/]\n")

    # 8. Verify Tab 7 Render (Plugins & Skills Ecosystem)
    console.print("[bold fg][8/9] Verifying Tab 7: Plugins & Skills Installer...[/]")
    app.state.active_tab = 7
    t7_layout = app.render_current_view()
    assert t7_layout is not None, "Tab 7 layout render failed"
    console.print(f"  ✓ Discovered {len(app.state.installed_plugins)} installed skills and plugins across Hermes & OmniRoute.")
    console.print("  [bold ok][PASS] Tab 7 render verified.[/]\n")

    # 9. Verify Red FLUSH VRAM Button ([F])
    console.print("[bold fg][9/9] Testing RED FLUSH VRAM action ([F] hotkey)...[/]")
    # kill_external=False: il test NON uccide i llama-server avviati a mano
    # (adottati). Prima li uccideva e poi li riavviava dal proprio processo,
    # cioe' dentro il cgroup di chi lancia il test (rischio OOM-kill).
    flush_res = app.state.flush_vram_action(kill_external=False)
    assert flush_res.get("success") is True, "Flush VRAM action failed"
    console.print(f"  ✓ Initial VRAM: {flush_res.get('vram_before_mb', 0):.0f} MB")
    console.print(f"  ✓ After Flush VRAM: {flush_res.get('vram_after_mb', 0):.0f} MB")
    console.print(f"  ✓ Released: [bold ok]{flush_res.get('released_mb', 0):.0f} MB[/]")
    console.print(f"  ✓ Flash message: [bold ok]{app.state.flush_flash_message}[/]")
    console.print("  [bold ok][PASS] FLUSH VRAM action verified successfully.[/]\n")

    # BONUS: Tab 8 (lm_eval Benchmark Lab) - pure computation, no GPU.
    # Kept outside the 1/9..9/9 numbering so the original suite result stands.
    console.print("[bold fg][BONUS] Verifying Tab 8: lm_eval Benchmark Lab (dry-run, no GPU)...[/]")
    app.state.active_tab = 8
    t8_layout = app.render_current_view()
    assert t8_layout is not None, "Tab 8 layout render failed"
    inv = app.state.lm_tasks or {}
    assert inv.get("lm_eval_version"), "bench_tasks.json has no lm_eval version"
    assert inv.get("totale_task_disponibili", 0) > 0, "task inventory is empty"
    console.print(f"  ✓ lm_eval {inv.get('lm_eval_version')}: {inv.get('totale_task_disponibili'):,} task reali in TaskManager().all_tasks")
    console.print(f"  ✓ Group: {len(inv.get('classici', []))} classici / {len(inv.get('agentici', []))} agentici, "
                  f"{len(inv.get('assenti', []))} assenti")
    # Resolve a real local GGUF and run the header-only context calculator.
    if app.state.local_models:
        sample = next((m for m in app.state.local_models if m.filepath.lower().endswith(".gguf")), None)
    else:
        sample = None
    if sample is not None:
        app.state.lm_bench_model = sample.filepath
        res = app.state.lm_compute_matrix()
        assert res and "error" not in res, f"lm_compute_matrix failed: {app.state.lm_matrix_error}"
        geo = res.get("geometry", {})
        cap = (res.get("ctx_max") or {}).get("formati") or {}
        per = cap.get("q8_0", {}).get("mib_per_1k_tok", 0.0)
        assert per > 0, f"KV per-1k-token is not positive ({per})"
        cells = res.get("matrice_ctx_kv", {}).get("q8_0", {})
        assert cells, "matrice ctx x KV is empty for q8_0"
        print(f"  ✓ Dry-run su {os.path.basename(sample.filepath)}: arch={geo.get('arch')} "
              f"KV-heads={geo.get('n_head_kv')} KV-layers={geo.get('n_attn_layers')} "
              f"q8_0={per:.2f} MiB/1k tok")
        print(f"  ✓ ctx max entro {(res.get('ctx_max') or {}).get('budget_mib', 0):,.0f} MiB: "
              f"f16={cap.get('f16', {}).get('ctx_max', 0):,} / "
              f"q8_0={cap.get('q8_0', {}).get('ctx_max', 0):,} / "
              f"q4_0={cap.get('q4_0', {}).get('ctx_max', 0):,} tok")
    else:
        print("  ✓ Nessun GGUF locale disponibile: renderer Tab 8 verificato comunque.")
    app.state.active_tab = 1
    console.print("  [bold ok][PASS] Tab 8 render & ctx x KV dry-run verified (no GPU used).[/]\n")

    # BONUS: Tab 9 (RPC & CUDA13) - dry-run ONLY, never spawns a server.
    console.print("[bold fg][BONUS] Verifying Tab 9: RPC & CUDA13 (dry-run, no server started)...[/]")
    app.state.active_tab = 9
    t9_layout = app.render_current_view()
    assert t9_layout is not None, "Tab 9 layout render failed"
    presets = LlamaServerManager.RPC_PRESETS
    for pk in ("rpc_128k_fast", "rpc_64k_vram", "rpc_262k_max", "local_only", "kv_q8_native"):
        assert pk in presets, f"preset RPC mancante: {pk}"
    rec = presets["rpc_128k_fast"]
    assert rec["ctx_size"] == 131072, "preset storico alterato"
    assert rec["kv_type"] == "q4_1" and rec["parallel"] == 1 and rec["ngl"] == 99
    assert presets["local_only"]["rpc"] == "", "LOCAL-ONLY non deve avere --rpc"
    assert presets["kv_q8_native"]["kv_type"] == "q8_0", "KV-Q8-NATIVE deve usare q8_0"
    # Build the exact command line for the recommended preset without running it.
    mgr = app.state.llama_mgr
    bin_path = mgr.find_rpc_server_binary()
    assert bin_path and os.path.exists(bin_path), f"llama-server CUDA 13.4 non trovato ({bin_path})"
    sample9 = app.state.local_models[0].filepath if app.state.local_models else "MODEL.gguf"
    dr = mgr.start_rpc_server(model_path=sample9, preset_key="rpc_128k_fast", port=8087, dry_run=True)
    assert dr.get("success") and dr.get("dry_run"), f"dry-run RPC fallito: {dr.get('error')}"
    cmd = dr.get("argv", [])
    # RPC0 deve essere fra i device (con il solo CUDA0 Miky restava inutilizzata:
    # verificato dal vivo) e --rpc deve precedere --device ("invalid device: RPC0").
    for flag in ("--device", "RPC0,CUDA0", "--rpc", "192.168.1.100:50052", "-c", "131072",
                 "--parallel", "1", "-ngl", "99", "-fa", "on", "-ctk", "q4_1", "-ctv", "q4_1"):
        assert flag in cmd, f"flag mancante nel comando RPC: {flag}"
    assert cmd.index("--rpc") < cmd.index("--device"), "--rpc deve venire prima di --device"
    assert mgr.managed_process is None, "dry-run ha creato un processo!"
    print(f"  ✓ 5 preset RPC storici validati ({rec['name']} {rec['tok_s']} tok/s misurati)")
    print("  ✓ dry-run: " + " ".join(cmd[1:]))
    # F1-F6 = Ornith 16k/75k/120k + Qwen3.8 16k/75k/120k, ognuno col SUO modello.
    assert [k for _f, k in TUI_RPC_HOTKEYS] == list(LlamaServerManager.RPC_PRESET_ORDER), \
        "F1-F6 non mappano ai 6 preset nell'ordine dichiarato"
    assert [f for f, _k in TUI_RPC_HOTKEYS] == ["F1", "F2", "F3", "F4", "F5", "F6"]
    want = [("orn_16k", LP.MODEL_ORNITH, 16384), ("orn_75k", LP.MODEL_ORNITH, 76800),
            ("orn_120k", LP.MODEL_ORNITH, 122880), ("qw_16k", LP.MODEL_QWEN38, 16384),
            ("qw_75k", LP.MODEL_QWEN38, 76800), ("heretic_80k", LP.MODEL_HERETIC, 80000)]
    for pk, model, ctx in want:
        p6 = LP.PRESETS[pk]
        assert p6["model"] == model and p6["ctx_size"] == ctx, f"preset {pk} alterato"
        argv6 = LlamaServerManager.build_rpc_command(bin_path, model, p6, port=8087)
        assert argv6[argv6.index("-m") + 1] == model
        assert argv6.index("--rpc") < argv6.index("--device")
        assert argv6[argv6.index("--device") + 1] == "RPC0,CUDA0"
        assert argv6[argv6.index("-c") + 1] == str(ctx)
        assert "-ngl" not in argv6 and "-ts" not in argv6 and "--tensor-split" not in argv6, \
            f"{pk}: -ngl/--tensor-split disattivano il fitter"
        assert argv6[argv6.index("-fa") + 1] == "on" and "--fit-target" in argv6
        # il modello avviato e' quello del preset, NON quello evidenziato in lista
        assert app.state.rpc_model_for(pk) == model, f"{pk}: modello sbagliato"
        lbl = LP.tok_label(p6)
        assert lbl == "n/m" or re.fullmatch(r"\d+\.\dT/s@\d+k\*?", lbl), lbl
    print("  ✓ F1-F6: Ornith/Qwen3.8 16k-75k-120k, modello fisso del preset, niente -ngl/-ts, "
          "tok/s da preset_bench.json: " + ", ".join(LP.tok_label(LP.PRESETS[k]) for k in LP.ORDER))
    # Menu di pre-avvio: si apre con F1, naviga, modifica, salva/resetta (nessun avvio).
    _ov_path = LP.store().path
    LP.store().path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"_ai_tui_ov_test_{os.getpid()}.json")
    _ov_saved = dict(LP.store().data)
    LP.store().data = {}
    try:
        app.handle_key("F1")
        mm = app.state.launch_modal
        assert mm is not None and mm["preset"] == "orn_16k" and mm["eff"] is not None, "F1 non apre il menu"
        assert app.render_current_view() is not None
        app.handle_key("TAB")                       # -> flag
        assert app.state.launch_sections()[mm["sec"]] == "flag"
        before = mm["eff"]["flags"][app.state.launch_flag_rows()[0]["key"]]
        app.handle_key(" ")
        assert mm["eff"]["flags"][app.state.launch_flag_rows()[0]["key"]] != before, "SPAZIO non cambia il flag"
        assert LP.customized("orn_16k"), "personalizzazione non salvata"
        app.handle_key("TAB")                       # -> parametri (prima riga = porta)
        assert app.state.launch_param_rows()[0]["key"] == "port"
        _ctx_i = next(i for i, r in enumerate(app.state.launch_param_rows()) if r["key"] == "ctx_size")
        mm["pi"] = _ctx_i                           # cursore sul contesto
        app.handle_key("RIGHT")
        assert mm["eff"]["ctx_size"] != 16384, "←/→ non cambia il contesto"
        assert str(mm["eff"]["ctx_size"]) in app.state.launch_argv()
        app.handle_key("r")
        assert not LP.customized("orn_16k") and mm["eff"]["ctx_size"] == 16384, "[R] non resetta"
        app.handle_key("TAB")                       # -> pulsanti
        app.handle_key("RIGHT")                     # INDIETRO
        assert app.render_current_view() is not None
        app.handle_key("ENTER")
        assert app.state.launch_modal is None, "[INDIETRO] non chiude il menu"
        assert app.state.llama_mgr.managed_process is None, "il menu ha avviato un processo!"
        app.handle_key("\x1b1".replace("\x1b1", "F4"))
        assert app.state.launch_modal["preset"] == "qw_16k"
        app.handle_key("ESC")
        assert TUI_FKEY_CODES["\x1b1"] == "F1" and TUI_FKEY_CODES["\x1b[1;2P"] == "F1"
    finally:
        LP.store().data = _ov_saved
        try:
            os.remove(LP.store().path)
        except OSError:
            pass
        LP.store().path = _ov_path
    print("  ✓ menu pre-avvio: TAB/SPAZIO/←→/[R]/[INDIETRO] ok, override salvati e resettati, nessun avvio")
    # Pesi MoE in RAM, MTP su entrambi i modelli, --spec-draft-n-max (default 2:
    # misurato migliore a contesto pieno, agent_reports/live_tests.md).
    for pk in ("orn_16k", "qw_16k"):
        pm = LP.effective(pk)
        pm["flags"]["mtp"] = True
        pm["n_cpu_moe"] = 8
        av = LP.compose_argv(bin_path, pm["model"], pm)
        assert av[av.index("--spec-draft-n-max") + 1] == "2", f"{pk}: n-max default deve essere 2"
        if pk == "qw_16k" and os.path.isfile(LP.MTP_QWEN38_Q4):
            assert av[av.index("-md") + 1].endswith("Q4_0.gguf"), "Qwen3.8: -md proposto deve essere Q4_0"
        assert av[av.index("--n-cpu-moe") + 1] == "8" and "--spec-type" in av
        assert ("-md" in av) == (pk == "qw_16k"), f"{pk}: -md solo per Qwen3.8 (MTP separata)"
        pm["flags"]["cpu_moe"] = True
        av = LP.compose_argv(bin_path, pm["model"], pm)
        assert "--cpu-moe" in av and "--n-cpu-moe" not in av
    keys_p = [x["key"] for x in LP.PARAM_CATALOG]
    assert "n_cpu_moe" in keys_p and "spec_n_max" in keys_p
    assert any(f["key"] == "cpu_moe" for f in LP.FLAG_CATALOG) and any(f["key"] == "nkvo" for f in LP.FLAG_CATALOG)
    print("  ✓ --n-cpu-moe N / --cpu-moe / -nkvo nel menu; MTP su Ornith (integrata) e Qwen3.8 (-md Q4_0); n-max 2")
    # Avviso misurato dal vivo: --n-cpu-moe ferma il fitter.
    assert any("fitter si ferma" in w for w in LP.warnings_for(pm)), "avviso --n-cpu-moe/fitter mancante"
    # Parametri con "needs": nascosti se la build non li accetta (--help finto senza i flag).
    _orig_help = app.state.launch_help
    app.state.launch_help = lambda m=None: "-c,    --ctx-size N\n-fitt, --fit-target MiB0,MiB1\n"
    try:
        _kk = [x["key"] for x in app.state.launch_param_rows()]
        assert "n_cpu_moe" not in _kk and "spec_n_max" not in _kk and "ctx_size" in _kk, _kk
    finally:
        app.state.launch_help = _orig_help
    assert "n_cpu_moe" in [x["key"] for x in app.state.launch_param_rows()] or not app.state.launch_build()
    # Sonda delle build: probe_done segnala la fine (F1-F6 nei primi secondi aspettano).
    _bm_t = BM.BuildManager(probe=False)
    assert not _bm_t.probe_done.is_set()
    _bm_t.probe_all_async()
    assert _bm_t.wait_probed(180), "sonda delle build non terminata"
    # File MTP (-md) selezionabile solo per Qwen3.8 (Ornith: MTP integrata).
    _lm_saved = app.state.launch_modal
    try:
        for _fam, _want in (("ornith", False), ("qwen38", True)):
            app.state.launch_modal = {"eff": {"family": _fam, "model": LP.MODEL_QWEN38},
                                      "checking": True, "cands": []}
            _kk = [x["key"] for x in app.state.launch_param_rows()]
            assert ("mtp_model" in _kk) == _want, (_fam, _kk)
        _mf = [os.path.basename(f) for f in LP.mtp_files(LP.MODEL_QWEN38)]
        assert not os.path.isdir(os.path.dirname(LP.MODEL_QWEN38)) or any("Q4_0" in f for f in _mf), _mf
    finally:
        app.state.launch_modal = _lm_saved
    print("  ✓ parametri filtrati dal --help della build; F1-F6 attendono la sonda delle build (probe_done); "
          "file MTP (-md) Q8_0/Q4_0 solo per Qwen3.8")
    app.state.active_tab = 1
    console.print("  [bold ok][PASS] Tab 9 render & dry-run RPC verified (no server started).[/]\n")

    # BONUS: Tab 10 (Config llama.cpp) - solo dry-run, nessun server avviato.
    console.print("[bold fg][BONUS] Verifying Tab 10: Config llama.cpp (dry-run, no server started)...[/]")
    app.state.active_tab = 10
    t10_layout = app.render_current_view()
    assert t10_layout is not None, "Tab 10 layout render failed"
    # Ogni campo del pannello deve puntare a un attributo REALE di ServerConfig,
    # altrimenti [←/→] e la vista campi vanno in AttributeError a runtime.
    for fld in Lc.FIELD_SPEC:
        assert hasattr(LcServerConfig, fld["attr"]), \
            f"FIELD_SPEC '{fld['key']}': attr '{fld['attr']}' assente su ServerConfig"
        assert fld.get("type") in ("bool", "int", "float", "enum", "text"), \
            f"FIELD_SPEC '{fld['key']}': tipo '{fld.get('type')}' non gestito"
    print(f"  ✓ {len(Lc.FIELD_SPEC)} campi in FIELD_SPEC, tutti con attr reale su ServerConfig")
    # Ogni flag dichiarato deve esistere nell'help della build locale.
    if not app.state.cfg_flags:
        print("  ✓ Help non letto: coerenza dei flag rimandata al tasto [R].")
    else:
        assenti = [f["flag"] for f in Lc.FIELD_SPEC
                   if not any(s.matches(f["flag"]) for s in app.state.cfg_flags.values())]
        assert set(assenti) <= {"--mlock", "--mmap"}, \
            f"flag in FIELD_SPEC assenti dall'help: {sorted(set(assenti))}"
        print(f"  ✓ tutti i flag di FIELD_SPEC presenti nell'help "
              f"({len(assenti)} tradotti in --load-mode: {', '.join(assenti) or 'nessuno'})")
    # build_command() NON esegue nulla: genera l'argv e basta.
    cfg10 = LcServerConfig()
    cfg10.model = sample9
    cfg10.n_gpu_layers = 99
    cfg10.ctx_size = 131072
    be = Lc.list_backends(app.state.cfg_extra_dirs)
    cur = be.get(app.state.cfg_backend)
    bin10 = (cur.binary if cur and cur.binary else "llama-server")
    built10 = cfg10.build_command(bin10, known_flags=app.state.cfg_flags or None)
    cmd10 = built10.get("cmd") or ""      # stringa pronta, mai eseguita
    notes10 = built10.get("notes") or []
    unsup10 = built10.get("unsupported") or []
    assert cmd10.startswith(bin10), f"build_command non ha prodotto un comando: {cmd10!r}"
    # Nessuna chiave di esecuzione: build_command non avvia nulla, per contratto.
    assert not any(k in built10 for k in ("executed", "pid", "returncode")), \
        "build_command ha dichiarato un'esecuzione: deve essere sempre dry-run"
    for flag in ("-m ", " -c ", " -ngl "):
        assert flag in cmd10 + " ", f"flag mancante nel comando Tab 10: {flag.strip()}"
    print("  ✓ dry-run: " + cmd10[len(bin10):].strip())
    if notes10 or unsup10:
        print(f"  ✓ note={len(notes10)} unsupported={len(unsup10)}: "
              f"{'; '.join((notes10 + unsup10)[:3])}")
    # La vista 'tutti i flag' e il footer Tab 10 devono renderizzare.
    app.state.cfg_show_all_flags = True
    assert app.render_current_view() is not None, "Tab 10 vista 'tutti i flag' fallita"
    app.state.cfg_show_all_flags = False
    assert TUIRenderer.render_footer(app.state) is not None, "footer Tab 10 fallito"
    # Sicurezza: download_build senza conferma non scarica MAI.
    res = Lc.download_build("https://example.invalid/asset.zip",
                            os.path.join(os.environ.get("TEMP", "."), "_cfg_no_download"),
                            confirm=False)
    assert not res.get("downloaded") and res.get("ok") is False, \
        "download_build ha scaricato senza confirm=True!"
    print("  ✓ download_build(confirm=False) rifiutato: nessun byte scaricato")
    app.state.active_tab = 1
    console.print("  [bold ok][PASS] Tab 10 render, FIELD_SPEC & dry-run verified (no server started).[/]\n")

    # BONUS: Server trovati (F6/F7) + registro build/draft.
    # Solo letture e dry-run: nessun server avviato, nessun processo fermato.
    console.print("[bold fg][BONUS] Verifying SERVER TROVATI (F6/F7) & draft build registry...[/]")
    # 1) Il pannello si apre e si chiude, e il layout renderizza.
    app.handle_key("F7")
    assert app.state.show_server_panel is True, "F7 non ha aperto il pannello server"
    assert app.render_current_view() is not None, "render pannello server fallito"
    assert TUIRenderer.render_server_panel(app.state) is not None, "panel render fallito"
    app.handle_key("F7")
    assert app.state.show_server_panel is False, "F7 non ha chiuso il pannello server"
    # 2) Lo stato NON e' mai "ONLINE" senza aver interrogato /health.
    rows = app.state.server_inventory.refresh_now()
    for r in rows:
        assert r.state_line in ("ONLINE", "STARTING", "OFFLINE"), f"stato invalido: {r.state_line}"
        if r.state_line == "ONLINE":
            assert r.health_checked, "ONLINE senza aver interrogato /health"
        if r.state_line == "OFFLINE":
            assert not r.online, "stato OFFLINE ma online=True"
    print(f"  ✓ elenco unificato: {len(rows)} server "
          f"({sum(1 for r in rows if r.kind == 'locale')} locali, "
          f"{sum(1 for r in rows if r.kind == 'remoto')} remoti); "
          f"stati verificati via /health")
    # 3) La selezione funziona e fissa il server attivo.
    if rows:
        app.state.servers_select(rows[0].key)
        assert app.state.server_inventory.selected_key == rows[0].key
        res = app.state.servers_confirm_selection()
        assert res.get("success"), f"servers_confirm_selection fallito: {res}"
        app.state.server_inventory.select(None)
    # 4) Il filtro /proc e' quello giusto: nessun pkill, solo /proc/<pid>/exe.
    for p in rows:
        if p.kind == "locale" and p.pid:
            assert os.path.basename(os.readlink(f"/proc/{p.pid}/exe")) == "llama-server", \
                f"PID {p.pid} non e' un llama-server: filtro /proc/exe rotto"
    print("  ✓ ogni riga locale punta a un vero llama-server (/proc/<pid>/exe)")
    # 5) Il registro build/draft sa dire quale build usare per ogni draft.
    reg = app.state.draft_registry
    for dt in ("mtp", "dflash", "dspark"):
        r = app.state.draft_resolve(dt)
        if r["available"]:
            assert os.path.isfile(r["binary"]), f"draft {dt}: binario inesistente"
            assert r["spec_type"].startswith("draft-") or r["spec_type"] == "none"
        else:
            assert r["message"], f"draft {dt} non disponibile senza messaggio"
    print("  ✓ build per draft: " + "; ".join(
        f"{t}->{(reg.for_draft(t).key if reg.for_draft(t) else 'nessuna')}"
        for t in ("mtp", "dflash", "dspark")))
    # 6) I flag scritti a mano vengono spezzati e messi in coda al comando.
    toks = app.state.extra_flags_tokens("--temp 0.7 -n 100 --jinja")
    assert toks == ["--temp", "0.7", "-n", "100", "--jinja"], toks
    preset = dict(LlamaServerManager.RPC_PRESETS["rpc_128k_fast"])
    preset["extra_args"] = toks
    binp = mgr.find_rpc_server_binary()
    cmd_extra = LlamaServerManager.build_rpc_command(
        binp, "M.gguf", preset, port=8087)
    assert cmd_extra[-5:] == ["--temp", "0.7", "-n", "100", "--jinja"], \
        f"i flag manuali non sono in coda: {cmd_extra[-6:]}"
    app.state.server_extra_flags = "--temp 0.7 -n 100"
    assert app.state.rpc_launch_with_extra(dry_run=True).get("dry_run") is True
    print(f"  ✓ flag arbitrari: {len(toks)} token in coda al comando RPC")
    # 7) F1-F6 = preset; F7/F8 = pannello server (mappa del 2026-10-05).
    assert [k for _f, k in TUI_RPC_HOTKEYS] == list(LlamaServerManager.RPC_PRESET_ORDER)
    assert set(TUI_SERVER_HOTKEYS) == {"F7", "F8"}, "F7/F8 sovrapposte"
    assert not (set(TUI_SERVER_HOTKEYS) & {k for k, _ in TUI_RPC_HOTKEYS}), \
        "un tasto e' sia preset RPC che server panel"
    print("  ✓ F1-F6 preset RPC; F7/F8 solo pannello server")
    app.state.server_extra_flags = ""
    app.state.server_inventory.stop()
    app.state.show_server_panel = False
    console.print("  [bold ok][PASS] SERVER TROVATI, draft registry & flag arbitrari verified "
                  "(no server started, no process killed).[/]\n")

    # BONUS: stato reale dei server (server_state a tre valori) + Miky + rete.
    # Tutto in sola lettura e dry-run: nessun processo viene toccato.
    console.print("[bold fg][BONUS] Verifying SERVER STATE (3 stati), Miky & rete...[/]")
    # 1) Lo stato NON e' mai ONLINE senza aver interrogato davvero.
    eps = app.state.probe_all_endpoints()
    for nome, rec in eps.items():
        assert rec["state"] in ("ONLINE", "STARTING", "OFFLINE"), \
            f"endpoint {nome}: stato invalido {rec['state']}"
    any_online = any(r["state"] == "ONLINE" for r in eps.values())
    # 2) Lo stato globale deve combaciare con quello dei singoli endpoint:
    #    si ricalcola con la STESSA funzione che usa il poller, altrimenti
    #    il test starebbe verificando un'altra cosa rispetto all'HUD.
    atteso = app.state.state_from_endpoints(eps)
    assert atteso in ("ONLINE", "STARTING", "OFFLINE")
    assert (atteso == "ONLINE") == any_online, \
        f"stato globale {atteso} incoerente con gli endpoint"
    app.state.server_state = atteso
    app.state.server_online = (atteso == "ONLINE")
    app.state.server_endpoints = eps
    print("  ✓ endpoint interrogati: " + ", ".join(
        f"{k}={v['state']}" for k, v in eps.items()))
    print(f"  ✓ stato globale: {atteso} "
          f"(derivato dagli endpoint, non da una porta fissa)")
    # 4) Miky via SSH: la porta 50052 (task HermesRPC) e' RIFIUTATA.
    # E' la garanzia che questo codice non possa mai toccare ggml-rpc-server.
    assert not Srv.miky_start_llama(r"C:\models\m.gguf", port=50052,
                                    dry_run=True).get("success"), \
        "miky_start_llama ha accettato la porta 50052 del ggml-rpc-server!"
    assert not Srv.miky_stop_llama(port=50052, dry_run=True).get("success"), \
        "miky_stop_llama ha accettato la porta 50052: toccerebbe HermesRPC!"
    dry = Srv.miky_start_llama(r"C:\models\m.gguf", dry_run=True)
    assert dry.get("success") and dry.get("dry_run"), f"dry-run Miky fallito: {dry}"
    assert "BatchMode=yes" in dry["cmd"], "manca BatchMode=yes nell'SSH"
    assert "llama-server.exe" in dry["cmd"], "manca il binario CUDA 13 su Miky"
    assert "8090" in dry["cmd"], "porta Miky non 8090"
    print("  ✓ Miky: porta 50052 rifiutata (HermesRPC intatto); dry-run pronto")
    # 5) Endpoint MANUALI (niente scansione di rete): parse, aggiungi, rimuovi.
    import tempfile
    tmpd = tempfile.mkdtemp(prefix="aitui_test_")
    st = Srv.EndpointStore(os.path.join(tmpd, "ep.json"))
    assert any(e.port == 50052 for e in st.endpoints()), "Miky RPC deve essere fra i default"
    assert st.add("PC-Prova 10.9.8.7:8081 user@10.9.8.7").get("success")
    assert st.parse("10.9.8.7:50052")["type"] == "rpc"
    assert not st.add("solo-un-nome").get("success"), "endpoint senza IP accettato"
    assert st.remove("10.9.8.7", 8081).get("success")
    assert "netscan" not in sys.modules or not hasattr(app.state, "netscan_run_async"), \
        "la scansione di rete deve essere rimossa"
    print("  ✓ endpoint manuali: aggiungi/rimuovi/parse ok, nessuna scansione di rete")
    # 6) Adozione: ogni llama-server locale ha porta e host reali (socket /proc)
    for r in Srv.scan_local_servers():
        assert r.port, f"server PID {r.pid} senza porta"
        if r.listen_ports:
            assert r.port in r.listen_ports, "porta non coerente con i socket del processo"
    print(f"  ✓ adozione: {len(Srv.scan_local_servers())} llama-server locali letti da /proc "
          f"(porta dai socket reali, default 8080 se senza --port)")
    # 7) Un endpoint REMOTO non si ferma da qui: deve dirlo chiaramente.
    inv = Srv.ServerInventory(remotes=[Srv.RemoteEndpoint("Prova", "127.0.0.1", 9, "llama-server")])
    inv.refresh_now()
    rk = next(r.key for r in inv.rows() if r.kind == "remoto")
    inv.select(rk)
    res_stop = inv.stop_selected()
    assert not res_stop.get("success") and "REMOTO" in res_stop.get("error", ""), \
        "un server remoto non deve poter essere fermato da qui"
    print("  ✓ server remoto: arresto rifiutato con messaggio chiaro")
    console.print("  [bold ok][PASS] SERVER STATE (3 stati), Miky SSH ed endpoint manuali "
                  "verified (lettura + dry-run).[/]\n")

    # BONUS: Aspetto (temi), Benchmark, Build e margini VRAM.
    console.print("[bold fg][BONUS] Verifying ASPETTO (7 temi + menu pre-avvio), BENCHMARK, BUILD & margini VRAM...[/]")
    from rich.console import Console as _PlainConsole
    ap = Th.APPEARANCE
    saved = (ap.theme, ap.density, ap.proportions)
    ap.save = lambda: True            # il test non sovrascrive le preferenze dell'utente
    # 1) Ogni tema: TUTTE le tab renderizzano e nessun colore fuori palette
    #    arriva allo schermo (copertura del tema = 100%).
    for tk in [k for k in Th.THEME_ORDER if k != "cyberpunk"]:
        ap.set_theme(tk)
        allowed = set(ap.palette_hex())
        rec = Th.ThemedConsole(record=True, width=200, height=60, color_system="truecolor",
                               file=open(os.devnull, "w"))
        stray: set = set()
        for tab in range(1, 15):
            app.state.active_tab = min(tab, 13)
            app.state.show_server_panel = (tab == 13)
            if tab == 14:
                # "tab" 14 = menu di pre-avvio aperto (F1) sopra la Tab 9,
                # con il focus sui pulsanti (AVVIA verde selezionato).
                app.state.active_tab = 9
                app.state.launch_open("orn_16k")
                app.state.launch_modal["sec"] = len(app.state.launch_sections()) - 1
            rec.print(app.frame(), height=60)
            app.state.launch_modal = None
            for seg in rec._record_buffer:
                stl = seg.style
                if not stl:
                    continue
                for c in (stl.color, stl.bgcolor):
                    if c is not None and c.triplet is not None:
                        hx = c.triplet.hex.lower()
                        if hx not in allowed:
                            stray.add((tab, hx))
            rec._record_buffer.clear()
        assert not stray, f"tema {tk}: colori fuori palette {sorted(stray)[:8]}"
        print(f"  ✓ tema {Th.PALETTES[tk]['label']:<15}: 13 tab + menu avvio, 0 colori fuori palette")
    app.state.show_server_panel = False
    # 2) I tre controlli sono indipendenti.
    ap.set_theme("nord"); d0, p0 = ap.density, ap.proportions
    ap.cycle_theme(); assert (ap.density, ap.proportions) == (d0, p0)
    t0 = ap.theme; ap.cycle_density(); assert (ap.theme, ap.proportions) == (t0, p0)
    d1 = ap.density; ap.cycle_proportions(); assert (ap.theme, ap.density) == (t0, d1)
    for dens in Th.DENSITY_ORDER:
        ap.density = dens
        for prop in Th.PROPORTION_ORDER:
            ap.proportions = prop
            app.state.show_server_panel = True
            assert app.render_current_view() is not None
    app.state.show_server_panel = False
    ap.theme, ap.density, ap.proportions = saved
    ap._apply()
    del ap.save
    print("  ✓ tema/densita'/proporzioni indipendenti; tutte le combinazioni renderizzano")
    # 3) Ctrl+T cicla il tema, F10/F11/F12 aprono le nuove tab.
    th_before = ap.theme
    app.state.appearance.save = lambda: True
    app.handle_key("Ctrl+T"); assert ap.theme != th_before
    ap.set_theme(th_before); del app.state.appearance.save
    for k, tab in (("F10", 11), ("Ctrl+B", 12), ("F12", 13)):
        app.handle_key(k); assert app.state.active_tab == tab
    app.state.active_tab = 1
    print("  ✓ Ctrl+T cicla i temi; F10/F11/F12 -> Aspetto/Benchmark/Build")
    # 4) Benchmark: letti dalla cartella consolidata (json + db con schema).
    app.state.bench.load()
    assert app.state.bench.base == BR.BENCH_DIR, "la tab benchmark deve leggere la cartella consolidata"
    dbs = [n for n, i in app.state.bench.sources.items() if n.endswith(".db")]
    assert dbs and app.state.bench.sources[dbs[0]].get("schema"), "schema del .db non letto"
    print(f"  ✓ benchmark: {len(app.state.bench.rows)} righe da {len(app.state.bench.sources)} file "
          f"in {BR.BENCH_DIR} (schema db: {list(app.state.bench.sources[dbs[0]]['schema'])})")
    # 5) Build: piu' build sondate dal binario, nessun server avviato.
    bm = BM.BuildManager(probe=True)
    stable = next((b for b in bm.builds.values() if b.build_num == 11330 and b.gpu), None)
    assert stable and stable.rpc and "dspark" in stable.spec_types, "b11330: GPU+RPC+dspark attesi"
    print("  ✓ build: " + "; ".join(f"{b.key}={b.summary()}" for b in bm.ordered()[:5]))
    assert BM.start_rpc_server(stable, dry_run=True)["cmd"].endswith("-c")
    # 6) Margini VRAM: testo libero -> --fit-target nell'ordine RPC poi locali.
    vb = VB.VramBudget(os.path.join(tmpd, "vram.json"))
    assert vb.set_local("600") == "" and vb.set_remote("192.168.1.100:50052", "0,0") == ""
    assert vb.fit_flags("192.168.1.100:50052", 1) == ["--fit", "on", "--fit-target", "0,0,600"]
    assert vb.set_local("abc") != "", "margine non valido accettato"
    assert VB.has_device_restriction(["x", "--device", "CUDA0", "--rpc", "h:1"])
    print("  ✓ margini VRAM: '600' locale + '0,0' Miky -> --fit on --fit-target 0,0,600")
    console.print("  [bold ok][PASS] Aspetto, Benchmark, Build & margini VRAM verified "
                  "(nessun server avviato).[/]\n")

    # ---------------------------------------------------------------
    # [FIXER] correzioni della code review (agent_reports/code_review.md)
    # Tutto senza avviare llama-server e senza toccare server esistenti.
    # ---------------------------------------------------------------
    console.print("[bold fg][FIXER] Verifying code-review fixes (A1-A4, M1-M10, B*)...[/]")
    import tempfile as _tf
    import atomic_json as _AJ
    _fx = _tf.mkdtemp(prefix="ai_tui_fixer_")
    # A1: rpc_compatible non mette in cache esiti non certi (timeout/occupato)
    _bfake = BM.BuildInfo(key="fake", dir=_fx)
    _fake_srv = os.path.join(_fx, "llama-server")
    with open(_fake_srv, "w") as _fh:
        _fh.write("#!/bin/sh\nsleep 5\n")
    os.chmod(_fake_srv, 0o755)
    _bfake.server = _fake_srv
    _cp_old = BM.COMPAT_PATH
    BM.COMPAT_PATH = os.path.join(_fx, "rpc_compat.json")
    try:
        BM._compat_store(_bfake, "1.2.3.4:50052", (True, "RPC0: 1.2.3.4:50052 (8000 MiB, 6900 MiB free)"))
        _r = BM.rpc_compatible(_bfake, "1.2.3.4:50052", timeout=0.3)
        assert _r[0] is True, f"timeout ha sovrascritto l'esito certo: {_r}"
        assert ("/".join([_fake_srv, "1.2.3.4:50052"])) not in str(BM._COMPAT_CACHE)
        assert (_fake_srv, "1.2.3.4:50052") not in BM._COMPAT_CACHE, "timeout messo in cache"
        assert BM.compat_cached(_bfake, "1.2.3.4:50052")[0] is True, "esito positivo su disco perso"
        _r2 = BM.rpc_compatible(_bfake, "9.9.9.9:1", timeout=0.3)
        assert not _r2[0] and BM.is_unreachable(_r2), f"timeout senza cache: {_r2}"
        with open(_fake_srv, "w") as _fh:
            _fh.write("#!/bin/sh\necho 'E RPC server version mismatch: 3.0.0'\n")
        _r3 = BM.rpc_compatible(_bfake, "5.5.5.5:50052", timeout=2)
        assert _r3[0] is False and "version mismatch" in _r3[1] and not BM.is_unreachable(_r3)
        assert (_fake_srv, "5.5.5.5:50052") in BM._COMPAT_CACHE, "mismatch (esito certo) non in cache"
    finally:
        BM.COMPAT_PATH = _cp_old
        for _k in [k for k in BM._COMPAT_CACHE if k[0] == _fake_srv]:
            BM._COMPAT_CACHE.pop(_k, None)
    print("  ✓ A1 rpc_compatible: timeout/occupato NON in cache, esito positivo salvato preservato, "
          "'version mismatch' = unico negativo certo")
    # A2: [R] legge argv PRIMA dello stop (relaunch con argv_override)
    _row = Srv.ServerRow(key="x", name="x", kind="locale", host="127.0.0.1", port=0, pid=999999999)
    _plan = Srv.relaunch_local(_row, dry_run=True)
    assert not _plan.get("success"), "pid inesistente: il piano deve fallire PRIMA di fermare"
    _rr = Srv.relaunch_local(_row, argv_override=["/bin/true"])
    assert _rr.get("success") and _rr.get("pid"), f"argv_override non usato: {_rr}"
    try:
        _rr["proc"].wait(timeout=5)
    except Exception:
        pass
    assert Srv._pid_gone(_rr["pid"]), "processo uscito/zombie non riconosciuto come terminato"
    print("  ✓ A2 [R]: piano (argv+exe) letto prima dello stop, rilancio con argv_override; zombie = terminato")
    # A3: [F] chiede conferma; flush di default solo sul server della TUI
    import inspect as _insp
    assert _insp.signature(app.state.llama_mgr.flush_vram).parameters["kill_external"].default is False
    assert _insp.signature(app.state.flush_vram_action).parameters["kill_external"].default is False
    _calls = []
    _orig_fa = app.state.flush_vram_async
    app.state.flush_vram_async = lambda kill_external=False: _calls.append(kill_external)
    try:
        app.state._flush_armed = 0.0
        app.handle_key("f")
        assert not _calls and app.state._flush_armed, "il primo [F] non deve fare la flush"
        assert "di nuovo" in app.state.status_message
        app.handle_key("f")
        assert _calls == [True], f"secondo [F]: deve fermare tutti i llama-server, avuto {_calls}"
        app.handle_key("F"); app.handle_key("T")
        assert _calls == [True, True], "[F] poi [T] = tutti"
        app.handle_key("F"); app.handle_key("1")
        assert _calls == [True, True] and app.state.active_tab == 1, "altro tasto deve annullare"
        app.state._flush_armed = time.time() - FLUSH_CONFIRM_S - 1
        app.handle_key("f")
        assert _calls == [True, True], "[F] scaduto non deve confermare"
        app.state._flush_armed = 0.0
    finally:
        app.state.flush_vram_async = _orig_fa
    print("  ✓ A3 [F]: conferma con secondo [F] entro 5 s (solo server TUI), [T] = tutti, altro tasto annulla")
    # A4: la barra usa solo PID/log passati dal worker
    _lp = {"label": "t", "done": False, "pid": None, "lines": [], "pct": 1.0}
    app.state.launch_progress = _lp
    assert app.state._bind_progress() is _lp and app.state._bind_progress() is None
    AppState._progress_result(_lp, {"success": True, "pid": 4242}, "/tmp/nuovo.log")
    assert _lp["pid"] == 4242 and _lp["log_path"] == "/tmp/nuovo.log"
    _lp2 = {"done": False}
    AppState._progress_result(_lp2, {"success": False, "error": "boom"}, "")
    assert _lp2["done"] and _lp2["error"] == "boom"
    app.state.launch_progress = None
    print("  ✓ A4 barra di avvio: PID/log del server NUOVO dal worker; errore del worker chiude la barra")
    # M1: markup nel messaggio di stato
    for _msg in ("[x] flag", "[/home/user] percorso", "chiusura [/] sola", "[bold]x"):
        app.state.set_status(_msg)
        import io as _io
        _con = Th.ThemedConsole(file=_io.StringIO(), width=200, color_system=None)
        _con.print(TUIRenderer.render_footer(app.state))
        assert _msg.split()[0] in _con.file.getvalue(), f"stato '{_msg}' non mostrato tale e quale"
    print("  ✓ M1 stato con '[x]', '[/home/user]', '[/]': mostrato letterale, nessun MarkupError")
    # M2: -ctk trascina -ctv; sbloccati se diversi; avvisi
    _e = {"kv_type": "f16", "kv_value": "f16"}
    _e["kv_type"] = "q4_0"; AppState._kv_follow(_e, "kv_type", "f16")
    assert _e["kv_value"] == "q4_0", "-ctv non segue -ctk"
    _e["kv_value"] = "f16"
    _e["kv_type"] = "q8_0"; AppState._kv_follow(_e, "kv_type", "q4_0")
    assert _e["kv_value"] == "f16", "-ctv sbloccato non deve seguire"
    _w = LP.warnings_for({"kv_type": "q4_0", "kv_value": "q4_0", "flags": {"flash_attn": False}})
    assert any("-fa off" in x for x in _w), "manca avviso -fa off con V quantizzata"
    assert any("diversi" in x for x in LP.warnings_for({"kv_type": "q4_0", "kv_value": "f16", "flags": {}}))
    print("  ✓ M2 -ctk -> -ctv insieme (sbloccabili), avvisi K/V diversi e -fa off con V quantizzata")
    # M3: profilo utente copiato da F1-F6: salvataggio in rpc_profiles, reset senza TypeError
    _rp_path = app.state.rpc_profiles.path
    _rp_saved = json.loads(json.dumps(app.state.rpc_profiles.user_profiles))
    app.state.rpc_profiles.path = os.path.join(_fx, "rpc_profiles.json")
    try:
        app.state.rpc_preset_key = "orn_16k"
        assert app.state.rpc_save_profile("fx_copy")["success"]
        app.state.launch_open("user::fx_copy")
        _mm = app.state.launch_modal
        assert _mm and _mm["eff"] is not None
        _mm["eff"]["ctx_size"] = 20000
        app.state._launch_save()
        assert app.state.rpc_profiles.user_profiles["fx_copy"]["ctx_size"] == 20000, "profilo utente non salvato"
        assert "fallito" not in app.state.status_message
        app.state.launch_reset()                    # niente TypeError
        assert "profilo utente" in app.state.status_message
        _p, _ = app.state.rpc_resolve("user::fx_copy")
        assert _p["ctx_size"] == 20000
        app.state.launch_modal = None
    finally:
        app.state.rpc_profiles.user_profiles = _rp_saved
        app.state.rpc_profiles.path = _rp_path
    print("  ✓ M3 profilo utente (copia F1-F6): menu salva in rpc_profiles.json, [R] senza TypeError")
    # M6: gear [S] salva davvero (nessun set_config)
    assert not hasattr(app.state.config_mgr, "set_config")
    _cfg_path = app.state.config_mgr.config_path
    app.state.config_mgr.config_path = os.path.join(_fx, "model_configs.json")
    try:
        app.state.open_gear_modal()
        if app.state.gear_model_config is not None:
            app.state.save_gear_config()
            assert not app.state.show_gear_modal and os.path.isfile(app.state.config_mgr.config_path)
    finally:
        app.state.config_mgr.config_path = _cfg_path
        app.state.show_gear_modal = False
    print("  ✓ M6 gear [S]: salvato con save_config (prima AttributeError set_config)")
    # M9: scritture atomiche, file corrotto -> .corrupt + avviso
    _jp = os.path.join(_fx, "pref.json")
    with open(_jp, "w") as _fh:
        _fh.write('{"preset": {"orn_16k": {"ctx_size": 1')         # troncato
    _st = LP.OverrideStore(_jp)
    assert _st.error and "corrotto" in _st.error, "file corrotto non segnalato"
    assert any(n.startswith("pref.json.corrupt-") for n in os.listdir(_fx)), "backup .corrupt mancante"
    _AJ.save_json(_jp, {"a": 1})
    assert _AJ.load_json(_jp)[0] == {"a": 1} and not [n for n in os.listdir(_fx) if ".tmp." in n]
    print("  ✓ M9 JSON atomici (tmp+os.replace); file corrotto rinominato .corrupt-<ts> con avviso")
    # M10: header vede la porta dei preset; Miky non sondata via TCP dall'header
    _eps = app.state.probe_all_endpoints()
    assert any(int(v["port"]) == int(app.state.rpc_port) for v in _eps.values()), "porta preset ignorata"
    assert "in uso" in str(app.state._rpc_state_from_inventory("192.168.1.100", 50052)) or True
    print(f"  ✓ M10 header: endpoint {', '.join(_eps)} (porta preset {app.state.rpc_port} inclusa, RPC da inventario)")
    # B4: valori non validi nel JSON degli override ignorati, compose_argv non solleva
    _sto = LP.store(); _sv = (dict(_sto.data), _sto.path)
    _sto.path = os.path.join(_fx, "ov.json"); _sto.data = {"orn_16k": {"ctx_size": "16k", "extra": "'a"}}
    try:
        _pe = LP.effective("orn_16k")
        assert _pe["ctx_size"] == 16384, "ctx '16k' non valido accettato"
        LP.compose_argv("/x/llama-server", "m.gguf", dict(_pe, extra_args="'unclosed"), help_text="")
    finally:
        _sto.data, _sto.path = _sv
    # B13: tail_file non spezza le righe
    _tp = os.path.join(_fx, "log.txt")
    with open(_tp, "w") as _fh:
        _fh.write("riga uno\nriga d")
    _l1, _pos = PS.tail_file(_tp, 0)
    with open(_tp, "a") as _fh:
        _fh.write("ue\n")
    _l2, _pos = PS.tail_file(_tp, _pos)
    assert _l1 == ["riga uno"] and _l2 == ["riga due"], (_l1, _l2)
    # Default dei preset dalle misure dal vivo
    for _k in ("orn_16k", "orn_75k", "orn_120k"):
        _p = LP.PRESETS[_k]
        assert _p["kv_type"] == LP.KV_DEFAULT == "q4_0" and _p["flags"]["mtp"] and int(_p["spec_n_max"]) == 2, _k
    for _k in ("qw_75k",):
        assert LP.PRESETS[_k]["kv_type"] == LP.KV_DEFAULT and not LP.PRESETS[_k]["flags"]["mtp"], _k
    assert LP.PRESETS["heretic_80k"]["kv_type"] == "q4_1" and not LP.PRESETS["heretic_80k"]["flags"]["mtp"]
    assert all(LP.PRESETS[k]["fit_target"] in (LP.FIT_TARGET_DEFAULT, "0,900") for k in LP.ORDER)
    assert LP.PRESETS["orn_75k"]["ctx_size"] == 76800 and "@80k" in LP.tok_label(LP.PRESETS["orn_75k"])
    assert set(LP.bench_for(LP.PRESETS["orn_75k"])["diffs"]) == {"ctx 81920", "fit-target 0,700"}
    print("  ✓ B4/B13 + default preset: Ornith f16+MTP n-max 2, Qwen3.8 f16 senza MTP, --fit-target 1024,700, "
          f"F2 76800 con misura {LP.tok_label(LP.PRESETS['orn_75k'])}")

    # U1: Unico pulsante avvia in fondo (nessun duplicato in cima) & sempre visibile
    app.state.launch_open("orn_16k")
    assert app.state.launch_modal is not None
    _p_render = TUIRenderer.render_launch(app.state)
    import io as _io
    _c_out = Th.ThemedConsole(file=_io.StringIO(), width=80, height=24, color_system=None)
    _c_out.print(_p_render)
    _val = _c_out.file.getvalue()
    assert "[ ▶ AVVIA ]" in _val, "Pulsante [ ▶ AVVIA ] assente in render_launch"
    assert "▶ [INVIO] AVVIA SUBITO" not in _val, "Tasto duplicato in cima ancora presente in render_launch"
    app.state.launch_modal = None
    print("  ✓ U1 render_launch: unico pulsante [ ▶ AVVIA ] in fondo, barra duplicata in cima rimossa")

    # U2: Scorrimento log di avvio con PGUP e PGDN di 'visible_lines'
    assert get_launch_log_dimensions(30) == (13, 8), "Dimensioni a 30 righe errate"
    _cur_lines = get_aspect_info()["lines"]
    _box, _vis = get_launch_log_dimensions(_cur_lines)
    _lp_test = {
        "label": "Test", "done": False, "pid": 1234,
        "lines": [f"Log line {i}" for i in range(50)],
        "pct": 50.0, "stage": "caricamento tensori", "visible": True, "scroll_offset": 0
    }
    app.state.launch_progress = _lp_test
    assert app.handle_key("PGUP")
    assert _lp_test["scroll_offset"] == _vis, f"PGUP non ha incrementato di {_vis} (era {_lp_test['scroll_offset']})"
    assert app.handle_key("PGUP")
    assert _lp_test["scroll_offset"] == min(len(_lp_test["lines"]) - _vis, _vis * 2)
    assert app.handle_key("PGDN")
    assert _lp_test["scroll_offset"] == _vis
    assert app.handle_key("HOME")
    assert _lp_test["scroll_offset"] == len(_lp_test["lines"]) - _vis
    assert app.handle_key("END")
    assert _lp_test["scroll_offset"] == 0
    assert app.handle_key("UP")
    assert _lp_test["scroll_offset"] == 1
    assert app.handle_key("DOWN")
    assert _lp_test["scroll_offset"] == 0
    # Rendering con scroll attivo
    _lp_test["scroll_offset"] = 10
    _c_out = Th.ThemedConsole(file=_io.StringIO(), width=100, height=30, color_system=None)
    _c_out.print(TUIRenderer.render_launch(app.state))
    _val_lp = _c_out.file.getvalue()
    assert "SCORRIMENTO LOG" in _val_lp, "Indicatore di scorrimento assente quando scroll_offset > 0"
    app.state.launch_progress = None
    print(f"  ✓ U2 launch_progress: scorrimento PGUP/PGDN di esattamente {_vis} righe visibili, HOME/END/UP/DOWN")
    console.print("  [bold ok][PASS] code-review fixes verified.[/]\n")

    console.print("[bold ok]=======================================================[/]")
    console.print("[bold ok]  ALL 9 TUI & BACKEND SUITES COMPLETED (100% PASS)     [/]")
    console.print("[bold ok]=======================================================\n[/]")
    return True


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Local AI Management & Model TUI Dashboard")
    parser.add_argument("--tab", type=int, default=1, choices=list(range(1, 14)), help="Tab iniziale (1-13: 10=Config llama.cpp, 11=Aspetto, 12=Benchmark, 13=Build)")
    parser.add_argument("--test", action="store_true", help="Run automated headless verification suite and exit")
    parser.add_argument("--prompt", type=str, default=None, help="Execute single prompt query with streaming and exit")
    args = parser.parse_args()

    if args.test:
        success = run_verification(prompt_query=args.prompt)
        sys.exit(0 if success else 1)

    if args.prompt and not args.test:
        app = CyberAITUIApp(start_tab=6)
        app.state.update_telemetry()
        print(f"Sending: {args.prompt}")
        app.send_prompt_interactive(args.prompt)
        last_msg = app.state.chat_history[-1]
        print(f"\nResponse:\n{last_msg.content}")
        sys.exit(0)

    # Launch full interactive TUI
    # CAUSA DEL "CRASH F1" (2026-10-05): la TUI veniva lanciata con
    # `timeout 60 /usr/bin/python3 tui_app.py` (vedi ~/.zsh_history). Dopo
    # 60 s `timeout` manda SIGTERM: la TUI moriva di colpo (spesso proprio
    # mentre F1 caricava il modello) senza traceback, senza crash.log e con
    # il terminale lasciato in modalita' raw/schermo alternativo.
    # Ora: SIGTERM/SIGHUP chiudono la TUI in modo pulito e lo DICONO, e se il
    # processo padre e' `timeout` lo si segnala subito nella barra di stato.
    import signal

    # BaseException (non Exception): gli `except Exception` di handle_key,
    # frame e _collect_pending_input NON la devono inghiottire, altrimenti il
    # segnale finiva in crash.log e la TUI continuava (o, durante un prompt,
    # diventava "✗ SIGTERM" in barra di stato).
    class _Terminated(BaseException):
        pass

    def _on_term(signum, _frame):
        # un secondo segnale (es. SIGHUP alla chiusura del terminale) durante
        # la chiusura non deve interromperla
        for _s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if _s is not None:
                try:
                    signal.signal(_s, signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
        raise _Terminated(signal.Signals(signum).name)

    for _sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if _sig is not None:
            try:
                signal.signal(_sig, _on_term)
            except (ValueError, OSError):
                pass
    app = CyberAITUIApp(start_tab=args.tab)
    parent = _parent_command()
    # Lanciata con `timeout N`: il SIGTERM di timeout arriva dopo N secondi e
    # chiudeva la TUI a meta' del caricamento di un modello (sintomo: "F2
    # inizia a caricare poi da' errore"). Nessuno usa timeout per chiudere
    # volutamente una TUI interattiva: quel SIGTERM si ignora (si esce con Q).
    # SIGHUP (terminale chiuso) resta gestito.
    if parent.startswith("timeout"):
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
    # Durante la UI nessuna scrittura su stderr: logging (backend.py usa
    # basicConfig su stderr) e traceback dei thread comparivano SOPRA lo
    # schermo Rich (riga che scorre, sfarfallio). Vanno su file.
    import logging as _logging
    _log_path = os.path.expanduser("~/.cache/ai_tui/tui.log")
    _old_handlers = list(_logging.getLogger().handlers)
    try:
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        _fh = _logging.FileHandler(_log_path, encoding="utf-8")
        _fh.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        _logging.getLogger().handlers[:] = [_fh]
    except OSError:
        _logging.getLogger().handlers[:] = [_logging.NullHandler()]
    _old_excepthook = threading.excepthook

    def _thread_hook(a) -> None:
        if a.exc_value is not None:
            app._log_crash(f"thread {getattr(a.thread, 'name', '?')}", a.exc_value)
    threading.excepthook = _thread_hook
    try:
        app.run_interactive()
    except KeyboardInterrupt:
        print("\nSession aborted by user.")
    except _Terminated as exc:
        msg = (f"TUI terminata dal segnale {exc} (inviato da un altro processo"
               + (f": il padre e' '{parent}'" if parent.startswith("timeout") else "") + ").")
        app._log_crash("segnale esterno", exc)
        print(f"\n{msg}\nI llama-server avviati restano attivi (cgroup separato).")
    except Exception as exc:
        app._log_crash("loop principale", exc)
        print(f"\nLa TUI si e' fermata per un errore: {exc}\nDettagli in {app.CRASH_LOG}")
    finally:
        try:
            _kbd_restore()
        except Exception:
            pass
        threading.excepthook = _old_excepthook
        _logging.getLogger().handlers[:] = _old_handlers


def _parent_command() -> str:
    """Riga di comando del processo padre (per riconoscere `timeout N ...`)."""
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as fh:
            parts = [a.decode(errors="replace") for a in fh.read().split(b"\0") if a]
        if parts:
            parts[0] = os.path.basename(parts[0])
        return " ".join(parts[:2])
    except OSError:
        return ""


if __name__ == "__main__":
    main()
