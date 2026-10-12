"""
rpc_tui.py - TUI per il ggml-rpc-server di Miky (Windows).

Cosa fa
-------
* elenca le build llama.cpp presenti su Miky che hanno ggml-rpc-server.exe,
  con versione, presenza di CUDA (cudart) e compatibilita' col Host Node;
* elenca i flag di ggml-rpc-server e le variabili d'ambiente GGML/CUDA che
  ESISTONO DAVVERO nella build b11254/b11330 (verificate nel sorgente), con
  una spiegazione per ognuna: [SPAZIO] attiva/disattiva, [INVIO] cambia valore;
* [A] applica: scrive rpc_config.json. Il supervisore (task HermesRPC,
  hermes_rpc_service.ps1 v2) se ne accorge entro 5 s e riavvia il server.
  Non servono i permessi di amministratore;
* mostra lo stato del server (rpc_status.json), la memoria che annuncia ai
  client e l'ultimo log.

Avvio:  python C:\\Users\\mikye\\hermes-rpc\\rpc_tui.py      (oppure rpc_tui.bat)
Test senza tastiera:  python rpc_tui.py --test
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HERE = r"C:\Users\mikye\hermes-rpc"
CFG_PATH = os.path.join(HERE, "rpc_config.json")
STATUS_PATH = os.path.join(HERE, "rpc_status.json")
LOG_PATH = os.path.join(HERE, "rpc-server.log")
STDOUT_LOG = os.path.join(HERE, "rpc-stdout.log")
STDERR_LOG = os.path.join(HERE, "rpc-stderr.log")
DEFAULT_BUILD = r"C:\Users\mikye\Documents\LLAMASERVER\builds\b11254-cuda13"
BUILD_ROOTS = [r"C:\Users\mikye\Documents\LLAMASERVER\builds"]
CLIENT_BUILD = 11330          # llama-server sul Host Node (Debian)

# Versione del protocollo RPC per build, letta da ggml/include/ggml-rpc.h ai
# rispettivi tag su GitHub. Client e server devono avere la STESSA versione
# principale, altrimenti la connessione viene rifiutata.
RPC_PROTO_KNOWN = {6750: 3, 9000: 4, 11254: 7, 11330: 7}

C = {"fg": "#f8f8f2", "muted": "#6272a4", "ok": "#50fa7b", "err": "#ff5555",
     "warn": "#f1fa8c", "metric": "#8be9fd", "accent": "#ff79c6", "sel": "#44475a",
     "bg": "#282a36"}

# ---------------------------------------------------------------------------
# Opzioni: flag della riga di comando + variabili d'ambiente
# kind: "bool" (spazio), "text"/"int"/"choice" (invio = nuovo valore)
# ---------------------------------------------------------------------------
OPTIONS: List[Dict[str, Any]] = [
    {"key": "host", "label": "-H  host di ascolto", "kind": "text", "default": "0.0.0.0", "group": "flag",
     "help": "Indirizzo su cui ggml-rpc-server accetta connessioni. 0.0.0.0 = tutte le "
             "interfacce (serve per farsi raggiungere dal Host Node). 127.0.0.1 = solo locale. "
             "ATTENZIONE: il protocollo RPC non ha autenticazione: usalo solo in LAN fidata."},
    {"key": "port", "label": "-p  porta", "kind": "int", "default": 50052, "group": "flag",
     "help": "Porta TCP. Il Host Node usa --rpc 192.168.1.100:50052: se la cambi, cambiala anche li'."},
    {"key": "threads", "label": "-t  thread CPU", "kind": "int", "default": 0, "group": "flag",
     "help": "Thread per il device CPU del rpc-server (0 = default del programma, 6 qui). "
             "Con la GPU CUDA attiva conta poco: serve solo se esponi anche la CPU."},
    {"key": "devices", "label": "-d  device esposti", "kind": "text", "default": "", "group": "flag",
     "help": "Elenco dei device da esporre (es. CUDA0). Vuoto = tutti quelli trovati. "
             "Su Miky c'e' una sola GPU CUDA (RTX 5050): la Intel UHD integrata non e' "
             "un device CUDA e non compare."},
    {"key": "cache", "label": "-c  cache locale dei tensori", "kind": "bool", "default": False, "group": "flag",
     "help": "Salva su disco i tensori grandi ricevuti dal client: al caricamento successivo "
             "dello stesso modello non vengono ritrasmessi in rete (avvio molto piu' veloce). "
             "Cartella: %LOCALAPPDATA%\\llama.cpp\\rpc o LLAMA_CACHE. Occupa spazio quanto il modello."},
    {"key": "GGML_CUDA_DISABLE_GRAPHS", "label": "disattiva CUDA graphs", "kind": "bool", "value": "1", "group": "env",
     "help": "Spegne i CUDA graphs (registrazione e riesecuzione dei kernel). Di norma rendono "
             "la generazione piu' veloce; spegnerli serve se vedi crash o memoria che cresce. "
             "Nota: issue #20315 segnala perdite di CUDA graphs nel rpc-server."},
    {"key": "GGML_CUDA_GRAPH_OPT", "label": "ottimizzazione grafi (=1)", "kind": "bool", "value": "1", "group": "env",
     "help": "Abilita il riordino sperimentale del grafo per eseguire nodi in parallelo su piu' "
             "stream. Spento di default nel sorgente: provalo e misura."},
    {"key": "GGML_CUDA_DISABLE_FUSION", "label": "disattiva fusione kernel (=1)", "kind": "bool", "value": "1",
     "group": "env",
     "help": "Spegne la fusione di operazioni consecutive in un solo kernel. Serve solo per "
             "diagnosi (risultati strani): di norma lasciala spenta = fusione attiva."},
    {"key": "GGML_CUDA_PDL", "label": "disattiva PDL (=0)", "kind": "bool", "value": "0", "group": "env",
     "help": "Programmatic Dependent Launch: sovrappone l'avvio dei kernel su GPU recenti "
             "(la 5050 e' Blackwell). Attivo di default se compilato; =0 lo spegne per diagnosi."},
    {"key": "GGML_CUDA_NO_PINNED", "label": "niente memoria host pinned", "kind": "bool", "value": "1", "group": "env",
     "help": "Non usa memoria RAM 'pinned' per i trasferimenti. Risparmia RAM bloccata ma "
             "rallenta le copie CPU<->GPU. Utile solo se la RAM di Miky e' al limite."},
    {"key": "GGML_CUDA_REGISTER_HOST", "label": "registra buffer host", "kind": "bool", "value": "1", "group": "env",
     "help": "Registra presso CUDA i buffer host gia' allocati (cudaHostRegister) per copie "
             "piu' veloci. Raramente utile per il rpc-server."},
    {"key": "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "label": "memoria unificata (sperim.)", "kind": "bool", "value": "1",
     "group": "env",
     "help": "Alloca con cudaMallocManaged. Nel sorgente il calcolo della memoria 'libera' da RAM "
             "di sistema e' SOLO Linux: su Windows NON fa annunciare piu' VRAM e la memoria "
             "gestita non va oltre la VRAM. Sconsigliata qui."},
    {"key": "GGML_CUDA_CUBLAS_COMPUTE_TYPE", "label": "tipo calcolo cuBLAS", "kind": "choice",
     "choices": ["", "f32", "f16"], "group": "env",
     "help": "Forza la precisione dei prodotti matriciali cuBLAS: f16 piu' veloce, f32 piu' "
             "preciso. Vuoto = scelta automatica (consigliato)."},
    {"key": "GGML_OP_OFFLOAD_MIN_BATCH", "label": "batch minimo offload", "kind": "int", "group": "env",
     "help": "Sotto questo numero di token le operazioni non vengono spostate sulla GPU "
             "(default 32). Vuoto = default."},
    {"key": "CUDA_VISIBLE_DEVICES", "label": "GPU CUDA visibili", "kind": "text", "group": "env",
     "help": "Variabile del driver: limita le GPU viste da CUDA (es. 0). Vuoto = tutte."},
    {"key": "LLAMA_CACHE", "label": "cartella cache (-c)", "kind": "text", "group": "env",
     "help": "Dove salvare la cache dei tensori quando -c e' attivo. Vuoto = default."},
    {"key": "GGML_RPC_DEBUG", "label": "log debug RPC", "kind": "bool", "value": "1", "group": "env",
     "help": "Stampa ogni comando RPC ricevuto in rpc-stderr.log. Solo per diagnosi: rallenta "
             "e fa crescere molto il log."},
    {"key": "GGML_SCHED_DEBUG", "label": "log debug scheduler", "kind": "bool", "value": "1", "group": "env",
     "help": "Log dello scheduler ggml. Solo per diagnosi."},
]
OPT_BY_KEY = {o["key"]: o for o in OPTIONS}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _run(argv: List[str], timeout: float = 15) -> str:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ERR {exc}"


def scan_builds(probe: bool = True) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for root in BUILD_ROOTS:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for n in names:
            d = os.path.join(root, n)
            exe = os.path.join(d, "ggml-rpc-server.exe")
            if not os.path.isfile(exe) or d.lower() in seen:
                continue
            seen.add(d.lower())
            b = {"dir": d, "name": n, "build": 0, "commit": "", "cuda": False, "cudart": False}
            b["cudart"] = any(f.lower().startswith("cudart64_") for f in os.listdir(d))
            b["cuda"] = b["cudart"] and os.path.isfile(os.path.join(d, "ggml-cuda.dll"))
            if probe:
                ls = os.path.join(d, "llama-server.exe")
                if os.path.isfile(ls):
                    m = re.search(r"build (\d+), commit (\w+)", _run([ls, "--version"]))
                    if m:
                        b["build"], b["commit"] = int(m.group(1)), m.group(2)
            out.append(b)
    return out


def proto_note(build: int) -> str:
    if not build:
        return "versione sconosciuta"
    if build == CLIENT_BUILD:
        return "= Host Node"
    p = RPC_PROTO_KNOWN.get(build)
    pc = RPC_PROTO_KNOWN[CLIENT_BUILD]
    if p is not None:
        return f"v{p} compatibile" if p == pc else f"v{p} != v{pc}: NON compatibile"
    return f"b{build}: verifica compatibilita' (client b{CLIENT_BUILD})"


# ---------------------------------------------------------------------------
# Config / stato
# ---------------------------------------------------------------------------
def load_cfg() -> Dict[str, Any]:
    cfg = {"build_dir": DEFAULT_BUILD, "host": "0.0.0.0", "port": 50052, "threads": 0,
           "devices": "", "cache": False, "env": {}, "extra_args": []}
    try:
        with open(CFG_PATH, "r", encoding="utf-8-sig") as fh:
            d = json.load(fh)
        cfg.update({k: v for k, v in d.items() if k in cfg})
    except (OSError, ValueError):
        pass
    return cfg


def save_cfg(cfg: Dict[str, Any]) -> None:
    cfg = dict(cfg)
    cfg["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")     # cambia l'hash -> riavvio
    tmp = CFG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, CFG_PATH)


def read_status() -> Dict[str, Any]:
    try:
        with open(STATUS_PATH, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def tail(path: str, n: int = 8) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return [l.rstrip() for l in fh.readlines()[-n:]]
    except OSError:
        return []


def announced_memory() -> List[str]:
    """Le righe 'CUDA0: ... (X MiB, Y MiB free)' stampate dal server all'avvio:
    e' la memoria che il client (llama.cpp sul Host Node) vede e usa per --fit."""
    out = []
    for path in (STDOUT_LOG, STDERR_LOG):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for l in fh:
                    if re.search(r"\(\d+ MiB, \d+ MiB free\)", l):
                        out.append(l.strip())
        except OSError:
            pass
    return out[-2:]


def gpu_line() -> str:
    o = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.reserved,memory.total,utilization.gpu,"
              "temperature.gpu", "--format=csv,noheader,nounits"], timeout=5)
    p = [x.strip() for x in o.strip().split(",")]
    if len(p) >= 6:
        return f"{p[0]}: usati {p[1]} MiB, riservati driver {p[2]}, totale {p[3]} MiB · {p[4]}% · {p[5]}°C"
    return "nvidia-smi non disponibile"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class App:
    def __init__(self) -> None:
        self.builds = scan_builds()
        self.cfg = load_cfg()
        self.pane = "opt"          # "build" | "opt"
        self.bi = max(0, next((i for i, b in enumerate(self.builds)
                               if os.path.normcase(b["dir"]) == os.path.normcase(self.cfg["build_dir"])), 0))
        self.oi = 0
        self.dirty = False
        self.msg = "[TAB] build/opzioni  [↑↓] muovi  [SPAZIO] attiva  [INVIO] valore/usa build  [A] applica  [Q] esci"
        self.gpu = gpu_line()
        self.last_gpu = time.time()

    # -- valori -----------------------------------------------------------
    def value(self, o: Dict[str, Any]) -> Any:
        if o["group"] == "flag":
            return self.cfg.get(o["key"], o.get("default"))
        return self.cfg["env"].get(o["key"])

    def is_on(self, o: Dict[str, Any]) -> bool:
        v = self.value(o)
        if o["kind"] == "bool":
            return bool(v) if o["group"] == "flag" else v is not None
        return v not in (None, "", 0) or (o["key"] == "port")

    def toggle(self, o: Dict[str, Any]) -> None:
        if o["kind"] == "bool":
            if o["group"] == "flag":
                self.cfg[o["key"]] = not bool(self.cfg.get(o["key"]))
            elif o["key"] in self.cfg["env"]:
                del self.cfg["env"][o["key"]]
            else:
                self.cfg["env"][o["key"]] = o.get("value", "1")
        elif o["kind"] == "choice":
            ch = o["choices"]
            cur = self.value(o) or ""
            nxt = ch[(ch.index(cur) + 1) % len(ch)] if cur in ch else ch[0]
            self.set_value(o, nxt)
        else:
            # spazio su un campo di testo = svuota (torna al default)
            self.set_value(o, "" if o["group"] == "env" else o.get("default"))
        self.dirty = True

    def set_value(self, o: Dict[str, Any], v: Any) -> Optional[str]:
        if o["kind"] == "int" and v not in ("", None):
            try:
                v = int(v)
            except (TypeError, ValueError):
                return f"'{v}' non e' un numero"
            if o["key"] == "port" and not 1 <= v <= 65535:
                return "porta fuori intervallo"
        if o["group"] == "flag":
            self.cfg[o["key"]] = v if v not in ("", None) else o.get("default")
        else:
            if v in ("", None):
                self.cfg["env"].pop(o["key"], None)
            else:
                self.cfg["env"][o["key"]] = str(v)
        self.dirty = True
        return None

    def cmdline(self) -> str:
        c = self.cfg
        a = [os.path.join(c["build_dir"], "ggml-rpc-server.exe"), "-H", str(c["host"]), "-p", str(c["port"])]
        if int(c.get("threads") or 0) > 0:
            a += ["-t", str(c["threads"])]
        if c.get("devices"):
            a += ["-d", c["devices"]]
        if c.get("cache"):
            a.append("-c")
        env = " ".join(f"{k}={v}" for k, v in c["env"].items())
        return (env + "  " if env else "") + " ".join(a)

    def apply(self) -> None:
        save_cfg(self.cfg)
        self.dirty = False
        self.msg = "✔ salvato: il supervisore riavvia il server entro ~5 s (vedi stato)"

    # -- render -----------------------------------------------------------
    def render(self) -> Layout:
        lay = Layout()
        lay.split_column(Layout(name="top", size=3), Layout(name="mid", ratio=1),
                         Layout(name="help", size=7), Layout(name="bottom", size=11))
        lay["mid"].split_row(Layout(name="builds", ratio=1), Layout(name="opts", ratio=1))
        st = read_status()
        state = st.get("state", "?")
        scol = {"running": C["ok"], "starting": C["warn"]}.get(state, C["err"])
        top = Text.assemble(("  ggml-rpc-server · Miky  ", f"bold {C['bg']} on {C['accent']}"), "  ",
                            (f"● {state}", f"bold {scol}"),
                            (f"  pid {st.get('pid') or '-'}  ·  {os.path.basename(st.get('build_dir') or '-')}",
                             C["fg"]),
                            ("   ✱ modifiche NON applicate: [A]" if self.dirty else "", f"bold {C['warn']}"))
        lay["top"].update(Panel(top, box=box.HEAVY, border_style=C["muted"]))

        bt = Table(expand=True, box=box.SIMPLE_HEAD, header_style=f"bold {C['metric']}")
        bt.add_column("", width=2, no_wrap=True)
        bt.add_column("Build", ratio=3, no_wrap=True, overflow="ellipsis")
        bt.add_column("Ver.", width=7, no_wrap=True)
        bt.add_column("CUDA", width=5, no_wrap=True)
        bt.add_column("Protocollo", ratio=3, no_wrap=True, overflow="ellipsis")
        for i, b in enumerate(self.builds):
            sel = self.pane == "build" and i == self.bi
            active = os.path.normcase(b["dir"]) == os.path.normcase(self.cfg["build_dir"])
            note = proto_note(b["build"])
            bt.add_row(Text("★" if active else ("▶" if sel else ""), style=f"bold {C['accent']}"),
                       Text(b["name"], style=f"bold {C['fg']}" if sel else C["fg"]),
                       Text(f"b{b['build']}" if b["build"] else "?", style=C["metric"]),
                       Text("sì" if b["cuda"] else "NO", style=C["ok"] if b["cuda"] else C["err"]),
                       Text(note, style=C["err"] if "NON" in note else (C["ok"] if "Host Node" in note
                                                                          or "compatibile" in note else C["warn"])),
                       style=f"on {C['sel']}" if sel else None)
        lay["builds"].update(Panel(bt, title=f"[bold {C['accent']}]Build[/]  [{C['muted']}]★ = in uso[/]",
                                   border_style=C["accent"] if self.pane == "build" else C["muted"]))

        ot = Table(expand=True, box=box.SIMPLE_HEAD, header_style=f"bold {C['metric']}")
        ot.add_column("", width=3)
        ot.add_column("Opzione", no_wrap=True)
        ot.add_column("Valore", no_wrap=True, overflow="ellipsis")
        last_group = None
        for i, o in enumerate(OPTIONS):
            if o["group"] != last_group:
                ot.add_row("", Text("— flag della riga di comando —" if o["group"] == "flag"
                                    else "— variabili d'ambiente GGML/CUDA —", style=C["muted"]), "")
                last_group = o["group"]
            sel = self.pane == "opt" and i == self.oi
            on = self.is_on(o)
            v = self.value(o)
            mark = "[x]" if on else "[ ]"
            ot.add_row(Text(mark, style=C["ok"] if on else C["muted"]),
                       Text(o["key"] if o["group"] == "env" else o["label"],
                            style=f"bold {C['fg']}" if sel else C["fg"]),
                       Text("" if v in (None, "") else str(v), style=C["metric"]),
                       style=f"on {C['sel']}" if sel else None)
        lay["opts"].update(Panel(ot, title=f"[bold {C['accent']}]Flag e variabili[/]",
                                 border_style=C["accent"] if self.pane == "opt" else C["muted"]))

        if self.pane == "opt":
            o = OPTIONS[self.oi]
            htxt = Text.assemble((o["label"] + "\n", f"bold {C['accent']}"), (o["help"], C["fg"]))
        else:
            b = self.builds[self.bi] if self.builds else {}
            htxt = Text.assemble((b.get("dir", "") + "\n", f"bold {C['accent']}"),
                                 (f"commit {b.get('commit') or '?'} · {proto_note(b.get('build', 0))}. "
                                  f"[INVIO] la usa (poi [A] per applicare). Client e server RPC devono avere "
                                  f"lo stesso protocollo: la build identica al Host Node (b{CLIENT_BUILD}) e' "
                                  f"la scelta piu' sicura.", C["fg"]))
        lay["help"].update(Panel(htxt, title=f"[bold {C['metric']}]Spiegazione[/]", border_style=C["muted"]))

        if time.time() - self.last_gpu > 5:
            self.gpu, self.last_gpu = gpu_line(), time.time()
        lines = [Text("comando: ", style=C["muted"]) + Text(self.cmdline(), style=C["metric"]),
                 Text("GPU: ", style=C["muted"]) + Text(self.gpu, style=C["fg"])]
        for l in announced_memory():
            lines.append(Text("annunciata ai client: ", style=C["muted"]) + Text(l, style=C["warn"]))
        for l in tail(LOG_PATH, 4):
            lines.append(Text(l, style=C["muted"]))
        lines.append(Text(self.msg, style=f"bold {C['ok']}"))
        lay["bottom"].update(Panel(Group(*lines), title=f"[bold {C['metric']}]Stato[/]", border_style=C["muted"]))
        return lay

    # -- tasti ------------------------------------------------------------
    def key(self, k: str, ask) -> bool:
        if k in ("q", "Q"):
            if self.dirty:
                self.msg = "modifiche non applicate: [A] per applicare, [Q] di nuovo per uscire comunque"
                self.dirty = False
                return True
            return False
        if k == "TAB":
            self.pane = "build" if self.pane == "opt" else "opt"
        elif k == "UP":
            if self.pane == "opt":
                self.oi = (self.oi - 1) % len(OPTIONS)
            elif self.builds:
                self.bi = (self.bi - 1) % len(self.builds)
        elif k == "DOWN":
            if self.pane == "opt":
                self.oi = (self.oi + 1) % len(OPTIONS)
            elif self.builds:
                self.bi = (self.bi + 1) % len(self.builds)
        elif k == " " and self.pane == "opt":
            self.toggle(OPTIONS[self.oi])
        elif k == "ENTER":
            if self.pane == "build" and self.builds:
                b = self.builds[self.bi]
                if not b["cuda"]:
                    self.msg = f"✗ {b['name']} non ha CUDA (manca cudart64_*.dll): girerebbe solo su CPU"
                else:
                    self.cfg["build_dir"] = b["dir"]
                    self.dirty = True
                    self.msg = f"build scelta: {b['name']} — premi [A] per applicare"
            else:
                o = OPTIONS[self.oi]
                if o["kind"] == "bool":
                    self.toggle(o)
                else:
                    v = ask(f"{o['label']} (attuale: {self.value(o)!r}, vuoto = default):")
                    if v is not None:
                        err = self.set_value(o, v.strip())
                        self.msg = f"✗ {err}" if err else f"{o['key']} = {v.strip() or '(default)'}"
        elif k in ("a", "A"):
            self.apply()
        elif k in ("r", "R"):
            self.builds = scan_builds()
            self.msg = "build e stato riletti"
        return True


def _getkey() -> Optional[str]:
    import msvcrt
    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        c2 = msvcrt.getwch()
        return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(c2)
    return {"\r": "ENTER", "\t": "TAB", "\x1b": "Q"}.get(ch, ch)


def main() -> None:
    for st in (sys.stdout, sys.stderr):       # console Windows: niente cp1252
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if "--test" in sys.argv:
        app = App()
        con = Console(width=150, height=48, record=True,
                      file=open(os.devnull, "w", encoding="utf-8"))
        con.print(app.render(), height=48)
        print(con.export_text()[:6000])
        for i, o in enumerate(OPTIONS):           # ogni opzione si accende e si spegne
            app.oi = i
            app.toggle(o)
            app.toggle(o)
        print("build trovate:", [(b["name"], b["build"], b["cuda"]) for b in app.builds])
        print("comando:", app.cmdline())
        return
    app = App()
    con = Console()
    with Live(app.render(), console=con, screen=True, refresh_per_second=4) as live:
        def ask(prompt: str) -> Optional[str]:
            live.stop()
            con.print(f"\n[bold {C['accent']}]{prompt}[/]")
            try:
                return input("❯ ")
            except (EOFError, KeyboardInterrupt):
                return None
            finally:
                live.start()
        last = 0.0
        while True:
            k = _getkey()
            if k is not None:
                if not app.key(k, ask):
                    break
                live.update(app.render(), refresh=True)
                last = time.time()
            else:
                time.sleep(0.05)
                if time.time() - last > 1.0:       # stato/log: 1 volta al secondo
                    live.update(app.render())
                    last = time.time()


if __name__ == "__main__":
    main()
