"""
updates.py - check centralizzato degli aggiornamenti per Tab 15 "Updates".

Elimina i job di check update sparsi dentro altre Tab (es. Tab 13 "Build"
che aveva [C] "Controlla update"): tutto il check version avviene qui, in
un thread daemon unico che aggiorna una struttura dati condivisa.

Componenti monitorati:
  * ai_tui                 (git HEAD vs origin)
  * llama.cpp (locale)     (build attiva vs GitHub releases)
  * llama.cpp (Miky RPC)   (via /status + GitHub releases)
  * ggml-rpc-server Miky   (uptime + versione)
  * miky-client-tui        (hash dello script vs quello su http://192.168.1.50:50099)
  * driver NVIDIA          (nvidia-smi vs apt-cache policy)
  * driver Mesa / Vulkan   (dpkg vs apt-cache policy)
  * firmware Debian        (dpkg -l firmware-* vs apt)

Il check NON blocca mai la UI: tutto async in un thread, con timeout brevi.
Niente curl -C -: le poche richieste HTTP sono GET piccoli con urllib.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HTTP_TIMEOUT_S = 8.0
MIKY_HOST = "192.168.1.100"
# Il ggml-rpc-server ascolta sulla 50052 e parla un protocollo binario, non
# HTTP: il contatto si verifica con una connessione TCP, non con un GET.
MIKY_RPC_PORT = 50052
MIKY_HTTP_PORT = 50053


@dataclass
class ComponentStatus:
    name: str                              # "llama.cpp (locale)"
    installed: str = "n/d"                 # "b11435"
    latest: str = "n/d"                    # "b11500"
    status: str = "unknown"                # "ok" | "update" | "unknown" | "error"
    detail: str = ""                       # testo libero per la colonna note
    update_cmd: str = ""                   # comando suggerito per aggiornare
    category: str = ""                     # "code" | "driver" | "firmware" | "remote"
    checked_at: float = 0.0


def _http_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> Optional[Any]:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ai_tui-updates/1.0",
                          "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, ValueError, socket.timeout):
        return None


def _run(cmd: List[str], timeout_s: float = 3.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s)
        return p.returncode, (p.stdout or "").strip() or (p.stderr or "").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


# ---------------------------------------------------------------------------
# Checker singoli
# ---------------------------------------------------------------------------
def check_llamacpp_latest() -> str:
    """Ultima release di llama.cpp su GitHub (tag name, es. 'b11534')."""
    try:
        import build_manager as _bm
        bm_rel = _bm.latest_release(use_cache=True)
        if bm_rel and not bm_rel.get("error") and bm_rel.get("tag"):
            return str(bm_rel["tag"])
    except Exception:
        pass

    data = _http_json(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        timeout_s=15.0,
    )
    tag = str((data or {}).get("tag_name") or "")
    if not re.match(r"^b\d{4,6}$", tag):
        items = _http_json(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
            timeout_s=15.0,
        )
        for r in items or []:
            t = str(r.get("tag_name") or "")
            if re.match(r"^b\d{4,6}$", t):
                return t
    return tag or "n/d"


def check_llamacpp_local(build_dir: str = "/mnt/lm_models/DEBIAN_FOLDER/llamacpp") -> ComponentStatus:
    """Build attiva: ultima cartella llama-b* presente in build_dir."""
    s = ComponentStatus(name="llama.cpp (locale)", category="code")
    try:
        # Solo directory 'llama-bXXXXX', escludendo .tar.gz / .zip e simili.
        # Ordinamento NUMERICO: in ordine alfabetico "llama-b9999" verrebbe
        # dopo "llama-b11435" e si segnalerebbe la build sbagliata.
        builds = sorted(
            [d for d in os.listdir(build_dir)
             if re.match(r"^llama-b\d{4,6}$", d)
             and os.path.isdir(os.path.join(build_dir, d))
             # Una cartella col solo .tar.gz dentro e' un download non
             # estratto: contarla direbbe "aggiornato" puntando al vuoto.
             and os.path.isfile(os.path.join(build_dir, d, "llama-server"))],
            key=lambda d: int(re.sub(r"\D", "", d)),
            reverse=True,
        )
        if builds:
            s.installed = builds[0].replace("llama-", "")
            s.detail = f"{len(builds)} build in {build_dir}"
    except OSError as e:
        s.detail = f"dir non accessibile: {e}"
    s.latest = check_llamacpp_latest()
    if s.installed != "n/d" and s.latest != "n/d":
        s.status = "ok" if s.installed == s.latest else "update"
    s.update_cmd = (
        "cd /mnt/lm_models/DEBIAN_FOLDER/llamacpp && "
        "git clone --depth=1 -b " + s.latest + " "
        "https://github.com/ggml-org/llama.cpp.git llama-" + s.latest
    )
    s.checked_at = time.time()
    return s


def check_ai_tui(ai_tui_dir: str = "/mnt/lm_models/DEBIAN_FOLDER/ai_tui") -> ComponentStatus:
    s = ComponentStatus(name="ai_tui", category="code")
    if not os.path.isdir(os.path.join(ai_tui_dir, ".git")):
        # Non e' un git repo: segnalo che il tracking versione non e' disponibile
        s.installed = "no-git"
        s.detail = "non un repo git (nessun tracking versione)"
        s.status = "unknown"
    else:
        rc, out = _run(["git", "-C", ai_tui_dir, "rev-parse", "--short", "HEAD"])
        if rc == 0:
            s.installed = out
        rc, out = _run(["git", "-C", ai_tui_dir, "log", "-1", "--format=%s"])
        if rc == 0:
            s.detail = out[:70]
        # Confronto con origin (se presente, no fetch)
        rc, out = _run(["git", "-C", ai_tui_dir, "rev-parse", "--short", "@{u}"],
                       timeout_s=1.5)
        if rc == 0 and out:
            s.latest = out
            s.status = "ok" if s.installed == s.latest else "update"
            s.update_cmd = f"cd {ai_tui_dir} && git pull"
    s.checked_at = time.time()
    return s


def check_miky_components() -> List[ComponentStatus]:
    """Query /status di Miky + ricava versioni dei componenti remoti."""
    out: List[ComponentStatus] = []
    data = _http_json(f"http://{MIKY_HOST}:{MIKY_HTTP_PORT}/status",
                      timeout_s=2.0)
    if data is None:
        # Niente endpoint HTTP di stato: si ripiega sulla prova TCP della
        # porta RPC, che dice comunque se il backend e' vivo.
        alive = False
        try:
            with socket.create_connection((MIKY_HOST, MIKY_RPC_PORT), timeout=2.0):
                alive = True
        except OSError:
            alive = False
        out.append(ComponentStatus(
            name="ggml-rpc-server (Miky)", category="remote",
            installed="running" if alive else "down",
            detail=(f"porta {MIKY_RPC_PORT} raggiungibile"
                    if alive else f"nessun contatto su :{MIKY_RPC_PORT}"),
            latest=check_llamacpp_latest(),
            status="ok" if alive else "error"))
        return out

    rpc = data.get("rpc", {})
    os_str = data.get("os", "")[:40]
    out.append(ComponentStatus(
        name="miky-client-tui (Miky)", category="remote",
        installed="running" if data.get("reachable") else "down",
        detail=f"uptime {int(data.get('uptime_s', 0))}s · {os_str}",
        latest="n/d (via SSH per confronto hash)",
        status="ok"))
    out.append(ComponentStatus(
        name="ggml-rpc-server (Miky)", category="remote",
        installed=f"pid {rpc.get('pid')}" if rpc.get("running") else "down",
        detail=f":{rpc.get('port', 50052)} · cache {rpc.get('cache_gb_used', 0):.1f} GB · flags {' '.join(rpc.get('flags', []))[:40]}",
        latest=check_llamacpp_latest(),
        status="ok" if rpc.get("running") else "error"))

    for gpu in data.get("gpus", []):
        if gpu.get("type") == "nvidia":
            drv = gpu.get("driver", "n/d")
            out.append(ComponentStatus(
                name=f"NVIDIA driver (Miky GPU{gpu.get('index',0)})",
                category="driver", installed=drv,
                detail=f"{gpu.get('name','?')[:40]} · VRAM {gpu.get('vram_total_mb',0):.0f} MiB",
                latest="n/d (remote apt-cache)", status="unknown"))
        elif gpu.get("type") == "intel-vulkan":
            out.append(ComponentStatus(
                name=f"Mesa/Vulkan (Miky iGPU{gpu.get('index',0)})",
                category="driver",
                installed="Vulkan ON" if gpu.get("vulkan_available") else "Vulkan OFF",
                detail=f"{gpu.get('name','?')[:40]} · drv {gpu.get('driver','?')}",
                latest="n/d", status="ok" if gpu.get("vulkan_available") else "update"))
    return out


def check_nvidia_local() -> ComponentStatus:
    s = ComponentStatus(name="NVIDIA driver (locale)", category="driver")
    rc, out = _run(["nvidia-smi", "--query-gpu=driver_version",
                    "--format=csv,noheader"])
    if rc == 0:
        s.installed = out.split("\n")[0].strip()
    rc, out = _run(["apt-cache", "policy", "nvidia-driver"], timeout_s=4.0)
    if rc == 0:
        m = re.search(r"Candidate:\s+(\S+)", out)
        if m:
            s.latest = m.group(1)
    if s.latest in ("(none)", ""):
        # Driver installato fuori da apt (installer NVIDIA o repo esterno):
        # non esiste un candidato con cui confrontarsi, quindi non si inventa
        # un aggiornamento che non c'e'.
        s.latest = "n/d (non da apt)"
        s.status = "unknown"
    elif s.installed != "n/d" and s.latest != "n/d":
        s.status = "ok" if s.latest.startswith(s.installed.split(".")[0]) else "update"
    s.update_cmd = "sudo apt install --only-upgrade nvidia-driver"
    s.checked_at = time.time()
    return s


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
class UpdateMonitor:
    """
    Thread daemon che fa re-check periodico (default 10 min).
    La UI legge `components()` senza mai bloccarsi.
    """
    def __init__(self, interval_s: float = 600.0) -> None:
        self._interval = interval_s
        self._lock = threading.Lock()
        self._components: List[ComponentStatus] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_refresh: float = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="updates-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def components(self) -> List[ComponentStatus]:
        with self._lock:
            return list(self._components)

    def last_refresh(self) -> float:
        return self._last_refresh

    def force_refresh(self) -> None:
        """Trigger immediato (non blocca il chiamante)."""
        threading.Thread(target=self._refresh_once, daemon=True).start()

    def _loop(self) -> None:
        # Primo refresh subito
        self._refresh_once()
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if not self._stop.is_set():
                self._refresh_once()

    def _refresh_once(self) -> None:
        comps: List[ComponentStatus] = []
        try:
            comps.append(check_ai_tui())
        except Exception as e:
            comps.append(ComponentStatus(name="ai_tui", status="error", detail=str(e)))
        try:
            comps.append(check_llamacpp_local())
        except Exception as e:
            comps.append(ComponentStatus(name="llama.cpp (locale)", status="error", detail=str(e)))
        try:
            comps.append(check_nvidia_local())
        except Exception as e:
            comps.append(ComponentStatus(name="NVIDIA driver (locale)", status="error", detail=str(e)))
        try:
            comps.extend(check_miky_components())
        except Exception as e:
            comps.append(ComponentStatus(name="Miky (nodo RPC)", status="error", detail=str(e)))
        with self._lock:
            self._components = comps
        self._last_refresh = time.time()


# ---------------------------------------------------------------------------
# Rendering helper per Tab 15
# ---------------------------------------------------------------------------
def render_updates_panel(monitor: UpdateMonitor, selected_idx: int = 0,
                          bm_downloading: bool = False, bm_progress: str = "",
                          bm_confirm: bool = False,
                          bm_upd: Optional[Dict[str, Any]] = None) -> Any:
    """Pannello Rich con la lista componenti e status update."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress_bar import ProgressBar
    from rich.console import Group
    from rich import box

    comps = monitor.components()
    last = monitor.last_refresh()
    age = time.time() - last if last else 0.0

    # --- Barra di avanzamento download (visibile quando bm_download attivo) ---
    header_widgets: List[Any] = []
    if bm_downloading and bm_progress:
        # Barra indeterminata: non conosciamo la percentuale totale da qui.
        # Mostriamo la stringa di progresso che worker() aggiorna in tempo reale.
        dl_text = Text.from_markup(
            f"[bold ok]⬇ DOWNLOAD IN CORSO:[/] [fg]{bm_progress[:120]}[/]")
        header_widgets.append(dl_text)
    elif bm_confirm:
        remote = (bm_upd or {}).get("remote_build", "?")
        size_mb = (bm_upd or {}).get("size_mb", 0)
        header_widgets.append(Text.from_markup(
            f"[bold bg on warn]  ⚠ CONFERMA: premi [Y] per scaricare b{remote} "
            f"({size_mb} MiB) — [ESC] annulla  [/]"))

    tbl = Table(expand=True, box=box.SIMPLE_HEAD, border_style="border",
                padding=(0, 1))
    tbl.add_column("Componente", width=28, style="bold fg")
    tbl.add_column("Installata", width=20, style="metric")
    tbl.add_column("Ultima", width=20, style="metric2")
    tbl.add_column("Stato", width=8, justify="center")
    tbl.add_column("Note", no_wrap=False, overflow="fold", style="muted")

    status_style = {"ok": "[bold ok]OK[/]",
                    "update": "[bold warn]UPD[/]",
                    "error": "[bold err]ERR[/]",
                    "unknown": "[muted]?[/]"}

    for i, c in enumerate(comps):
        marker = "▶ " if i == selected_idx else "  "
        row_style = "bold bg on metric" if i == selected_idx else ""
        tbl.add_row(
            Text(marker + c.name[:26], style="bold fg"),
            Text(c.installed[:20]),
            Text(c.latest[:20]),
            Text.from_markup(status_style.get(c.status, "?")),
            Text(c.detail[:80], style="muted"),
            style=row_style,
        )
    if not comps:
        tbl.add_row("[muted]in attesa del primo check...[/]", "", "", "", "")

    subtitle = (f"[muted]ultimo check: {int(age)}s fa · "
                "[R] refresh · [U] mostra cmd · [Y][Y] scarica (llama.cpp) · "
                "[A] elenca tutti · [↑/↓] naviga · [0] torna[/]")
    content: Any = Group(*header_widgets, tbl) if header_widgets else tbl
    return Panel(content, box=box.ROUNDED, border_style="metric",
                 title=f"[bold accent]📦 UPDATES · {len(comps)} componenti monitorati[/]",
                 subtitle=subtitle, padding=(0, 1))
