"""
miky_remote.py - telemetria multi-GPU e controllo remoto del nodo RPC "Miky".

Contratto JSON condiviso con `miky-client-tui` (Punto 3). Il client remoto
espone su http://192.168.1.100:50053 i seguenti endpoint:

* GET  /status   -> MikyNodeSnapshot serializzato (JSON, vedi dataclass)
* POST /start    -> avvia ggml-rpc-server (ritorna {"ok": true, "pid": N})
* POST /stop     -> ferma ggml-rpc-server (ritorna {"ok": true})
* POST /cleanup  -> rimuove .tmp/.part/scratch (ritorna {"ok": true, "removed": N})

Timeout brevi (1s): la UI non si deve bloccare mai se Miky è offline.
Fallback SSH automatico se la REST è irraggiungibile (es. Miky su Windows
senza il server HTTP ancora avviato).

Due paranoie già incontrate in questa codebase:
- MAI pkill -f: qui per fermare il server remoto si usa il PID ritornato
  da /status, mai pattern match sulla cmdline.
- MAI curl -C -: non si scarica niente qui, ma è la stessa regola.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

MIKY_HOST = "192.168.1.100"
MIKY_HTTP_PORT = 50053
MIKY_RPC_PORT = 50052
# Due identita' alternative per il fallback SSH:
# - 'mikye' quando Miky e' su Windows (OpenSSH con utente mikye)
# - 'root'  quando Miky e' sulla ISO Live Debian 13 che costruiamo qui
# Si prova sempre root PRIMA (Live ha sshd pre-configurato con la chiave
# in authorized_keys), poi mikye come fallback finale.
MIKY_SSH_USERS = ("root", "mikye")
MIKY_SSH_KEY = os.path.expanduser("~/.ssh/id_rsa_cluster")

HTTP_TIMEOUT_S = 1.0
POLL_INTERVAL_S = 2.0


# -------------------------------------------------------------------------
# Dataclass: contratto JSON condiviso Punto 2 <-> Punto 3
# -------------------------------------------------------------------------
@dataclass
class MikyGPU:
    """Una GPU sul nodo Miky. NVIDIA e Intel UHD condividono questo schema;
    i campi non applicabili (es. power_w per UHD) restano a 0.0."""
    index: int = 0
    name: str = ""
    type: str = "nvidia"               # "nvidia" | "intel-vulkan"
    driver: str = ""
    vram_total_mb: float = 0.0
    vram_used_mb: float = 0.0
    vram_free_mb: float = 0.0
    util_percent: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    power_limit_w: float = 0.0
    vulkan_available: bool = False
    shared_ram_total_mb: float = 0.0   # solo iGPU (RAM dinamica condivisa)
    shared_ram_used_mb: float = 0.0


@dataclass
class MikyRPCInfo:
    running: bool = False
    pid: Optional[int] = None
    port: int = MIKY_RPC_PORT
    cache_dir: str = "/mnt/nvme_cache/llama_rpc_cache"
    cache_gb_used: float = 0.0
    cache_tensor_count: int = 0
    cache_fs_free_gb: float = 0.0
    flags: List[str] = field(default_factory=list)
    uptime_s: float = 0.0


@dataclass
class MikyNodeSnapshot:
    reachable: bool = False
    error: str = ""
    rtt_ms: float = 0.0
    host: str = MIKY_HOST
    connected_to: str = ""              # testo libero: "cluster-client@192.168.1.50"
    uptime_s: float = 0.0
    os: str = ""                        # "Debian 13 Live" | "Windows 11"
    gpus: List[MikyGPU] = field(default_factory=list)
    cpu_name: str = ""
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    rpc: MikyRPCInfo = field(default_factory=MikyRPCInfo)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MikyNodeSnapshot":
        """Deserializza mantenendo robustezza su campi mancanti."""
        gpus = [MikyGPU(**{**MikyGPU().__dict__, **g}) for g in data.get("gpus", [])]
        rpc_data = data.get("rpc", {})
        rpc = MikyRPCInfo(**{**MikyRPCInfo().__dict__, **rpc_data})
        snap = cls(
            reachable=True,
            host=data.get("host", MIKY_HOST),
            connected_to=data.get("connected_to", ""),
            uptime_s=float(data.get("uptime_s", 0.0)),
            os=data.get("os", ""),
            cpu_name=data.get("cpu_name", ""),
            cpu_percent=float(data.get("cpu_percent", 0.0)),
            cpu_cores=int(data.get("cpu_cores", 0)),
            ram_total_gb=float(data.get("ram_total_gb", 0.0)),
            ram_used_gb=float(data.get("ram_used_gb", 0.0)),
            ram_percent=float(data.get("ram_percent", 0.0)),
            gpus=gpus,
            rpc=rpc,
            timestamp=time.time(),
        )
        return snap

    def to_json(self) -> Dict[str, Any]:
        """Serializza mantenendo i tipi primitivi (per il server HTTP di Miky)."""
        d = asdict(self)
        return d


# -------------------------------------------------------------------------
# Client HTTP + SSH fallback
# -------------------------------------------------------------------------
def _http_json(method: str, path: str, body: Optional[Dict[str, Any]] = None,
               timeout_s: float = HTTP_TIMEOUT_S) -> tuple[bool, Any]:
    """Chiama Miky REST. Ritorna (ok, dati|errore). Non solleva mai."""
    url = f"http://{MIKY_HOST}:{MIKY_HTTP_PORT}{path}"
    data_bytes = json.dumps(body).encode("utf-8") if body is not None else None
    try:
        req = urllib.request.Request(
            url, data=data_bytes, method=method,
            headers={"User-Agent": "ai_tui-miky/1.0", "Accept": "application/json",
                     "Content-Type": "application/json"} if body is not None else
                     {"User-Agent": "ai_tui-miky/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return True, json.loads(raw)
        except json.JSONDecodeError:
            return True, raw
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        return False, str(exc)


def _tcp_probe(host: str, port: int, timeout_s: float = 0.5) -> bool:
    """TCP connect probe: True se la porta accetta la connessione.
    Non manda bytes: solo un connect() veloce per sapere se c'e' qualcuno."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, socket.timeout):
        return False


def _ssh_cmd(cmd: str, timeout_s: float = 4.0) -> tuple[bool, str]:
    """Esegue un comando su Miky via SSH con la chiave dedicata.
    Prova in sequenza gli utenti noti (vedi MIKY_SSH_USERS) finche' uno risponde.
    Non solleva mai: ritorna (ok, stdout|err_concatenato)."""
    if not os.path.isfile(MIKY_SSH_KEY):
        return False, f"chiave SSH assente: {MIKY_SSH_KEY}"
    errs: List[str] = []
    for user in MIKY_SSH_USERS:
        try:
            proc = subprocess.run(
                ["ssh", "-i", MIKY_SSH_KEY,
                 "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=accept-new",
                 "-o", f"ConnectTimeout={int(max(1, timeout_s))}",
                 f"{user}@{MIKY_HOST}", cmd],
                capture_output=True, text=True, timeout=timeout_s + 1.0,
            )
            if proc.returncode == 0:
                return True, (proc.stdout or "").strip()
            errs.append(f"{user}: {(proc.stderr or proc.stdout or '').strip()[:120]}")
        except (OSError, subprocess.SubprocessError) as exc:
            errs.append(f"{user}: {exc}")
    return False, " | ".join(errs)


# -------------------------------------------------------------------------
# Monitor: thread daemon che pubblica l'ultimo snapshot
# -------------------------------------------------------------------------
class MikyRemoteMonitor:
    """
    Polling non-bloccante dello stato di Miky.

    Si avvia una volta sola; la UI legge `last_snapshot` quando vuole.
    Zero chiamate di rete nel thread della UI.
    """

    def __init__(self) -> None:
        self._last: MikyNodeSnapshot = MikyNodeSnapshot(reachable=False,
                                                       error="in attesa del primo polling")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_action: str = ""           # testo di stato per la TUI

    @property
    def last_snapshot(self) -> MikyNodeSnapshot:
        with self._lock:
            return self._last

    @property
    def last_action(self) -> str:
        return self._last_action

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="miky-remote-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(POLL_INTERVAL_S)

    def _poll_once(self) -> None:
        t0 = time.monotonic()
        ok, data = _http_json("GET", "/status")
        rtt = (time.monotonic() - t0) * 1000.0
        if ok and isinstance(data, dict):
            snap = MikyNodeSnapshot.from_json(data)
            snap.rtt_ms = rtt
            with self._lock:
                self._last = snap
            return
        # REST fallita -> probe TCP diretto sulla porta RPC (:50052).
        # Se risponde, Miky e' viva e ggml-rpc-server gira anche se il server
        # HTTP di controllo non c'e' ancora (es. ISO vecchia senza
        # miky-client-tui): la TUI mostra "RPC-only" invece di "OFFLINE".
        rpc_alive = _tcp_probe(MIKY_HOST, MIKY_RPC_PORT, timeout_s=0.8)
        # Probe SSH (lento, timeout 2s): puo' darci OS + PID.
        ssh_ok, out = _ssh_cmd(
            "pidof ggml-rpc-server 2>/dev/null; echo __OS__; uname -a 2>/dev/null || ver",
            timeout_s=2.0,
        )
        reachable = rpc_alive or ssh_ok
        snap = MikyNodeSnapshot(
            reachable=reachable,
            error="" if reachable else str(data),
            rtt_ms=rtt if reachable else 0.0,
        )
        if rpc_alive:
            # Modalita' RPC-only: non abbiamo GPU/CPU, ma almeno sappiamo che
            # il server e' su e la porta RPC risponde.
            snap.os = "RPC-only (installa miky-client-tui per telemetria completa)"
            snap.rpc.running = True
            snap.rpc.port = MIKY_RPC_PORT
        if ssh_ok:
            parts = out.split("__OS__", 1)
            pid_str = parts[0].strip()
            snap.os = parts[1].strip()[:80] if len(parts) > 1 else snap.os
            if pid_str:
                # L'accesso SSH a Miky stampa un banner su STDOUT: il PID non
                # e' il primo token (quello e' una cornice del riquadro), e
                # int() falliva in silenzio lasciando lo stato su "offline"
                # anche con il server vivo. Si cerca il primo token numerico.
                nums = [t for t in pid_str.split() if t.isdigit()]
                if nums:
                    snap.rpc.pid = int(nums[0])
                    snap.rpc.running = True
                    snap.rpc.port = MIKY_RPC_PORT
            # Auto-heal: prova ad avviare il servizio telemetria se non rispondeva
            _ssh_cmd("systemctl start miky-telemetry 2>/dev/null || true", timeout_s=1.5)
        with self._lock:
            self._last = snap

    # --- controllo remoto ---------------------------------------------------
    def remote_start(self) -> tuple[bool, str]:
        ok, data = _http_json("POST", "/start", {})
        if ok:
            self._last_action = f"REST /start → {data}"
            return True, self._last_action
        # Fallback SSH: lancia il server con -c e cache_dir su NVMe.
        cache = self._last.rpc.cache_dir if self._last.rpc.cache_dir else "/mnt/nvme_cache/llama_rpc_cache"
        cmd = ("mkdir -p %s && nohup ggml-rpc-server -H 0.0.0.0 -p %d -c %s "
               ">/var/log/ggml-rpc.log 2>&1 & echo $!") % (cache, MIKY_RPC_PORT, cache)
        ssh_ok, out = _ssh_cmd(cmd)
        self._last_action = f"SSH start → {'pid='+out if ssh_ok else 'FAIL '+out}"
        return ssh_ok, self._last_action

    def remote_stop(self) -> tuple[bool, str]:
        ok, data = _http_json("POST", "/stop", {})
        if ok:
            self._last_action = f"REST /stop → {data}"
            return True, self._last_action
        pid = self._last.rpc.pid
        if pid:
            ssh_ok, out = _ssh_cmd(f"kill {pid}")
            self._last_action = f"SSH kill {pid} → {'OK' if ssh_ok else 'FAIL '+out}"
            return ssh_ok, self._last_action
        self._last_action = "stop: nessun PID noto e REST irraggiungibile"
        return False, self._last_action

    def remote_cleanup(self) -> tuple[bool, str]:
        ok, data = _http_json("POST", "/cleanup", {})
        if ok:
            self._last_action = f"REST /cleanup → {data}"
            return True, self._last_action
        # Fallback SSH: puliamo solo .tmp/.part sotto /tmp (mai la cache NVMe).
        ssh_ok, out = _ssh_cmd(
            "find /tmp -maxdepth 2 \\( -name '*.tmp' -o -name '*.part' \\) -delete 2>/dev/null; echo done"
        )
        self._last_action = f"SSH cleanup → {'OK' if ssh_ok else 'FAIL '+out}"
        return ssh_ok, self._last_action


# -------------------------------------------------------------------------
# Rendering helper (Rich): pannelli pronti da inserire nel Layout
# -------------------------------------------------------------------------
def render_miky_panel(snap: MikyNodeSnapshot, compact: bool = False) -> Any:
    """
    Pannello Rich con telemetria Miky.
    compact=True: versione per Tab 1 (una colonna stretta).
    compact=False: versione estesa per Tab 9 (dettaglio GPU + RPC).
    """
    # Lazy import: miky_remote deve restare importabile anche in test headless.
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress_bar import ProgressBar
    from rich import box

    if not snap.reachable:
        body = Text.from_markup(
            f"[err]● Miky OFFLINE[/]\n"
            f"[muted]{snap.error[:120] if snap.error else 'nessun contatto'}[/]\n"
            f"[muted]endpoint:[/] http://{snap.host}:{MIKY_HTTP_PORT}/status"
        )
        return Panel(body, box=box.ROUNDED, border_style="red",
                     title="[bold err]🖥 MIKY (nodo RPC)[/]")

    # Modalita' RPC-only: Miky risponde su :50052 ma non su :50053 HTTP.
    # Nessuna GPU/CPU/RAM disponibile, pero' sappiamo che l'RPC gira.
    if not snap.gpus and snap.cpu_name == "" and snap.rpc.running:
        hint_tbl = Table(box=None, expand=True, show_header=False, padding=(0, 1))
        hint_tbl.add_column("K", style="muted", width=16)
        hint_tbl.add_column("V", style="bold fg", ratio=1)
        hint_tbl.add_row("Stato", Text.from_markup(
            f"[bold warn]● RPC-ONLY[/]  [muted]RTT {snap.rtt_ms:.1f} ms[/]"))
        hint_tbl.add_row("ggml-rpc-server",
                         f"[bold ok]:{snap.rpc.port} TCP ONLINE[/]"
                         + (f"  pid={snap.rpc.pid}" if snap.rpc.pid else ""))
        hint_tbl.add_row("Telemetria", "[warn]non disponibile[/]")
        hint_tbl.add_row("Causa",
                         "[muted]il server HTTP di controllo (:50053) non risponde[/]")
        hint_tbl.add_row("Fix", Text.from_markup(
            f"[muted]installa miky-client-tui su Miky:[/]\n"
            f"  [fg]curl -so /usr/local/bin/miky-client-tui \\\n"
            f"    http://192.168.1.50:50099/miky-client-tui && \\\n"
            f"  chmod +x /usr/local/bin/miky-client-tui && \\\n"
            f"  setsid /usr/local/bin/miky-client-tui </dev/null \\\n"
            f"    >/var/log/miky-client-tui.log 2>&1 &[/]"))
        return Panel(
            hint_tbl, box=box.ROUNDED, border_style="yellow",
            title=f"[bold warn]🖥 MIKY · {snap.host} · RPC-ONLY[/]",
            subtitle="[muted][K] start/stop · [P] probe[/]", padding=(0, 1),
        )

    tbl = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    tbl.add_column("K", style="muted", width=16 if compact else 18)
    tbl.add_column("V", style="bold fg", ratio=1)

    tbl.add_row("Stato", Text.from_markup(
        f"[bold ok]● ONLINE[/]  [muted]RTT {snap.rtt_ms:.1f} ms · {snap.os[:40]}[/]"))
    if snap.connected_to and not compact:
        tbl.add_row("Collegato a", snap.connected_to)
    if not compact:
        tbl.add_row("CPU", f"{snap.cpu_name[:40]} · {snap.cpu_percent:.0f}% · {snap.cpu_cores}c")
        tbl.add_row("RAM", f"{snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f} GB ({snap.ram_percent:.0f}%)")

    for gpu in snap.gpus:
        if gpu.type == "intel-vulkan":
            vk = "[ok]Vulkan OK[/]" if gpu.vulkan_available else "[warn]Vulkan n/d[/]"
            tbl.add_row(f"GPU{gpu.index} iGPU",
                        f"[bold]{gpu.name[:30]}[/]  {vk}  [muted]{gpu.driver[:28]}[/]")
            if gpu.shared_ram_total_mb:
                tbl.add_row("  RAM condivisa",
                            f"{gpu.shared_ram_used_mb:.0f}/{gpu.shared_ram_total_mb:.0f} MiB")
            continue
        # NVIDIA
        vram_pct = (gpu.vram_used_mb / max(1.0, gpu.vram_total_mb)) * 100.0
        # Vulkan funziona anche su NVIDIA tramite nvidia_icd: mostriamo lo
        # stato per non lasciare il campo misterioso a false quando l'ICD
        # c'e' ma vulkan-tools no.
        vk = ("[ok]Vulkan OK[/]" if gpu.vulkan_available
              else "[muted]Vulkan n/d[/]")
        tbl.add_row(f"GPU{gpu.index} dGPU",
                    f"[bold metric]{gpu.name[:30]}[/]  [muted]drv {gpu.driver[:12]}[/]  {vk}")
        util_col = "ok" if gpu.util_percent < 50 else ("warn" if gpu.util_percent < 85 else "err")
        temp_col = "ok" if gpu.temperature_c < 65 else ("warn" if gpu.temperature_c < 80 else "err")
        tbl.add_row("  Utilizzo", f"[{util_col}]{gpu.util_percent:.0f}%[/]  [{temp_col}]{gpu.temperature_c:.0f}°C[/]"
                    + (f"  [metric]{gpu.power_w:.0f}W[/]" if gpu.power_w else ""))
        bar_grid = Table.grid(expand=True, padding=(0, 1))
        bar_grid.add_column(ratio=1)
        bar_grid.add_column(width=18, justify="right")
        col = "ok" if vram_pct < 70 else ("warn" if vram_pct < 90 else "err")
        bar_grid.add_row(
            ProgressBar(total=100, completed=vram_pct, complete_style=col,
                        finished_style=col, style="sel"),
            Text(f"{gpu.vram_used_mb:.0f}/{gpu.vram_total_mb:.0f} MiB",
                 style=f"bold {col}"),
        )
        tbl.add_row("  VRAM", bar_grid)

    # RPC server
    rpc = snap.rpc
    rpc_state = "[bold ok]ONLINE[/]" if rpc.running else "[bold err]OFFLINE[/]"
    tbl.add_row("ggml-rpc-server",
                f"{rpc_state} :{rpc.port}" + (f" pid={rpc.pid}" if rpc.pid else ""))
    if not compact:
        tbl.add_row("Cache tensori",
                    f"[metric]{rpc.cache_gb_used:.1f} GB[/] · {rpc.cache_tensor_count} pesi "
                    f"[muted]{rpc.cache_dir}[/]")
        if rpc.cache_fs_free_gb:
            tbl.add_row("NVMe libero", f"[ok]{rpc.cache_fs_free_gb:.1f} GB[/]")
        if rpc.flags:
            tbl.add_row("Flag server", " ".join(rpc.flags))

    hint = ("[muted][K] start/stop · [C] cleanup remoto · [P] probe[/]"
            if not compact else "[muted]Tab 9 per controllo[/]")
    return Panel(
        tbl, box=box.ROUNDED,
        border_style="green" if rpc.running else "yellow",
        title=f"[bold metric]🖥 MIKY · {snap.host}[/]",
        subtitle=hint, padding=(0, 1),
    )
