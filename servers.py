"""
servers.py - elenco e gestione di TUTTI i server llama.cpp aperti.

Problema che risolve
--------------------
`LlamaServerManager` gestisce UNA sola istanza (managed_process): va bene per
avviare e fermare il server che la TUI ha lanciato lei, ma l'utente ha piu'
server aperti contemporaneamente (per esempio quello lanciato a mano su un
altro terminale, piu' quello via RPC verso Miky) e non li vede da nessuna
parte.

Qui si raccolgono in UNA lista unificata:

* i processi llama-server LOCALI, trovati in /proc;
* gli endpoint REMOTI configurati (Miky, ecc.), interrogati via HTTP.

Due trappole gia' incontrate in questa sessione, e come sono gestite qui
------------------------------------------------------------------------
1. **MAI `pkill -f llama-server`.** Il pattern `-f` matcha anche la riga di
   comando della shell che sta eseguendo il pkill, quindi uccide se stessa.
   Qui i processi sono identificati leggendo `/proc/<pid>/exe` (il link
   simbolico all'eseguibile reale) e si fermano con `kill <pid>` esplicito.
   Nessun pkill, nessun kill di gruppo: solo un PID verificato.

2. **MAI `curl -C -`.** Non usato qui (non si scarica niente), ma vale la
   regola: il resume con curl sovrascrive meta' sovrapposti e rovina i GGUF.

Il refresh non blocca la UI: `ServerInventory` interroghi gli endpoint in un
thread daemon e pubblica un'immagine (snapshot) in una lista. La UI legge
solo lo snapshot, quindi anche con la rete lenta non si blocca mai.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Timeout brevi: la UI aspetta al massimo qualche decimo di secondo per
# endpoint. Un server lento non deve poter congelare il pannello.
HTTP_TIMEOUT_S = 1.5

# Come viene riconosciuto un llama-server: nome dell'eseguibile. Lo si
# confronta con la BASENOME di /proc/<pid>/exe, mai con la riga di comando.
SERVER_EXE_NAMES = ("llama-server",)

# Estrae "b11330" da un percorso tipo ".../llama-b11330/llama-server".
_BUILD_RE = re.compile(r"\bb\d{4,6}\b")


def _http_get(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[bool, Any]:
    """GET JSON con timeout. Non solleva mai: ritorna (ok, dati|errore)."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ai_tui-servers/1.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            return True, json.loads(body)
        except json.JSONDecodeError:
            return True, body
    except (urllib.error.URLError, OSError, ValueError, socket.timeout) as exc:
        return False, str(exc)


def _http_get_status(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, Any]:
    """
    GET che distingue il codice HTTP, senza sollevare sui 4xx/5xx.

    Serve perche' llama.cpp risponde 503 su /health mentre sta caricando il
    modello e 200 quando e' pronto: senza il codice non si distinguono
    "sta caricando" da "rotto". urllib solleva HTTPError sui 503, quindi qui
    l'eccezione viene catturata e tradotta nel numero.
    """
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ai_tui-servers/1.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except Exception:
            return exc.code, ""
    except (urllib.error.URLError, OSError, ValueError, socket.timeout) as exc:
        return 0, str(exc)


# -------------------------------------------------------------------------
# Riga della lista: un server, locale o remoto
# -------------------------------------------------------------------------
@dataclass
class ServerRow:
    """Un server llama.cpp, visto dalla lista unificata."""

    key: str                       # identificatore univoco nella lista
    name: str                      # nome leggibile
    kind: str                      # "locale" | "remoto"
    host: str                      # "127.0.0.1" o l'IP remoto
    port: int = 0
    pid: Optional[int] = None      # solo per i locali
    # Tre stati distinti, non un booleano. Un processo llama-server puo'
    # essere VIVO ma non rispondere (caricamento del modello, attesa su un
    # server non raggiungibile): in quel caso non e' "online".
    #   OFFLINE  = nessun processo, o nessuna risposta HTTP
    #   STARTING = processo vivo ma /health non risponde
    #   ONLINE   = /health ha risposto davvero
    online: bool = False           # True SOLO se /health ha risposto
    health_checked: bool = False   # True se /health e' stato interrogato
    state: str = "OFFLINE"         # OFFLINE | STARTING | ONLINE
    model_path: str = ""
    n_ctx: int = 0
    tok_s: Optional[float] = None
    vram_used_mib: Optional[int] = None
    vram_total_mib: Optional[int] = None
    started_at: Optional[float] = None
    flags: List[str] = field(default_factory=list)   # flag di avvio, verbatim
    exe: str = ""                  # eseguibile reale (da /proc/pid/exe)
    build: str = ""                # es. "b11330"
    error_msg: str = ""
    managed: bool = False          # True se l'ha avviata questa TUI
    ssh_target: str = ""           # per i remoti: "utente@host"
    # "proc"   = trovato in /proc (exe = llama-server)
    # "health" = processo non leggibile (altro utente/container) ma la
    #            porta locale risponde come un llama-server su /props
    # "remote" = endpoint dichiarato dall'utente
    source: str = "proc"
    listen_ports: List[int] = field(default_factory=list)
    cgroup: str = ""               # cgroup del processo (isolamento OOM)
    details: Dict[str, Any] = field(default_factory=dict)   # /props /slots /metrics

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def state_line(self) -> str:
        """
        Stato in una parola, per la colonna 'stato'.

        Non si dice MAI "ONLINE" senza aver interrogato /health: il processo
        esistente non basta, perche' durante il caricamento del modello il
        server non accetta connessioni e la TUI mostrerebbe un falso verde.
        """
        return self.state

    @property
    def model_name(self) -> str:
        if not self.model_path:
            return "-"
        return os.path.basename(self.model_path)

    @property
    def vram_line(self) -> str:
        if not self.vram_used_mib:
            return "-"
        pct = ""
        if self.vram_total_mib:
            pct = f" ({self.vram_used_mib * 100 // max(self.vram_total_mib, 1)}%)"
        return f"{self.vram_used_mib} MiB{pct}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "kind": self.kind,
            "host": self.host, "port": self.port, "pid": self.pid,
            "online": self.online, "state": self.state_line,
            "model": self.model_name,
            "model_path": self.model_path, "n_ctx": self.n_ctx,
            "tok_s": self.tok_s, "vram": self.vram_line,
            "build": self.build, "managed": self.managed,
            "source": self.source, "listen_ports": self.listen_ports,
            "flags": " ".join(self.flags), "error": self.error_msg,
        }


# -------------------------------------------------------------------------
# Scansione dei processi LOCALI via /proc
# -------------------------------------------------------------------------
def _read_cmdline(pid: int) -> List[str]:
    """Argomenti del processo, gia' separati. [] se non leggibili."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except (OSError, PermissionError):
        return []
    return [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]


def _read_exe(pid: int) -> str:
    """
    Percorso dell'eseguibile REALE, letto dal link /proc/<pid>/exe.

    Questo e' il controllo che tiene lontano la trappola del pkill: se il
    link non e' un llama-server, il processo non e' di nostro interesse e
    non lo tocchiamo, qualunque cosa dica la sua riga di comando.
    """
    try:
        return os.path.basename(os.readlink(f"/proc/{pid}/exe"))
    except (OSError, PermissionError):
        return ""


def _proc_net_listeners() -> Dict[int, List[tuple]]:
    """
    inode del socket -> [(indirizzo, porta)] per tutti i socket TCP in LISTEN
    (stato 0A) letti da /proc/net/tcp e /proc/net/tcp6. Nessun comando
    esterno (ss/netstat): solo lettura di file del kernel.
    """
    out: Dict[int, List[tuple]] = {}
    for path, v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        try:
            with open(path, "r", encoding="ascii", errors="replace") as fh:
                next(fh, None)
                for line in fh:
                    parts = line.split()
                    if len(parts) < 10 or parts[3] != "0A":
                        continue
                    addr_hex, port_hex = parts[1].split(":")
                    port = int(port_hex, 16)
                    if v6:
                        addr = "::" if set(addr_hex) == {"0"} else (
                            "::1" if addr_hex.endswith("01000000") and addr_hex.startswith("0000000000000000") else "v6")
                    else:
                        b = bytes.fromhex(addr_hex)[::-1]
                        addr = ".".join(str(x) for x in b)
                    out.setdefault(int(parts[9]), []).append((addr, port))
        except (OSError, ValueError):
            continue
    return out


def _pid_listen(pid: int, listeners: Optional[Dict[int, List[tuple]]] = None) -> List[tuple]:
    """Le coppie (indirizzo, porta) su cui il processo `pid` e' in ascolto."""
    listeners = _proc_net_listeners() if listeners is None else listeners
    found: List[tuple] = []
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return found
    for fd in fds:
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            ino = int(link[8:-1])
            for pair in listeners.get(ino, []):
                if pair not in found:
                    found.append(pair)
    return found


def _probe_host(bind: str) -> str:
    """L'host da interrogare per un server in ascolto su `bind`."""
    if not bind or bind in ("0.0.0.0", "::", "v6", "localhost"):
        return "127.0.0.1"
    if bind == "::1":
        return "[::1]"
    return bind


def _flag_value(argv: List[str], *names: str) -> Optional[str]:
    """Valore di un flag presente in argv, per qualsiasi delle sue spellings."""
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
        for n in names:
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def _has_flag(argv: List[str], *names: str) -> bool:
    return any(a in names for a in argv)


def _parse_local(pid: int, argv: List[str], exe: str,
                 listeners: Optional[Dict[int, List[tuple]]] = None) -> ServerRow:
    """
    Trasforma /proc/<pid>/cmdline in una ServerRow.

    Non si stima niente che si possa leggere: porta, modello, contesto e KV
    vengono dai flag reali della riga di comando.
    """
    model = _flag_value(argv, "-m", "--model") or ""
    # llama-server senza --port ascolta sulla 8080 (default documentato).
    # La porta vera pero' si legge dai socket del processo: e' l'unica
    # fonte che non puo' sbagliare (env LLAMA_ARG_PORT, script wrapper...).
    port_s = _flag_value(argv, "--port") or "8080"
    host_flag = _flag_value(argv, "--host") or "127.0.0.1"
    ctx_s = _flag_value(argv, "-c", "--ctx-size", "--ctx") or "0"
    rpc = _flag_value(argv, "--rpc")
    kv_k = _flag_value(argv, "-ctk", "--cache-type-k") or ""
    kv_v = _flag_value(argv, "-ctv", "--cache-type-v") or ""

    # I flag di avvio sono tutto tranne il binario, il modello e --log-file:
    # la UI li mostra cosi' l'utente puo' copiarli e modificarli.
    skip = {"-m", "--model", "--log-file", "--port", "-p"}
    flags: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if i == 0:
            i += 1
            continue
        if a in ("-m", "--model", "--log-file", "--port", "-p"):
            i += 2
            continue
        flags.append(a)
        # se il flag successivo non inizia con '-', e' il suo valore: si
        # include cosi' il comando rimane copiabile.
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            flags.append(argv[i + 1])
            i += 2
        else:
            i += 1

    try:
        port = int(port_s)
    except (TypeError, ValueError):
        port = 0
    listen = _pid_listen(pid, listeners)
    listen_ports = sorted({pt for _a, pt in listen})
    if listen_ports and port not in listen_ports:
        # Il processo ascolta davvero su un'altra porta (es. LLAMA_ARG_PORT).
        port = listen_ports[0]
    bind = next((a for a, pt in listen if pt == port), host_flag)
    try:
        n_ctx = int(ctx_s)
    except (TypeError, ValueError):
        n_ctx = 0

    # Il nome del build lo prendo dal percorso: la cartella si chiama
    # "llama-b11330" e il build dentro e' "b11330". Serve una regex, non un
    # startswith("b"): "llama-b11330" non inizia con "b".
    build = ""
    exe_path = ""
    try:
        exe_path = os.readlink(f"/proc/{pid}/exe")
    except (OSError, PermissionError):
        pass
    m = _BUILD_RE.search(exe_path)
    if m:
        build = m.group(0)

    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            fields = fh.read().split()
        # /proc/<pid>/stat campo 22 (1-based) = starttime in clock tick.
        started = float(fields[21]) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        started = 0.0

    row = ServerRow(
        key=f"local:{pid}",
        name=f"locale:{port}" if port else f"locale:pid{pid}",
        kind="locale",
        host=_probe_host(bind),
        port=port,
        pid=pid,
        # Il processo esiste, ma NON e' ancora "online": quello lo decide
        # solo una risposta vera di /health (vedi probe_local_health).
        online=False,
        state="STARTING",
        model_path=model,
        n_ctx=n_ctx,
        flags=flags,
        exe=exe_path or exe,
        build=build,
        started_at=started,
        source="proc",
        listen_ports=listen_ports,
    )
    try:
        with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as fh:
            row.cgroup = fh.read().strip().split("::", 1)[-1]
    except OSError:
        pass
    if host_flag.endswith(".sock"):
        row.error_msg = f"in ascolto su socket UNIX {host_flag}: niente HTTP TCP"
    if rpc:
        row.error_msg = ""
        row.flags.append(f"[via --rpc {rpc}]")
    if kv_k or kv_v:
        row.flags.append(f"[KV {kv_k or '-'}/{kv_v or '-'}]")
    return row


def scan_local_servers() -> List[ServerRow]:
    """
    Tutti i llama-server LOCALI, letti da /proc.

    Il filtro e' /proc/<pid>/exe: si prende solo il processo il cui
    eseguibile reale si chiama llama-server. Non si usa pkill -f e non si
    tocca nessun altro processo.
    """
    rows: List[ServerRow] = []
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return rows
    listeners = _proc_net_listeners()
    for p in pids:
        exe = _read_exe(p)
        if exe not in SERVER_EXE_NAMES:
            continue
        argv = _read_cmdline(p)
        if not argv:
            continue
        # Sonde di un attimo (--help/--version/--list-devices, usate dal
        # registro build) non sono server: non vanno elencate ne' "riavviate".
        if any(a in ("--help", "-h", "--version", "--list-devices", "--completion-bash")
               for a in argv[1:]):
            continue
        try:
            row = _parse_local(int(p), argv, exe, listeners)
        except Exception:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r.port or 0)
    return rows


# Porte locali gia' classificate come "non llama-server": non si
# reinterrogano a ogni refresh (TTL), cosi' un servizio qualsiasi in
# ascolto non viene martellato di GET /props ogni 4 secondi.
_FOREIGN_CACHE: Dict[int, tuple] = {}
_FOREIGN_TTL_S = 120.0
# Porte di sistema/servizi noti che non si interrogano mai.
_SKIP_PORTS = {22, 25, 53, 111, 631, 5432, 3306, 6379, 50052}


def scan_foreign_local_servers(known_ports: List[int]) -> List[ServerRow]:
    """
    llama-server LOCALI il cui processo NON e' leggibile da /proc (altro
    utente, container, sandbox) ma che rispondono sulla porta.

    Si guardano solo le porte TCP in LISTEN su questa macchina che non sono
    gia' attribuite a un processo noto; ognuna viene identificata con
    GET /props (risposta JSON con default_generation_settings/build_info =
    llama-server). Risultato in cache per _FOREIGN_TTL_S.
    """
    rows: List[ServerRow] = []
    now = time.time()
    seen = set(known_ports)
    for _ino, pairs in _proc_net_listeners().items():
        for addr, port in pairs:
            if port in seen or port in _SKIP_PORTS or port < 1024:
                continue
            seen.add(port)
            if addr not in ("0.0.0.0", "127.0.0.1", "::", "::1", "v6"):
                continue
            cached = _FOREIGN_CACHE.get(port)
            if cached and now - cached[0] < _FOREIGN_TTL_S and not cached[1]:
                continue
            host = _probe_host(addr)
            ok, props = _http_get(f"http://{host}:{port}/props", timeout_s=0.4)
            is_llama = ok and isinstance(props, dict) and (
                "default_generation_settings" in props or "build_info" in props)
            _FOREIGN_CACHE[port] = (now, is_llama)
            if not is_llama:
                continue
            row = ServerRow(key=f"local-port:{port}", name=f"locale:{port}",
                            kind="locale", host=host, port=port, source="health",
                            listen_ports=[port])
            row.flags = ["[adottato via /props: processo non leggibile da /proc]"]
            rows.append(row)
    return rows


def probe_local_health(row: ServerRow) -> ServerRow:
    """
    Chiede /health al server locale e gli assegna lo stato VERO.

    Sequenza:
      * nessuna porta            -> OFFLINE (non si puo' interrogare)
      * /health risponde         -> ONLINE  (health_checked=True)
      * /health non risponde     -> STARTING se il processo e' vivo

    Restituisce la stessa riga, modificata: comodo da usare in-place.
    """
    if not row.port:
        row.online = False
        row.health_checked = False
        row.state = "OFFLINE"
        row.error_msg = row.error_msg or "porta non determinata dai flag"
        return row
    base = f"http://{row.host or '127.0.0.1'}:{row.port}"
    code, _body = _http_get_status(f"{base}/health")
    row.health_checked = True
    # llama.cpp risponde 503 su /health FINCHE' il modello non e' pronto e
    # 200 quando e' servibile. I due casi vanno distinti: 503 non e' "rotto",
    # e' "sta ancora caricando".
    if code == 200:
        row.online = True
        row.state = "ONLINE"
        row.error_msg = ""
    elif code == 503:
        row.online = False
        row.state = "STARTING"
        row.error_msg = "/health 503: il server sta ancora caricando il modello"
    elif code == 0:
        # Nessuna risposta: il processo c'e' ma il server non ascolta ancora.
        row.online = False
        alive = bool(row.pid) and os.path.exists(f"/proc/{row.pid}")
        # STARTING solo se il processo esiste e NON e' ancora in ascolto
        # (sta caricando). Se ascolta ma non risponde e' un errore vero.
        row.state = "STARTING" if alive else "OFFLINE"
        row.error_msg = f"/health non risponde ({str(_body)[:60]})"
    else:
        row.online = False
        row.state = "OFFLINE"
        row.error_msg = f"/health ha risposto HTTP {code}"
    if row.online:
        # /props aggiunge modello, contesto e build quando disponibili.
        pok, props = _http_get(f"{base}/props")
        if pok and isinstance(props, dict):
            row.model_path = str(props.get("model_path") or row.model_path)
            bi = str(props.get("build_info") or "")
            if bi:
                row.build = bi.split("-")[0]
            dgs = props.get("default_generation_settings") or {}
            if isinstance(dgs, dict) and dgs.get("n_ctx"):
                try:
                    row.n_ctx = int(dgs["n_ctx"])
                except (TypeError, ValueError):
                    pass
    return row


def _parse_metrics(text: str) -> Dict[str, float]:
    """Metriche Prometheus di llama-server (solo le righe llamacpp:*)."""
    out: Dict[str, float] = {}
    for line in (text or "").splitlines():
        if line.startswith("llamacpp:"):
            try:
                k, v = line.split()[:2]
                out[k[len("llamacpp:"):]] = float(v)
            except ValueError:
                continue
    return out


def fetch_details(row: ServerRow) -> Dict[str, Any]:
    """
    Info complete di un llama-server (locale, adottato o remoto HTTP):

    * /props  -> modello, n_ctx, build, chat template, modalita'
    * /slots  -> slot (abilitato di default, --no-slots lo spegne): per
                 ognuno n_ctx, se sta elaborando, token in cache
    * /metrics-> SOLO se il server e' partito con --metrics: tok/s di
                 prompt e generazione, richieste in corso, KV usata

    Nessuna stima: se un endpoint non c'e' lo si dice (campo *_error).
    """
    if row.kind == "remoto" and "rpc" in " ".join(row.flags).lower():
        return {"error": "endpoint ggml-rpc: non e' HTTP, niente /props /slots"}
    base = row.url if not row.host.startswith("[") else f"http://{row.host}:{row.port}"
    det: Dict[str, Any] = {"url": base, "t": time.time()}
    ok, props = _http_get(f"{base}/props")
    if ok and isinstance(props, dict):
        dgs = props.get("default_generation_settings") or {}
        det["props"] = {
            "model_path": props.get("model_path", ""),
            "n_ctx": (dgs or {}).get("n_ctx"),
            "total_slots": props.get("total_slots"),
            "build_info": props.get("build_info", ""),
            "modalities": props.get("modalities", {}),
            "chat_template": (str(props.get("chat_template") or "")[:60]
                              + ("…" if len(str(props.get("chat_template") or "")) > 60 else "")),
            "is_sleeping": props.get("is_sleeping", False),
        }
    else:
        det["props_error"] = str(props)[:100]
    ok, slots = _http_get(f"{base}/slots")
    if ok and isinstance(slots, list):
        det["slots"] = [{
            "id": sl.get("id"),
            "n_ctx": sl.get("n_ctx"),
            "busy": bool(sl.get("is_processing")),
            "speculative": sl.get("speculative"),
            "n_past": ((sl.get("next_token") or [{}])[0] or {}).get("n_decoded")
                      if isinstance(sl.get("next_token"), list) else None,
        } for sl in slots if isinstance(sl, dict)]
    else:
        det["slots_error"] = ("disabilitato (--no-slots)" if "501" in str(slots)
                              or "not supported" in str(slots).lower() else str(slots)[:100])
    code, body = _http_get_status(f"{base}/metrics")
    if code == 200:
        det["metrics"] = _parse_metrics(body)
    else:
        det["metrics_error"] = ("non attivo: il server va avviato con --metrics"
                                if code in (404, 501) else f"HTTP {code}")
    row.details = det
    m = det.get("metrics") or {}
    if m.get("predicted_tokens_seconds"):
        row.tok_s = round(m["predicted_tokens_seconds"], 2)
    return det


def nvidia_vram() -> Dict[str, int]:
    """VRAM della GPU locale, in MiB. Tutto 0 se nvidia-smi non c'e'."""
    smi = _which("nvidia-smi")
    if not smi:
        return {"used": 0, "total": 0}
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3.0,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"used": 0, "total": 0}
        used, total = proc.stdout.strip().splitlines()[0].split(",")
        return {"used": int(used.strip()), "total": int(total.strip())}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {"used": 0, "total": 0}


# -------------------------------------------------------------------------
# Endpoint remoti
# -------------------------------------------------------------------------
# Endpoint gia' noti. Miky:50052 e' il backend ggml-rpc (NON un HTTP server):
# non risponde a /health, e va trattato come tale.
DEFAULT_REMOTE_ENDPOINTS: List[Dict[str, str]] = [
    {"name": "Miky RPC", "host": "192.168.1.100", "port": "50052",
     "type": "rpc", "ssh": "root@192.168.1.100"},
    {"name": "Miky llama", "host": "192.168.1.100", "port": "8080",
     "type": "llama-server", "ssh": "root@192.168.1.100"},
]


ENDPOINTS_PATH = os.path.expanduser("~/.ai_tui_endpoints.json")


class EndpointStore:
    """
    Endpoint remoti inseriti A MANO dall'utente (IP + porta + nome), salvati
    in ~/.ai_tui_endpoints.json. Sostituisce la vecchia scansione della
    sottorete (netscan): niente /24 da sondare, nessun rallentamento.

    Al primo avvio il file viene creato con gli endpoint gia' noti (Miky).
    """

    def __init__(self, path: str = ENDPOINTS_PATH) -> None:
        self.path = path
        self.items: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self.items = [d for d in data if isinstance(d, dict) and d.get("host")]
                return
        except (OSError, ValueError):
            pass
        self.items = [dict(d) for d in DEFAULT_REMOTE_ENDPOINTS]
        self.save()

    def save(self) -> bool:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.items, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False

    def endpoints(self) -> List["RemoteEndpoint"]:
        out = []
        for d in self.items:
            try:
                out.append(_ep(d))
            except (KeyError, ValueError):
                continue
        return out

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """
        Testo libero dell'utente -> endpoint.  Formati accettati:
            192.168.1.20:8080
            192.168.1.20:50052 rpc
            PC-Studio 192.168.1.20:8080 utente@192.168.1.20
            nome=Miky host=192.168.1.100 port=8090 ssh=root@192.168.1.100 key=~/.ssh/id
        Una porta 50052 o la parola "rpc" -> tipo ggml-rpc.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("testo vuoto")
        kv = dict(re.findall(r"(\w+)=(\S+)", text))
        rest = re.sub(r"\w+=\S+", " ", text).split()
        host = kv.get("host", "")
        port = kv.get("port", "")
        name = kv.get("nome") or kv.get("name") or ""
        ssh = kv.get("ssh", "")
        typ = kv.get("type") or kv.get("tipo") or ""
        for tok in rest:
            if "@" in tok and not ssh:
                ssh = tok
            elif tok.lower() in ("rpc", "llama", "llama-server", "http"):
                typ = "rpc" if tok.lower() == "rpc" else "llama-server"
            elif re.fullmatch(r"[\w.\-]+:\d{1,5}", tok) and not host:
                host, port = tok.rsplit(":", 1)
            elif re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", tok) and not host:
                host = tok
            elif not name:
                name = tok
        if not host:
            raise ValueError("manca l'IP/host (es. 192.168.1.20:8080)")
        port = int(port or 8080)
        if not 1 <= port <= 65535:
            raise ValueError(f"porta non valida: {port}")
        if not typ:
            typ = "rpc" if port == 50052 else "llama-server"
        d = {"name": name or host, "host": host, "port": str(port), "type": typ}
        if ssh:
            d["ssh"] = ssh
        if kv.get("key"):
            d["ssh_key"] = os.path.expanduser(kv["key"])
        return d

    def add(self, text: str) -> Dict[str, Any]:
        try:
            d = self.parse(text)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        for it in self.items:
            if it["host"] == d["host"] and str(it["port"]) == d["port"]:
                it.update(d)
                self.save()
                return {"success": True, "updated": True, "endpoint": d}
        self.items.append(d)
        self.save()
        return {"success": True, "endpoint": d}

    def remove(self, host: str, port: int) -> Dict[str, Any]:
        before = len(self.items)
        self.items = [it for it in self.items
                      if not (it["host"] == host and int(it["port"]) == int(port))]
        if len(self.items) == before:
            return {"success": False, "error": f"{host}:{port} non e' tra gli endpoint salvati"}
        self.save()
        return {"success": True}


def ssh_gpu_status(ssh_target: str, key: str = "", timeout_s: int = 25) -> Dict[str, Any]:
    """
    Stato GPU di una macchina remota via SSH (nvidia-smi), per quando il
    probe HTTP/RPC non basta (es. ggml-rpc-server non dice quanta VRAM e'
    libera). Solo lettura. BatchMode=yes: niente richieste di password.
    """
    key = key or SSH_KEY_MIKY
    q = ("nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
         "utilization.gpu,temperature.gpu,memory.reserved,driver_model.current "
         "--format=csv,noheader,nounits")
    argv = ["ssh", "-i", key, "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            ssh_target, q]
    t0 = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"SSH timeout ({timeout_s}s)"}
    except OSError as exc:
        return {"success": False, "error": str(exc)[:100]}
    gpus = []
    for line in (proc.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 6 and parts[0].isdigit():
            def _n(v):
                try:
                    return int(float(v))
                except ValueError:
                    return None
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "used": _n(parts[2]), "total": _n(parts[3]),
                         "util": _n(parts[4]), "temp": _n(parts[5]),
                         "reserved": _n(parts[6]) if len(parts) > 6 else None,
                         "driver_model": parts[7] if len(parts) > 7 else ""})
    return {"success": bool(gpus), "gpus": gpus, "rtt_s": round(time.time() - t0, 2),
            "error": "" if gpus else ((proc.stderr or proc.stdout or "nessun output").strip()[:160])}


_LISTDEV_RE = re.compile(r"^\s*(\w+\d*):\s*(.*?)\s*\((\d+) MiB, (\d+) MiB free\)")


def llama_device_memory(binary: str, rpc: str = "", timeout_s: int = 25) -> Dict[str, Any]:
    """
    La memoria dei device COME LA VEDE llama.cpp (`llama-server [--rpc ...]
    --list-devices`): e' il valore che usa --fit. Su Windows/WDDM e' piu'
    basso del "libero" di nvidia-smi/Task Manager (driver riservato +
    contesto CUDA + budget WDDM per processo). Non carica modelli.
    """
    if not binary or not os.path.isfile(binary):
        return {"success": False, "error": "binario llama-server non trovato"}
    argv = [binary] + (["--rpc", rpc] if rpc else []) + ["--list-devices"]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"success": False, "error": str(exc)[:100]}
    devs = []
    for line in (p.stdout or "").splitlines() + (p.stderr or "").splitlines():
        m = _LISTDEV_RE.match(line)
        if m:
            devs.append({"device": m.group(1), "desc": m.group(2),
                         "total": int(m.group(3)), "free": int(m.group(4))})
    return {"success": bool(devs), "devices": devs,
            "error": "" if devs else (p.stderr or p.stdout or "")[-160:]}


@dataclass
class RemoteEndpoint:
    """Un endpoint remoto noto alla TUI."""
    name: str
    host: str
    port: int
    kind: str = "llama-server"        # "llama-server" o "rpc"
    ssh: str = ""                     # "utente@host" per l'avvio via SSH
    auto: bool = True                 # includerlo nel refresh automatico
    ssh_key: str = ""                 # chiave privata (default: quella di Miky)


class RemoteProbe:
    """Interroga un endpoint remoto: /health e /props, con timeout breve."""

    @staticmethod
    def probe(ep: RemoteEndpoint) -> ServerRow:
        row = ServerRow(
            key=f"remote:{ep.host}:{ep.port}",
            name=f"{ep.name}:{ep.port}",
            kind="remoto",
            host=ep.host,
            port=ep.port,
            ssh_target=ep.ssh,
            source="remote",
        )
        if ep.kind == "rpc":
            # ggml-rpc non e' HTTP. Si verifica solo che la porta risponda:
            # e' la stessa sonda che usa probe_rpc_backend().
            t0 = time.time()
            try:
                with socket.create_connection((ep.host, ep.port), timeout=HTTP_TIMEOUT_S):
                    row.online = True
                    row.health_checked = True
                    row.state = "ONLINE"
                    row.flags = ["ggml-rpc backend (non HTTP: niente /props)"]
                row.error_msg = f"RPC ok, RTT {(time.time() - t0) * 1000:.1f} ms"
            except OSError as exc:
                row.online = False
                row.health_checked = True
                row.state = "OFFLINE"
                row.error_msg = f"non raggiungibile: {exc}"
            return row

        ok, _health = _http_get(f"http://{ep.host}:{ep.port}/health")
        row.health_checked = True
        if not ok:
            row.online = False
            row.state = "OFFLINE"
            row.error_msg = str(_health)[:120]
            return row
        # /health ha risposto: il server e' davvero online.
        row.online = True
        row.state = "ONLINE"
        # /props aggiunge modello, contesto e build (/health dice solo "ok").
        ok2, data = _http_get(f"http://{ep.host}:{ep.port}/props")
        if ok2 and isinstance(data, dict):
            row.model_path = str(data.get("model_path") or "")
            row.build = str(data.get("build_info") or "").split("-")[0]
            dgs = data.get("default_generation_settings") or {}
            if isinstance(dgs, dict) and dgs.get("n_ctx"):
                try:
                    row.n_ctx = int(dgs["n_ctx"])
                except (TypeError, ValueError):
                    pass
            if data.get("is_sleeping"):
                row.error_msg = "sleeping"
        return row


# -------------------------------------------------------------------------
# Azioni: ferma, browser, avvio via SSH
# -------------------------------------------------------------------------
SSH_KEY_MIKY = os.path.expanduser(os.environ.get("AI_TUI_SSH_KEY", "~/.ssh/id_rsa_cluster"))
SSH_TIMEOUT_S = 20


def stop_server_pid(pid: int, verify_exe: bool = True) -> Dict[str, Any]:
    """
    Ferma UN server locale con kill <pid> esplicito.

    verify_exe=True ricontrolla /proc/<pid>/exe prima di mandare il segnale:
    se nel frattempo il PID e' stato riciclato dal sistema per un altro
    programma, non lo si tocca. E' la difesa contro l'errore che in questa
    sessione ha ucciso la shell esecutrice (pkill -f).
    """
    if verify_exe:
        exe = _read_exe(pid)
        if exe not in SERVER_EXE_NAMES:
            return {
                "success": False,
                "error": f"PID {pid} non e' piu' un llama-server (ora e' '{exe or 'sconosciuto'}'): "
                         f"non lo tocco.",
            }
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"success": False, "error": f"PID {pid} gia' terminato"}
    except PermissionError:
        return {"success": False, "error": f"permesso negato su PID {pid}"}
    # Si dà tempo al SIGTERM di fare effetto, poi SIGKILL se serve.
    for _ in range(20):
        time.sleep(0.1)
        if _pid_gone(pid):
            return {"success": True, "stopped_pid": pid, "message": f"fermato PID {pid} (SIGTERM)"}
    # riverifica prima del SIGKILL: il PID potrebbe essere stato riciclato
    if verify_exe and _read_exe(pid) not in SERVER_EXE_NAMES:
        if _pid_gone(pid):
            return {"success": True, "stopped_pid": pid, "message": f"fermato PID {pid} (SIGTERM)"}
        return {"success": False, "error": f"PID {pid} cambiato durante l'attesa: non mando SIGKILL"}
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return {"success": True, "stopped_pid": pid, "message": f"fermato PID {pid} (SIGTERM)"}
    except OSError as exc:
        return {"success": False, "error": f"impossibile fermare PID {pid}: {exc}"}
    for _ in range(20):                 # attende l'uscita vera dopo il SIGKILL
        if _pid_gone(pid):
            break
        time.sleep(0.1)
    return {"success": True, "stopped_pid": pid, "message": f"fermato PID {pid} (SIGKILL)"}


def _pid_gone(pid: int) -> bool:
    """True se il processo non esiste piu' o e' uno zombie (terminato ma non
    ancora raccolto). Se e' figlio della TUI lo si raccoglie (waitpid)."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            st = fh.read()
        state = st[st.rindex(b")") + 2:st.rindex(b")") + 3]
        return state in (b"Z", b"X")
    except (OSError, ValueError):
        return True


def wait_port_free(port: int, timeout_s: float = 10.0, host: str = "127.0.0.1") -> bool:
    """Attende che nessuno ascolti piu' su `port` (dopo uno stop)."""
    if not port:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        busy = False
        for h in (host, "::1"):
            try:
                with socket.create_connection((h, port), timeout=0.3):
                    busy = True
            except OSError:
                pass
        if not busy:
            return True
        time.sleep(0.2)
    return False


def open_in_browser(url: str) -> Dict[str, Any]:
    """
    Apre l'URL nel browser di default.

    Su questa macchina non c'e' un browser grafico quindi xdg-open puo' non
    esistere: in quel caso si restituisce l'URL, che e' comunque utile
    per il copia-incolla. Non viene mai sollevata un'eccezione.
    """
    opener = next((c for c in ("xdg-open", "gio", "gnome-open", "sensible-browser")
                   if _which(c)), None)
    if not opener:
        return {"success": False, "url": url, "error": "nessun opener installato", "manual": url}
    try:
        subprocess.Popen([opener, url], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return {"success": True, "url": url, "message": f"aperto con {opener}"}
    except OSError as exc:
        return {"success": False, "url": url, "error": str(exc), "manual": url}


def _which(name: str) -> Optional[str]:
    for d in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep):
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


# Il comando da eseguire su Miky passa per `cmd /c` (Windows con bash/SMB
# o WSL: Miky e' una macchina Windows raggiunta via SSH).
# Miky e' Linux: il comando va a /bin/sh via SSH, nessun prefisso di shell.
MIKY_CMD_PREFIX = ""


# BatchMode=yes: se su quell'host non c'e' una chiave che TUI conosce, il
# fallimento arriva IMMEDIATAMENTE invece di chiedere una password che
# nessuno puo' digitare dentro una TUI. Serve a non tentare SSH su host
# casuali trovati dalla scansione di rete.
def remote_ssh_command(ssh_target: str, command: str,
                       key: str = SSH_KEY_MIKY) -> List[str]:
    """Compono l'invocazione SSH. Nessun processo viene avviato qui."""
    return [
        "ssh", "-i", key, "-o", "ConnectTimeout=%d" % SSH_TIMEOUT_S,
        "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
        ssh_target, MIKY_CMD_PREFIX + command,
    ]


# Miky e' una build live Debian custom (toram + persistenza su SSD).
# Si usano i symlink stabili di /usr/local/bin invece del percorso con il
# numero di build: restano validi anche aggiornando /opt/llamacpp.
MIKY_BUILD_DIR = "/opt/llamacpp"
MIKY_RPC_EXE = "/usr/local/bin/ggml-rpc-server"
MIKY_LLAMA_EXE = "/usr/local/bin/llama-server"


def relaunch_local(row: ServerRow, extra_args: Optional[List[str]] = None,
                   dry_run: bool = False,
                   argv_override: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Riavvia un llama-server LOCALE con gli stessi identici flag.

    E' il "legare a se'" il server che l'utente ha avviato a mano: si
    ricostruisce la riga di comando da /proc/<pid>/cmdline (non da una
    lista supposta) e la si rilancia. Il binario si prende dal link
    /proc/<pid>/exe, quindi e' lo stesso che stava girando, non uno
    "llama-server" qualsiasi dal PATH.

    dry_run=True non avvia nulla: restituisce solo l'argv.
    """
    if row.kind != "locale" or not row.pid:
        return {"success": False,
                "error": "solo i server LOCALI si riavviano da qui"}
    if argv_override:
        # argv (con exe vero) letto PRIMA di fermare il processo: dopo lo stop
        # /proc/<pid> non esiste piu' (era il bug di [R]: fermava e non riavviava)
        new_argv = list(argv_override)
        import proc_spawn as _PS
        res = _PS.spawn_detached(new_argv, name=f"llama-{row.port or row.pid}")
        if not res.get("success"):
            return {"success": False, "error": res.get("error")}
        return {"success": True, "pid": res["pid"], "cmd": " ".join(new_argv),
                "log_path": res["log_path"], "isolated": res["isolated"],
                "proc": res.get("proc")}
    if not os.path.exists(f"/proc/{row.pid}"):
        return {"success": False, "error": f"PID {row.pid}: processo gia' terminato"}
    argv = _read_cmdline(row.pid)
    if not argv:
        return {"success": False, "error": f"PID {row.pid}: cmdline non leggibile"}
    exe = ""
    try:
        exe = os.readlink(f"/proc/{row.pid}/exe")
    except OSError:
        exe = argv[0]
    new_argv = [exe] + argv[1:]
    if extra_args:
        new_argv.extend(extra_args)
    if dry_run:
        return {"success": True, "dry_run": True, "argv": new_argv,
                "cmd": " ".join(new_argv)}
    import proc_spawn as _PS
    res = _PS.spawn_detached(new_argv, name=f"llama-{row.port or row.pid}")
    if not res.get("success"):
        return {"success": False, "error": res.get("error")}
    return {"success": True, "pid": res["pid"], "cmd": " ".join(new_argv),
            "log_path": res["log_path"], "isolated": res["isolated"]}


def start_remote_server(ep: RemoteEndpoint, command: str,
                        key: str = SSH_KEY_MIKY,
                        dry_run: bool = False) -> Dict[str, Any]:
    """
    Avvia un llama-server SU un'altra macchina via SSH.

    dry_run=True restituisce solo la riga di comando (usato dai test: non
    si tocca Miky). Altrimenti esegue davvero.

    Vincoli: NON tocca il task "HermesRPC" di Miky (che tiene su
    ggml-rpc-server): chi lo lancia e' l'utente con un llama-server, e
    la porta va indicata esplicitamente per non investire quella del RPC.
    """
    argv = remote_ssh_command(ep.ssh or f"root@{ep.host}", command, key)
    if dry_run:
        return {"success": True, "dry_run": True, "argv": argv, "cmd": " ".join(argv)}
    if not ep.ssh:
        return {"success": False, "error": f"endpoint {ep.name} senza target SSH configurato"}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "SSH: timeout dopo 180 s"}
    except OSError as exc:
        return {"success": False, "error": f"SSH non eseguibile: {exc}"}
    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "cmd": " ".join(argv),
    }


# -------------------------------------------------------------------------
# Azioni su Miky via SSH: avvio e arresto di un llama-server
# -------------------------------------------------------------------------
# Il task "HermesRPC" di Miky tiene su ggml-rpc-server e NON si tocca mai:
# nessuna delle funzioni qui sotto lo nomina, e la porta 50052 e' rifiutata
# per principio. L'utente avvia un llama-server con una porta esplicita,
# cosi' non si investe quella del RPC.
def miky_start_llama(model_path: str, port: int = 8090,
                     ctx: int = 32768, ngl: int = 99,
                     kv: str = "q8_0",
                     ssh_target: str = "root@192.168.1.100",
                     dry_run: bool = False) -> Dict[str, Any]:
    """
    Avvia un llama-server SU MIKY con la build CUDA 13 gia' presente.

    La porta di default e' 8090, NON 50052: quella e' del ggml-rpc-server
    tenuto su dal task HermesRPC, che questo codice non tocca.

    dry_run=True restituisce solo la riga di comando (la UI la mostra
    prima di eseguire davvero).
    """
    if not model_path:
        return {"success": False, "error": "serve un percorso .gguf su Miky"}
    if port == 50052:
        return {"success": False,
                "error": "la porta 50052 e' del ggml-rpc-server (task HermesRPC): "
                         "non la si tocca"}
    # setsid + redirezione: il server sopravvive alla chiusura della sessione
    # SSH, che altrimenti lo porterebbe giu' insieme a se'.
    cmd = (f"setsid {MIKY_LLAMA_EXE} -m '{model_path}' --host 0.0.0.0 "
           f"--port {port} -c {ctx} -ngl {ngl} -fa on -ctk {kv} -ctv {kv} "
           f"> /tmp/llama-{port}.log 2>&1 < /dev/null & echo $!")
    ep = RemoteEndpoint(name="Miky llama", host="192.168.1.100", port=port,
                        kind="llama-server", ssh=ssh_target)
    return start_remote_server(ep, cmd, dry_run=dry_run)


def miky_stop_llama(port: int = 8090,
                    ssh_target: str = "root@192.168.1.100",
                    dry_run: bool = False) -> Dict[str, Any]:
    """
    Ferma un llama-server SU MIKY, per PID.

    Non si usa taskkill /IM: ucciderebbe per NOME e potrebbe prendere di petto
    ggml-rpc-server. Si cerca il PID che ascolta sulla porta e si uccide
    quello: lo stesso criterio "kill <PID> esplicito" usato in locale.
    """
    if port == 50052:
        return {"success": False,
                "error": "la porta 50052 e' del ggml-rpc-server (task HermesRPC): "
                         "non la si tocca"}
    # ss filtra gia' per porta LOCALE in ascolto, quindi non si rischia di
    # prendere le connessioni dei client (che hanno quella porta come remota).
    # sed estrae il PID: si uccide QUEL pid, mai per nome, altrimenti si
    # rischierebbe di investire anche ggml-rpc-server.
    cmd = (f"pid=$(ss -ltnpH 'sport = :{port}' 2>/dev/null"
           " | sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | head -1); "
           "if [ -n \"$pid\" ]; then kill -TERM \"$pid\" && echo \"terminato PID $pid\"; "
           "else echo 'nessun processo in ascolto su quella porta'; fi")
    ep = RemoteEndpoint(name="Miky stop", host="192.168.1.100", port=port,
                        kind="llama-server", ssh=ssh_target)
    return start_remote_server(ep, cmd, dry_run=dry_run)


def miky_sync_build(local_dir: str, ssh_target: str = "root@192.168.1.100",
                    key: str = SSH_KEY_MIKY, activate: bool = True,
                    restart: bool = True, dry_run: bool = False) -> Dict[str, Any]:
    """
    Copia una build llama.cpp locale su Miky e la rende il backend attivo.

    Serve perche' il protocollo RPC deve combaciare tra le due macchine:
    aggiornare un solo lato lo rompe con "RPC server version mismatch".
    Quando si aggiorna qui, Miky va aggiornata insieme.

    La destinazione /opt/llamacpp sta sulla persistenza (overlay su SSD),
    quindi la build sopravvive al riavvio nonostante Miky sia un live system.

    Passi: rsync della cartella, riaggancio dei symlink in /usr/local/bin,
    riavvio di miky-rpc.service. dry_run=True mostra solo i comandi.
    """
    name = os.path.basename(os.path.normpath(local_dir))
    if not name.startswith("llama-b"):
        return {"success": False, "error": f"nome build inatteso: {name}"}
    if not os.path.isfile(os.path.join(local_dir, "ggml-rpc-server")):
        return {"success": False,
                "error": f"{name} non contiene ggml-rpc-server: build incompleta"}
    remote = f"/opt/llamacpp/{name}"
    ssh_opts = (f"ssh -i {key} -o ConnectTimeout={SSH_TIMEOUT_S} "
                "-o StrictHostKeyChecking=accept-new -o BatchMode=yes")
    rsync_argv = ["rsync", "-a", "--delete", "-e", ssh_opts,
                  local_dir.rstrip("/") + "/", f"{ssh_target}:{remote}/"]
    # I symlink si rifanno con ln -sfn: 'f' sovrascrive, 'n' evita di
    # scrivere DENTRO il link vecchio se punta a una directory.
    post = (f"mkdir -p /opt/llamacpp && "
            f"ln -sfn {remote}/ggml-rpc-server /usr/local/bin/ggml-rpc-server && "
            f"ln -sfn {remote}/llama-server /usr/local/bin/llama-server && "
            f"chmod +x {remote}/ggml-rpc-server {remote}/llama-server")
    if restart:
        post += " && systemctl restart miky-rpc"
    post_argv = remote_ssh_command(ssh_target, post, key)
    if dry_run:
        return {"success": True, "dry_run": True, "build": name, "remote": remote,
                "rsync": " ".join(rsync_argv), "post": " ".join(post_argv)}
    try:
        pr = subprocess.run(rsync_argv, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "rsync: timeout dopo 30 min"}
    except OSError as exc:
        return {"success": False, "error": f"rsync non eseguibile: {exc}"}
    if pr.returncode != 0:
        return {"success": False, "error": (pr.stderr or "")[-300:], "step": "rsync"}
    if not activate:
        return {"success": True, "build": name, "remote": remote, "activated": False}
    try:
        pp = subprocess.run(post_argv, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"success": False, "error": str(exc)[:200], "step": "attivazione"}
    return {"success": pp.returncode == 0, "build": name, "remote": remote,
            "activated": pp.returncode == 0, "restarted": restart,
            "stderr": (pp.stderr or "")[-300:]}


def miky_rpc_running(ssh_target: str = "root@192.168.1.100") -> Dict[str, Any]:
    """
    Legge (NON tocca) se ggml-rpc-server gira su Miky.

    E' una sola lettura: serve per dire all'utente che il backend RPC e'
    vivo. Non avvia, non ferma, non modifica niente.
    """
    argv = ["ssh", "-i", SSH_KEY_MIKY, "-o", "ConnectTimeout=%d" % SSH_TIMEOUT_S,
            "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            ssh_target, MIKY_CMD_PREFIX + "pgrep -x ggml-rpc-server"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return {"running": False, "error": "SSH: timeout"}
    except OSError as exc:
        return {"running": False, "error": str(exc)[:80]}
    out = proc.stdout or ""
    # pgrep stampa un PID per riga ed esce con 1 se non trova niente.
    pids = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
    running = bool(pids)
    pid = pids[0] if pids else ""
    return {"running": running, "pid": pid, "stdout": out[-800:],
            "stderr": (proc.stderr or "")[-300:]}


# -------------------------------------------------------------------------
# Inventario: la lista unificata, aggiornata in background
# -------------------------------------------------------------------------
class ServerInventory:
    """
    La lista di TUTTI i server, locali e remoti.

    Il refresh gira in un thread daemon: la UI non aspetta mai la rete,
    legge solo l'ultimo snapshot. `refresh_now()` restituisce l'elenco
    calcolato al momento della chiamata (usato dai test).
    """

    def __init__(self, remotes: Optional[List[RemoteEndpoint]] = None,
                 refresh_s: float = 4.0) -> None:
        self.refresh_s = max(1.0, float(refresh_s))
        # Gli endpoint remoti vengono dallo store (inseriti dall'utente),
        # salvo che il chiamante ne passi una lista esplicita (test).
        self.store: Optional[EndpointStore] = None
        if remotes is None:
            self.store = EndpointStore()
            remotes = self.store.endpoints()
        self.remotes: List[RemoteEndpoint] = list(remotes)
        # PID avviati da questa TUI: la riga viene marcata "managed",
        # tutte le altre sono "adottate" (avviate a mano altrove).
        self.managed_pids: set = set()
        self.adopt_foreign: bool = True
        self._rows: List[ServerRow] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.selected_key: Optional[str] = None
        self.last_error: str = ""
        self.gpu_remote: Dict[str, Dict[str, Any]] = {}   # host -> ssh_gpu_status
        # Dettagli letti con [I] (/props /slots /metrics), per chiave di riga:
        # sopravvivono ai refresh del thread (che ricrea le righe).
        self.details: Dict[str, Dict[str, Any]] = {}

    # -- ciclo di vita ---------------------------------------------------
    def start(self) -> None:
        """Avvia il thread di refresh. Idempotente."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="server-inventory")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_now()
            except Exception as exc:      # un errore non deve uccidere il thread
                self.last_error = str(exc)
            self._stop.wait(self.refresh_s)

    # -- dati ------------------------------------------------------------
    def refresh_now(self, probe_remote: bool = True) -> List[ServerRow]:
        """
        Ricalcola l'elenco e lo pubblica. Chiamata dal thread o dai test.

        I locali si leggono da /proc (istantaneo, nessuna I/O di rete).
        I remoti si interrogano in parallelo con un ThreadPoolExecutor,
        cosi' un endpoint lento non ritarda gli altri.
        """
        rows = scan_local_servers()
        if self.adopt_foreign:
            known = [pt for r in rows for pt in (r.listen_ports or [r.port])]
            try:
                rows.extend(scan_foreign_local_servers(known))
            except Exception as exc:
                self.last_error = f"adozione via porta: {exc}"
        for r in rows:
            r.managed = bool(r.pid and r.pid in self.managed_pids)
        if probe_remote and self.remotes:
            rows.extend(self._probe_remotes(rows))
        # Lo stato NON e' quello "processo vivo": ogni server locale viene
        # interrogato davvero su /health, cosi' la lista non dice mai ONLINE
        # per un server che non sta rispondendo.
        for row in rows:
            if row.kind == "locale" and row.source == "proc":
                probe_local_health(row)
            elif row.kind == "locale":
                probe_local_health(row)
        # Riattacca i dettagli gia' letti ([I]) alle righe nuove.
        for row in rows:
            det = self.details.get(row.key)
            if det:
                row.details = det
                m = det.get("metrics") or {}
                if m.get("predicted_tokens_seconds"):
                    row.tok_s = round(m["predicted_tokens_seconds"], 2)
        if rows:
            vram = nvidia_vram()
            for row in rows:
                if row.kind == "locale":
                    row.vram_used_mib = vram["used"] or None
                    row.vram_total_mib = vram["total"] or None
                else:
                    g = (self.gpu_remote.get(row.host) or {}).get("gpus") or []
                    if g:
                        row.vram_used_mib = sum(x.get("used") or 0 for x in g) or None
                        row.vram_total_mib = sum(x.get("total") or 0 for x in g) or None
        with self._lock:
            self._rows = rows
            # Se il server selezionato e' sparito, la selezione si azzera:
            # meglio nessuna selezione che una selezione su un server morto.
            keys = {r.key for r in rows}
            if self.selected_key and self.selected_key not in keys:
                self.selected_key = None
        return list(rows)

    # Il ggml-rpc-server serve UN client alla volta (backlog 1): la sonda TCP
    # si fa al massimo ogni RPC_PROBE_EVERY_S secondi e MAI mentre un
    # llama-server locale e' collegato a lui (sarebbe "in uso", non offline).
    RPC_PROBE_EVERY_S = 20.0

    def _rpc_user(self, ep: RemoteEndpoint, local_rows: List[ServerRow]) -> Optional[ServerRow]:
        target = f"{ep.host}:{ep.port}"
        for r in local_rows:
            if r.kind != "locale" or not r.pid:
                continue
            rpc = _flag_value(_read_cmdline(r.pid), "--rpc") or ""
            if target in rpc.split(","):
                return r
        return None

    def _probe_remotes(self, local_rows: Optional[List[ServerRow]] = None) -> List[ServerRow]:
        out: List[ServerRow] = []
        threads = []
        results: Dict[str, ServerRow] = {}
        if not hasattr(self, "_rpc_cache"):
            self._rpc_cache: Dict[str, tuple] = {}

        def work(ep: RemoteEndpoint) -> None:
            results[f"{ep.host}:{ep.port}"] = RemoteProbe.probe(ep)

        now = time.time()
        probed_now: set = set()
        for ep in self.remotes:
            if not ep.auto:
                continue
            k = f"{ep.host}:{ep.port}"
            if ep.kind == "rpc":
                user = self._rpc_user(ep, local_rows or [])
                if user is not None:
                    row = ServerRow(key=f"remote:{k}", name=f"{ep.name}:{ep.port}", kind="remoto",
                                    host=ep.host, port=ep.port, ssh_target=ep.ssh, source="remote",
                                    # ONLINE dedotto dal llama-server locale collegato (/health
                                    # gia' verificato su di lui): Miky non viene sondata a parte.
                                    online=True, health_checked=True, state="ONLINE")
                    row.flags = [f"ggml-rpc backend IN USO (llama-server PID {user.pid} :{user.port})"]
                    row.error_msg = ("in uso dal llama-server locale (/health verificato su di lui): "
                                     "non sondato a parte, il rpc-server serve un client alla volta")
                    results[k] = row
                    continue
                cached = self._rpc_cache.get(k)
                if cached and now - cached[0] < self.RPC_PROBE_EVERY_S:
                    results[k] = cached[1]
                    continue
            probed_now.add(f"{ep.host}:{ep.port}")
            t = threading.Thread(target=work, args=(ep,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=HTTP_TIMEOUT_S * 2 + 1.0)
        for ep in self.remotes:
            k = f"{ep.host}:{ep.port}"
            r = results.get(k)
            if r is not None:
                out.append(r)
                # cache SOLO quando c'e' stata una sonda vera (prima la cache
                # veniva aggiornata anche quando il risultato era GIA' dalla
                # cache: non scadeva mai e Miky restava "ONLINE" per sempre).
                if ep.kind == "rpc" and r.health_checked and k in probed_now:
                    self._rpc_cache[k] = (now, r)
        return out

    # -- endpoint manuali ------------------------------------------------
    def reload_remotes(self) -> None:
        if self.store is not None:
            self.store.load()
            self.remotes = self.store.endpoints()

    def add_endpoint(self, text: str) -> Dict[str, Any]:
        if self.store is None:
            self.store = EndpointStore()
        res = self.store.add(text)
        if res.get("success"):
            self.remotes = self.store.endpoints()
        return res

    def remove_endpoint(self, host: str, port: int) -> Dict[str, Any]:
        if self.store is None:
            return {"success": False, "error": "nessuno store endpoint"}
        res = self.store.remove(host, port)
        if res.get("success"):
            self.remotes = self.store.endpoints()
            with self._lock:
                self._rows = [r for r in self._rows
                              if not (r.kind == "remoto" and r.host == host and r.port == int(port))]
        return res

    def endpoint_for(self, row: ServerRow) -> Optional[RemoteEndpoint]:
        for ep in self.remotes:
            if ep.host == row.host and ep.port == row.port:
                return ep
        return None

    def rows(self) -> List[ServerRow]:
        """L'ultimo snapshot pubblicato. Non tocca la rete: e' un semplice peek."""
        with self._lock:
            return list(self._rows)

    def selected(self) -> Optional[ServerRow]:
        with self._lock:
            if not self.selected_key:
                return None
            for r in self._rows:
                if r.key == self.selected_key:
                    return r
        return None

    def select(self, key: Optional[str]) -> Optional[ServerRow]:
        """Seleziona per chiave; None o una chiave inesistente deseleziona."""
        with self._lock:
            self.selected_key = key
            for r in self._rows:
                if r.key == key:
                    return r
        return None

    def move_selection(self, delta: int) -> Optional[ServerRow]:
        """Sposta il cursore di `delta` posizioni e seleziona la nuova riga."""
        with self._lock:
            if not self._rows:
                return None
            keys = [r.key for r in self._rows]
            cur = keys.index(self.selected_key) if self.selected_key in keys else -1
            nxt = max(0, min(len(keys) - 1, cur + delta))
            self.selected_key = keys[nxt]
            for r in self._rows:
                if r.key == keys[nxt]:
                    return r
        return None

    # -- azioni ----------------------------------------------------------
    def stop_selected(self) -> Dict[str, Any]:
        row = self.selected()
        if row is None:
            return {"success": False, "error": "nessun server selezionato"}
        if row.kind != "locale":
            # Per "rete" non si puo' fermare nulla: il processo e' su una
            # macchina diversa e non c'e' azione senza SSH esplicito
            # (vedi miky_stop_remote). Si dice perche', non si fallisce in
            # silenzio come se fosse un bug.
            return {
                "success": False,
                "error": f"{row.name} e' REMOTO: si ferma dalla macchina stessa "
                         f"(non via SSHkill).",
            }
        if not row.pid:
            return {"success": False,
                    "error": f"{row.name}: adottato via porta, PID non leggibile "
                             f"(processo di un altro utente/container): fermalo da li'"}
        res = stop_server_pid(row.pid)
        if res.get("success"):
            self.refresh_now()
        return res

    def browser_url(self, row: Optional[ServerRow] = None) -> str:
        row = row or self.selected()
        if row is None:
            return ""
        return row.url


def _ep(d: Dict[str, Any]) -> RemoteEndpoint:
    return RemoteEndpoint(
        name=str(d.get("name") or d["host"]), host=str(d["host"]), port=int(d["port"]),
        kind=str(d.get("type", "llama-server")), ssh=str(d.get("ssh", "")),
        ssh_key=str(d.get("ssh_key", "")),
    )


# -------------------------------------------------------------------------
# Flag arbitrari: dal testo dell'utente alla riga di comando
# -------------------------------------------------------------------------
def split_extra_flags(text: str) -> List[str]:
    """
    Divide una riga di flag scritta a mano in una lista di token.

    Usa shlex con posix=True cosi' si possono usare le virgolette come in
    una shell. Una riga vuota dà una lista vuota. Non viene mai eseguito
    niente: e' solo parsing.
    """
    if not text or not text.strip():
        return []
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        # Virgolette non chiuse: si ripiega su whitespace, che e' meglio
        # che rifiutare i flag dell'utente.
        return text.split()


def read_running_flags(pid: int) -> List[str]:
    """
    I flag con cui il server e' davvero in esecuzione, da /proc/<pid>/cmdline.

    Serve perche' l'utente vuole vedere il comando attivo, copiarlo e
    modificarlo. Non richiede che la TUI abbia lanciato quel processo.
    """
    return _read_cmdline(pid)

# -------------------------------------------------------------------------
# Memoria di un llama-server: RAM del processo, VRAM per device, dove sta la KV
# -------------------------------------------------------------------------
_BUF_RE = re.compile(r"(RPC\d+\[[^\]]*\]|CUDA\d+|CPU|Vulkan\d+)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB")


def process_ram_mib(pid: int) -> float:
    """RAM residente (RSS) del processo in MiB, da /proc. 0 se non leggibile."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def memory_breakdown(log_path: str, pid: int = 0) -> Dict[str, Any]:
    """
    Ripartizione REALE della memoria di un server, letta dal suo log di avvio
    (llama-server stampa i buffer: model / KV / compute per device).

    Ritorna {"devices": {nome: {"model": MiB, "kv": MiB, "compute": MiB}},
             "ram_mib": RSS del processo, "kv_mib": totale KV, "kv_where": {...}}.
    Se il log non ha i buffer (serve -lv 4) lo dice in "note".
    """
    out: Dict[str, Any] = {"devices": {}, "ram_mib": process_ram_mib(pid) if pid else 0.0,
                           "kv_mib": 0.0, "kv_where": {}, "note": ""}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            text = re.sub(r"\x1b\[[0-9;]*m", "", fh.read())
    except OSError:
        out["note"] = "log non leggibile"
        return out
    for dev, kind, mib in _BUF_RE.findall(text):
        d = out["devices"].setdefault(dev, {"model": 0.0, "kv": 0.0, "compute": 0.0})
        d[kind.lower()] += float(mib)
    out["kv_mib"] = sum(d["kv"] for d in out["devices"].values())
    for dev, d in out["devices"].items():
        if d["kv"] > 0:
            out["kv_where"][dev] = d["kv"]
    if not out["devices"]:
        out["note"] = "buffer non nel log: avvia con -lv 4 per vedere la ripartizione"
    return out


# -------------------------------------------------------------------------
# Scan passivo della rete locale: cerca llama-server/ggml-rpc negli IP vicini
# -------------------------------------------------------------------------
_LAN_SCAN_PORTS = (50052, 8080, 8087, 8088, 9292, 9205, 18434)


def _local_subnets() -> List[str]:
    """IP /24 delle interfacce non-loopback (es. '192.168.1.0'). Senza
    dipendenze esterne: parsa /proc/net/route o `ip -4 addr`."""
    out: List[str] = []
    try:
        with open("/proc/net/fib_trie", "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        for line in txt.splitlines():
            m = re.search(r"(\d+\.\d+\.\d+)\.\d+/24", line)
            if m:
                sub = m.group(1) + ".0"
                if sub not in out and not sub.startswith("127."):
                    out.append(sub)
    except OSError:
        pass
    return out[:3]


def scan_lan(hosts: Optional[List[str]] = None, ports: tuple = _LAN_SCAN_PORTS,
             timeout: float = 0.25, max_workers: int = 32) -> List[Dict[str, Any]]:
    """
    Prova (ip, porta) in parallelo. Ritorna i soli endpoint che accettano la
    connessione TCP. NON fa HTTP; solo lo scheletro per capire chi risponde.
    Pensata per girare in un thread daemon, non blocca la UI.
    """
    if hosts is None:
        hosts = []
        for sub in _local_subnets():
            parts = sub.split(".")
            base = ".".join(parts[:3]) + "."
            for i in range(1, 255):
                hosts.append(base + str(i))
    found: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def probe(host: str, port: int) -> None:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                with lock:
                    found.append({"host": host, "port": port, "kind": ("rpc" if port == 50052 else "llama-server")})
        except (OSError, socket.timeout):
            return

    pairs = [(h, p) for h in hosts for p in ports]
    thr: List[threading.Thread] = []
    for h, p in pairs:
        while sum(1 for t in thr if t.is_alive()) >= max_workers:
            time.sleep(0.02)
        t = threading.Thread(target=probe, args=(h, p), daemon=True)
        t.start()
        thr.append(t)
    for t in thr:
        t.join(timeout=timeout + 0.5)
    return found
