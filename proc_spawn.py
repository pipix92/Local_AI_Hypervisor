"""
proc_spawn.py - avvio "staccato" dei processi pesanti (llama-server & co).

Perche' esiste
--------------
`subprocess.Popen(..., start_new_session=True)` crea una nuova SESSIONE
(il processo non muore con Ctrl+C della TUI e non riceve SIGHUP alla
chiusura del terminale) ma NON cambia il CGROUP: il figlio resta nello
stesso cgroup di chi lo lancia. Se quel cgroup e' quello di un'app (per
esempio l'agente di coding o il terminale) e il llama-server cresce in RAM,
l'OOM-killer del kernel puo' scegliere l'app intera, non il server. E' gia'
successo due volte.

La soluzione su Linux con systemd e' `systemd-run --user --scope`: il
processo viene messo in uno scope suo (`ai-tui-<nome>-<ts>.scope`) sotto
`app.slice`, quindi la sua memoria e' contabilizzata a parte. `--scope`
esegue il comando IN PLACE (exec), quindi:

* il PID restituito da Popen e' quello vero del llama-server;
* stdout/stderr ereditano i file descriptor passati (log su file).

Verificato su questa macchina con un processo leggero (`sleep`): stesso
PID, cgroup `.../app.slice/ai-tui-probe-<ts>.scope`.

Se systemd-run non c'e' (o fallisce), si ripiega su start_new_session=True
e lo si DICHIARA nel risultato (`isolated=False`), senza fingere.

Su Windows (Miky) l'equivalente e' CREATE_NEW_PROCESS_GROUP |
DETACHED_PROCESS: il processo non e' figlio della console che lo lancia.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

# I log dei server avviati dalla TUI. Su /tmp sarebbero persi al riavvio e
# la root e' piccola: si usa la home, che e' sullo stesso disco della TUI.
LOG_DIR = os.path.expanduser("~/.cache/ai_tui/server-logs")

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _systemd_run() -> Optional[str]:
    """Percorso di systemd-run se c'e' un manager utente raggiungibile."""
    if os.name == "nt":
        return None
    exe = shutil.which("systemd-run")
    if not exe:
        return None
    # Senza XDG_RUNTIME_DIR il manager --user non e' raggiungibile.
    if not os.environ.get("XDG_RUNTIME_DIR") and not os.path.isdir(f"/run/user/{os.getuid()}"):
        return None
    return exe


def log_path_for(name: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"{_SAFE.sub('_', name)}-{time.strftime('%Y%m%d-%H%M%S')}.log")


def spawn_detached(argv: List[str], name: str = "llama",
                   log_path: Optional[str] = None,
                   env: Optional[Dict[str, str]] = None,
                   cwd: Optional[str] = None,
                   memory_max: Optional[str] = None) -> Dict[str, Any]:
    """
    Avvia `argv` staccato dal processo/cgroup chiamante.

    Ritorna {"success", "proc", "pid", "log_path", "isolated", "unit", "cmd"}.
    stdout+stderr vanno in `log_path` (creato se None): un PIPE verso la
    TUI non va bene, perche' quando la TUI esce il server prenderebbe
    SIGPIPE alla prima riga di log e morirebbe.

    memory_max (es. "40G") imposta un tetto di memoria SOLO allo scope del
    server: se sfora, il kernel uccide lui e non la TUI.
    """
    if not argv:
        return {"success": False, "error": "argv vuoto"}
    log_path = log_path or log_path_for(name)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logf = open(log_path, "ab", buffering=0)
    except OSError as exc:
        return {"success": False, "error": f"log non apribile: {exc}"}

    full = list(argv)
    unit = ""
    isolated = False
    sd = _systemd_run()
    if sd:
        unit = f"ai-tui-{_SAFE.sub('-', name)[:40]}-{int(time.time() * 1000) % 10_000_000}"
        prefix = [sd, "--user", "--scope", "--quiet", "--collect", f"--unit={unit}"]
        if memory_max:
            prefix.append(f"--property=MemoryMax={memory_max}")
        full = prefix + full
        isolated = True

    kwargs: Dict[str, Any] = dict(stdout=logf, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, cwd=cwd,
                                  env=env if env is not None else None)
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                   | getattr(subprocess, "DETACHED_PROCESS", 0)
                                   | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(full, **kwargs)
    except OSError as exc:
        if isolated:
            # systemd-run presente ma non funzionante: si riprova senza,
            # dicendolo nel risultato.
            try:
                proc = subprocess.Popen(list(argv), **kwargs)
                isolated, unit = False, ""
            except OSError as exc2:
                logf.close()
                return {"success": False, "error": f"avvio fallito: {exc2}"}
        else:
            logf.close()
            return {"success": False, "error": f"avvio fallito: {exc}"}
    finally:
        pass
    # Il figlio ha il suo fd; il nostro si puo' chiudere.
    try:
        logf.close()
    except OSError:
        pass
    return {"success": True, "proc": proc, "pid": proc.pid, "log_path": log_path,
            "isolated": isolated, "unit": unit, "cmd": " ".join(argv)}


def cgroup_of(pid: int) -> str:
    """Il cgroup di un processo (per mostrarlo/verificarlo nella UI)."""
    try:
        with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as fh:
            return fh.read().strip().split("::", 1)[-1]
    except OSError:
        return ""


def tail_file(path: str, start_pos: int = 0, max_bytes: int = 65536,
              final: bool = False) -> tuple[List[str], int]:
    """Righe nuove da `path` a partire da start_pos. Ritorna (righe, nuova_pos).
    final=True (processo uscito): restituisce anche l'ultima riga senza \\n."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end < start_pos:          # file troncato/ruotato
                start_pos = 0
            fh.seek(max(start_pos, end - max_bytes) if end - start_pos > max_bytes else start_pos)
            data = fh.read()
            pos = fh.tell()
    except OSError:
        return [], start_pos
    # L'ultima riga senza "\n" e' ancora in scrittura: la si lascia per il
    # giro dopo (pos torna al suo inizio), altrimenti arriverebbe in due pezzi.
    nl = data.rfind(b"\n")
    if final:
        pass
    elif nl < 0:
        if len(data) < max_bytes:
            return [], pos - len(data)
    else:
        pos -= len(data) - (nl + 1)
        data = data[:nl + 1]
    text = data.decode("utf-8", errors="replace")
    return [l for l in text.splitlines() if l.strip()], pos
