"""
peers.py - le TUI ai_tui si vedono tra loro sulla rete locale.

Come funziona
-------------
Ogni TUI manda ogni 4 secondi un piccolo messaggio JSON in broadcast UDP
(porta PEER_PORT, 9390) con: nome macchina, IP, porta e stato dei suoi
llama-server. Le altre TUI in ascolto lo ricevono e tengono una lista dei
"pari" (peers) vivi: un pari sparisce se non si fa sentire per 15 secondi.

Cosa serve a chi usa la TUI
---------------------------
* vedere quali altre TUI sono aperte in rete e quali server gestiscono;
* scegliere una porta libera anche rispetto ai server degli altri (es. se
  su 9292 c'e' gia' il server di Miky o di un'altra TUI, si passa a 9205).

Non controlla i server degli altri: li mostra soltanto. Il messaggio non
contiene token, chiavi o percorsi di file.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

PEER_PORT = 9390
ANNOUNCE_EVERY_S = 4.0
PEER_TTL_S = 15.0
MAX_MSG = 4096


def _local_ip() -> str:
    """IP della LAN usato per uscire (senza mandare nulla in rete)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.1.100", 9))      # route verso Miky
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "?"


class PeerNet:
    """Annuncia questa TUI e ascolta quelle degli altri (thread daemon)."""

    def __init__(self, servers_fn: Callable[[], List[Dict[str, Any]]], version: str = "2.6") -> None:
        self.me = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.servers_fn = servers_fn            # -> [{"port":..,"model":..,"state":..}]
        self.version = version
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.error: str = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx: Optional[socket.socket] = None
        self._tx: Optional[socket.socket] = None

    # -- avvio / arresto ----------------------------------------------------
    def start(self) -> None:
        try:
            rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            rx.bind(("", PEER_PORT))
            rx.settimeout(1.0)
            self._rx = rx
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            tx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._tx = tx
        except OSError as exc:
            self.error = f"rete TUI non attiva ({exc}): la porta {PEER_PORT} e' occupata?"
            return
        threading.Thread(target=self._rx_loop, daemon=True, name="peers-rx").start()
        threading.Thread(target=self._tx_loop, daemon=True, name="peers-tx").start()

    def stop(self) -> None:
        self._stop.set()

    # -- invio ----------------------------------------------------------------
    def _message(self) -> bytes:
        try:
            servers = [{"port": int(s.get("port") or 0), "model": str(s.get("model") or "")[:60],
                        "state": str(s.get("state") or "")}
                       for s in (self.servers_fn() or [])]
        except Exception:
            servers = []
        msg = {"app": "ai_tui", "id": self.me, "host": socket.gethostname(), "ip": _local_ip(),
               "ver": self.version, "servers": servers[:12], "ts": time.time()}
        return json.dumps(msg).encode("utf-8")[:MAX_MSG]

    def _tx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._tx is not None:
                    self._tx.sendto(self._message(), ("255.255.255.255", PEER_PORT))
            except OSError as exc:
                self.error = f"invio annuncio: {exc}"
            self._stop.wait(ANNOUNCE_EVERY_S)

    # -- ricezione --------------------------------------------------------------
    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._rx.recvfrom(MAX_MSG)
            except socket.timeout:
                self._expire()
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            if msg.get("app") != "ai_tui" or msg.get("id") == self.me:
                continue
            msg["seen"] = time.time()
            msg["addr"] = addr[0]
            with self._lock:
                self.peers[msg["id"]] = msg
            self._expire()

    def _expire(self) -> None:
        now = time.time()
        with self._lock:
            for k in [k for k, v in self.peers.items() if now - v.get("seen", 0) > PEER_TTL_S]:
                del self.peers[k]

    # -- lettura (per la UI) --------------------------------------------------
    def live_peers(self) -> List[Dict[str, Any]]:
        self._expire()
        with self._lock:
            return sorted(self.peers.values(), key=lambda p: p.get("host", ""))

    def ports_in_use(self) -> set:
        """Porte dei server annunciati dalle altre TUI."""
        out = set()
        for p in self.live_peers():
            for s in p.get("servers", []):
                if s.get("port"):
                    out.add(int(s["port"]))
        return out


def porta_libera_in_rete(preferite: tuple, occupate: set) -> int:
    """Prima porta tra quelle preferite che non e' occupata ne' da un server
    locale ne' da un server annunciato da una TUI in rete. Altrimenti la prima
    libera a partire dall'ultima preferita."""
    import socket as _s

    def libera(p: int) -> bool:
        if p in occupate:
            return False
        with _s.socket(_s.AF_INET6, _s.SOCK_STREAM) as x:
            try:
                x.bind(("::1", p))
                return True
            except OSError:
                return False

    for p in preferite:
        if libera(p):
            return p
    base = preferite[-1] if preferite else 9292
    for p in range(base, base + 200):
        if libera(p):
            return p
    return preferite[0] if preferite else 9292
