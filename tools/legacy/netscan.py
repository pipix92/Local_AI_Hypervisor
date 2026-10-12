"""
netscan.py - scansione della RETE LOCALE alla ricerca di llama-server attivi.

Problema che risolve
--------------------
`llama_mgr` conosce UN endpoint (quello configurato) e `servers.py` conosce i
remoti dichiarati a mano. Ma l'utente ha server sparsi sulle macchine della
casa/rete e non li vede da nessuna parte: un llama-server lanciato a mano
sulla porta 8080 di un altro PC e' invisibile finche' nessuno lo cerca.

Qui si scandisce la sottorete e si restituiscono i server trovati, con:
* lo stato vero (interrogato su /health, non dedotto);
* modello e contesto (da /props) e stato degli slot (da /slots);
* la latenza misurata, che serve a distinguere una macchina lenta da una
  porta chiusa.

Perche' una scansione parallela e non seriale
---------------------------------------------
254 host x 6 porte in seriale, con timeout di 1 s, sono ~25 minuti: la
scarsa banda dell'hotspot rende la cosa inutilizzabile. Qui:
* timeout brevissimi per porta (PORT_TIMEOUT_S = 0.35 s);
* ThreadPoolExecutor con worker = min(64, ...): le connessioni sono I/O
  bloccante, quindi i thread fanno esattamente quello che serve;
* un BUDGET di tempo complessivo per rete (NET_BUDGET_S): se il budget
  scade, si restituisce quello che e' stato trovato finora invece di
  bloccare la UI all'infinito.

Cosa NON fa
-----------
Non tocca la configurazione di rete, non apre porte, non invia nulla a
host che non rispondono: solo connessioni TCP e due GET in sola lettura
(/health, /props, /slots) sugli host che rispondono. Nessun SSH viene
tentato durante la scansione: su host per cui non si ha la chiave
fallirebbe in fretta ma inutilmente, quindi l'SSH e' un'azione separata e
voluta (vedi servers.start_remote_server, che usa BatchMode).

I host trovati vengono salvati in una cache JSON: la seconda volta non si
riscandisce tutto, si riprova solo quello che era scaduto. Il TTL e' corto
(15 minuti) perche' un server puo' essere spento in qualsiasi momento.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Timeout per singola porta: breve di proposito. Un host che non risponde
# costa 0.35 s, non 3 s.
PORT_TIMEOUT_S = 0.35

# Budget di tempo per una rete intera. Superato, si restituisce quello che
# e' stato trovato: la UI non aspetta mai la scansione.
NET_BUDGET_S = 12.0

# Timeout per le richieste HTTP (/health, /props, /slots) verso un host che
# ha gia' dimostrato di avere la porta aperta.
HTTP_TIMEOUT_S = 1.2

# Le porte tipiche di llama.cpp. 8080 e' quella di default, 18434 quella di
# LM Studio, 1234 quella del progetto originale, le altre sono in uso nei
# preset misurati in questa sessione.
SCAN_PORTS: List[int] = [8080, 8081, 8090, 8099, 18434, 1234]

# Cache degli host trovati.
CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "ai_tui_netscan.json")
CACHE_TTL_S = 900.0        # 15 minuti

# Reti da scandire. La prima e' la rete di Miky (cavo diretto, enp3s0), la
# seconda e' quella del wifi di casa, raggiunta attraverso il gateway.
DEFAULT_NETWORKS: List[str] = ["192.168.1.0/24", "192.168.1.0/24"]

# Interfacce da NON usare: il loopback non contiene server, tailscale e'
# una VPN che non contiene nulla di nostro.
SKIP_IFACES = {"lo", "tailscale0", "docker0", "veth"}


def _which(name: str) -> Optional[str]:
    for d in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep):
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


# -------------------------------------------------------------------------
# Rotte: quale interfaccia usa davvero ogni rete
# -------------------------------------------------------------------------
def route_for(ip: str) -> Tuple[str, str]:
    """
    Interfaccia (dev) e gateway usati per raggiungere `ip`.

    Si chiede al kernel con "ip route get", che e' l'unica risposta
    affidabile: la stessa rete puo' essere raggiunta via cavo su una
    macchina e via hotspot su un'altra, e la tabella di routing cambia.
    Restituisce ("", "") se l'host non e' instradabile.
    """
    ipbin = _which("ip")
    if not ipbin:
        return ("", "")
    try:
        proc = subprocess.run([ipbin, "route", "get", ip],
                              capture_output=True, text=True, timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        return ("", "")
    if proc.returncode != 0:
        return ("", "")
    dev = ""
    via = ""
    parts = proc.stdout.split()
    for i, tok in enumerate(parts):
        if tok == "dev" and i + 1 < len(parts):
            dev = parts[i + 1]
        elif tok == "via" and i + 1 < len(parts):
            via = parts[i + 1]
    if not dev or any(dev.startswith(p) for p in SKIP_IFACES):
        return ("", via)
    return (dev, via)


def local_networks() -> List[str]:
    """Le sottorete direttamente configurate sulle interfacce utili."""
    nets: List[str] = []
    for d in os.listdir("/sys/class/net"):
        if d in SKIP_IFACES:
            continue
        try:
            with open(f"/sys/class/net/{d}/address", "r") as fh:
                fh.read()
        except OSError:
            continue
        # Le sottorete stanno nell'output di "ip addr", non in un file.
        try:
            out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", d],
                                 capture_output=True, text=True, timeout=3.0).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in out.splitlines():
            # "2: enp3s0    inet 192.168.1.50/24 brd ..." -> la CIDR e' il
            # terzo campo. Prendere il secondo dava il NOME del device.
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "inet":
                nets.append(parts[3])
    return nets


# -------------------------------------------------------------------------
# Cache
# -------------------------------------------------------------------------
def load_cache() -> Dict[str, Any]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_cache(data: Dict[str, Any]) -> str:
    """Salva la cache. Un errore in scrittura non e' un errore: si ignora."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
    except OSError:
        pass
    return CACHE_PATH


def cache_entries(ttl_s: float = CACHE_TTL_S) -> List[Dict[str, Any]]:
    """Gli host in cache che non sono ancora scaduti."""
    data = load_cache()
    now = time.time()
    out: List[Dict[str, Any]] = []
    for host, rec in (data.get("hosts") or {}).items():
        if not isinstance(rec, dict):
            continue
        if now - float(rec.get("seen_at") or 0) > ttl_s:
            continue
        rec = dict(rec)
        rec["host"] = host
        rec["from_cache"] = True
        out.append(rec)
    return out


def remember_host(host: str, port: int, extra: Optional[Dict[str, Any]] = None) -> None:
    """Ricorda un host trovato, per non riscansionarlo subito."""
    data = load_cache()
    hosts = data.setdefault("hosts", {})
    rec = {"port": port, "seen_at": time.time()}
    if extra:
        rec.update(extra)
    hosts[host] = rec
    save_cache(data)


def forget_host(host: str) -> None:
    data = load_cache()
    hosts = data.get("hosts") or {}
    if host in hosts:
        del hosts[host]
        data["hosts"] = hosts
        save_cache(data)


# -------------------------------------------------------------------------
# Sonda: TCP prima, poi HTTP
# -------------------------------------------------------------------------
def port_open(host: str, port: int, timeout_s: float = PORT_TIMEOUT_S) -> Tuple[bool, str]:
    """(porta aperta?, dettaglio). Dettaglio: vuoto se aperta."""
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return (True, f"{(time.time() - t0) * 1000:.0f} ms")
    except socket.timeout:
        return (False, "timeout")
    except OSError as exc:
        return (False, str(exc)[:40])


def http_json(url: str, timeout_s: float = HTTP_TIMEOUT_S) -> Tuple[bool, Any, int]:
    """
    GET JSON. Restituisce (ok, dati|errore, codice_http).

    Il codice HTTP serve: llama.cpp risponde 503 su /health e su /props
    FINCHE' il modello non e' pronto. Quel 503 non e' un errore di rete,
    e' "sta caricando": senza il codice non si distingue da una connessione
    fallita e il server verrebbe scartato dalla scansione.
    """
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ai_tui-netscan/1.0",
                           "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            return True, json.loads(body), resp.status
        except json.JSONDecodeError:
            return True, body[:200], resp.status
    except urllib.error.HTTPError as exc:
        # 503 = "Loading model": si conserva il codice, non si getta.
        return False, f"HTTP {exc.code}", exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, str(exc)[:80], 0


def identify_llama(host: str, port: int) -> Optional[Dict[str, Any]]:
    """
    Dice se su host:port c'e' un llama-server, e con quali parametri.

    /health da solo dice "qualcosa risponde": per distinguishinge un
    llama-server da qualsiasi altro servizio si guarda /props, che e' la
    rotta specifica di llama.cpp (model_path, build_info, n_ctx).
    Se /props non c'e' ma /health risponde, si registra lo stesso come
    "http non identificato": utile comunque, ma etichettato per bene.
    """
    base = f"http://{host}:{port}"
    ok_h, health, code_h = http_json(f"{base}/health", HTTP_TIMEOUT_S)
    # 503 su /health significa "sono un llama-server, sto caricando il
    # modello": e' un ritrovamento valido, non uno scarto. 0 = nessuna
    # risposta HTTP, ma la porta TCP era aperta: si registra lo stesso,
    # etichettato per bene come non identificato.
    if code_h == 0 and not ok_h:
        return None
    ok_p, props, code_p = http_json(f"{base}/props", HTTP_TIMEOUT_S)
    rec: Dict[str, Any] = {
        "host": host, "port": port,
        "identified": bool(ok_p and isinstance(props, dict)),
        "health_code": code_h,
        "loading": code_h == 503,
        "health_raw": str(health)[:80] if isinstance(health, str) else "",
    }
    if ok_p and isinstance(props, dict):
        rec["model"] = str(props.get("model_path") or "")
        rec["build"] = str(props.get("build_info") or "").split("-")[0]
        dgs = props.get("default_generation_settings") or {}
        if isinstance(dgs, dict) and dgs.get("n_ctx"):
            try:
                rec["n_ctx"] = int(dgs["n_ctx"])
            except (TypeError, ValueError):
                rec["n_ctx"] = 0
        rec["sleeping"] = bool(props.get("is_sleeping"))
    # /slots: stato dei worker. Assente in molte build, quindi facoltativo.
    ok_s, slots, _code_s = http_json(f"{base}/slots", HTTP_TIMEOUT_S)
    if ok_s and isinstance(slots, list):
        rec["slots"] = len(slots)
        busy = sum(1 for s in slots
                   if isinstance(s, dict) and (s.get("is_processing") or s.get("is_processing")))
        rec["slots_busy"] = busy
    return rec


# -------------------------------------------------------------------------
# Scansione
# -------------------------------------------------------------------------
def scan_network(cidr: str, ports: Optional[List[int]] = None,
                 budget_s: float = NET_BUDGET_S,
                 max_workers: int = 64,
                 on_progress=None) -> List[Dict[str, Any]]:
    """
    Scansiona una sottorete alla ricerca di llama-server.

    Restituisce i record di TUTTO quello che ha risposto su una porta nota,
    identificato oppure no. Il budget e' un tetto: se scade, si restituisce
    quello che e' stato trovato invece di continuare all'infinito.
    """
    ports = list(ports or SCAN_PORTS)
    try:
        net = ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return []
    if net.version != 4:
        return []
    # Verifica la rotta PRIMA di sparare 1500 connessioni: una rete non
    # instradata va detto subito, non fatta scadere per timeout.
    probe_ip = str(net.network_address + 1)
    dev, _via = route_for(probe_ip)
    if not dev:
        return []

    targets = [(str(ip), p) for ip in net.hosts() for p in ports]
    if on_progress:
        on_progress(f"scansione {cidr} su {dev}: {len(targets)} combinazioni")

    found: Dict[Tuple[str, int], Dict[str, Any]] = {}
    deadline = time.time() + budget_s
    workers = max(8, min(max_workers, len(targets)))

    def one(job: Tuple[str, int]) -> Optional[Tuple[str, int, str]]:
        host, port = job
        ok, detail = port_open(host, port)
        return (host, port, detail) if ok else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, t): t for t in targets}
        for fut in as_completed(futures, timeout=None):
            if time.time() > deadline:
                # Budget esaurito: si lascia perdere il resto della coda e
                # si lavora quello che si ha. La UI non deve aspettare.
                for f in futures:
                    f.cancel()
                break
            try:
                res = fut.result()
            except Exception:
                continue
            if res:
                found[(res[0], res[1])] = {"rtt": res[2]}

    # Identificazione dei porti aperti: sono pochi, si puo' fare in
    # sequenza senza penalizzare (e con un timeout un po' piu' generoso,
    # perche' sappiamo gia' che la porta c'e').
    out: List[Dict[str, Any]] = []
    for (host, port), meta in sorted(found.items()):
        rec = identify_llama(host, port)
        if rec is None:
            rec = {"host": host, "port": port, "identified": False,
                   "health_raw": "porta aperta, nessun /health"}
        rec["rtt"] = meta.get("rtt", "")
        rec["iface"] = dev
        rec["seen_at"] = time.time()
        out.append(rec)
        remember_host(host, port, {"model": rec.get("model", ""),
                                   "n_ctx": rec.get("n_ctx", 0)})
    return out


def scan_all(networks: Optional[List[str]] = None,
             budget_s: float = NET_BUDGET_S,
             use_cache: bool = True,
             on_progress=None) -> List[Dict[str, Any]]:
    """
    Scansione di tutte le reti richieste, mettendo dentro anche la cache.

    La cache non e' una scorciatoia: e' quello che permette a [F8] di
    ricaricare l'elenco in un secondo invece di rifare 1500 connessioni a
    ogni giro. I record scaduti vengono ripescati con una scansione.
    """
    networks = list(networks or DEFAULT_NETWORKS)
    out: List[Dict[str, Any]] = []
    seen: set = set()

    if use_cache:
        for rec in cache_entries():
            key = (rec["host"], int(rec.get("port") or 0))
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
        if on_progress and out:
            on_progress(f"{len(out)} host dalla cache")

    for cidr in networks:
        if on_progress:
            on_progress(f"scansione {cidr}...")
        try:
            rows = scan_network(cidr, budget_s=budget_s, on_progress=on_progress)
        except Exception as exc:      # una rete rotta non ferma le altre
            if on_progress:
                on_progress(f"{cidr}: {exc}")
            continue
        for rec in rows:
            key = (rec["host"], int(rec["port"]))
            if key in seen:
                # Riscansione: la riga fresca aggiorna quella in cache.
                for i, old in enumerate(out):
                    if (old["host"], int(old.get("port") or 0)) == key:
                        out[i] = rec
                        break
                continue
            seen.add(key)
            out.append(rec)
    out.sort(key=lambda r: (r["host"], int(r.get("port") or 0)))
    return out


def probe_known(hosts: Optional[List[Tuple[str, int]]] = None) -> List[Dict[str, Any]]:
    """Riprova solo gli endpoint gia' noti: niente scansione."""
    hosts = hosts or [(r["host"], int(r["port"])) for r in cache_entries()]
    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(4, min(16, len(hosts) or 1))) as pool:
        futs = {pool.submit(identify_llama, h, p): (h, p) for h, p in hosts}
        for fut in as_completed(futs):
            h, p = futs[fut]
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec:
                rec["seen_at"] = time.time()
                out.append(rec)
            else:
                forget_host(h)
    return out


if __name__ == "__main__":
    import sys
    nets = sys.argv[1:] or DEFAULT_NETWORKS
    rows = scan_all(nets, on_progress=print)
    print(f"\n{len(rows)} host trovati:")
    for r in rows:
        print(f"  {r['host']}:{r['port']}  ident={r.get('identified')} "
              f"modello={os.path.basename(r.get('model') or '-')} "
              f"ctx={r.get('n_ctx', 0)} slots={r.get('slots', '-')} "
              f"rtt={r.get('rtt', '')}")