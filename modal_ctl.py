"""
modal_ctl.py - integrazione Modal nella TUI (Tab 14).

Parla con il CLI 'modal' installato in ~/.local/bin, in thread daemon per non
bloccare la UI. Usa output JSON (--json) per massima velocita' e affidabilita'.

Scenario operativo:
  - account modal:  lexander3692
  - app principale: modal-hermes-ornith  (deployed, serve)
  - endpoint:       https://lexander3692--modal-hermes-ornith-serve.modal.run/v1
  - volume:         modal-ornith-volume  (modelli GGUF persistenti: 27.6 GB / 50 GB)
  - modelli:        Ornith-1.5-35B-A3B (Mudler) & Tiel-Coder-35B-A3B (IsValorum)
  - scaledown:      600s (10 min inattivita' -> container spento, 0$/h)
  - server file:    /home/user/scratch/modal-tui-manager/
                    hermes_ornith_server.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MODAL_BIN = shutil.which("modal") or os.path.expanduser("~/.local/bin/modal")
MODAL_DIR = "/home/user/scratch/modal-tui-manager"
SERVER_FILE = os.path.join(MODAL_DIR, "hermes_ornith_server.py")

APP_NAME = "modal-hermes-ornith"
ENDPOINT = "https://lexander3692--modal-hermes-ornith-serve.modal.run"

# Costo orario pubblico delle GPU su Modal (USD/ora)
GPU_COSTS: Dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "A100-40GB": 2.10,
    "A100-80GB": 3.40,
    "H100": 4.50,
}

# Modelli pre-caricati nel volume modal-ornith-volume
KNOWN_MODELS: List[Dict[str, Any]] = [
    {
        "key": "ornith",
        "file": "Ornith-1.5-35B-A3B-APEX-MTP-I-Mini.gguf",
        "label": "Ornith 1.5 35B A3B APEX-MTP",
        "author": "mudler",
        "size_gb": 13.38,
        "desc": "MoE 35B (3B attivi) · Velocita' ~75-85 tok/s su L4 · 80k CTX",
    },
    {
        "key": "tiel-coder",
        "file": "Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf",
        "label": "Tiel-Coder 35B A3B APEX-I MiniPlus",
        "author": "IsValorum",
        "size_gb": 14.18,
        "desc": "MoE 35B Coding/Agentic · 80k CTX · Ottimizzato Python/Rust",
    },
]


def _run(argv: List[str], timeout: float = 20.0) -> Tuple[int, str, str]:
    """Lancia modal con PATH esteso. Ritorna (rc, stdout, stderr)."""
    env = dict(os.environ)
    env["PATH"] = f"{os.path.dirname(MODAL_BIN) or ''}:{env.get('PATH', '')}"
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout dopo {timeout} s"
    except OSError as exc:
        return 1, "", f"modal non eseguibile: {exc}"


def modal_available() -> bool:
    return os.path.isfile(MODAL_BIN)


def list_apps() -> List[Dict[str, Any]]:
    """`modal app list --json`: lista app attive o deployate."""
    rc, out, err = _run([MODAL_BIN, "app", "list", "--json"], timeout=15.0)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            rows: List[Dict[str, Any]] = []
            for a in data:
                desc = a.get("description") or a.get("name") or ""
                created = (a.get("created_at") or "")[:16]
                stopped = (a.get("stopped_at") or "")[:16]
                info = created if not stopped else f"stop {stopped}"
                rows.append({
                    "id": a.get("app_id") or a.get("id", ""),
                    "name": desc,
                    "state": a.get("state", ""),
                    "tasks": int(a.get("tasks") or 0),
                    "created_at": created,
                    "stopped_at": stopped,
                    "info": info,
                })
            return rows
        except Exception:
            pass

    # Fallback su tabella testuale
    rc2, out2, err2 = _run([MODAL_BIN, "app", "list"], timeout=15.0)
    if rc2 != 0:
        return [{"error": err.strip() or out.strip() or err2.strip()}]
    rows = []
    for line in out2.splitlines():
        m = re.match(r"\s*(ap-[A-Za-z0-9]+)\s+(.+?)\s{2,}(\w+)\s+(\d+)?\s+(.*)", line)
        if m:
            rows.append({
                "id": m.group(1),
                "name": m.group(2).strip(),
                "state": m.group(3),
                "tasks": int(m.group(4) or 0),
                "info": m.group(5).strip(),
            })
    return rows


def list_containers() -> List[Dict[str, Any]]:
    """`modal container list --json`: restituisce SOLO i container attualmente accesi."""
    rc, out, err = _run([MODAL_BIN, "container", "list", "--json"], timeout=10.0)
    if rc == 0 and out.strip():
        try:
            return json.loads(out)
        except Exception:
            pass
    return []


def active_app() -> Optional[Dict[str, Any]]:
    for a in list_apps():
        name = a.get("name") or a.get("description") or ""
        if name == APP_NAME and a.get("state") == "deployed":
            return a
    return None


def list_volumes() -> List[Dict[str, Any]]:
    rc, out, err = _run([MODAL_BIN, "volume", "list", "--json"], timeout=15.0)
    if rc == 0 and out.strip():
        try:
            return json.loads(out)
        except Exception:
            pass
    rc, out, err = _run([MODAL_BIN, "volume", "list"], timeout=15.0)
    if rc != 0:
        return [{"error": err.strip() or out.strip()}]
    rows: List[Dict[str, Any]] = []
    for line in out.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_\-]+)\s+(\d{4}-\d\d-\d\d.*)", line)
        if m and "name" not in m.group(1).lower():
            rows.append({"name": m.group(1), "created": m.group(2).strip()})
    return rows


def volume_contents(name: str = "modal-ornith-volume") -> Tuple[List[Dict[str, Any]], str]:
    """`modal volume ls <name> --json`: elenco file nel volume con dimensioni esatte."""
    rc, out, err = _run([MODAL_BIN, "volume", "ls", name, "--json"], timeout=25.0)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            items: List[Dict[str, Any]] = []
            for f in data:
                fname = f.get("filename") or ""
                size_b = int(f.get("size") or 0)
                mib = size_b / (1024.0 * 1024.0)
                items.append({
                    "name": fname,
                    "type": f.get("type", "file"),
                    "bytes": size_b,
                    "mib": mib,
                    "created": f.get("created_modified", ""),
                })
            return items, ""
        except Exception as exc:
            return [], f"errore parsing volume: {exc}"
    return [], (err.strip() or out.strip())[:200]


def get_active_model(volume_name: str = "modal-ornith-volume") -> str:
    """Legge il modello attivo da active_model.json nel volume persistente."""
    rc, out, _ = _run([MODAL_BIN, "volume", "get", volume_name, "active_model.json", "-"], timeout=15.0)
    if rc == 0 and out:
        m = re.search(r'\{.*?"active_model"\s*:\s*"([^"]+)".*?\}', out)
        if m:
            return m.group(1).lower().strip()
    return "ornith"


def set_active_model(model_key: str) -> Dict[str, Any]:
    """Imposta il modello attivo nel volume ed effettua il reload."""
    choice = model_key.lower().strip()
    valid = [m["key"] for m in KNOWN_MODELS]
    if choice not in valid:
        return {"ok": False, "error": f"Modello non valido: {model_key}. Opzioni: {valid}"}
    rc, out, err = _run([MODAL_BIN, "run", f"{SERVER_FILE}::set_active_model", "--model-name", choice], timeout=60.0)
    if rc != 0:
        return {"ok": False, "error": (err.strip() or out.strip())[-150:]}
    # Ferma i container attivi per forzare il reload al prossimo prompt
    for c in list_containers():
        if "modal-hermes-ornith" in c.get("app_name", ""):
            stop_container(c.get("container_id", ""))
    return {"ok": True, "model": choice, "stdout": out.strip()}


def ping_endpoint(timeout: float = 10.0) -> Dict[str, Any]:
    """GET /v1/models sull'URL pubblico. Ritorna {ok, latency_ms, model, error}."""
    url = f"{ENDPOINT}/v1/models"
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        lat = (time.time() - t0) * 1000.0
        model = ""
        if isinstance(data, dict):
            arr = data.get("data") or data.get("models") or []
            if arr and isinstance(arr, list):
                m0 = arr[0]
                model = (m0.get("id") if isinstance(m0, dict) else str(m0)) or ""
        return {"ok": True, "latency_ms": round(lat, 1), "model": model}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}", "latency_ms": round((time.time() - t0) * 1000.0, 1)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:160], "latency_ms": round((time.time() - t0) * 1000.0, 1)}


def chat_once(prompt: str, timeout: float = 180.0) -> Dict[str, Any]:
    """Un prompt di test al server Modal (/v1/chat/completions)."""
    url = f"{ENDPOINT}/v1/chat/completions"
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": False,
    }).encode("utf-8")
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = d["choices"][0]["message"]["content"]
        t = d.get("timings") or {}
        return {
            "ok": True,
            "text": text,
            "latency_ms": round((time.time() - t0) * 1000.0, 1),
            "gen_tok_s": t.get("predicted_per_second"),
            "gen_n": t.get("predicted_n"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200], "latency_ms": round((time.time() - t0) * 1000.0, 1)}


def deploy() -> Dict[str, Any]:
    if not os.path.isfile(SERVER_FILE):
        return {"ok": False, "error": f"file del server non trovato: {SERVER_FILE}"}
    rc, out, err = _run([MODAL_BIN, "deploy", SERVER_FILE], timeout=600.0)
    return {"ok": rc == 0, "stdout": out[-800:], "stderr": err[-400:]}


def stop_container(container_id: str) -> Dict[str, Any]:
    """Arresta un singolo container attivo con -y (nessuna conferma bloccante)."""
    rc, out, err = _run([MODAL_BIN, "container", "stop", "-y", container_id], timeout=30.0)
    return {"ok": rc == 0, "stdout": out[-300:], "stderr": err[-200:]}


def stop_app(app_name: str = APP_NAME) -> Dict[str, Any]:
    """Ferma l'intera applicazione con -y."""
    rc, out, err = _run([MODAL_BIN, "app", "stop", "-y", app_name], timeout=60.0)
    return {"ok": rc == 0, "stdout": out[-400:], "stderr": err[-200:]}


def workspace_spend(timeout_s: float = 10.0) -> Dict[str, Any]:
    """
    Spesa REALE del ciclo di fatturazione in corso, letta da Modal.

    Prima il pannello usava un valore fisso (spent_usd=0.05) e mostrava quindi
    un credito inventato invece di quello consumato davvero. L'SDK espone
    Workspace.billing.summary(): 'metered_cost' e' il costo lordo del ciclo,
    mentre la voce 'Credits' dentro 'adjustments' e' il credito effettivamente
    bruciato (negativa, quindi si prende il valore assoluto).
    """
    try:
        import modal
        summary = modal.Workspace.from_context().billing.summary()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
    def _f(x) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    adj = {str(k): _f(v) for k, v in (getattr(summary, "adjustments", None) or {}).items()}
    brk = {str(k): _f(v) for k, v in (getattr(summary, "metered_cost_breakdown", None) or {}).items()}
    return {
        "ok": True,
        "metered_cost": round(_f(getattr(summary, "metered_cost", 0)), 4),
        "credits_used": round(abs(adj.get("Credits", 0.0)), 4),
        "breakdown": {k: round(v, 4) for k, v in brk.items() if abs(v) > 1e-9},
        "adjustments": {k: round(v, 4) for k, v in adj.items() if abs(v) > 1e-9},
        "start": str(getattr(summary, "start", ""))[:10],
        "end": str(getattr(summary, "end", ""))[:10],
    }


def hours_remaining(spent_usd: float = 0.05, monthly_credit: float = 30.0) -> Dict[str, float]:
    budget = max(0.0, monthly_credit - spent_usd)
    return {gpu: round(budget / cost, 2) for gpu, cost in GPU_COSTS.items()}


@dataclass
class ModalStatus:
    checked_at: float = 0.0
    cli_ok: bool = False
    account: str = ""
    active_model: str = "ornith"
    apps: List[Dict[str, Any]] = field(default_factory=list)
    active_containers: List[Dict[str, Any]] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    volume_files: List[Dict[str, Any]] = field(default_factory=list)
    endpoint_ping: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def refresh(self, do_ping: bool = False) -> None:
        self.checked_at = time.time()
        self.cli_ok = modal_available()
        if not self.cli_ok:
            self.error = f"modal CLI non trovato in {MODAL_BIN}"
            return
        rc, out, _ = _run([MODAL_BIN, "profile", "current"], timeout=10.0)
        if rc == 0 and out.strip():
            self.account = out.strip().splitlines()[-1].strip()
        try:
            self.apps = list_apps()
            self.active_containers = list_containers()
            self.volumes = list_volumes()
            self.volume_files, verr = volume_contents()
            if verr and not self.volume_files:
                self.error = verr
            self.active_model = get_active_model()

            # REGOLA ANTI-SPRECO:
            # NON facciamo una chiamata HTTP a /v1/models se non ci sono container attivi!
            # Altrimenti risveglieremmo il container dal sonno azzerando il timer dei 10 min.
            # Eseguiamo il ping solo se ci sono container attivi OPPURE se l'utente ha premuto [P].
            if do_ping or self.active_containers:
                self.endpoint_ping = ping_endpoint()
            else:
                self.endpoint_ping = {
                    "ok": False,
                    "status": "sleep",
                    "info": "💤 Container in Sleep (0.00 $/h)",
                }
        except Exception as exc:
            self.error = str(exc)[:200]


class ModalCtl:
    """Controller Modal integrato nella TUI."""

    def __init__(self) -> None:
        self.status = ModalStatus()
        self.cursor: int = 0
        self.last_action: str = ""
        self.busy: bool = False
        self.monthly_credit: float = 30.0
        self.spent_usd: float = 0.05        # sostituito dal dato reale al refresh
        self.spend: Dict[str, Any] = {}     # esito di workspace_spend()

    def refresh_async(self, do_ping: bool = False) -> None:
        if self.busy:
            return
        self.busy = True

        def _w():
            try:
                self.status.refresh(do_ping=do_ping)
                # Spesa vera del ciclo: se l'API non risponde si tiene il
                # valore precedente invece di mostrare uno zero fittizio.
                sp = workspace_spend()
                self.spend = sp
                if sp.get("ok"):
                    self.spent_usd = sp.get("credits_used") or sp.get("metered_cost") or self.spent_usd
            finally:
                self.busy = False

        threading.Thread(target=_w, daemon=True, name="modal-refresh").start()

    def cycle_model_async(self) -> None:
        """Alterna tra i modelli nel volume persistente (Ornith <-> Tiel-Coder)."""
        if self.busy:
            return
        self.busy = True
        curr = (self.status.active_model or "ornith").lower()
        next_m = "tiel-coder" if "ornith" in curr else "ornith"
        self.last_action = f"🔄 Cambio modello a {next_m}..."

        def _w():
            try:
                r = set_active_model(next_m)
                if r.get("ok"):
                    self.last_action = f"✓ Modello impostato su: {next_m} (riavvio al prossimo prompt)"
                    self.status.active_model = next_m
                else:
                    self.last_action = f"✗ Cambio fallito: {r.get('error', '')[:100]}"
            finally:
                self.busy = False
                self.status.refresh()

        threading.Thread(target=_w, daemon=True, name="modal-switch-model").start()

    def deploy_async(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.last_action = "⚡ deploy in corso (avvio container e GPU L4)..."

        def _w():
            try:
                r = deploy()
                self.last_action = (
                    "✓ deploy completato con successo"
                    if r["ok"]
                    else f"✗ deploy fallito: {(r['stderr'] or r['stdout'])[-120:]}"
                )
            finally:
                self.busy = False
                self.status.refresh()

        threading.Thread(target=_w, daemon=True, name="modal-deploy").start()

    def stop_async(self, app_name: str = APP_NAME) -> None:
        """Ferma istantaneamente tutti i container attivi e l'app."""
        if self.busy:
            return
        self.busy = True
        self.last_action = f"⏹ stop container e app {app_name}..."

        def _w():
            try:
                # Prima killa i container attivi per interrompere subito la fatturazione
                cnts = list_containers()
                stopped_n = 0
                for c in cnts:
                    cid = c.get("container_id", "")
                    if cid:
                        stop_container(cid)
                        stopped_n += 1
                r = stop_app(app_name)
                self.last_action = f"✓ stop ok (terminati {stopped_n} container)"
            except Exception as e:
                self.last_action = f"✗ errore stop: {e}"
            finally:
                self.busy = False
                self.status.refresh()

        threading.Thread(target=_w, daemon=True, name="modal-stop").start()

    def ping_async(self) -> None:
        """Ping esplicito richiesto dall'utente con [P]."""
        if self.busy:
            return
        self.busy = True
        self.last_action = "📡 ping endpoint /v1/models..."

        def _w():
            try:
                r = ping_endpoint()
                self.last_action = (
                    f"✓ /v1/models · {r['latency_ms']:.0f} ms · {r.get('model') or 'OK'}"
                    if r.get("ok")
                    else f"✗ ping fallito: {r.get('error')} ({r['latency_ms']:.0f} ms)"
                )
                self.status.endpoint_ping = r
            finally:
                self.busy = False

        threading.Thread(target=_w, daemon=True, name="modal-ping").start()

    def chat_async(self, prompt: str) -> None:
        if self.busy or not prompt.strip():
            return
        self.busy = True
        self.last_action = f"💬 invio prompt ({len(prompt)} car): {prompt[:30]}..."

        def _w():
            try:
                r = chat_once(prompt)
                if r.get("ok"):
                    tps = r.get("gen_tok_s") or 0.0
                    resp_text = (r.get("text") or "").strip().replace("\n", " ")
                    self.last_action = (
                        f"✓ ({r['latency_ms']:.0f} ms, {tps:.1f} tok/s): {resp_text[:120]}"
                    )
                else:
                    self.last_action = f"✗ chat: {r.get('error')}"
            finally:
                self.busy = False

        threading.Thread(target=_w, daemon=True, name="modal-chat").start()
