"""
vram_budget.py - margine VRAM per OGNI GPU (locale e remote via RPC).

Cosa fa llama.cpp davvero (verificato su `llama-server --help`, b11330)
-----------------------------------------------------------------------
* ``-fit, --fit [on|off]`` (default ON): regola gli argomenti NON impostati
  (``-ngl`` che di default e' ``auto``, ``-c`` che di default e' 0, la
  ripartizione fra device) per entrare nella memoria dei device.
* ``-fitt, --fit-target MiB0,MiB1,...``: margine da lasciare LIBERO su ogni
  device, in MiB. Un valore solo vale per tutti. **Default: 1024 MiB** -> se
  non lo si imposta si perde 1 GB per scheda.
* Se ``-ngl`` o ``-c`` sono passati esplicitamente, --fit non li tocca: il
  margine influenza solo cio' che e' rimasto "auto" (es. la ripartizione dei
  layer fra CUDA0 e RPC0 quando ``--tensor-split`` non e' dato).

Ordine dei device
-----------------
La lista di --fit-target segue l'ordine dei device del modello. In
llama.cpp i device RPC vengono messi IN TESTA ("add RPC servers at the
front of the list to minimize network transfers"), poi le GPU locali. Qui
si genera quindi: [GPU degli host RPC, nell'ordine di --rpc] + [GPU
locali]. L'ordine effettivo si legge nel log del server (righe "using
device ..."): `devices_from_log()` lo estrae e la UI avvisa se non combacia.

Configurazione (testo libero, salvata in ~/.ai_tui_vram.json)
-------------------------------------------------------------
    locale : "600"        -> CUDA0 lascia 600 MiB al sistema
    locale : "600,0"      -> due GPU locali
    remoti : {"192.168.1.100:50052": "0,0"}   -> Miky: 2 GPU al 100%
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

CONFIG_PATH = os.path.expanduser("~/.ai_tui_vram.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    # RTX 3050: 600 MiB lasciati al desktop/compositor.
    "local": "512",     # 3050: 512 MiB di margine
    # Miky (RTX 5050 Laptop + seconda GPU): tutto al modello.
    "remotes": {"192.168.1.100:50052": "512"},   # Miky: 512 MiB di margine
    # Aggiungere --fit-target anche ai preset F1-F5 (in coda, come i flag
    # manuali: la logica dei preset resta identica).
    "apply_to_presets": False,
}


def parse_margins(text: str) -> Tuple[List[int], str]:
    """
    "600, 0" -> ([600, 0], "").  Accetta MiB interi, "100%" / "max" = 0,
    suffissi "M"/"MiB"/"G"/"GiB". Ritorna (lista, errore).
    """
    out: List[int] = []
    if text is None:
        return out, ""
    for raw in re.split(r"[,\s;]+", str(text).strip()):
        if not raw:
            continue
        t = raw.lower()
        if t in ("100%", "max", "all", "tutto"):
            out.append(0)
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|mb|mib|g|gb|gib)?", t)
        if not m:
            return [], f"valore non valido: '{raw}' (usa MiB, es. 600 oppure 0.5G)"
        val = float(m.group(1))
        if m.group(2) and m.group(2).startswith("g"):
            val *= 1024
        if val < 0 or val > 65536:
            return [], f"margine fuori scala: {raw}"
        out.append(int(round(val)))
    return out, ""


class VramBudget:
    """Margini VRAM per device, persistiti su file."""

    def __init__(self, path: str = CONFIG_PATH) -> None:
        self.path = path
        self.cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
        self.load()

    # -- persistenza ------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self.cfg.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
        except (OSError, ValueError):
            pass

    def save(self) -> bool:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.cfg, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False

    # -- modifica (testo dell'utente) -------------------------------------
    def set_local(self, text: str) -> str:
        vals, err = parse_margins(text)
        if err:
            return err
        self.cfg["local"] = ",".join(str(v) for v in vals) if vals else ""
        self.save()
        return ""

    def set_remote(self, endpoint: str, text: str) -> str:
        endpoint = endpoint.strip()
        if not re.fullmatch(r"[\w.\-]+:\d{1,5}", endpoint):
            return f"endpoint RPC non valido: '{endpoint}' (formato host:porta)"
        if not str(text).strip():
            self.cfg["remotes"].pop(endpoint, None)
            self.save()
            return ""
        vals, err = parse_margins(text)
        if err:
            return err
        self.cfg["remotes"][endpoint] = ",".join(str(v) for v in vals)
        self.save()
        return ""

    def toggle_presets(self) -> bool:
        self.cfg["apply_to_presets"] = not bool(self.cfg.get("apply_to_presets"))
        self.save()
        return self.cfg["apply_to_presets"]

    # -- calcolo ----------------------------------------------------------
    def local_margins(self) -> List[int]:
        return parse_margins(self.cfg.get("local", ""))[0]

    def remote_margins(self, endpoint: str) -> List[int]:
        return parse_margins((self.cfg.get("remotes") or {}).get(endpoint, ""))[0]

    def fit_target(self, rpc: str = "", local_devices: int = 1) -> Tuple[List[int], List[str]]:
        """
        Lista per --fit-target e le etichette dei device, nell'ordine
        di llama.cpp: prima le GPU degli host RPC (ordine di --rpc), poi le
        locali. Un host RPC senza margini configurati prende 1 device con
        il default di llama.cpp (1024), cosi' la lista resta allineata.
        """
        vals: List[int] = []
        labels: List[str] = []
        rpc_idx = 0
        for ep in [e.strip() for e in (rpc or "").split(",") if e.strip()]:
            margins = self.remote_margins(ep) or [1024]
            for m in margins:
                vals.append(m)
                labels.append(f"RPC{rpc_idx}[{ep}]")
                rpc_idx += 1
        loc = self.local_margins() or [1024]
        loc = (loc + [loc[-1]] * local_devices)[:max(local_devices, len(loc))]
        for i, m in enumerate(loc):
            vals.append(m)
            labels.append(f"CUDA{i}")
        return vals, labels

    def fit_flags(self, rpc: str = "", local_devices: int = 1) -> List[str]:
        """["--fit", "on", "--fit-target", "0,600"]."""
        vals, _ = self.fit_target(rpc, local_devices)
        if not vals:
            return []
        return ["--fit", "on", "--fit-target", ",".join(str(v) for v in vals)]

    def describe(self, rpc: str = "", local_devices: int = 1) -> str:
        vals, labels = self.fit_target(rpc, local_devices)
        return "  ".join(f"{l}={'100%' if v == 0 else f'-{v} MiB'}" for l, v in zip(labels, vals))


_DEV_RE = re.compile(r"using device (\S+)")


def devices_from_log(lines: List[str]) -> List[str]:
    """Device effettivi letti dal log di llama-server ("using device CUDA0 ...")."""
    out: List[str] = []
    for ln in lines:
        m = _DEV_RE.search(ln)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def has_device_restriction(argv: List[str]) -> Optional[str]:
    """
    Se argv contiene --device senza un device RPC ma con --rpc, ritorna un
    avviso: con --device esplicito llama.cpp usa SOLO i device elencati,
    quindi il device RPC potrebbe restare escluso.
    """
    dev = None
    for i, a in enumerate(argv):
        if a in ("--device", "-dev") and i + 1 < len(argv):
            dev = argv[i + 1]
        elif a.startswith("--device="):
            dev = a.split("=", 1)[1]
    has_rpc = any(a == "--rpc" or a.startswith("--rpc=") for a in argv)
    if dev and has_rpc and "RPC" not in dev.upper():
        return (f"--device {dev} insieme a --rpc: llama.cpp usa solo i device "
                f"elencati, il device RPC rischia di restare inutilizzato "
                f"(usa ad es. --device {dev},RPC0 oppure togli --device)")
    return None
