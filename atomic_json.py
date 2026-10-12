"""
atomic_json.py - lettura/scrittura sicura dei file JSON di preferenze.

* save_json(): scrive su un file temporaneo nella stessa cartella, fsync,
  poi os.replace(). Se la TUI viene chiusa a meta' (SIGTERM, timeout,
  crash) il file resta quello vecchio, mai troncato.
* load_json(): se il file esiste ma e' illeggibile (JSON corrotto), NON lo
  si azzera in silenzio: lo si rinomina in <file>.corrupt-<timestamp> e si
  restituisce un messaggio d'errore da mostrare all'utente. Cosi' il primo
  salvataggio successivo non cancella i dati senza che nessuno se ne accorga.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional, Tuple

_LOCK = threading.Lock()


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    """Scrittura atomica (tmp + fsync + os.replace). Solleva OSError/TypeError."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_json(path: str, default: Any = None) -> Tuple[Any, Optional[str]]:
    """
    Ritorna (dati, errore). File assente -> (default, None).
    File corrotto -> rinominato in .corrupt-<ts>, (default, "messaggio").
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return default, None
    except OSError as exc:
        return default, f"{os.path.basename(path)} non leggibile: {exc}"
    except ValueError as exc:
        bak = f"{path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
        with _LOCK:
            try:
                os.replace(path, bak)
                where = f"salvato in {bak}"
            except OSError:
                where = "non rinominato"
        return default, (f"{os.path.basename(path)} corrotto ({exc.__class__.__name__}): "
                         f"{where}, ripartito da vuoto")


def update_json(path: str, key: str, value: Any) -> None:
    """Read-modify-write di una chiave, serializzato da un lock (piu' thread)."""
    with _LOCK:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if not isinstance(d, dict):
                d = {}
        except (OSError, ValueError):
            d = {}
        d[key] = value
        save_json(path, d, indent=1)
