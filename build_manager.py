"""
build_manager.py - gestione di PIU' build llama.cpp (e dei loro rpc-server).

Cosa fa (tab "Build" della TUI)
-------------------------------
* trova le build su disco: catalogo noto + cartelle aggiunte dall'utente
  (salvate in ~/.ai_tui_builds.json) + scansione delle radici note;
* per ogni build legge DAL BINARIO, senza stime:
    - versione/commit (--version)
    - device visibili (--list-devices): CUDA0..., oppure "(none)" = solo CPU
    - presenza di ggml-rpc-server e libggml-rpc.so (supporto RPC)
    - valori ammessi di --spec-type, --split-mode, --cache-type-k
    - flag presenti: --fit, --load-mode, -nkvo, --spec-sched (fork DSPARK)...
* DOCUMENTAZIONE PER BUILD in ~/docs/llamacpp/builds/<chiave>/:
    - help.txt         = `llama-server --help` di QUELLA build (sempre esatto)
    - rpc-help.txt     = `ggml-rpc-server --help`
    - server-README.md / rpc-README.md scaricati da GitHub al TAG della
      build (ggml-org/llama.cpp, tag bNNNNN) solo per le build ufficiali;
      i fork non hanno un tag upstream: per loro fa fede help.txt.
* avvio/arresto di ggml-rpc-server LOCALI (staccati con proc_spawn) e
  elenco di quelli gia' in esecuzione (/proc).

Nessuna sonda carica un modello: --help/--version/--list-devices escono
subito e non allocano VRAM.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import proc_spawn as _PS
import atomic_json as _AJ

USER_BUILDS_PATH = os.path.expanduser("~/.ai_tui_builds.json")
DOCS_ROOT = os.path.expanduser("~/docs/llamacpp/builds")
PROBE_TIMEOUT_S = 10.0

# Radici dove cercare build (una cartella = una build se contiene llama-server).
SCAN_ROOTS = [
    # Tutte le build scaricate dalla TUI stanno qui (disco condiviso con
    # Miky): un'unica cartella BUILD dentro LLAMASERVER, non sparse.
    "/mnt/lm_models/DEBIAN_FOLDER/llamacpp",
    "/mnt/lm_models/llamacpp",
    "/mnt/lm_models/DEBIAN-Files",
    os.path.expanduser("~/.unsloth/llama.cpp/build/bin"),
    os.path.expanduser("~/.lmstudio/extensions/backends"),
]

# Note verificate a mano su questa macchina (2026-10-05) con --help e
# --list-devices: cosa usare per cosa.
KNOWN_NOTES: Dict[str, str] = {
    "llama-b11435": "ULTIMA (v0.6.0-dev, build 11435). CUDA 13.4 + RPC. cublas via symlink → cublas/cuda-13.4/. "
                    "Usa al posto di b11429 per nuove sessioni.",
    "llama-b11429": "CUDA 13.4 + RPC. cublas via symlink → cublas/cuda-13.4/. Stabile.",
    "llama-b11330": "STABILE. CUDA 13.4 + RPC. Tutti i --spec-type (anche draft-dspark nativo), "
                    "-sm tensor (sperimentale), --load-mode, --fit. Build di riferimento per F1-F6.",
    "llama-b10373": "Fork speed.llama.cpp (DSPARK con scheduler: --spec-sched, --spec-sts...). "
                    "NESSUN backend CUDA: --list-devices = (none) -> in locale gira SOLO su CPU. "
                    "Ha senso solo con --rpc verso GPU remote (Miky).",
    "llama-cpp-compelte-tools": "Copia di b11330 (stesso commit c061df198) con tutti i tool.",
    "unsloth": "b9987 (Unsloth). CUDA + RPC, NO draft-dspark, NO --load-mode. Solo per compatibilita'.",
    "lmstudio": "Backend interno di LM Studio: gestito da LM Studio, non lanciarlo a mano.",
}


@dataclass
class BuildInfo:
    key: str
    dir: str
    server: str = ""
    rpc_server: str = ""
    version: str = ""
    build_num: int = 0
    commit: str = ""
    devices: List[str] = field(default_factory=list)
    gpu: bool = False
    rpc: bool = False
    spec_types: List[str] = field(default_factory=list)
    split_modes: List[str] = field(default_factory=list)
    cache_types: List[str] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    probed: bool = False
    probe_error: str = ""
    note: str = ""
    user_added: bool = False
    docs: Dict[str, str] = field(default_factory=dict)   # nome -> percorso

    @property
    def official(self) -> bool:
        """Build upstream (ha un tag bNNNNN su ggml-org/llama.cpp)."""
        return (self.build_num >= 1000 and "speed" not in self.dir and "dspark" not in self.dir
                and "lmstudio" not in self.dir)

    def summary(self) -> str:
        bits = [f"b{self.build_num}" if self.build_num else "?"]
        bits.append("GPU" if self.gpu else "solo CPU")
        if self.rpc:
            bits.append("RPC")
        if "dspark" in self.spec_types:
            bits.append("DSPARK")
        if self.flags.get("--spec-sched"):
            bits.append("sched")
        return " · ".join(bits)


def _run(argv: List[str], timeout: float = PROBE_TIMEOUT_S) -> str:
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout,
                           env=dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(argv[0])
                                    + ":" + os.environ.get("LD_LIBRARY_PATH", "")))
        return p.stdout or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"__ERR__ {exc}"


def _choices(help_text: str, flag_re: str) -> List[str]:
    m = re.search(flag_re, help_text)
    if not m:
        return []
    return [v.strip() for v in re.split(r"[,|]", m.group(1)) if v.strip()]


def probe_build(b: BuildInfo) -> BuildInfo:
    if not b.server or not os.path.isfile(b.server):
        b.probe_error = "llama-server assente"
        return b
    ver = _run([b.server, "--version"])
    m = re.search(r"build (\d+), commit (\w+)", ver) or re.search(r"version: (\d{3,}) \((\w+)\)", ver)
    if m:
        b.build_num, b.commit = int(m.group(1)), m.group(2)
    b.version = next((l.strip() for l in ver.splitlines() if "version" in l), ver.strip()[:80])
    devs = _run([b.server, "--list-devices"])
    b.devices = [l.strip().split(":")[0] for l in devs.splitlines()
                 if re.match(r"\s+\w+\d*:", l)]
    b.gpu = any(d.startswith(("CUDA", "Vulkan", "ROCm", "Metal", "SYCL")) for d in b.devices)
    h = _run([b.server, "--help"])
    if h.startswith("__ERR__"):
        b.probe_error = h[8:]
        return b
    b.spec_types = [v[len("draft-"):] if v.startswith("draft-") else v
                    for v in _choices(h, r"--spec-type ([\w,\-]+)")]
    b.split_modes = _choices(h, r"--split-mode \{([\w,]+)\}")
    m = re.search(r"--cache-type-k TYPE.*?allowed values: ([\w, ]+)", h, re.S)
    b.cache_types = [v.strip() for v in m.group(1).split(",")] if m else []
    # nuovo dict assegnato alla fine: la Tab 13 puo' scorrere b.flags mentre
    # la sonda gira in un thread (niente "dictionary changed size")
    flags: Dict[str, bool] = dict(b.flags)
    for f in ("--fit", "--fit-target", "--load-mode", "--no-kv-offload", "--rpc",
              "--spec-sched", "--spec-draft-dspark-n-cand", "--metrics", "--slots",
              "--mlock", "--no-mmap", "--kv-unified", "--swa-full"):
        flags[f] = (f + " ") in h or (f + ",") in h or (f + "\n") in h or (f + "=") in h
    b.flags = flags
    rpc_exe = os.path.join(b.dir, "ggml-rpc-server")
    if not os.path.isfile(rpc_exe):
        rpc_exe = os.path.join(b.dir, "rpc-server")
    b.rpc_server = rpc_exe if os.path.isfile(rpc_exe) else ""
    b.rpc = bool(b.flags.get("--rpc")) and (bool(b.rpc_server)
                                            or os.path.isfile(os.path.join(b.dir, "libggml-rpc.so")))
    b.probed = True
    return b


# Indice dei risultati della sonda: all'avvio la lista build compare subito e
# si risondano solo le build nuove o il cui llama-server e' cambiato.
INDEX_PATH = os.path.expanduser("~/.cache/ai_tui/builds_index.json")
_PROBED_FIELDS = ("version", "build_num", "commit", "devices", "gpu", "rpc", "rpc_server",
                  "spec_types", "split_modes", "cache_types", "flags")


def _signature(b: BuildInfo) -> Optional[List[int]]:
    try:
        st = os.stat(b.server)
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _key_for(d: str) -> str:
    base = os.path.basename(os.path.normpath(d))
    if "lmstudio" in d:
        m = re.search(r"linux-x86_64-(.*?)-?avx2", base)
        return "lmstudio-" + ((m.group(1) or "cpu") if m else base)
    if ".unsloth" in d:
        return "unsloth"
    if base in ("bin", "build"):
        parent = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(d))))
        base = parent.strip(".") or base
    return base


def _note_for(d: str, key: str) -> str:
    for k, v in KNOWN_NOTES.items():
        if k in d or k == key:
            return v
    return ""


class BuildManager:
    def __init__(self, probe: bool = False) -> None:
        self.builds: Dict[str, BuildInfo] = {}
        self.user_dirs: List[str] = []
        self.active_key: str = ""
        self.lock = threading.Lock()
        # impostato quando probe_all() ha finito (vedi probe_all_async)
        self.probe_done = threading.Event()
        self.status: str = ""
        self.docs_busy: bool = False
        self._load_user()
        self.discover()
        if self._apply_index():
            self.probe_done.set()
        if probe:
            self.probe_all()

    # -- indice sonde -----------------------------------------------------
    def _apply_index(self) -> bool:
        """Riempie le build dall'indice. True se tutte risultano gia' sondate."""
        idx, _err = _AJ.load_json(INDEX_PATH, {})
        entries = idx.get("builds", {}) if isinstance(idx, dict) else {}
        for b in list(self.builds.values()):
            e = entries.get(b.dir)
            if b.probed or not isinstance(e, dict):
                continue
            if e.get("sig") != _signature(b):
                continue
            for f in _PROBED_FIELDS:
                if f in e:
                    setattr(b, f, e[f])
            b.probed = True
        return all(b.probed for b in self.builds.values())

    def _save_index(self) -> None:
        entries = {}
        for b in list(self.builds.values()):
            sig = _signature(b)
            if b.probed and not b.probe_error and sig:
                entries[b.dir] = dict({f: getattr(b, f) for f in _PROBED_FIELDS}, sig=sig)
        try:
            _AJ.save_json(INDEX_PATH, {"builds": entries})
        except OSError as exc:
            self.status = f"salvataggio indice build fallito: {exc}"

    # -- persistenza ------------------------------------------------------
    def _load_user(self) -> None:
        d, err = _AJ.load_json(USER_BUILDS_PATH, {})
        if err:
            # file corrotto: rinominato in .corrupt-<ts>, l'avviso resta in status
            self.status = "⚠ " + err
        if not isinstance(d, dict):
            d = {}
        self.user_dirs = [x for x in d.get("dirs", []) if isinstance(x, str)]
        self.active_key = str(d.get("active", ""))

    def _save_user(self) -> None:
        try:
            # scrittura atomica: un'interruzione non lascia il file troncato
            _AJ.save_json(USER_BUILDS_PATH, {"dirs": self.user_dirs, "active": self.active_key})
        except OSError as exc:
            self.status = f"salvataggio {USER_BUILDS_PATH} fallito: {exc}"


    # -- scoperta ---------------------------------------------------------
    def _candidate_dirs(self) -> List[str]:
        out: List[str] = []

        def add(d: str) -> None:
            d = os.path.normpath(os.path.expanduser(d))
            if os.path.isfile(os.path.join(d, "llama-server")) and d not in out:
                out.append(d)

        for root in SCAN_ROOTS:
            if not os.path.isdir(root):
                continue
            add(root)
            try:
                for n in sorted(os.listdir(root)):
                    sub = os.path.join(root, n)
                    if os.path.isdir(sub) and not n.startswith("dl-"):   # download in corso
                        add(sub)
                        try:
                            for n2 in sorted(os.listdir(sub)):
                                if os.path.isdir(os.path.join(sub, n2)):
                                    add(os.path.join(sub, n2))
                        except OSError:
                            pass
            except OSError:
                pass
        for d in self.user_dirs:
            add(d)
        return out

    def discover(self) -> None:
        found: Dict[str, BuildInfo] = {}
        for d in self._candidate_dirs():
            key = _key_for(d)
            if key in found:
                key = f"{key}@{os.path.basename(os.path.dirname(d))}"
            old = self.builds.get(key)
            b = old if (old and old.dir == d) else BuildInfo(key=key, dir=d,
                                                            server=os.path.join(d, "llama-server"))
            b.note = _note_for(d, key)
            b.user_added = d in [os.path.normpath(os.path.expanduser(x)) for x in self.user_dirs]
            b.docs = self._existing_docs(key)
            found[key] = b
        with self.lock:
            self.builds = found
        if self.active_key not in found or not self.active_key:
            ord_b = self.ordered()
            self.active_key = ord_b[0].key if ord_b else next(iter(found), "")

    def probe_all(self, force: bool = False) -> None:
        """Sonda le build non coperte dall'indice (tutte con force=True) e aggiorna l'indice."""
        if not force:
            self._apply_index()
        todo = [b for b in list(self.builds.values()) if force or not b.probed]
        for b in todo:
            if force:
                b.probed, b.probe_error = False, ""
            try:
                probe_build(b)
            except Exception as exc:
                b.probe_error = str(exc)[:100]
        if todo:
            self._save_index()
        self.status = (f"{len(todo)} build sondate, {len(self.builds) - len(todo)} dall'indice"
                       if not force else f"{len(todo)} build risondate")
        self.probe_done.set()

    def probe_all_async(self, force: bool = False) -> None:
        # Se l'indice copre gia' tutto, probe_done resta impostato e il menu di
        # pre-avvio non aspetta. Altrimenti si azzera SUBITO (non nel thread): chi
        # apre il menu nei primi secondi deve aspettare la sonda, o nessuna build
        # risulterebbe "probed" ("nessuna build compatibile").
        if force or not self._apply_index():
            self.probe_done.clear()
            threading.Thread(target=self.probe_all, args=(force,), daemon=True).start()

    def refresh_after_install(self) -> List[str]:
        """Dopo un download: riscansiona, sonda SOLO le build nuove, aggiorna l'indice."""
        before = set(self.builds)
        self.discover()
        self.probe_all()
        return sorted(set(self.builds) - before)

    def wait_probed(self, timeout: float = 60.0) -> bool:
        """Attende la fine della sonda delle build (True = finita)."""
        return self.probe_done.wait(timeout)

    def ordered(self) -> List[BuildInfo]:
        return sorted(self.builds.values(), key=lambda b: (not b.gpu, -b.build_num, b.key))

    def add_dir(self, path: str) -> Dict[str, Any]:
        path = os.path.normpath(os.path.expanduser(path.strip()))
        cand = path
        if os.path.isfile(path) and os.path.basename(path) == "llama-server":
            cand = os.path.dirname(path)
        if not os.path.isfile(os.path.join(cand, "llama-server")):
            return {"success": False, "error": f"in {cand} non c'e' un llama-server"}
        if cand not in self.user_dirs:
            self.user_dirs.append(cand)
            self._save_user()
        self.discover()
        b = next((x for x in self.builds.values() if x.dir == cand), None)
        if b:
            probe_build(b)
            self._save_index()
        return {"success": True, "key": b.key if b else cand}

    def remove_user(self, key: str) -> Dict[str, Any]:
        b = self.builds.get(key)
        if not b or not b.user_added:
            return {"success": False, "error": "solo le build aggiunte a mano si tolgono dall'elenco "
                                              "(i file su disco non vengono mai cancellati)"}
        self.user_dirs = [d for d in self.user_dirs
                          if os.path.normpath(os.path.expanduser(d)) != b.dir]
        self._save_user()
        self.discover()
        return {"success": True}

    def set_active(self, key: str) -> None:
        if key in self.builds:
            self.active_key = key
            self._save_user()

    def active(self) -> Optional[BuildInfo]:
        return self.builds.get(self.active_key)

    # -- documentazione ---------------------------------------------------
    def _existing_docs(self, key: str) -> Dict[str, str]:
        d = os.path.join(DOCS_ROOT, key)
        out: Dict[str, str] = {}
        try:
            for n in sorted(os.listdir(d)):
                out[n] = os.path.join(d, n)
        except OSError:
            pass
        return out

    def fetch_docs(self, key: str, online: bool = True) -> Dict[str, Any]:
        """
        Salva la documentazione di UNA build: help locale sempre, README
        upstream al tag bNNNNN se la build e' ufficiale e online=True.
        """
        b = self.builds.get(key)
        if not b:
            return {"success": False, "error": f"build {key} sconosciuta"}
        if not b.probed:
            probe_build(b)
        d = os.path.join(DOCS_ROOT, key)
        os.makedirs(d, exist_ok=True)
        saved, errors = [], []
        h = _run([b.server, "--help"])
        if not h.startswith("__ERR__"):
            with open(os.path.join(d, "help.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"# {b.server}\n# {b.version}\n# salvato {time.ctime()}\n\n{h}")
            saved.append("help.txt")
        if b.rpc_server:
            rh = _run([b.rpc_server, "--help"])
            with open(os.path.join(d, "rpc-help.txt"), "w", encoding="utf-8") as fh:
                fh.write(rh)
            saved.append("rpc-help.txt")
        with open(os.path.join(d, "build.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(b), fh, indent=2, default=str)
        saved.append("build.json")
        if online and b.official:
            tag = f"b{b.build_num}"
            for name, path in (("server-README.md", "tools/server/README.md"),
                               ("rpc-README.md", "tools/rpc/README.md"),
                               ("build.md", "docs/build.md"),
                               ("multimodal.md", "docs/multimodal.md")):
                url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/{path}"
                try:
                    with urllib.request.urlopen(urllib.request.Request(
                            url, headers={"User-Agent": "ai_tui-docs/1.0"}), timeout=15) as r:
                        data = r.read()
                    with open(os.path.join(d, name), "wb") as fh:
                        fh.write(data)
                    saved.append(name)
                except Exception as exc:
                    errors.append(f"{name}: {str(exc)[:60]}")
        elif online:
            errors.append("fork senza tag upstream: fa fede help.txt")
        b.docs = self._existing_docs(key)
        return {"success": bool(saved), "saved": saved, "errors": errors, "dir": d}

    def fetch_all_docs_async(self, on_done=None) -> None:
        if self.docs_busy:
            return
        self.docs_busy = True

        def work() -> None:
            res = {}
            try:
                for k in list(self.builds):
                    res[k] = self.fetch_docs(k)
                ok = sum(1 for r in res.values() if r.get("success"))
                self.status = f"documentazione salvata per {ok}/{len(res)} build in {DOCS_ROOT}"
            finally:
                self.docs_busy = False
                if on_done:
                    on_done(res)

        threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------------
# Scelta della build all'avvio di un preset (F1-F5)
# ---------------------------------------------------------------------------
# Per cosa l'abbiamo usata / vantaggi. Solo fatti verificati in questa macchina.
USE_NOTES: Dict[str, str] = {
    "llama-b11330": "Stabile di riferimento: preset F1-F6, benchmark RPC con Miky, "
                    "DSPARK nativo, --fit/--load-mode. Stessa build di Miky.",
    "llama-cpp-compelte-tools": "Copia di b11330 (stesso commit) con tutti i tool "
                                "(bench, quantize, perplexity): comportamento identico.",
    "unsloth": "Build Unsloth b9987: piu' vecchia, niente DSPARK ne' --load-mode. "
               "Solo per confronto/compatibilita'.",
    "llama-b10373": "Fork speed.llama.cpp con scheduler DSPARK: nessuna GPU locale.",
}


def use_note(b: "BuildInfo") -> str:
    for k, v in USE_NOTES.items():
        if k == b.key or k in b.dir:
            return v
    return b.note or b.summary()


_COMPAT_CACHE: Dict[tuple, tuple] = {}
COMPAT_PATH = os.path.expanduser("~/.cache/ai_tui/rpc_compat.json")


_REMOTE_TAG: Dict[str, tuple] = {}
REMOTE_TAG_TTL_S = 15.0


def remote_rpc_tag(endpoint: str) -> str:
    """Build del ggml-rpc-server dietro l'endpoint (es. "b11435"), letta via ssh dal
    link /usr/local/bin/ggml-rpc-server. "" se non si sa (host senza ssh, spento).
    Fa parte della chiave dei verdetti RPC: quando l'host cambia build si riverifica."""
    host = endpoint.rsplit(":", 1)[0]
    hit = _REMOTE_TAG.get(host)
    if hit and time.time() - hit[1] < REMOTE_TAG_TTL_S:
        return hit[0]
    tag = ""
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=2", host,
                            "readlink -f /usr/local/bin/ggml-rpc-server"],
                           capture_output=True, text=True, timeout=4)
        m = re.search(r"llama-(b\d+)", p.stdout)
        tag = m.group(1) if m else ""
    except (OSError, subprocess.SubprocessError):
        pass
    _REMOTE_TAG[host] = (tag, time.time())
    return tag


def _compat_key(b: "BuildInfo", endpoint: str) -> str:
    try:
        mt = int(os.path.getmtime(b.server))
    except OSError:
        mt = 0
    tag = remote_rpc_tag(endpoint)
    return f"{b.server}|{mt}|{endpoint}" + (f"|{tag}" if tag else "")


def compat_cached(b: "BuildInfo", endpoint: str) -> Optional[tuple]:
    """Risultato salvato di un test precedente (valido finche' il binario non cambia)."""
    try:
        with open(COMPAT_PATH, "r", encoding="utf-8") as fh:
            v = json.load(fh).get(_compat_key(b, endpoint))
        return (bool(v[0]), str(v[1])) if v else None
    except (OSError, ValueError, TypeError, IndexError):
        return None


def _compat_store(b: "BuildInfo", endpoint: str, res: tuple) -> None:
    try:
        # read-modify-write sotto lock + scrittura atomica (piu' thread di verifica)
        _AJ.update_json(COMPAT_PATH, _compat_key(b, endpoint), [res[0], res[1]])
    except (OSError, TypeError):
        pass


def rpc_compatible(b: "BuildInfo", endpoint: str, timeout: float = 8.0,
                   use_cache: bool = True) -> tuple:
    """
    Test REALE di compatibilita' con un ggml-rpc-server: si esegue
    `llama-server --rpc <endpoint> --list-devices` (nessun modello caricato).
    Se il protocollo RPC combacia compare la riga "RPC0: host:porta (X MiB,
    Y MiB free)"; se non combacia (versione diversa) il device non compare.
    Ritorna (compatibile: bool, riga_device: str).
    """
    key = (b.server, endpoint, remote_rpc_tag(endpoint))
    if use_cache and key in _COMPAT_CACHE:
        return _COMPAT_CACHE[key]
    out = _run([b.server, "--rpc", endpoint, "--list-devices"], timeout=timeout)
    line = next((l.strip() for l in out.splitlines() if l.strip().startswith("RPC")), "")
    if line:
        # unico esito CERTO positivo: il device RPC0 e' comparso
        res = (True, line)
        _COMPAT_CACHE[key] = res
        _compat_store(b, endpoint, res)
        return res
    if "version mismatch" in out:
        # incompatibilita' VERA di protocollo (ggml-rpc.cpp: "RPC server version mismatch")
        res = (False, next(l for l in out.splitlines() if "version mismatch" in l).strip()[:120])
        _COMPAT_CACHE[key] = res
        _compat_store(b, endpoint, res)
        return res
    # Timeout ("timed out after ..."), "Failed to connect", RPC occupato da un
    # altro client (serve UN client alla volta): NON e' un verdetto, quindi
    # niente cache. Si usa l'ultimo esito certo salvato, se c'e'.
    cached = compat_cached(b, endpoint)
    if cached:
        return cached
    tail = out.replace("__ERR__", "").strip()
    tail = tail.splitlines()[-1][-100:] if tail else "nessun output"
    return (False, UNREACHABLE_PREFIX + tail)


UNREACHABLE_PREFIX = "RPC non raggiungibile/occupato: "


def is_unreachable(res: tuple) -> bool:
    """True se l'esito di rpc_compatible NON e' un verdetto (timeout/occupato/spento)."""
    return (not res[0]) and str(res[1]).startswith(UNREACHABLE_PREFIX)


def launch_block_reason(b: "BuildInfo", device_spec: str) -> str:
    """Perche' launch_candidates() scarta questa build ("" = non la scarta)."""
    want = [d.strip() for d in (device_spec or "").split(",") if d.strip()]
    missing = [d for d in want if not d.upper().startswith("RPC") and d not in b.devices]
    if not os.path.isfile(b.server):
        return "llama-server assente"
    if not b.probed:
        return "non ancora sondata"
    if missing:
        return f"senza {','.join(missing)}" + (f" (ha {','.join(b.devices)})" if b.devices else " (solo CPU)")
    if any(d.upper().startswith("RPC") for d in want) and not b.rpc:
        return "senza supporto RPC"
    return ""


def launch_candidates(builds: List["BuildInfo"], device_spec: str) -> List["BuildInfo"]:
    """Build che hanno TUTTI i device locali richiesti dal preset (es. CUDA0)
    e, se il preset usa RPC0, il supporto RPC. Il test di protocollo vero si
    fa poi con rpc_compatible()."""
    want = [d.strip() for d in (device_spec or "").split(",") if d.strip()]
    local = [d for d in want if not d.upper().startswith("RPC")]
    needs_rpc = any(d.upper().startswith("RPC") for d in want)
    out = []
    for b in builds:
        if not b.probed or not os.path.isfile(b.server):
            continue
        if any(d not in b.devices for d in local):
            continue
        if needs_rpc and not b.rpc:
            continue
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# ggml-rpc-server locali
# ---------------------------------------------------------------------------
RPC_EXE_NAMES = ("ggml-rpc-server", "rpc-server")


def running_rpc_servers() -> List[Dict[str, Any]]:
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            exe = os.readlink(f"/proc/{p}/exe")
        except OSError:
            continue
        if os.path.basename(exe) not in RPC_EXE_NAMES:
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as fh:
                argv = [a.decode(errors="replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            argv = []
        port, host = "50052", "127.0.0.1"
        for i, a in enumerate(argv):
            if a in ("-p", "--port") and i + 1 < len(argv):
                port = argv[i + 1]
            if a in ("-H", "--host") and i + 1 < len(argv):
                host = argv[i + 1]
        out.append({"pid": int(p), "exe": exe, "host": host, "port": int(port), "argv": argv})
    return out


def start_rpc_server(build: BuildInfo, port: int = 50052, host: str = "127.0.0.1",
                     device: str = "", cache: bool = True,
                     dry_run: bool = False) -> Dict[str, Any]:
    """
    Avvia un ggml-rpc-server LOCALE dalla build scelta, staccato (cgroup
    proprio). host=0.0.0.0 lo espone in rete: il protocollo RPC NON e'
    autenticato, va usato solo su LAN fidata (avviso della doc ufficiale).
    """
    if not build.rpc_server:
        return {"success": False, "error": f"la build {build.key} non ha ggml-rpc-server"}
    argv = [build.rpc_server, "-H", host, "-p", str(port)]
    if device:
        argv += ["--device", device]
    if cache:
        argv.append("-c")
    if dry_run:
        return {"success": True, "dry_run": True, "cmd": " ".join(argv)}
    env = dict(os.environ, LD_LIBRARY_PATH=build.dir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    res = _PS.spawn_detached(argv, name=f"rpc-{port}", env=env)
    res.pop("proc", None)
    return res


# ---------------------------------------------------------------------------
# Aggiornamenti: ultima build llama.cpp con binari (tag bNNNN), controllata
# all'avvio e salvata per 6 ore (cache) per non ripetere la chiamata.
# ---------------------------------------------------------------------------
RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
UPDATE_CACHE = os.path.expanduser("~/.cache/ai_tui/llama_update.json")
UPDATE_TTL_S = 6 * 3600
LINUX_ASSET = "llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz"
CUDART_ASSET = "cudart-llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz"


def latest_release(use_cache: bool = True) -> Dict[str, Any]:
    """{'tag','build','assets':[...]} della build piu' recente con binari Linux CUDA."""
    try:
        if use_cache and time.time() - os.path.getmtime(UPDATE_CACHE) < UPDATE_TTL_S:
            with open(UPDATE_CACHE, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, ValueError):
        pass
    try:
        req = urllib.request.Request(RELEASES_URL, headers={"User-Agent": "ai_tui-update/1.0",
                                                            "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            rels = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)[:120]}
    for rel in rels:
        tag = str(rel.get("tag_name") or "")
        by_name = {a.get("name"): a for a in rel.get("assets", [])}
        if re.fullmatch(r"b\d+", tag) and LINUX_ASSET.format(tag=tag) in by_name:
            main_a = by_name[LINUX_ASSET.format(tag=tag)]
            cudart_a = by_name.get(CUDART_ASSET.format(tag=tag))
            out = {
                "tag": tag, "build": int(tag[1:]), "published": rel.get("published_at", "")[:10],
                "assets": [
                    {"name": main_a.get("name"), "url": main_a.get("browser_download_url"),
                     "size": int(main_a.get("size") or 0)},
                ] + ([{"name": cudart_a.get("name"), "url": cudart_a.get("browser_download_url"),
                       "size": int(cudart_a.get("size") or 0)}] if cudart_a else []),
            }
            try:
                os.makedirs(os.path.dirname(UPDATE_CACHE), exist_ok=True)
                with open(UPDATE_CACHE, "w", encoding="utf-8") as fh:
                    json.dump(out, fh)
            except OSError:
                pass
            return out
    return {"error": "nessuna release con binari Linux CUDA"}


def newest_local_build(builds: List["BuildInfo"]) -> int:
    return max((b.build_num for b in builds if b.build_num), default=0)


# ---------------------------------------------------------------------------
# Download della nuova build (solo Linux: due .tar.gz, binari + runtime CUDA)
# dentro BUILD_DIR, con callback di percentuale. Non tocca MAI una build
# gia' presente: estrae sempre in una sottocartella nuova (llama-bNNNNN).
# ---------------------------------------------------------------------------
BUILD_DIR = "/mnt/lm_models/DEBIAN_FOLDER/llamacpp"


def download_release(rel: Dict[str, Any], progress: Optional[Any] = None) -> Dict[str, Any]:
    """Scarica ed estrae gli asset di `rel` (da latest_release()) in
    BUILD_DIR/llama-b<NNN>/. `progress(fase, got, total)` e' chiamata
    durante ogni download (fase = nome file corrente)."""
    import tarfile
    import tempfile

    tag = rel.get("tag")
    assets = rel.get("assets") or []
    if not tag or not assets:
        return {"ok": False, "error": "release senza asset utilizzabili"}
    dest = os.path.join(BUILD_DIR, f"llama-{tag}")
    if os.path.exists(dest):
        return {"ok": False, "error": f"{dest} esiste gia'"}
    os.makedirs(BUILD_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"dl-{tag}-", dir=BUILD_DIR)
    try:
        for a in assets:
            url, name, size = a.get("url"), a.get("name"), int(a.get("size") or 0)
            if not url:
                continue
            tgz_path = os.path.join(tmp_dir, name)
            req = urllib.request.Request(url, headers={"User-Agent": "ai_tui-update/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tgz_path, "wb") as out:
                got = 0
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
                    if progress:
                        progress(name, got, size or got)
            with tarfile.open(tgz_path, "r:gz") as tf:
                tf.extractall(tmp_dir)
            os.remove(tgz_path)
        # gli archivi llama.cpp si espandono in build/bin/ (o simile): si
        # appiattisce tutto cio' che e' eseguibile/libreria dentro dest/
        # si prepara tutto in stage/ e lo si rinomina in dest/ in un colpo solo:
        # la scansione delle build non vede mai una cartella a meta'
        stage = os.path.join(tmp_dir, "_stage")
        os.makedirs(stage)
        moved = 0
        for root, _dirs, files in os.walk(tmp_dir):
            if root.startswith(stage):
                continue
            for fn in files:
                dst = os.path.join(stage, fn)
                if os.path.exists(dst):
                    continue
                os.replace(os.path.join(root, fn), dst)
                moved += 1
        if moved == 0:
            return {"ok": False, "error": "archivio estratto ma nessun file trovato"}
        if not os.path.isfile(os.path.join(stage, "llama-server")):
            return {"ok": False, "error": "archivio estratto ma senza llama-server"}
        os.replace(stage, dest)
        return {"ok": True, "dir": dest, "files": moved}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
    finally:
        try:
            import shutil as _sh
            _sh.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pulizia: oltre alle build protette si tengono solo le KEEP_RECENT piu' recenti
# (stessa regola sul PC in BUILD_DIR e su Miky in /opt/llamacpp).
# ---------------------------------------------------------------------------
PROTECTED_BUILDS = ("llama-b11330", "llama-b11429", "llama-b11435")   # base su cui e' costruita la TUI
NEVER_PRUNE = PROTECTED_BUILDS + ("llama-b10373",)   # fork speed.llama.cpp: non si riscarica da GitHub
KEEP_RECENT = 6
MIKY_LLAMA_DIR = "/opt/llamacpp"


def prune_plan(names: List[str], in_use: set) -> List[str]:
    """Build da cancellare: solo cartelle llama-bNNNNN fuori da NEVER_PRUNE, oltre le
    KEEP_RECENT piu' recenti, e mai quelle in uso."""
    cands = sorted((n for n in names if re.fullmatch(r"llama-b\d+", n) and n not in NEVER_PRUNE),
                   key=lambda n: int(n[len("llama-b"):]))
    old = cands[:-KEEP_RECENT] if len(cands) > KEEP_RECENT else []
    return [n for n in old if n not in in_use]


def prune_local(in_use: set) -> Dict[str, Any]:
    import shutil as _sh
    names = [n for n in os.listdir(BUILD_DIR) if os.path.isdir(os.path.join(BUILD_DIR, n))]
    removed, errors = [], []
    for n in prune_plan(names, in_use):
        try:
            _sh.rmtree(os.path.join(BUILD_DIR, n))
            removed.append(n)
        except OSError as exc:
            errors.append(f"{n}: {exc}")
    return {"removed": removed, "errors": errors}


def prune_remote(host: str = "miky") -> Dict[str, Any]:
    """Stessa regola su Miky; la build che usa miky-rpc non si tocca mai."""
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host]
    try:
        p = subprocess.run(ssh + [f"ls -1 {MIKY_LLAMA_DIR}; echo __CUR__; "
                                  f"readlink -f /usr/local/bin/ggml-rpc-server"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"removed": [], "errors": [str(exc)]}
    if p.returncode != 0 or "__CUR__" not in p.stdout:
        return {"removed": [], "errors": [(p.stderr or "Miky non raggiungibile").strip()[:120]]}
    listing, cur = p.stdout.split("__CUR__", 1)
    m = re.search(r"(llama-b\d+)", cur)
    in_use = {m.group(1)} if m else set()
    if not in_use:
        return {"removed": [], "errors": ["build in uso su Miky sconosciuta: non cancello nulla"]}
    plan = prune_plan(listing.split(), in_use)
    if not plan:
        return {"removed": [], "errors": []}
    targets = " ".join(f"{MIKY_LLAMA_DIR}/{n} {MIKY_LLAMA_DIR}/.bad-b{n[len('llama-b'):]}" for n in plan)
    r = subprocess.run(ssh + [f"rm -rf {targets}"], capture_output=True, text=True, timeout=120)
    return {"removed": plan if r.returncode == 0 else [],
            "errors": [] if r.returncode == 0 else [(r.stderr or "rm fallito").strip()[:120]]}
