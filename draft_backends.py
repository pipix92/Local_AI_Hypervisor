"""
draft_backends.py - registro delle build llama.cpp e dei draft che supportano.

Perche' questo modulo esiste
----------------------------
La TUI deve poter scegliere il draft piu' veloce. Ma non tutte le build
llama.cpp accettano tutti i tipi di draft: il tipo di draft e' un flag
(`--spec-type`), e se la build non lo conosce il server non parte.

Questo modulo tiene l'unica fonte di verita' su:

* quali build llama.cpp sono presenti su disco,
* quali tipi di draft (`--spec-type`) ciascuna build accetta davvero,
* quale build usare quando l'utente sceglie un certo draft.

Le liste di `spec_types` NON sono scritte a mano: sono il risultato di
`probe_spec_types()`, che esegue `llama-server --help` e legge la riga
`--spec-type`. Se la build sul disco cambia, la lista si aggiorna da sola e
non puo' becoming stale. Solo se l'esecuzione fallisce si ripiega sui valori
dichiarati in `DRAFT_TYPES_FALLBACK` (e il campo `probed` resta False, cosi'
la UI puo' avvisare che la lista non e' stata verificata).

Tipi di draft (valori di --spec-type)
-------------------------------------
* ``draft-mtp``    : multi-token prediction, il draft piu' comune.
* ``draft-dflash`` : DFlash (richiede llama.cpp b9831+).
* ``draft-dspark`` : DSpark. VERIFICATO 2026-10-05: e' NATIVO anche nella
  build ufficiale b11330 (riga --spec-type del suo --help). Il fork
  speed.llama.cpp (b10373) aggiunge solo lo SCHEDULER (--spec-sched,
  --spec-sts, --spec-draft-dspark-n-cand...) ma e' compilato SENZA CUDA:
  `--list-devices` = (none), quindi in locale gira solo su CPU.
* ``ngram-*``      : speculative senza modello draft (ngram-simple,
  ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache), presenti in b11330.

Vincoli del progetto gia' rispettati qui
-----------------------------------------
* i binari stanno su ``/mnt/lm_models``, MAI sulla root: quando la root e'
  arrivata al 100% una pulizia automatica li aveva cancellati;
* nessun probe avvia un server: si esegue solo ``--help`` e ``--version``,
  che non caricano modelli e non toccano la VRAM;
* nessun ``pkill``: qui non si killa niente.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Timeout per le sonde: --help e' un comando che termina subito, ma se il
# filesystem di /mnt/lm_models e' lento il rischio di hang c'e'. 6 secondi
# bastano e lasciano comunque la UI libera.
PROBE_TIMEOUT_S = 6.0


# -------------------------------------------------------------------------
# Tipi di draft: nome "pulito" -> valore di --spec-type
# -------------------------------------------------------------------------
# La chiave e' quella usata dalla TUI e nei preset; il valore e' la stringa
# esatta che llama.cpp si aspetta. Non si indovina: sono i valori elencati
# nella riga --spec-type dell'help delle build reali di questa macchina
# (verificato con `llama-server --help` su b11330 e b10373).
SPEC_TYPE_VALUES: Dict[str, str] = {
    "none": "none",
    "mtp": "draft-mtp",
    "dflash": "draft-dflash",
    "dspark": "draft-dspark",
    "simple": "draft-simple",
    "eagle3": "draft-eagle3",
}

# Usato solo se la sonda --help non e' disponibile. La UI mostra comunque
# che la lista non e' stata verificata (probed=False).
DRAFT_TYPES_FALLBACK: Dict[str, List[str]] = {
    "stable": ["none", "simple", "eagle3", "mtp", "dflash", "dspark",
               "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"],
    "speed-dspark": ["none", "simple", "eagle3", "mtp", "dflash", "dspark",
                     "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"],
    "unsloth": ["none", "simple", "eagle3", "mtp", "dflash",
                "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"],
}


def spec_type_value(draft_type: str) -> str:
    """Converte il nome del draft nel valore di --spec-type."""
    key = (draft_type or "none").strip().lower()
    return SPEC_TYPE_VALUES.get(key, key)


@dataclass
class DraftBuild:
    """Una build llama.cpp su disco e i draft che sa usare."""

    key: str
    label: str
    binary: str                      # percorso assoluto di llama-server
    dir: str = ""                    # cartella che contiene i .so
    spec_types: List[str] = field(default_factory=list)
    probed: bool = False             # spec_types letto davvero da --help?
    note: str = ""
    # Flag che esistono SOLO in questa build. La TUI non li genera su altre
    # build: build_command() li filtra gia' con known_flags, ma qui serves
    # a decidere il messaggio "questa build serve per DSPARK".
    dspark_flags: List[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return bool(self.binary) and os.path.isfile(self.binary)

    def supports(self, draft_type: str) -> bool:
        """True se questa build accetta il tipo di draft richiesto."""
        if not self.spec_types:
            return False
        return (draft_type or "none").strip().lower() in self.spec_types

    def supports_dspark(self) -> bool:
        return self.supports("dspark")

    def status_line(self) -> str:
        """Riga breve per la UI: presenza su disco + draft supportati."""
        if not self.exists:
            return f"✗ non installata ({self.binary})"
        mark = "✓" if self.probed else "~"
        src = "help" if self.probed else "stima"
        return (f"{mark} {self.binary} | draft: "
                f"{', '.join(self.spec_types) or 'nessuno'} (da {src})")


# Le due build di questa macchina. La prima e' quella STABILE, gia' usata da
# tutto il resto della TUI: non viene mai sovrascritta ne' spostata.
# La seconda e' il fork speed.llama.cpp, installata in una cartella SEPARATA.
# La lista non e' un': se l'utente sposta una build, si aggiunge la riga.
BUILD_CATALOG: List[Dict[str, object]] = [
    {
        "key": "stable",
        "label": "STABLE (b11330, CUDA 13.4)",
        "dir": "/mnt/lm_models/llamacpp/bin/llama-b11330",
        "note": "Build gia' validata con tutto il resto della TUI. CUDA 13.4 + RPC. "
                "Supporta MTP, DFLASH, DSPARK (nativo, senza scheduler), EAGLE3 e "
                "ngram-*; -sm none/layer/row/tensor (tensor = SPERIMENTALE). "
                "NON viene mai sovrascritta.",
        "dspark_flags": [],
    },
    {
        "key": "speed-dspark",
        "label": "SPEED-DSPARK (fork KoopahTManiac)",
        "dir": "/mnt/lm_models/llamacpp/speed-dspark",
        "note": "Fork speed.llama.cpp b10373: aggiunge lo SCHEDULER DSPARK "
                "(--spec-sched, --spec-sts, --spec-draft-dspark-n-cand). "
                "ATTENZIONE: compilato senza CUDA (nessun device locale) -> "
                "in locale solo CPU; ha senso solo con --rpc verso GPU remote.",
        "cpu_only": True,
        "dspark_flags": [
            "--spec-sched",
            "--spec-sched-adapt",
            "--spec-sched-cache",
            "--spec-conf-log",
            "--spec-sts",
            "--spec-sts-bias",
            "--spec-sched-beta",
            "--spec-sched-draft-cost",
            "--spec-sched-probe",
            "--spec-draft-dspark-n-cand",
        ],
    },
    {
        "key": "unsloth",
        "label": "UNSLOTH (b9987, CUDA)",
        "dir": "/home/user/.unsloth/llama.cpp/build/bin",
        "note": "Build di Unsloth b9987: CUDA + RPC, NO draft-dspark, NO --load-mode. "
                "Solo compatibilita'; la stabile b11330 la supera in tutto.",
        "dspark_flags": [],
    },
]


# -------------------------------------------------------------------------
# Sonda: quali --spec-type accetta davvero questo binario?
# -------------------------------------------------------------------------
def _resolve_binary(build_dir: str) -> str:
    """
    Trova llama-server dentro build_dir.

    La release e' un tar.gz che contiene a sua volta una sottocartella
    (es. speed-dspark/llama-b10373/llama-server), quindi si scende di un
    livello invece di assumere che il binario sia in cima.
    """
    direct = os.path.join(build_dir, "llama-server")
    if os.path.isfile(direct):
        return os.path.normpath(direct)
    try:
        for name in sorted(os.listdir(build_dir)):
            cand = os.path.join(build_dir, name, "llama-server")
            if os.path.isfile(cand):
                return os.path.normpath(cand)
    except OSError:
        pass
    return ""


def _parse_spec_types(help_text: str) -> List[str]:
    """
    Estrae i valori di --spec-type dalla riga di help.

    La riga reale e' lunga e va a capo, per esempio:
        --spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,...
    Si prende tutto what segue "--spec-type" fino a fine riga e si pulisce.
    """
    if not help_text:
        return []
    m = re.search(r"--spec-type[= ]+([A-Za-z0-9_,\-]+)", help_text)
    if not m:
        return []
    raw = m.group(1)
    # dai valori llama.cpp (draft-dflash) ricaviamo i nomi della TUI (dflash)
    out: List[str] = []
    for v in raw.split(","):
        v = v.strip()
        if not v:
            continue
        short = v[len("draft-"):] if v.startswith("draft-") else v
        out.append(short)
    return out


def probe_spec_types(binary: str, timeout_s: float = PROBE_TIMEOUT_S) -> Tuple[List[str], bool]:
    """
    Esegue `binary --help` e restituisce (tipi_di_draft, sondato).

    Non avvia un server e non tocca la VRAM: --help esce subito. Se il
    binario non e' eseguibile o fallisce, si restituisce ([], False) e il
    chiamante decide se ripiegare sui valori di fallback.
    """
    if not binary or not os.path.isfile(binary):
        return [], False
    try:
        proc = subprocess.run(
            [binary, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return [], False
    return _parse_spec_types(proc.stdout or ""), True


def probe_build_version(binary: str, timeout_s: float = PROBE_TIMEOUT_S) -> str:
    """Legge la versione del binario (per mostrarlo nella UI). Stringa vuota se fallisce."""
    if not binary or not os.path.isfile(binary):
        return ""
    try:
        proc = subprocess.run(
            [binary, "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return (proc.stdout or "").strip().splitlines()[0] if proc.stdout else ""


# -------------------------------------------------------------------------
# Registro
# -------------------------------------------------------------------------
class DraftBackendRegistry:
    """
    L'elenco delle build disponibili, con i draft che ciascuna supporta.

    Uso tipico:

        reg = DraftBackendRegistry()
        build = reg.for_draft("dflash")     # sceglie la build giusta
        if not build.exists:
            print(build.status_line())      # messaggio chiaro, niente silenzio
    """

    def __init__(self, catalog: Optional[List[Dict[str, object]]] = None,
                 probe: bool = True) -> None:
        self._builds: Dict[str, DraftBuild] = {}
        for entry in (catalog if catalog is not None else BUILD_CATALOG):
            key = str(entry["key"])
            build_dir = str(entry["dir"])
            binary = _resolve_binary(build_dir)
            build = DraftBuild(
                key=key,
                label=str(entry.get("label", key)),
                binary=binary,
                dir=build_dir,
                spec_types=[],
                probed=False,
                note=str(entry.get("note", "")),
                dspark_flags=list(entry.get("dspark_flags", [])),  # type: ignore[arg-type]
            )
            if build.exists and probe:
                types, ok = probe_spec_types(binary)
                if ok and types:
                    build.spec_types = types
                    build.probed = True
            if not build.spec_types:
                # Nessuna sonda riuscita: ripiego sui valori dichiarati.
                # probed resta False, cosi' la UI puo' dirlo.
                build.spec_types = list(DRAFT_TYPES_FALLBACK.get(key, ["none"]))
            self._builds[key] = build

    # -- accesso ---------------------------------------------------------
    def get(self, key: str) -> Optional[DraftBuild]:
        return self._builds.get(key)

    def all(self) -> List[DraftBuild]:
        return list(self._builds.values())

    def installed(self) -> List[DraftBuild]:
        return [b for b in self._builds.values() if b.exists]

    def supports_dspark(self) -> List[DraftBuild]:
        return [b for b in self.installed() if b.supports_dspark()]

    # -- scelta ----------------------------------------------------------
    def for_draft(self, draft_type: str) -> Optional[DraftBuild]:
        """
        La build che sa usare `draft_type`, o None se nessuna la supporta.

        Si preferisce SEMPRE la build stabile quando basta: e' quella
        gia' validata con tutto il resto. La build del fork entra solo
        quando serve qualcosa che la stabile non sa fare (oggi: DSPARK).
        """
        want = (draft_type or "none").strip().lower()
        if want == "none":
            stable = self._builds.get("stable")
            return stable if (stable and stable.exists) else (
                self.installed()[0] if self.installed() else None
            )
        candidates = [b for b in self.installed() if b.supports(want)]
        if not candidates:
            return None
        # 'stable' vince a parita' di supporto.
        for b in candidates:
            if b.key == "stable":
                return b
        return candidates[0]

    def explain(self, draft_type: str) -> str:
        """
        Messaggio esplicito quando il draft scelto non e' disponibile.

        Serve a NON fallire in silenzio: se l'utente sceglie DSPARK e la
        build non c'e', qui si dice perche' e cosa fare.
        """
        want = (draft_type or "none").strip().lower()
        build = self.for_draft(want)
        if build is not None:
            return (f"Draft '{want}': build {build.key} -> {build.binary}")
        if not self.installed():
            return (f"Nessuna build llama.cpp trovata: il draft '{want}' non e' "
                    f"usabile. Installa almeno /mnt/lm_models/llamacpp/bin/llama-b11330.")
        # Ci sono build, ma nessuna sa fare questo draft.
        lines = [f"Nessuna build installata supporta il draft '{want}'."]
        if want == "dspark":
            lines.append(
                "draft-dspark e' nativo da b11330 (ufficiale); lo scheduler "
                "DSPARK e' nel fork speed.llama.cpp (solo CPU in locale).")
            lines.append(
                "Alternativa gia' funzionante: DFLASH (draft-dflash) sulla "
                "build stabile b11330.")
        else:
            lines.append("Tipi supportati dalle build presenti: " + "; ".join(
                f"{b.key}={','.join(b.spec_types)}" for b in self.installed()))
        return " ".join(lines)

    # -- stato per la UI -------------------------------------------------
    def report(self) -> List[str]:
        """Righe di riepilogo, una per build, con versione e draft supportati."""
        out: List[str] = []
        for b in self.all():
            ver = probe_build_version(b.binary) if b.exists else ""
            head = f"{b.key}: {b.label}"
            if ver:
                head += f" [{ver.split('(')[0].replace('version:', '').strip()}]"
            out.append(f"{head} -> {b.status_line()}")
        return out
