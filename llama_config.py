"""
llama_config.py — Pannello di configurazione COMPLETO di llama-server
=======================================================================
Estensione del pannello "RPC & CUDA13" (Tab 9): qui l'utente non e' limitato
ai 5 preset validati ma puo' modificare QUALSIASI flag di llama-server.

Tre principi non negoziabili di questo modulo:

1. **Elenco dei flag DINAMICO.** La lista arriva dal parsing del testo di
   ``llama-server -h`` della build locale (``parse_help``). Nulla e'
   hardcodato: se l'utente aggiorna llama.cpp i flag nuovi compaiono da soli
   e i flag rimossi smettono di essere generati (finiscono in ``unsupported``).
   Le uniche liste fisse sono le *spiegazioni* (``KV_TYPE_INFO``) e le chiavi
   logiche del comando (``COMMAND_KEYS``), che non cambiano fra build.

2. **Dry-run totale.** Ogni funzione che produce un comando si chiama
   ``build_command`` / ``dry_run_command`` e NON esegue mai nulla. Non c'e'
   un solo ``Popen`` in questo file. Il server che gira sulla 8087 resta
   intatto e la VRAM non viene toccata.

3. **Niente kill.** Nessun ``taskkill``, nessun ``kill()``, nessun
   terminazione di processi, in nessun percorso di questo modulo.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "FlagSpec",
    "parse_help",
    "read_help_text",
    "BACKENDS",
    "BackendInfo",
    "detect_backend",
    "list_backends",
    "parse_build_number",
    "local_build_numbers",
    "quote_arg",
    "quote_argv",
    "normalize_tensor_split",
    "tensor_split_ok",
    "KV_TYPE_INFO",
    "ServerConfig",
    "COMMAND_KEYS",
    "ConfigProfileStore",
    "fetch_latest_release",
    "check_update",
    "plan_download",
    "download_build",
    "GITHUB_LATEST_URL",
    "GITHUB_ASSET_PATTERNS",
]


# =====================================================================
# 1. PARSING DINAMICO DELL'HELP  (la parte che rende tutto futuro-proof)
# =====================================================================

# Una riga che INIZIA con un flag. Esempi reali dell'help di llama-server:
#   -t,    --threads N
#   -ctk,  --cache-type-k TYPE
#   --spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE
#   -cb,   --cont-batching, -nocb, --no-cont-batching
#   -sm,   --split-mode {none,layer,row,tensor}
#   --cpu-strict <0|1>
#   --kvo,  --kv-offload, -nkvo, --no-kv-offload        (continua' a capo)
_FLAG_START_RE = re.compile(r"^(-{1,2}[A-Za-z0-9][A-Za-z0-9_.-]*)(.*)$")
# Placeholder che indicano che il flag ACCETTA un valore.
_PLACEHOLDERS = {
    "n", "m", "type", "f", "fn", "fname", "mode", "servers", "sim", "simi",
    "simil", "similarity", "lo-hi", "host", "port", "index", "path", "dir",
    "url", "string", "filename", "level", "format", "template", "seconds",
    "second", "template_file", "device", "jinja_template", "p", "value",
    "usage", "target", "miB0", "mib0,miB1,miB2,...", "value", "args",
}
_ENUM_INLINE_RE = re.compile(r"\{([^{}]+)\}")
# Un token e' un NOME DI FLAG (non un placeholder) se e' -x / --xxx-xxx.
_FLAGLIKE_RE = re.compile(r"^-{1,2}[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ALLOWED_RE = re.compile(r"allowed values:\s*([^\n]+)", re.IGNORECASE)
_DEFAULT_RE = re.compile(r"\(default:\s*([^)]*)\)")
# Alcuni help scrivono "(default: auto, env: LLAMA_ARG_N_GPU_LAYERS)": il
# riferimento all'ambiente e' dentro la parentesi del default e va staccato,
# altrimenti finirebbe nel testo del default.
_DEFAULT_ENV_RE = re.compile(r"\s*,?\s*env:\s*[A-Za-z0-9_]+", re.IGNORECASE)
_ENV_RE = re.compile(r"\(env:\s*([A-Za-z0-9_]+)\s*\)")
# "- on-demand reading ..." : le righe di opzioni descrivono i valori di enum.
_BULLET_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_.-]+):")
_SECTION_RE = re.compile(r"^-{2,}\s*(.*?)\s*-{2,}\s*$")


@dataclass
class FlagSpec:
    """Un singolo flag di llama-server, come letto dall'help della build."""

    canonical: str                 # "--ctx-size" (forma lunga, senza valori)
    aliases: List[str]             # ["-c"]
    takes_value: bool
    arg_placeholder: str           # "N", "TYPE", "{none,layer}" grezzo, ""
    value_kind: str                # bool|int|float|enum|string
    choices: List[str]             # per value_kind == "enum"
    default: str
    env: str
    description: str
    section: str
    raw_lines: List[str] = field(default_factory=list)

    def all_names(self) -> List[str]:
        return [self.canonical] + list(self.aliases)

    def matches(self, name: str) -> bool:
        """True se ``name`` e' il nome canonico o una delle sue alias."""
        return name == self.canonical or name in self.aliases

    def negative_name(self) -> str:
        """
        Nome con cui SPECIFICARE lo spegnimento, per un flag booleano.

        "--cont-batching" con alias "--no-cont-batching" -> "--no-cont-batching".
        Restituisce "" se l'help non espone una forma negativa (per i flag enum
        come --flash-attn si passa il valore esplicito "off").
        """
        if self.value_kind != "bool":
            return ""
        cands = [n for n in self.aliases + [self.canonical] if n.startswith("--no-")]
        return cands[0] if cands else ""


def describe_kv_type(kv_type: Optional[str]) -> Dict[str, str]:
    """
    Scheda descrittiva di un tipo di cache KV.

    Restituisce dizionario vuoto per un tipo sconosciuto, cosi' la TUI puo'
    mostrare "non documentato" senza dover gestire eccezioni a ogni frame.
    """
    key = (kv_type or "").strip().lower()
    if key in KV_TYPE_INFO:
        return dict(KV_TYPE_INFO[key])
    #Ricadi sugli alias piu' comuni usati nelle guide.
    alias = {"f16_k": "f16", "kv16": "f16", "q4_k": "q4_0", "q4": "q4_0",
             "q8": "q8_0", "q5": "q5_0"}
    if alias.get(key) in KV_TYPE_INFO:
        return dict(KV_TYPE_INFO[alias[key]])
    return {"label": kv_type or "(non impostato)", "detail": "", "use": ""}


def _is_placeholder(tok: str) -> bool:
    """Un token dopo il flag e' un placeholder se NON e' un altro flag."""
    if not tok:
        return False
    if _FLAGLIKE_RE.match(tok):
        return False
    if _ENUM_INLINE_RE.match(tok):
        return True
    # <0|1>, [on|off|auto], <0...100>: racchiusi fra parentesi angolari/quadre
    if re.match(r"^[<\[].*[>\]]$", tok):
        return True
    if tok.lower() in _PLACEHOLDERS:
        return True
    # Placeholder tipo "MiB0,MiB1,MiB2,..." o "lo-hi" o "N0,N1,N2,..."
    if re.match(r"^[A-Za-z][A-Za-z0-9]*(0,?)+", tok):
        return True
    if re.match(r"^[a-z]+-[a-z]+$", tok, re.IGNORECASE):
        return True
    return False


# Placeholder espliciti: numero intero, float, o "SIMILARITY" di -sps.
_TOKEN_INT_RE = re.compile(r"^-?[A-Z]?$|^-?[A-Z]\d*(,[A-Z0-9.]*)*$")
_TOKEN_FLOAT_RE = re.compile(r"^(?:\d+\.\d+|\d*\.\d+)$|^(?i:sim|similarity|similarity)$")


def _infer_value_kind(tok: str, desc: str) -> Tuple[bool, str, List[str]]:
    """
    Decide se il flag prende un valore e con che tipo.
    Ritorna (takes_value, kind, choices).

    Gli help di llama.cpp usano placeholder tipo "N", "M", "TYPE", "lo-hi",
    "<0|1>", "{none,layer}", "SIMILARITY", "MiB0,MiB1,MiB2,...".
    """
    if not tok:
        return False, "bool", []
    enum = _ENUM_INLINE_RE.search(tok)
    if enum:
        return True, "enum", [c.strip() for c in enum.group(1).split(",") if c.strip()]
    if "|" in tok:                                  # <0|1>, [on|off|auto]
        return True, "enum", [c.strip() for c in tok.strip("<>[]").split("|") if c.strip()]
    if _TOKEN_FLOAT_RE.match(tok):
        return True, "float", []
    # "N", "M", "P", "MiB0,MiB1,MiB2,..."
    if _TOKEN_INT_RE.match(tok):
        return True, "int", []
    # "TYPE", "FNAME", "HOST", "PATH", "MODE", "FORMAT", "LEVEL", "URL", ...
    if re.match(r"^[A-Za-z][A-Za-z0-9_.]*(\.\.\.)?$", tok):
        low = desc.lower()
        if "number" in low or "how many" in low or "max." in low or "index" in low:
            return True, "int", []
        if "float" in low or "proportion" in low or "ratio" in low:
            return True, "float", []
        return True, "string", []
    # "lo-hi"
    return True, "string", []


def _join_wrapped(lines: List[str]) -> str:
    """Ricomponie le righe di continuazione del testo dell'help."""
    return " ".join(x.strip() for x in lines if x.strip())


def _tokenize_names(rest: str) -> List[str]:
    """
    Spezza la parte destra di una riga di flag in token.

    Si separa su spazi E virgole (perche' "-ncmoe,--n-cpu-moe N" scrive due
    nomi senza spazio), ma NON all'interno di graffe, parentesi quadre o
    angolari: un enum inline come ``{none,layer,row,tensor}`` o
    ``<dev1,dev2,...>`` deve restare un token unico, altrimenti il
    placeholder si frantuma e i valori enum vengono persi.
    """
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in rest:
        if ch in "{[<":
            depth += 1
        elif ch in "}>]":
            depth = max(0, depth - 1)
        if depth == 0 and (ch.isspace() or ch == ","):
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def parse_help(help_text: str) -> Dict[str, FlagSpec]:
    """
    Trasforma il testo di ``llama-server -h`` in {nome_canonico: FlagSpec}.

    Robusto a: righe avvolte a 80 colonne, flag con alias multipli, flag senza
    alias, valori enumerativi scritti sia come {a,b} sia come elenco
    "allowed values:", flag booleani con alias negativi (-nkvo) e flag il cui
    nome e' stato spostato a capo come ``--spec-draft-type-k, -ctkd, ...``.

    Non solleva mai eccezioni: un help illeggibile produce semplicemente un
    dizionario vuoto, e il chiamante deve gestirlo.
    """
    out: Dict[str, FlagSpec] = {}
    if not help_text:
        return out

    section = "common"
    # Split in blocchi logici: un blocco parte da una riga di flag e finisce
    # alla riga che introduce il blocco successivo.
    blocks: List[Tuple[str, List[str]]] = []
    current: Optional[List[str]] = None
    for raw in help_text.splitlines():
        line = raw.rstrip()
        sec = _SECTION_RE.match(line.strip())
        if sec:
            # Le intestazioni sono "----- sampling params -----": non possono
            # essere confuse con un flag perche' il testo e' racchiuso fra
            # trattini su ENTRAMBI i lati (un flag ha invece "--nome, -alias").
            section = sec.group(1).strip() or section
            current = None
            continue
        if _FLAG_START_RE.match(line):
            # La section va agganciata al blocco: viene archiviata qui e non
            # letta in un secondo giro, altrimenti tutti i flag prenderebbero
            # l'ultima section incontrata nell'help.
            current = [line]
            blocks.append((section, current))
        elif current is not None:
            current.append(line)

    for section, blk in blocks:
        head = blk[0]
        m = _FLAG_START_RE.match(head)
        if not m:
            continue
        # group(1) e' l'alias corto ("-c"), group(2) tutto il resto della riga.
        # I due pezzi si ricompongono: si analizzano i token di group(2) e
        # l'alias di group(1) si aggiunge in testa alla lista dei nomi.
        first, rest = m.group(1), m.group(2)
        # Spezza su spazi: gli alias sono token che iniziano con "-", il primo
        # token non-flag e' il placeholder, il resto e' la descrizione (per
        # questo NON si accumula carattere per carattere: altrimenti
        # "--usage        print usage" diventerebbe un unico token).
        names: List[str] = [first]
        placeholder = ""
        consumed = 0
        # Riga tipo "-ncmoe,--n-cpu-moe N": il nome successivo e' attaccato
        # alla virgola senza spazio, quindi si separa anche sulle virgole
        # interne.  La separazione deve pero' rispettare le graffe: un enum
        # inline come "--split-mode {none,layer,row,tensor}" contiene virgole
        # che NON sono separatori, altrimenti il placeholder si frantuma.
        toks = _tokenize_names(rest)
        # Un flag puo' comparire piu' volte nella riga, ognuno con la propria
        # forma di valore: "--threads, -t N, -c N", "--top-p N, --top-pa N".
        # Finche' il token e' un nome di flag seguito da un placeholder, si
        # continua a consumare invece di fermarsi al primo placeholder.
        # NB consumed parte da 0 perche' `toks` NON contiene l'alias corto
        # di group(1) ("first"), che e' gia' in `names`.
        consumed = 0
        idx = 0
        while idx < len(toks):
            t = toks[idx]
            if _FLAGLIKE_RE.match(t) or t.startswith("--"):
                names.append(t)
                consumed += 1
                # il token successivo e' il placeholder di QUESTO nome?
                if idx + 1 < len(toks) and _is_placeholder(toks[idx + 1]):
                    if not placeholder:
                        placeholder = toks[idx + 1]
                    consumed += 1
                    idx += 2
                    continue
                idx += 1
                continue
            if not placeholder and _is_placeholder(t):
                placeholder = t
                consumed += 1
                idx += 1
                continue
            # testo descrittivo: il blocco dei nomi e' finito
            break

        if not names:
            continue
        # Il flag canonico e' il lungo, salvo "--no-xxx" (si tiene com'e').
        # Nome canonico: si preferisce il PRIMO flag lungo che non sia una
        # forma negativa.  Infatti "-cb, --cont-batching, -nocb,
        # --no-cont-batching" e' un solo flag booleano e la sua forma
        # canonica e' --cont-batching, non --no-cont-batching.  Tutte le
        # altre forme restano in `aliases`, quindi --no-cont-batching,
        # --webui ecc. restano comunque raggiungibili via alias_index.
        longs = [n for n in names if n.startswith("--")]
        positive = [n for n in longs if not n[2:].startswith("no-")]
        if positive:
            canonical = positive[0]
        elif longs:
            canonical = longs[0]
        else:
            canonical = names[0]
        aliases = [n for n in names if n != canonical]

        # Corpo descrittivo: dalla parte della riga di intestazione dopo i
        # nomi, piu' tutte le righe di continuazione.
        # I nomi vanno tolti per TOKEN, non con replace(): altrimenti
        # togliendo l'alias "-c" da "--ctx-size" nel testo residuo
        # comparirebbe il pezzo spurio "-tx-size".  Si consumano solo i
        # primi `consumed` token, quelli che costituivano nomi+placeholder.
        tail_tokens = toks[consumed:][0:]
        desc_parts: List[str] = []
        if placeholder:
            desc_parts.append(placeholder)
        desc_parts.extend(tail_tokens)
        desc_parts.extend(blk[1:])

        raw_desc = _join_wrapped(desc_parts)
        desc = _SECTION_RE.sub("", raw_desc).strip()

        # Togli "allowed values:" dal testo ma tieni la lista.
        # Scelta fra le tre fonti di valori, in ordine di affidabilita':
        #  1. enum scritto nel placeholder "--flag {a,b,c}" o "[a|b|c]":
        #    e' l'elenco ufficiale del flag, wins sempre.
        #  2. "allowed values: a, b, c" nell'aiuto.
        #  3. elenco a punti "- a: ..." nelle righe di continuazione
        #    (fonte piu' fragile: per --split-mode le righe descrivono solo
        #     {none,row,tensor} e perderebbero il valore "layer").
        inline: List[str] = []
        if _ENUM_INLINE_RE.search(placeholder):
            inline = [c.strip() for c in _ENUM_INLINE_RE.search(placeholder).group(1).split(",") if c.strip()]
        elif "|" in placeholder:
            inline = [c.strip() for c in placeholder.strip("<>[]{}").split("|") if c.strip()]

        choices: List[str] = []
        am = _ALLOWED_RE.search(raw_desc)
        if am:
            allowed_txt = am.group(1)
            # "allowed values:" e' a volte seguito da altro testo sulla stessa
            # # riga logica ("...q5_1 (default: f16) (env: ...)"): si taglia
            # la coda a partire dalla prima parentesi.
            cut = re.search(r"\s*\(default:", allowed_txt)
            if cut:
                allowed_txt = allowed_txt[:cut.start()]
            allowed = [c.strip() for c in allowed_txt.rstrip(".").split(",") if c.strip()]
            raw_desc = raw_desc[:am.start()] + raw_desc[am.end():]
            choices = inline or allowed
        if not choices:
            choices = list(inline)
        if not choices:
            for line in blk[1:]:
                bm = _BULLET_RE.match(line)
                if bm:
                    choices.append(bm.group(1).rstrip(":"))

        takes_value, kind, _k = _infer_value_kind(placeholder, desc)
        if choices and not _TOKEN_FLOAT_RE.match(placeholder):
            kind = "enum"
        # Correzione: un placeholder numerico puro con scelte testuali resta enum.
        if choices and kind in ("int", "float") and any(
            not re.match(r"^-?\d", c) for c in choices
        ):
            kind = "enum"

        # "(default: auto, env: LLAMA_ARG_N_GPU_LAYERS)": l'env va staccato
        # dal default, altrimenti il default conterrebbe il nome della var.
        raw_desc = _DEFAULT_ENV_RE.sub(" ", raw_desc)
        dm = _DEFAULT_RE.search(raw_desc)
        default = dm.group(1).strip() if dm else ""
        em = _ENV_RE.search(raw_desc)
        env = em.group(1) if em else ""
        if not env:
            # l'env era dentro la parentesi: "(default: auto, env: LLAMA_...)"
            ie = re.search(r"env:\s*([A-Za-z0-9_]+)", raw_desc)
            if ie:
                env = ie.group(1)

        # Descrizione pulita: togli ambienti di default/env, URL e spazi extra.
        desc = _ENV_RE.sub("", desc)
        desc = _DEFAULT_RE.sub("", desc)
        desc = re.sub(r"\(\s*\)", "", desc)
        desc = re.sub(r"https?://\S+", "", desc)
        desc = re.sub(r"\s{2,}", " ", desc).strip(" ,.")

        spec = FlagSpec(
            canonical=canonical,
            aliases=aliases,
            takes_value=takes_value,
            arg_placeholder=placeholder,
            value_kind=kind,
            choices=choices,
            default=default,
            env=env,
            description=desc,
            section=section,
            raw_lines=list(blk),
        )
        if canonical not in out:      # il primo blocco vince (help lineare)
            out[canonical] = spec
    return out


def read_help_text(binary_path: str, timeout: float = 20.0) -> str:
    """
    Esegue ``llama-server -h`` e restituisce il testo dell'help.

    ATTENZIONE: ``-h`` stampa l'help e esce immediatamente: NON avvia un
    server, non apre una porta, non carica pesi, non occupa VRAM. E' l'unico
    modo per avere l'elenco dei flag *della build locale*.
    """
    if not binary_path or not os.path.isfile(binary_path):
        return ""
    import subprocess  # import locale: questo modulo resta importabile ovunque

    try:
        cp = subprocess.run(
            [binary_path, "-h"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(binary_path)),
        )
    except Exception:
        return ""
    return (cp.stdout or "") + (cp.stderr or "")


# =====================================================================
# 2. BACKEND E RILEVAMENTO BINARI  (solo filesystem: nessun avvio)
# =====================================================================

@dataclass
class BackendInfo:
    """Un backend possibile e dove trovarne il binario."""

    key: str
    label: str
    dll: str                  # DLL che identifica il backend ("" = CPU puro)
    marker: str               # sottostringa che il nome della dir deve contenere
    dirs: List[str]
    device_flag: str          # valore da passare a --device
    description: str
    binary: str = ""          # risolto solo se esiste
    build: int = 0            # bNNNNN ricavato dal nome della dir
    exists: bool = False

    def note(self) -> str:
        if self.exists:
            return f"✓ {self.binary} (build b{self.build})" if self.build else f"✓ {self.binary}"
        return "✗ non trovato"


def _name_has_marker(name: str, marker: str) -> bool:
    """
    True se il nome della directory contiene il marker come PAROLA.

    Serve confrontare i token separati da "-", "_" e "." invece di usare un
    semplice `in`: altrimenti il marker "cpu" matcherebbe dentro "cuda"
    (cu-pu-d-a contiene "cpu" come sottostringa) e il backend CPU punterebbe
    alla build CUDA.
    """
    if not marker:
        return True
    low = name.lower()
    tokens = re.split(r"[-_.\s]+", low)
    if marker in tokens:
        return True
    # "vulkan" non e' mai un token isolato ("win-vulkan-x64" lo e', ma
    # "llamavulkan" no): per i marker lunghi accetto anche la sottostringa.
    return len(marker) >= 6 and marker in low


# Ricerca: prima le directory esplicite dell'utente, poi una scansione di
# ~/Downloads e /mnt/lm_models/models per nome che contiene il marker del backend.
def _candidate_dirs(marker: str, extra: Optional[List[str]] = None) -> List[str]:
    out: List[str] = []
    for base in ["/mnt/lm_models/DEBIAN_FOLDER/llamacpp",
                 "/home/user/Downloads", "/mnt/lm_models/models", "/mnt/lm_models/models/Anbeeld",
                 "C:/Users/User/Documents", "C:/Users/User/llama"]:
        if not os.path.isdir(base):
            continue
        try:
            for name in sorted(os.listdir(base)):
                # Su Linux la directory si chiama solo "llama-bNNNNN": il
                # marker non compare nel nome, e a filtrare e' la libreria di
                # backend che _resolve_backend verifica dentro la directory.
                if _name_has_marker(name, marker) or re.match(r"^llama-b\d{4,6}$", name):
                    out.append(os.path.join(base, name))
        except OSError:
            continue
    for e in (extra or []):
        if e not in out:
            out.append(e)
    return out


BACKENDS: Dict[str, BackendInfo] = {
    "cuda": BackendInfo(
        key="cuda",
        label="CUDA 13.4 (RTX 3050 locale)",
        dll="ggml-cuda.dll",
        marker="cuda",
        dirs=[
            "/home/user/Downloads/llama-b11259-bin-win-cuda-13.4-x64",
            "/mnt/lm_models/models/llama-b11259-bin-win-cuda-13.4-x64",
        ],
        device_flag="CUDA0",
        description="Build CUDA gia' validata e funzionante. Non viene mai sovrascritta.",
    ),
    "vulkan": BackendInfo(
        key="vulkan",
        label="Vulkan (RTX 3050 / Intel / AMD)",
        dll="ggml-vulkan.dll",
        marker="vulkan",
        dirs=["/home/user/Downloads/llama-b11259-bin-win-vulkan-x64"],
        device_flag="Vulkan0",
        description="Build Vulkan verificata: ggml-vulkan.dll presente in directory.",
    ),
    "cpu": BackendInfo(
        key="cpu",
        label="CPU (nessun offload GPU)",
        dll="",
        marker="cpu",
        dirs=[
            "/home/user/Downloads/llama-b11259-bin-win-cpu-x64",
        ],
        device_flag="none",
        description="Ogni build llama.cpp gira su CPU: basta -ngl 0 / --device none.",
    ),
    "rpc": BackendInfo(
        key="rpc",
        label="RPC (tensori su Miky 192.168.1.100)",
        dll="ggml-rpc.dll",
        marker="cuda",
        dirs=[
            "/home/user/Downloads/llama-b11259-bin-win-cuda-13.4-x64",
        ],
        device_flag="CUDA0",
        description="Client RPC: il client gira in locale, i tensori su Miky via --rpc.",
    ),
}


def parse_build_number(text: str) -> int:
    """
    Estrae bNNNNN da un nome di directory o da un tag tipo "b11259".
    Ritorna 0 se non c'e' nessun numero.
    """
    m = re.search(r"b(\d{4,6})", str(text or ""), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _resolve_backend(info: BackendInfo) -> BackendInfo:
    """Cerca il binario nei candidati. Solo os.path: nessun processo avviato."""
    info.exists = False
    info.binary = ""
    info.build = 0
    dirs = list(info.dirs) + _candidate_dirs(info.marker, info.dirs)
    seen = set()
    for d in dirs:
        d = os.path.normpath(d)
        if d in seen:
            continue
        seen.add(d)
        exe = os.path.join(d, "llama-server")
        if not (os.path.isfile(exe) and os.path.isdir(d)):
            continue
        # Per i backend GPU la libreria di backend e' la prova che sia
        # quello. Il nome cambia per sistema operativo: "ggml-cuda.dll" su
        # Windows, "libggml-cuda.so" su Linux.
        if info.dll:
            alt = ("lib" + info.dll[:-4] + ".so") if info.dll.endswith(".dll") else info.dll
            if not (os.path.isfile(os.path.join(d, info.dll))
                    or os.path.isfile(os.path.join(d, alt))):
                continue
        # Si tiene la build col numero PIU' ALTO, non la prima trovata:
        # os.listdir e' in ordine alfabetico e "b9999" precede "b11435".
        n = parse_build_number(d)
        if not info.exists or n > info.build:
            info.binary = exe
            info.build = n
            info.exists = True
    return info


def list_backends(extra_dirs: Optional[Dict[str, List[str]]] = None) -> Dict[str, BackendInfo]:
    """Risolvi tutti i backend noti. Verifica solo l'esistenza dei file."""
    for key, info in BACKENDS.items():
        _resolve_backend(info)
    for key, dirs in (extra_dirs or {}).items():
        if key in BACKENDS:
            for d in dirs:
                exe = os.path.join(d, "llama-server")
                if os.path.isfile(exe):
                    info = BACKENDS[key]
                    info.binary = exe
                    info.build = parse_build_number(d)
                    info.exists = True
                    info.dirs.insert(0, d)
                    break
    return BACKENDS


def detect_backend(key: str, extra_dirs: Optional[List[str]] = None) -> BackendInfo:
    """
    Restituisce il BackendInfo richiesto, con binary/build risolti.
    Per 'cpu' il fallback e' il binario CUDA: qualunque build gira su CPU.
    """
    info = BACKENDS.get(key) or BACKENDS["cuda"]
    _resolve_backend(info)
    if extra_dirs:
        for d in extra_dirs:
            exe = os.path.join(d, "llama-server")
            if os.path.isfile(exe):
                info.binary = exe
                info.build = parse_build_number(d)
                info.exists = True
                break
    if not info.exists and key in ("cpu", "rpc"):
        cuda = BACKENDS["cuda"]
        _resolve_backend(cuda)
        if cuda.exists:
            info.binary = cuda.binary
            info.build = cuda.build
            info.exists = True
    return info


def local_build_numbers() -> List[Tuple[str, int, str]]:
    """
    Tutte le build trovate in locale: [(backend, build, directory)].
    Serve al confronto con l'ultima release GitHub.
    """
    out: List[Tuple[str, int, str]] = []
    for key, info in list_backends().items():
        if info.exists and info.build:
            out.append((key, info.build, os.path.dirname(info.binary)))
    return out


# =====================================================================
# 3. ESCAPING DEI PATH E DEGLI ARGOMENTI (Windows)
# =====================================================================

def quote_arg(arg: str) -> str:
    """
    Applica le regole di quoting di cmd.exe/PowerShell a un singolo argomento.

    - Path con spazi (C:/Users/User/My Models/x.gguf) -> "..." con doppi apici.
    - Virgolette embedded -> raddoppiate, che e' la regola di cmd.exe
      (``\"`` vale solo in POSIX/bash e romperebbe sotto cmd.exe).
    - Stringa vuota -> "" (altrimenti sparisce dalla riga di comando).
    """
    s = "" if arg is None else str(arg)
    if s == "":
        return '""'
    if not re.search(r'[\s"^&|<>()]', s):
        return s
    escaped = s.replace('"', '""')
    return f'"{escaped}"'


def quote_argv(argv: List[str]) -> str:
    """Versione stringa della riga di comando, pronta da copiare."""
    return " ".join(quote_arg(a) for a in argv)


# =====================================================================
# 4. TENSOR-SPLIT: validazione dei casi limite
# =====================================================================

def normalize_tensor_split(value: Any) -> List[float]:
    """
    Normalizza --tensor-split in una lista di frazioni.

    Solleva ValueError sui casi non sensati cosi' la TUI possa mostrarlo:
    stringa vuota, valori non numerici, frazione negativa, piu' di un
    separatore non consentito. Lo zero e' LEGITTIMO (GPU esclusa dallo split).
    """
    if value is None:
        raise ValueError("tensor-split vuoto")
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [p.strip() for p in str(value).split(",")]
    if not parts or all(p == "" for p in parts):
        raise ValueError("tensor-split vuoto")
    if len(parts) > 16:
        raise ValueError(f"troppe GPU nello split: {len(parts)} (max 16)")
    out: List[float] = []
    for p in parts:
        if p == "":
            raise ValueError(f"elemento vuoto in tensor-split '{value}'")
        if not re.match(r"^-?\d+(\.\d+)?$", p):
            raise ValueError(f"valore non numerico in tensor-split: '{p}'")
        f = float(p)
        if f < 0:
            raise ValueError(f"frazione negativa in tensor-split: '{p}'")
        out.append(f)
    if not any(f > 0 for f in out):
        raise ValueError("tensor-split tutto zeri: niente verrebbe offloadato")
    return out


def tensor_split_ok(value: Any, n_devices: Optional[int] = None) -> Tuple[bool, str]:
    """(ok, motivo) — comodo per la TUI e per i test."""
    try:
        fr = normalize_tensor_split(value)
    except ValueError as e:
        return False, str(e)
    if n_devices is not None and len(fr) != n_devices:
        return False, (f"{len(fr)} frazioni per {n_devices} device: "
                       f"una frazione per GPU")
    if max(fr) == 0:
        return False, "tutto zero"
    return True, f"{len(fr)} device, somma {sum(fr):g}"


# =====================================================================
# 5. SPIEGAZIONI DEI TIPI DI KV QUANT
# =====================================================================

KV_TYPE_INFO: Dict[str, Dict[str, str]] = {
    "f32": {
        "label": "F32 (32 bit float)",
        "detail": "Massima fedelta' assoluta, 4 byte per elemento. Raddoppia la "
                  "memoria della KV rispetto a f16. Praticamente inutile.",
        "use": "solo diagnostica / confronto",
    },
    "f16": {
        "label": "F16 (16 bit float, default)",
        "detail": "Il default di llama.cpp: nessuna perdita percepibile, "
                  "compatibile con Flash Attention su tutte le build.",
        "use": "qualita' massima, contesti medio-brevi",
    },
    "bf16": {
        "label": "BF16 (bfloat 16 bit)",
        "detail": "Stessa ampiezza di f16 ma con lo stesso esponente di f32: "
                  "robusto su hardware senza supporto fp16 nativo.",
        "use": "CPU, ARM, Vulkan su GPU recenti",
    },
    "q8_0": {
        "label": "Q8_0 (8 bit, blocchi da 32)",
        "detail": "Quantizzazione per blocchi, errore minimo: il 90% della "
                  "precisione di f16 con meta' della memoria. Nessun warning "
                  "di Flash Attention.",
        "use": "compromesso consigliato quando q4 non regge la qualita'",
    },
    "q6_k": {
        "label": "Q6_K (6 bit k-quant)",
        "detail": "Superblocco k-quant, ottima qualita' su larga parte dei "
                  "layer. Molto vicino a f16.",
        "use": "KV 6 bit, buon compromesso qualita'/memoria",
    },
    "q5_0": {
        "label": "Q5_0 (5 bit, blocchi da 32)",
        "detail": "Leggermente sotto q8_0, circa il 60% della memoria di f16. "
                  "Alcune build richiedono Flash Attention attivo.",
        "use": "contesti lunghi con qualita' accettabile",
    },
    "q5_1": {
        "label": "Q5_1 (5 bit, blocchi da 32, con minimo)",
        "detail": "Variante di q5_0 che conserva il minimo per blocco: piu' "
                  "stabile di q5_0, margine di errore simile.",
        "use": "alternativa a q5_0",
    },
    "q5_K": {
        "label": "Q5_K (5 bit k-quant)",
        "detail": "k-quant a 5 bit: qualita' migliore di q5_0 a parita' di "
                  "memoria, sufficiente per prompt lunghi.",
        "use": "KV 5 bit k-quant",
    },
    "q4_1": {
        "label": "Q4_1 (4 bit, blocchi da 32)",
        "detail": "Il preset RACCOMANDATO misurato su cluster dual-node: 32.32 tok/s a "
                  "128k di contesto, errore percettibile solo su task molto "
                  "delicati. Richiede Flash Attention ON.",
        "use": "contesti lunghissimi su VRAM limitata",
    },
    "q4_0": {
        "label": "Q4_0 (4 bit, blocchi da 32)",
        "detail": "Stessa memoria di q4_1, scala leggermente peggiore. "
                  "Flash Attention obbligatorio.",
        "use": "massimo contesto possibile",
    },
    "q3_K": {
        "label": "Q3_K (3 bit k-quant)",
        "detail": "3 bit k-quant: qualita' iniziale degradata, utile solo per "
                  " KV molto grandi su GPU piccole.",
        "use": "ultimo ricorso, contesti estremi",
    },
    "iq4_nl": {
        "label": "IQ4_NL (4 bit non lineare)",
        "detail": "Quantizzazione a importance matrix (i-quants): il 4 bit con "
                  "la migliore qualita' percepita, pensata per i pesi ma "
                  "usabile anche sulla KV.",
        "use": "4 bit di qualita'",
    },
    "iq4_xs": {
        "label": "IQ4_XS (4 bit XS)",
        "detail": "Variante i-quant a 4 bit con blocchi da 256, ottima "
                  "compressione. Non sempre accettata per la cache K.",
        "use": "4 bit i-quant",
    },
    "iq5_1": {
        "label": "IQ5_1 (5 bit i-quant)",
        "detail": "5 bit con importance matrix: quasi f16, memoria ridotta "
                  "circa del 65%.",
        "use": "5 bit di qualita' con poca memoria",
    },
    "iq3_xxs": {
        "label": "IQ3_XXS (3 bit i-quant)",
        "detail": "3 bit estremo, qualita' heavily compromessa.",
        "use": "solo per test",
    },
    "iq2_xxs": {
        "label": "IQ2_XXS (2 bit i-quant)",
        "detail": "2 bit: inadeguato per un server di produzione.",
        "use": "solo per test",
    },
    "bf16_1": {
        "label": "BF16 (1 byte di ricostruzione)",
        "detail": "Variante compressa usata internamente da alcune build.",
        "use": "specifico della build",
    },
    "q8_1": {
        "label": "Q8_1 (8 bit con minimo)",
        "detail": "Variante di q8_0 con minimo per blocco.",
        "use": "variante q8",
    },
    "q4_K": {
        "label": "Q4_K (4 bit k-quant)",
        "detail": "k-quant a 4 bit: qualita' migliore di q4_0 con la stessa "
                  "memoria approssimativa.",
        "use": "4 bit k-quant",
    },
    "q2_K": {
        "label": "Q2_K (2 bit k-quant)",
        "detail": "2 bit k-quant: qualita' insufficiente per uso reale.",
        "use": "solo confronto",
    },
}


# =====================================================================
# 6. CONFIGURAZIONE COMPLETA E GENERAZIONE DEL COMANDO (DRY-RUN)
# =====================================================================

@dataclass
class ServerConfig:
    """
    Tutti i parametri che la TUI lascia modificare. I valori None significano
    "non passare il flag" e lasciano il default di llama-server.
    """

    # modello / rete
    model: str = ""
    host: str = "localhost"
    port: int = 8087

    # contesto e slot
    ctx_size: Optional[int] = 131072
    parallel: Optional[int] = 1
    n_predict: Optional[int] = None
    batch: Optional[int] = None
    ubatch: Optional[int] = None
    kv_unified_per_slot: Optional[int] = None

    # GPU / offload
    n_gpu_layers: Optional[Any] = 99            # int | "auto" | "all"
    # None = deriva dal backend scelto (vedi effective_device).  Impostarlo a
    # mano sovrascrive la derivazione, utile per build multi-GPU esplicite.
    device: Optional[str] = None
    split_mode: Optional[str] = None            # none|layer|row|tensor
    tensor_split: Optional[str] = None          # "3,1"
    main_gpu: Optional[int] = None
    fit: Optional[str] = None                   # on|off
    fit_target: Optional[str] = None
    no_kv_offload: bool = False
    no_host: bool = False
    no_warmup: bool = False
    no_perf: bool = False

    # MoE su CPU
    cpu_moe: bool = False
    n_cpu_moe: Optional[int] = None
    n_cpu_ffn: Optional[int] = None

    # KV cache
    cache_type_k: Optional[str] = "q4_1"
    cache_type_v: Optional[str] = "q4_1"
    flash_attn: Optional[str] = "on"            # on|off|auto
    cont_batching: Optional[bool] = True
    cache_prompt: Optional[bool] = None
    cache_reuse: Optional[int] = None
    defrag_thold: Optional[float] = None

    # CPU / threading
    threads: Optional[int] = None
    threads_batch: Optional[int] = None
    threads_http: Optional[int] = None
    poll: Optional[int] = None
    prio: Optional[int] = None
    cpu_range: Optional[str] = None
    cpu_mask: Optional[str] = None
    numa: Optional[str] = None
    mlock: bool = False
    mmap: bool = True
    load_mode: Optional[str] = None            # auto|mmap|mlock|mmap+mlock|none|dio
    lazy_mode: Optional[str] = None
    ctx_shift: bool = False

    # sampling / chat
    temp: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_n_sigma: Optional[float] = None         # -1 = disabilitato
    typical_p: Optional[float] = None           # 1.0 = disabilitato
    min_p: Optional[float] = None               # 0.0 = disabilitato
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None                  # -1 = seed casuale
    jinja: Optional[bool] = True
    reasoning: Optional[str] = None            # on|off|auto
    reasoning_effort: Optional[str] = None
    reasoning_budget: Optional[int] = None
    reasoning_format: Optional[str] = None
    reasoning_preserve: Optional[bool] = None
    chat_template: Optional[str] = None
    chat_template_kwargs: Optional[str] = None

    # server extra
    timeout: Optional[int] = None
    sleep_idle_seconds: Optional[int] = None
    sse_ping_interval: Optional[int] = None
    slot_prompt_similarity: Optional[float] = None
    slots: Optional[bool] = None
    metrics: bool = False
    webui: Optional[bool] = None
    verbosity: Optional[int] = None

    # backend
    backend: str = "cuda"
    rpc: Optional[str] = None                   # "192.168.1.100:50052"

    # speculative decoding / draft
    spec_draft_model: Optional[str] = None
    spec_draft_n_max: Optional[int] = None
    spec_draft_n_min: Optional[int] = None
    spec_draft_ngl: Optional[int] = None
    spec_draft_device: Optional[str] = None
    spec_draft_threads: Optional[int] = None
    spec_draft_type_k: Optional[str] = None
    spec_draft_type_v: Optional[str] = None
    spec_draft_n_cpu_moe: Optional[int] = None
    spec_draft_p_min: Optional[float] = None
    spec_draft_p_split: Optional[float] = None
    spec_draft_cpu_moe: bool = False
    spec_draft_backend_sampling: Optional[bool] = None
    spec_draft_hf: Optional[str] = None

    # flags liberi (--qualcosa senza chiave dedicata) inseriti a mano
    extra_args: List[str] = field(default_factory=list)
    # override espliciti per chiavi non mappate: {"--jitter": "5"}
    extra_flags: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Risoluzione del backend: il campo `backend` e' la sorgente di verita'
    # per device/n_gpu_layers/rpc, cosi' non resta mai un --device CUDA0
    # "dimenticato" quando si passa a Vulkan o CPU.
    # -----------------------------------------------------------------
    def effective_device(self) -> Optional[str]:
        """Il valore di --device da usare, None per non passarlo."""
        if self.device not in (None, ""):
            return self.device
        return {"cuda": "CUDA0", "vulkan": "Vulkan0", "cpu": None,
                "rpc": "CUDA0"}.get(self.backend, None)

    def effective_ngl(self) -> Any:
        """Il valore di -ngl: su CPU e' sempre 0 (forza il massimo offload)."""
        if self.backend == "cpu":
            return 0
        return self.n_gpu_layers

    # -----------------------------------------------------------------
    # (A) Generazione del comando — DRY-RUN PURAMENTE
    # -----------------------------------------------------------------
    def build_command(
        self,
        binary_path: str,
        known_flags: Optional[Dict[str, FlagSpec]] = None,
    ) -> Dict[str, Any]:
        """
        Costruisce l'argv di llama-server. **Non esegue nulla.**

        Se ``known_flags`` e' il dizionario prodotto da ``parse_help`` (help
        della build locale), i flag che quella build non espone NON vengono
        generati e finiscono in ``unsupported``: e' cosi' che il comando resta
        valido anche se l'utente aggiorna o declassa llama.cpp.
        Se ``known_flags`` e' None si genera tutto (utile per i test).
        """
        argv: List[str] = [binary_path]
        unsupported: List[str] = []
        notes: List[str] = []

        def known(name: str) -> bool:
            if known_flags is None:
                return True
            if name in known_flags:
                return True
            for spec in known_flags.values():
                if spec.matches(name):
                    return True
            return False

        def add(name: str, value: Any = None) -> None:
            """Aggiunge un flag, verificando che la build lo esponga."""
            if not known(name):
                unsupported.append(name)
                return
            if value is None or value is True:
                argv.append(name)
            elif value is False:
                return
            else:
                argv.extend([name, str(value)])

        # --- modello (sempre per primo: e' il posizionale obbligatorio) ---
        if self.model:
            add("-m", self.model)
        elif known_flags is not None and "-m" in known_flags:
            notes.append("nessun modello selezionato: aggiungi -m <modello.gguf>")

        # --- rete ---
        if self.host:
            add("--host", self.host)
        if self.port:
            add("--port", self.port)
        if self.webui is False:
            add("--no-webui")
        if self.verbosity is not None:
            add("--verbosity", self.verbosity)

        # --- contesto e slot ---
        if self.ctx_size is not None:
            add("-c", self.ctx_size)
        if self.parallel is not None:
            add("-np", self.parallel)
        if self.n_predict is not None:
            add("-n", self.n_predict)
        if self.batch is not None:
            add("-b", self.batch)
        if self.ubatch is not None:
            add("-ub", self.ubatch)
        if self.kv_unified_per_slot is not None:
            add("--kv-unified-per-slot", self.kv_unified_per_slot)
        if self.ctx_shift:
            add("--context-shift")

        # --- GPU / offload ---
        ngl = self.effective_ngl()
        if ngl is not None:
            add("-ngl", ngl)
        dev = self.effective_device()
        if dev:
            add("--device", dev)
        if self.split_mode:
            add("--split-mode", self.split_mode)
        if self.tensor_split:
            ok, why = tensor_split_ok(self.tensor_split)
            if ok:
                add("--tensor-split", str(self.tensor_split).replace(" ", ""))
                if self.backend == "cpu":
                    notes.append("--tensor-split su backend CPU: le frazioni "
                                 "verranno ignorate (nessun device GPU)")
            else:
                unsupported.append(f"--tensor-split ({why})")
        if self.main_gpu is not None:
            add("--main-gpu", self.main_gpu)
        if self.fit:
            add("--fit", self.fit)
        if self.fit_target:
            add("--fit-target", self.fit_target)
        if self.no_kv_offload:
            add("--no-kv-offload")
        if self.no_host:
            add("--no-host")
        if self.no_warmup:
            add("--no-warmup")
        if self.no_perf:
            add("--no-perf")
        if self.rpc and self.backend == "rpc":
            add("--rpc", self.rpc)
        elif self.rpc and self.backend != "rpc":
            notes.append(f"--rpc ({self.rpc}) impostato ma il backend e' "
                         f"'{self.backend}': il flag verra' omesso")

        # --- MoE su CPU ---
        if self.cpu_moe:
            add("--cpu-moe")
        if self.n_cpu_moe is not None:
            add("-ncmoe", self.n_cpu_moe)
        if self.n_cpu_ffn is not None:
            add("-ncffn", self.n_cpu_ffn)

        # --- KV cache ---
        if self.cache_type_k:
            add("-ctk", self.cache_type_k)
        if self.cache_type_v:
            add("-ctv", self.cache_type_v)
        if self.flash_attn:
            add("-fa", self.flash_attn)
        if self.cont_batching is not None:
            add("-cb" if self.cont_batching else "--no-cont-batching")
        if self.cache_prompt is not None:
            add("--cache-prompt" if self.cache_prompt else "--no-cache-prompt")
        if self.cache_reuse is not None:
            add("--cache-reuse", self.cache_reuse)
        if self.defrag_thold is not None:
            add("-dt", self.defrag_thold)

        # --- CPU / threading ---
        if self.threads is not None:
            add("-t", self.threads)
        if self.threads_batch is not None:
            add("-tb", self.threads_batch)
        if self.threads_http is not None:
            add("--threads-http", self.threads_http)
        if self.poll is not None:
            add("--poll", self.poll)
        if self.prio is not None:
            add("--prio", self.prio)
        if self.cpu_range:
            add("--cpu-range", self.cpu_range)
        if self.cpu_mask:
            add("--cpu-mask", self.cpu_mask)
        if self.numa:
            add("--numa", self.numa)
        # mmap/mlock: dalla build b11259 in poi NON esistono piu' come flag
        # standalone, sono valori di --load-mode.  Si emettono i flag vecchi
        # solo se la build li espone davvero, altrimenti si traduce in
        # --load-mode, cosi' l'opzione non sparisce silenziosamente.
        load_mode = self.load_mode
        # Ogni sotto-flag va verificato: una build vecchia puo' avere solo
        # --mlock e non --no-mmap (o viceversa).
        if self.mlock and known("--mlock"):
            add("--mlock")
        if (not self.mmap) and known("--no-mmap"):
            add("--no-mmap")
        if load_mode and known("--load-mode"):
            add("--load-mode", load_mode)
        elif not known("--mlock") and not known("--no-mmap"):
            # Build nuova (b11259+): mmap/mlock sono VALORI di --load-mode.
            if load_mode is None:
                if self.mlock and self.mmap:
                    load_mode = "mmap+mlock"
                elif self.mlock:
                    load_mode = "mlock"
                elif not self.mmap:
                    load_mode = "none"
            if load_mode:
                add("--load-mode", load_mode)
        if self.lazy_mode:
            add("--lazy-mode", self.lazy_mode)

        # --- sampling / chat ---
        if self.temp is not None:
            add("--temp", self.temp)
        if self.top_k is not None:
            add("--top-k", self.top_k)
        if self.top_p is not None:
            add("--top-p", self.top_p)
        if self.top_n_sigma is not None:
            add("--top-n-sigma", self.top_n_sigma)
        if self.typical_p is not None:
            add("--typical-p", self.typical_p)
        if self.min_p is not None:
            add("--min-p", self.min_p)
        if self.repeat_penalty is not None:
            add("--repeat-penalty", self.repeat_penalty)
        if self.presence_penalty is not None:
            add("--presence-penalty", self.presence_penalty)
        if self.frequency_penalty is not None:
            add("--frequency-penalty", self.frequency_penalty)
        if self.seed is not None:
            add("--seed", self.seed)
        if self.jinja is not None:
            add("--jinja" if self.jinja else "--no-jinja")
        if self.reasoning:
            add("--reasoning", self.reasoning)
        if self.reasoning_effort:
            add("--reasoning-effort", self.reasoning_effort)
        if self.reasoning_budget is not None:
            add("--reasoning-budget", self.reasoning_budget)
        if self.reasoning_format:
            add("--reasoning-format", self.reasoning_format)
        if self.reasoning_preserve is not None:
            add("--reasoning-preserve" if self.reasoning_preserve
                else "--no-reasoning-preserve")
        if self.chat_template:
            add("--chat-template", self.chat_template)
        if self.chat_template_kwargs:
            add("--chat-template-kwargs", self.chat_template_kwargs)

        # --- server extra ---
        if self.timeout is not None:
            add("-to", self.timeout)
        if self.sleep_idle_seconds is not None:
            add("--sleep-idle-seconds", self.sleep_idle_seconds)
        if self.sse_ping_interval is not None:
            add("--sse-ping-interval", self.sse_ping_interval)
        if self.slot_prompt_similarity is not None:
            add("-sps", self.slot_prompt_similarity)
        if self.slots is not None:
            add("--slots" if self.slots else "--no-slots")
        if self.metrics:
            add("--metrics")

        # --- draft / speculative decoding ---
        if self.spec_draft_model:
            add("--spec-draft-model", self.spec_draft_model)
        if self.spec_draft_hf:
            add("--spec-draft-hf", self.spec_draft_hf)
        if self.spec_draft_n_max is not None:
            add("--spec-draft-n-max", self.spec_draft_n_max)
        if self.spec_draft_n_min is not None:
            add("--spec-draft-n-min", self.spec_draft_n_min)
        if self.spec_draft_ngl is not None:
            add("--spec-draft-ngl", self.spec_draft_ngl)
        if self.spec_draft_device:
            add("--spec-draft-device", self.spec_draft_device)
        if self.spec_draft_threads is not None:
            add("--spec-draft-threads", self.spec_draft_threads)
        if self.spec_draft_type_k:
            add("--spec-draft-type-k", self.spec_draft_type_k)
        if self.spec_draft_type_v:
            add("--spec-draft-type-v", self.spec_draft_type_v)
        if self.spec_draft_n_cpu_moe is not None:
            add("--spec-draft-n-cpu-moe", self.spec_draft_n_cpu_moe)
        if self.spec_draft_p_min is not None:
            add("--spec-draft-p-min", self.spec_draft_p_min)
        if self.spec_draft_p_split is not None:
            add("--spec-draft-p-split", self.spec_draft_p_split)
        if self.spec_draft_cpu_moe:
            add("--spec-draft-cpu-moe")
        if self.spec_draft_backend_sampling is not None:
            add("--spec-draft-backend-sampling" if self.spec_draft_backend_sampling
                else "--no-spec-draft-backend-sampling")

        # --- flag liberi ---
        for k, v in (self.extra_flags or {}).items():
            add(k, v)
        for a in (self.extra_args or []):
            argv.append(str(a))

        return {
            "argv": argv,
            "cmd": quote_argv(argv),
            "unsupported": sorted(set(unsupported)),
            "notes": notes,
            "backend": self.backend,
            "device": self.effective_device(),
            "n_gpu_layers": self.effective_ngl(),
            "tensor_split": (
                normalize_tensor_split(self.tensor_split)
                if self.tensor_split and tensor_split_ok(self.tensor_split)[0] else None
            ),
        }

    # -----------------------------------------------------------------
    def dry_run(self, binary_path: str,
                known_flags: Optional[Dict[str, FlagSpec]] = None) -> str:
        """La stringa del comando, senza eseguire NULLA. Usata da [T] e dai test."""
        return self.build_command(binary_path, known_flags)["cmd"]

    # -----------------------------------------------------------------
    def estimate_kv_mib(self) -> float:
        """
        Stima grezza della KV cache in MiB per capire se il contesto entra.
        Serve solo a dare un'idea in pannello: il calcolo vero dipende da
        n_layer, n_head_kv e head_dim del modello, che qui non leggiamo.
        """
        bits = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 8, "q8_1": 8, "q5_0": 5,
                "q5_1": 5, "q4_0": 4, "q4_1": 4, "q5_K": 5, "q4_K": 4,
                "q6_k": 6, "q3_K": 3, "q2_K": 2, "iq4_nl": 4, "iq4_xs": 4,
                "iq5_1": 5, "iq3_xxs": 3, "iq2_xxs": 2}.get(self.cache_type_k or "f16", 16)
        bitsv = {"f32": 32, "f16": 16, "bf16": 16, "q8_0": 8, "q8_1": 8, "q5_0": 5,
                 "q5_1": 5, "q4_0": 4, "q4_1": 4, "q5_K": 5, "q4_K": 4,
                 "q6_k": 6, "q3_K": 3, "q2_K": 2, "iq4_nl": 4, "iq4_xs": 4,
                 "iq5_1": 5, "iq3_xxs": 3, "iq2_xxs": 2}.get(self.cache_type_v or "f16", 16)
        ctx = int(self.ctx_size or 0)
        parallel = max(1, int(self.parallel or 1))
        # 8 KV head a 128 dim per layer e' una stima media per i modelli 7-35B.
        k_dim, v_dim, n_layer = 1024, 1024, 32
        per_tok = (k_dim * bits + v_dim * bitsv) / 8
        total = per_tok * ctx * parallel * n_layer
        return total / (1024.0 * 1024.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerConfig":
        """
        Costruisce da un dizionario JSON, scartando le chiavi sconosciute
        (profilo creato da una versione futura della TUI).
        """
        if not isinstance(data, dict):
            return cls()
        payload = data.get("config", data)
        if not isinstance(payload, dict):
            payload = {}
        known_fields = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in payload.items() if k in known_fields}
        if isinstance(clean.get("extra_args"), str):
            clean["extra_args"] = [clean["extra_args"]]
        return cls(**clean)


# Mappa chiave logica -> flag canonico nell'help (per la UI e per i test).
COMMAND_KEYS: Dict[str, str] = {
    "ctx_size": "--ctx-size",
    "n_gpu_layers": "--gpu-layers",
    "parallel": "--parallel",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "flash_attn": "--flash-attn",
    "split_mode": "--split-mode",
    "tensor_split": "--tensor-split",
    "main_gpu": "--main-gpu",
    "cpu_moe": "--cpu-moe",
    "n_cpu_moe": "--n-cpu-moe",
    "threads": "--threads",
    "batch": "--batch-size",
    "ubatch": "--ubatch-size",
    "mlock": "--mlock",
    "mmap": "--mmap",
    "cont_batching": "--cont-batching",
    "reasoning_budget": "--reasoning-budget",
    "reasoning_format": "--reasoning-format",
    "reasoning_effort": "--reasoning-effort",
    "spec_draft_model": "--spec-draft-model",
    "spec_draft_type_k": "--spec-draft-type-k",
    "spec_draft_type_v": "--spec-draft-type-v",
    "port": "--port",
    "host": "--host",
    "rpc": "--rpc",
    "device": "--device",
    "load_mode": "--load-mode",
    "fit": "--fit",
    "fit_target": "--fit-target",
}


# =====================================================================
# 6b. FIELD_SPEC: come i campi sono disposti nel pannello
# =====================================================================
# Ogni voce descrive UNA riga del Tab 10:
#   key          identificativo usato dagli hotkey e dai test
#   label        testo mostrato a sinistra
#   attr         attributo di ServerConfig che il campo modifica
#   flag         nome canonico del flag, per interrogare l'help
#   type         bool | int | enum | text  (decide il tipo di hotkey)
#   group        sezione del pannello
#   step/min/max solo per type=int: passo e limiti di [LEFT]/[RIGHT]
#
# NON e' la lista dei flag: quella arriva da parse_help().  Qui ci mette
# solo quello che la UI sa fare con le frecce, quindi i campi navigabili.
# Tutto il resto si vede comunque con [A] e si accetta via extra_args.

FIELD_SPEC: List[Dict[str, Any]] = [
    # --- Rete / servizio ---
    {"key": "host", "label": "Host", "attr": "host", "flag": "--host",
     "type": "text", "group": "Rete / servizio"},
    {"key": "port", "label": "Porta", "attr": "port", "flag": "--port",
     "type": "int", "step": 1, "min": 1, "max": 65535,
     "group": "Rete / servizio"},
    {"key": "parallel", "label": "Slot paralleli (-np)", "attr": "parallel",
     "flag": "--parallel", "type": "int", "step": 1, "min": 1, "max": 512,
     "group": "Rete / servizio"},
    {"key": "cont_batching", "label": "Continuous batching", "attr": "cont_batching",
     "flag": "--cont-batching", "type": "bool", "group": "Rete / servizio"},
    {"key": "poll", "label": "Polling (-p)", "attr": "poll", "flag": "--poll",
     "type": "int", "step": 10, "min": 0, "max": 100, "group": "Rete / servizio"},
    {"key": "timeout", "label": "Timeout (-to)", "attr": "timeout",
     "flag": "--timeout", "type": "int", "step": 10, "min": 0, "max": 100000,
     "group": "Rete / servizio"},
    {"key": "sleep_idle_seconds", "label": "Sleep idle (s)",
     "attr": "sleep_idle_seconds",
     "flag": "--sleep-idle-seconds", "type": "int", "step": 30, "min": 0,
     "max": 86400, "group": "Rete / servizio"},

    # --- Contesto e cache KV ---
    {"key": "ctx_size", "label": "Context size (-c)", "attr": "ctx_size",
     "flag": "--ctx-size", "type": "int", "step": 4096, "min": 0, "max": 2097152,
     "group": "Contesto e cache KV"},
    {"key": "cache_type_k", "label": "KV quant K (-ctk)", "attr": "cache_type_k",
     "flag": "--cache-type-k", "type": "enum", "group": "Contesto e cache KV"},
    {"key": "cache_type_v", "label": "KV quant V (-ctv)", "attr": "cache_type_v",
     "flag": "--cache-type-v", "type": "enum", "group": "Contesto e cache KV"},
    {"key": "flash_attn", "label": "Flash attention (-fa)", "attr": "flash_attn",
     "flag": "--flash-attn", "type": "enum", "group": "Contesto e cache KV"},
    {"key": "no_kv_offload", "label": "KV su CPU (no offload)",
     "attr": "no_kv_offload", "flag": "--no-kv-offload", "type": "bool",
     "group": "Contesto e cache KV"},
    {"key": "cache_reuse", "label": "Cache reuse (-cr)", "attr": "cache_reuse",
     "flag": "--cache-reuse", "type": "int", "step": 256, "min": 0, "max": 1048576,
     "group": "Contesto e cache KV"},
    {"key": "defrag_thold", "label": "Defrag soglia (-dt)",
     "attr": "defrag_thold", "flag": "--defrag-thold", "type": "float",
     "step": 0.05, "group": "Contesto e cache KV"},
    {"key": "ctx_shift", "label": "Context shift", "attr": "ctx_shift",
     "flag": "--context-shift", "type": "bool", "group": "Contesto e cache KV"},

    # --- GPU / offload ---
    {"key": "n_gpu_layers", "label": "GPU layers (-ngl)", "attr": "n_gpu_layers",
     "flag": "--gpu-layers", "type": "text", "group": "GPU / offload"},
    {"key": "device", "label": "Device", "attr": "device", "flag": "--device",
     "type": "text", "group": "GPU / offload"},
    {"key": "split_mode", "label": "Split mode (-sm)", "attr": "split_mode",
     "flag": "--split-mode", "type": "enum", "group": "GPU / offload"},
    {"key": "tensor_split", "label": "Tensor split (-ts)", "attr": "tensor_split",
     "flag": "--tensor-split", "type": "text", "group": "GPU / offload"},
    {"key": "main_gpu", "label": "Main GPU (-mg)", "attr": "main_gpu",
     "flag": "--main-gpu", "type": "int", "step": 1, "min": 0, "max": 15,
     "group": "GPU / offload"},
    {"key": "fit", "label": "Fit auto", "attr": "fit", "flag": "--fit",
     "type": "enum", "group": "GPU / offload"},
    {"key": "fit_target", "label": "Fit target (-fitt)", "attr": "fit_target",
     "flag": "--fit-target", "type": "text", "group": "GPU / offload"},
    {"key": "rpc", "label": "RPC servers", "attr": "rpc", "flag": "--rpc",
     "type": "text", "group": "GPU / offload"},

    # --- MoE su CPU ---
    {"key": "cpu_moe", "label": "MoE su CPU (-cmoe)", "attr": "cpu_moe",
     "flag": "--cpu-moe", "type": "bool", "group": "MoE su CPU"},
    {"key": "n_cpu_moe", "label": "Layer MoE su CPU (-ncmoe)",
     "attr": "n_cpu_moe", "flag": "--n-cpu-moe", "type": "int", "step": 1,
     "min": 0, "max": 999, "group": "MoE su CPU"},
    {"key": "n_cpu_ffn", "label": "Layer FFN su CPU (-ncffn)", "attr": "n_cpu_ffn",
     "flag": "--n-cpu-ffn", "type": "int", "step": 1, "min": 0, "max": 999,
     "group": "MoE su CPU"},
    {"key": "cpu_range", "label": "CPU range (-cr)", "attr": "cpu_range",
     "flag": "--cpu-range", "type": "text", "group": "MoE su CPU"},
    {"key": "cpu_mask", "label": "CPU mask (-cm)", "attr": "cpu_mask",
     "flag": "--cpu-mask", "type": "text", "group": "MoE su CPU"},

    # --- CPU / threading ---
    {"key": "threads", "label": "Threads (-t)", "attr": "threads",
     "flag": "--threads", "type": "int", "step": 2, "min": 1, "max": 512,
     "group": "CPU / threading"},
    {"key": "threads_batch", "label": "Threads batch (-tb)", "attr": "threads_batch",
     "flag": "--threads-batch", "type": "int", "step": 2, "min": 1, "max": 512,
     "group": "CPU / threading"},
    {"key": "threads_http", "label": "Threads HTTP", "attr": "threads_http",
     "flag": "--threads-http", "type": "int", "step": 2, "min": 1, "max": 512,
     "group": "CPU / threading"},
    {"key": "batch", "label": "Batch size (-b)", "attr": "batch",
     "flag": "--batch-size", "type": "int", "step": 256, "min": 8, "max": 65536,
     "group": "CPU / threading"},
    {"key": "ubatch", "label": "Ubatch size (-ub)", "attr": "ubatch",
     "flag": "--ubatch-size", "type": "int", "step": 128, "min": 8, "max": 65536,
     "group": "CPU / threading"},
    {"key": "numa", "label": "NUMA", "attr": "numa", "flag": "--numa",
     "type": "enum", "group": "CPU / threading"},
    {"key": "mlock", "label": "mlock (forza in RAM)", "attr": "mlock",
     "flag": "--mlock", "type": "bool", "group": "CPU / threading"},
    {"key": "mmap", "label": "mmap (memory-map)", "attr": "mmap",
     "flag": "--mmap", "type": "bool", "group": "CPU / threading"},
    {"key": "load_mode", "label": "Load mode (-lm)", "attr": "load_mode",
     "flag": "--load-mode", "type": "enum", "group": "CPU / threading"},

    # --- Sampling / chat ---
    {"key": "temp", "label": "Temp", "attr": "temp", "flag": "--temp",
     "type": "float", "group": "Sampling / chat"},
    {"key": "top_k", "label": "Top-k", "attr": "top_k", "flag": "--top-k",
     "type": "int", "step": 1, "min": 0, "max": 1000, "group": "Sampling / chat"},
    {"key": "top_p", "label": "Top-p", "attr": "top_p", "flag": "--top-p",
     "type": "float", "group": "Sampling / chat"},
    {"key": "top_n_sigma", "label": "Top-n-sigma", "attr": "top_n_sigma",
     "flag": "--top-n-sigma", "type": "float", "group": "Sampling / chat"},
    {"key": "typical_p", "label": "Typical-p", "attr": "typical_p",
     "flag": "--typical-p", "type": "float", "group": "Sampling / chat"},
    {"key": "min_p", "label": "Min-p", "attr": "min_p", "flag": "--min-p",
     "type": "float", "group": "Sampling / chat"},
    {"key": "repeat_penalty", "label": "Repeat penalty", "attr": "repeat_penalty",
     "flag": "--repeat-penalty", "type": "float", "group": "Sampling / chat"},
    {"key": "presence_penalty", "label": "Presence penalty",
     "attr": "presence_penalty", "flag": "--presence-penalty", "type": "float",
     "group": "Sampling / chat"},
    {"key": "frequency_penalty", "label": "Frequency penalty",
     "attr": "frequency_penalty", "flag": "--frequency-penalty", "type": "float",
     "group": "Sampling / chat"},
    {"key": "n_predict", "label": "Max token (-n)", "attr": "n_predict",
     "flag": "--predict", "type": "int", "step": 256, "min": -1, "max": 1048576,
     "group": "Sampling / chat"},
    {"key": "seed", "label": "Seed", "attr": "seed", "flag": "--seed",
     "type": "int", "step": 1, "min": -1, "max": 4294967295,
     "group": "Sampling / chat"},
    {"key": "jinja", "label": "Chat template Jinja", "attr": "jinja",
     "flag": "--jinja", "type": "bool", "group": "Sampling / chat"},
    {"key": "chat_template", "label": "Chat template", "attr": "chat_template",
     "flag": "--chat-template", "type": "text", "group": "Sampling / chat"},

    # --- Ragionamento ---
    {"key": "reasoning_budget", "label": "Reasoning budget (-rb)",
     "attr": "reasoning_budget", "flag": "--reasoning-budget", "type": "int",
     "step": 256, "min": -1, "max": 1048576, "group": "Ragionamento"},
    {"key": "reasoning_format", "label": "Reasoning format (-rf)",
     "attr": "reasoning_format", "flag": "--reasoning-format", "type": "enum",
     "group": "Ragionamento"},
    # L'help la prende come LEVEL libero ("'default' to keep ...", altri livelli
    # dipendono dal template): non e' un enum chiuso, quindi resta testo.
    {"key": "reasoning_effort", "label": "Reasoning effort (-re)",
     "attr": "reasoning_effort", "flag": "--reasoning-effort", "type": "text",
     "group": "Ragionamento"},
    {"key": "reasoning", "label": "Reasoning on/off", "attr": "reasoning",
     "flag": "--reasoning", "type": "enum", "group": "Ragionamento"},
    {"key": "reasoning_preserve", "label": "Reasoning preserve",
     "attr": "reasoning_preserve", "flag": "--reasoning-preserve",
     "type": "bool", "group": "Ragionamento"},

    # --- Speculative decoding (draft) ---
    {"key": "spec_draft_model", "label": "Draft model (-md)",
     "attr": "spec_draft_model", "flag": "--spec-draft-model", "type": "text",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_n_max", "label": "Draft token (-nd)",
     "attr": "spec_draft_n_max", "flag": "--spec-draft-n-max", "type": "int",
     "step": 1, "min": 0, "max": 100, "group": "Speculative (draft)"},
    {"key": "spec_draft_n_min", "label": "Draft n-min (-nmin)",
     "attr": "spec_draft_n_min", "flag": "--spec-draft-n-min", "type": "int",
     "step": 1, "min": 0, "max": 100, "group": "Speculative (draft)"},
    {"key": "spec_draft_ngl", "label": "Draft GPU layers (-ngld)",
     "attr": "spec_draft_ngl", "flag": "--spec-draft-ngl", "type": "text",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_type_k", "label": "Draft cache K (-ctkd)",
     "attr": "spec_draft_type_k", "flag": "--spec-draft-type-k", "type": "enum",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_type_v", "label": "Draft cache V (-ctvd)",
     "attr": "spec_draft_type_v", "flag": "--spec-draft-type-v", "type": "enum",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_p_min", "label": "Draft p-min (-pmin)",
     "attr": "spec_draft_p_min", "flag": "--spec-draft-p-min", "type": "float",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_p_split", "label": "Draft p-split (-psplit)",
     "attr": "spec_draft_p_split", "flag": "--spec-draft-p-split", "type": "float",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_cpu_moe", "label": "Draft MoE su CPU",
     "attr": "spec_draft_cpu_moe", "flag": "--spec-draft-cpu-moe", "type": "bool",
     "group": "Speculative (draft)"},
    {"key": "spec_draft_n_cpu_moe", "label": "Draft n-cpu-moe",
     "attr": "spec_draft_n_cpu_moe", "flag": "--spec-draft-n-cpu-moe",
     "type": "int", "step": 1, "min": 0, "max": 999,
     "group": "Speculative (draft)"},
    {"key": "spec_draft_threads", "label": "Draft threads (-td)",
     "attr": "spec_draft_threads", "flag": "--spec-draft-threads",
     "type": "int", "step": 1, "min": 0, "max": 512,
     "group": "Speculative (draft)"},
]


# =====================================================================
# 7. PROFILI JSON: salvataggio, export, import
# =====================================================================

class ConfigProfileStore:
    """
    Profili di configurazione in un JSON unico (flag_profiles.json).

    Tre operazioni distinte, con formati diversi:
      * save()    -> scrivete nel profilo di lavoro
      * export()  -> dump di UN profilo in un file a scelta
      * import()  -> lettura di un file esterno (validato, mai cieco)
    """

    FILENAME = "flag_profiles.json"

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        # La directory puo' non esistere (percorso di import scelto a mano):
        # si crea qui, altrimenti il primo save() fallisce con ENOENT.
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except OSError:
            pass
        self.path = os.path.join(self.base_dir, self.FILENAME)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.last_error: str = ""
        self.load()

    # ---------------------------------------------------------------
    def load(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.path):
            self.profiles = {}
            return self.profiles
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("profili") or {}
            self.profiles = {k: v for k, v in raw.items() if isinstance(v, dict)}
            self.last_error = ""
        except Exception as e:
            self.profiles = {}
            self.last_error = f"{self.FILENAME} illeggibile: {e}"
        return self.profiles

    def save(self) -> bool:
        payload = {
            "_meta": "Profili di configurazione llama-server di ai_tui (Tab 10).",
            "_formato": 1,
            "profili": self.profiles,
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.last_error = ""
            return True
        except Exception as e:
            self.last_error = f"salvataggio fallito: {e}"
            return False

    # ---------------------------------------------------------------
    def save_profile(self, name: str, config: ServerConfig,
                     notes: str = "", binary: str = "") -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"success": False, "error": "Nome profilo vuoto"}
        self.profiles[name] = {
            "name": name,
            "binary": binary,
            "backend": config.backend,
            "notes": notes,
            "saved_at": None,          # riempito dal chiamante se serve
            "config": config.to_dict(),
        }
        ok = self.save()
        return {"success": ok, "name": name, "path": self.path,
                "error": self.last_error}

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.profiles.get(name)

    def load_config(self, name: str) -> Optional[ServerConfig]:
        p = self.get(name)
        if not p:
            return None
        return ServerConfig.from_dict(p)

    def delete(self, name: str) -> Dict[str, Any]:
        if name in self.profiles:
            self.profiles.pop(name)
            return {"success": self.save(), "error": self.last_error}
        return {"success": False, "error": f"'{name}' non e' un profilo salvato"}

    def names(self) -> List[str]:
        return sorted(self.profiles.keys())

    def all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Tutti i profili caricati, in copia (per non esporre lo stato)."""
        return {k: dict(v) for k, v in self.profiles.items()}

    # ---------------------------------------------------------------
    def export(self, name: str, dest_path: str) -> Dict[str, Any]:
        """Esporta un profilo in un JSON standalone, leggibile da altre macchine."""
        p = self.get(name)
        if not p:
            return {"success": False, "error": f"profilo '{name}' inesistente"}
        try:
            out = {
                "_meta": "Profilo llama-server esportato da ai_tui.",
                "_formato": 1,
                "profilo": p,
            }
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            return {"success": True, "path": dest_path,
                    "size": os.path.getsize(dest_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def import_file(self, src_path: str, name: str = "") -> Dict[str, Any]:
        """
        Importa un profilo da un JSON esterno. Non fidarsi del contenuto:
        ogni chiave viene validata con ServerConfig.from_dict.
        """
        if not os.path.isfile(src_path):
            return {"success": False, "error": f"file non trovato: {src_path}"}
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"success": False, "error": f"JSON non valido: {e}"}
        prof = data.get("profilo") if isinstance(data, dict) else None
        if not isinstance(prof, dict):
            prof = data if isinstance(data, dict) else None
        if not isinstance(prof, dict):
            return {"success": False, "error": "nessun profilo trovato nel file"}
        cfg = ServerConfig.from_dict(prof)
        nm = (name or prof.get("name") or "").strip()
        if not nm:
            return {"success": False, "error": "il profilo non ha un nome"}
        return self.save_profile(nm, cfg,
                                 notes=prof.get("notes", "importato"),
                                 binary=prof.get("binary", ""))

    def export_all(self, dest_path: str) -> Dict[str, Any]:
        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump({"_meta": "Tutti i profili ai_tui", "_formato": 1,
                           "profili": self.profiles}, f, indent=2,
                          ensure_ascii=False)
            return {"success": True, "path": dest_path}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =====================================================================
# 8. TASTO AGGIORNA: confronto build locale vs ultima release GitHub
# =====================================================================

GITHUB_LATEST_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
GITHUB_RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
GITHUB_ASSET_PATTERNS = [
    re.compile(r"llama-b(\d+)-bin-ubuntu-cuda.*\.tar\.gz$", re.IGNORECASE),
    re.compile(r"llama-b(\d+)-bin-ubuntu-.*\.tar\.gz$", re.IGNORECASE),
    re.compile(r"llama-b(\d+)-bin-linux-.*\.tar\.gz$", re.IGNORECASE),
    re.compile(r"llama-b(\d+)-bin-win-cuda.*\.zip$", re.IGNORECASE),
    re.compile(r"llama-b(\d+)-bin-win-vulkan.*\.zip$", re.IGNORECASE),
    re.compile(r"llama-b(\d+)-bin-win-cpu.*\.zip$", re.IGNORECASE),
    re.compile(r"llama-b(\d+)-bin-win-.*x64\.zip$", re.IGNORECASE),
]


def _first_build_release(timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    La release piu' recente taggata bNNNNN.

    Serve perche' /releases/latest restituisce anche release semantiche
    ("v0.6.0") che stanno in testa ma non hanno numero di build: da quelle
    non si ricava nulla da confrontare.
    """
    req = urllib.request.Request(
        GITHUB_RELEASES_URL,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "ai_tui-llama-config"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            items = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None
    if not isinstance(items, list):
        return None
    for r in items:
        if parse_build_number(str(r.get("tag_name") or "")):
            return r
    return None


def _local_cuda_version() -> str:
    """Versione CUDA usata in locale (es. "13.4"), letta dalla cartella cublas."""
    for base in ("/mnt/lm_models/DEBIAN_FOLDER/llamacpp/cublas",):
        try:
            for name in sorted(os.listdir(base), reverse=True):
                m = re.search(r"cuda-(\d+\.\d+)", name)
                if m:
                    return m.group(1)
        except OSError:
            continue
    return ""


def _pick_asset(assets: List[Dict[str, Any]], backend: str) -> Optional[Dict[str, Any]]:
    """
    Sceglie l'archivio adatto a QUESTA macchina.

    Scarta i pacchetti "cudart-": contengono solo il runtime CUDA, non
    llama-server, e pesano il quadruplo. Su Linux scarta Windows, macOS e le
    architetture diverse da x86-64. A parita' di condizioni preferisce la
    stessa versione CUDA gia' installata in locale.
    """
    linux = os.name != "nt"
    want = {"cuda": "cuda", "vulkan": "vulkan"}.get(backend, "")
    cuda_ver = _local_cuda_version() if want == "cuda" else ""
    best, best_score = None, -1
    for a in assets or []:
        name = str(a.get("name") or "").lower()
        if not a.get("url") or name.startswith("cudart-"):
            continue
        if linux:
            if "win" in name or "macos" in name:
                continue
            if "ubuntu" not in name and "linux" not in name:
                continue
        if "arm64" in name or "s390x" in name:
            continue
        if want:
            if want not in name:
                continue
        elif any(k in name for k in ("cuda", "vulkan", "rocm", "sycl", "openvino")):
            continue            # backend "cpu": si vuole l'archivio semplice
        score = 10
        if cuda_ver and f"cuda-{cuda_ver}" in name:
            score += 5
        if "x64" in name:
            score += 1
        if score > best_score:
            best, best_score = a, score
    return best


def fetch_latest_release(timeout: float = 25.0,
                         url: str = "") -> Dict[str, Any]:
    """
    Ritorna la release piu' recente di llama.cpp con binari Linux CUDA.
    Usa build_manager (con cache 6h) per evitare chiamate lente, timeout
    o tag semantici (v0.6.0) privi di build number.
    """
    try:
        import build_manager as _bm
        bm_rel = _bm.latest_release(use_cache=True)
        if bm_rel and not bm_rel.get("error") and bm_rel.get("build"):
            tag = str(bm_rel["tag"])
            build = int(bm_rel["build"])
            return {
                "ok": True,
                "tag": tag,
                "build": build,
                "published": str(bm_rel.get("published", "")),
                "url": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}",
                "assets": bm_rel.get("assets", []),
                "cached": True,
            }
        elif bm_rel and bm_rel.get("error"):
            return {"ok": False, "error": str(bm_rel["error"])}
    except Exception:
        pass

    cache_path = os.path.expanduser("~/.cache/ai_tui/llama_update.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                c = json.load(fh)
                if c and c.get("tag") and c.get("build"):
                    tag = str(c["tag"])
                    build = int(c["build"])
                    return {
                        "ok": True,
                        "tag": tag,
                        "build": build,
                        "published": str(c.get("published", "")),
                        "url": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}",
                        "assets": c.get("assets", []),
                        "cached": True,
                    }
        except Exception:
            pass

    return {"ok": False, "error": "Release non disponibile (timeout o errore GitHub API)"}


def check_update(backend: str = "cuda", timeout: float = 25.0,
                 offline_release: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """
    Confronta la build locale (bNNNNN dalla directory) con l'ultima release.

    Ritorna un report con: build locale, build remota, se c'e' un aggiornamento,
    gli asset candidati e — PRIMA di qualsiasi download — dimensione del file
    e spazio libero su disco. **Non scarica niente.**
    """
    info = detect_backend(backend)
    local_build = info.build or 0
    rel = offline_release if offline_release is not None else fetch_latest_release(timeout=timeout)
    if not rel.get("ok"):
        return {
            "ok": False, "backend": backend, "local_build": local_build,
            "error": rel.get("error", "release non disponibile"),
        }
    remote_build = int(rel.get("build") or 0)
    newer = remote_build > local_build

    candidates = [a for a in rel.get("assets", []) if a.get("url")]
    want = {"cuda": "cuda", "vulkan": "vulkan", "cpu": "cpu"}.get(backend, "")
    chosen = _pick_asset(candidates, backend)

    report: Dict[str, Any] = {
        "ok": True,
        "backend": backend,
        "local_build": local_build,
        "local_dir": os.path.dirname(info.binary) if info.binary else "",
        "local_binary": info.binary,
        "remote_build": remote_build,
        "tag": rel.get("tag"),
        "remote_url": rel.get("url"),
        "newer": newer,
        "asset": chosen,
    }
    if chosen:
        report["asset_name"] = chosen.get("name", "")
        report["download_size"] = int(chosen.get("size") or 0)
        report["free_mib"] = None
        report["target_dir"] = ""
        try:
            # La directory di destinazione: SEMPRE nuova e SEMPRE sorella
            # della build locale, mai dentro.  Scaricare un archivio dentro
            # la directory della build CUDA funzionante la inquinerebbe (o,
            # peggio, la estrarrebbe per sopra), quindi si sale di un livello
            # finche' non si arriva alla radice comune con ~/Downloads.
            local_dir = os.path.dirname(info.binary) if info.binary else ""
            # La destinazione e' SORELLA della build locale, nella stessa
            # cartella llamacpp: scaricare centinaia di MiB in ~/Downloads
            # riempirebbe il disco di root. Il nome "llama-bNNNNN" e' quello
            # che build_manager si aspetta per riconoscerla da sola.
            parent = os.path.dirname(local_dir) if local_dir else _downloads_root("")
            dest = _unique_dir(os.path.join(parent, f"llama-b{remote_build}"))
            report["target_dir"] = dest
            usage = shutil.disk_usage(os.path.dirname(dest) or parent)
            report["free_mib"] = usage.free / (1024.0 * 1024.0)
            need = report["download_size"] / (1024.0 * 1024.0)
            report["needed_mib"] = need
            # 3x la dimensione: zip + estratto.
            report["enough_space"] = (usage.free / (1024.0 * 1024.0)) > (need * 3.0)
        except Exception as e:
            report["space_error"] = str(e)
    return report


def _downloads_root(local_dir: str) -> str:
    """
    Cartella in cui scaricare la nuova build: ~/Downloads.

    Sale dalla directory della build locale finche' non trova ~/Downloads, e
    ripiega su ~/Downloads se il percorso locale e' altrove.  Serve a NON
    scrivere dentro la directory della build CUDA funzionante.
    """
    want = os.path.abspath("/home/user/Downloads")
    cur = os.path.abspath(local_dir) if local_dir else ""
    if not cur:
        return want
    # se local_dir e' gia' sotto ~/Downloads, si usa direttamente
    if os.path.commonpath([cur, want]) == want:
        return want
    return want


def _unique_dir(path: str) -> str:
    """
    Restituisce un percorso che NON esiste gia', aggiungendo _2, _3, ...
    Serve a non sovrascrivere MAI la build CUDA funzionante.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def download_build(asset_url: str, target_dir: str, confirm: bool = False,
                   timeout: float = 60.0, progress: Optional[Any] = None) -> Dict[str, Any]:
    """
    Scarica un asset release in una directory NUOVA.

    Regole ferree:
      * senza ``confirm=True`` NON si scarica niente (ritorna un avviso);
      * se ``target_dir`` esiste gia' si rifiuta di scriverci dentro;
      * nessun processo viene terminato, nessun server viene avviato;
      * lo zip viene lasciato com'e': l'estrazione e' un passo separato
        (``extract_zip``) cosi' l'utente vede cosa c'e' prima di usarlo.
    """
    if not confirm:
        return {
            "ok": False,
            "downloaded": False,
            "error": "download annullato: serve confirm=True esplicito",
        }
    if not asset_url:
        return {"ok": False, "downloaded": False, "error": "URL asset vuoto"}
    if not target_dir or os.path.exists(target_dir):
        return {
            "ok": False, "downloaded": False,
            "error": f"directory di destinazione gia' esistente o vuota: {target_dir}",
        }
    try:
        os.makedirs(target_dir)
    except Exception as e:
        return {"ok": False, "downloaded": False, "error": f"mkdir fallito: {e}"}

    name = asset_url.rsplit("/", 1)[-1]
    dest = os.path.join(target_dir, name)
    try:
        with urllib.request.urlopen(
            urllib.request.Request(asset_url,
                                   headers={"User-Agent": "ai_tui-llama-config"}),
            timeout=timeout,
        ) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress:
                    progress(got, total)
    except Exception as e:
        # Non lasciare directory mezze vuote: si rimuove solo quanto creato ora.
        try:
            if os.path.isfile(dest):
                os.remove(dest)
            os.rmdir(target_dir)
        except OSError:
            pass
        return {"ok": False, "downloaded": False, "error": str(e)}

    return {"ok": True, "downloaded": True, "path": dest,
            "dir": target_dir, "size": os.path.getsize(dest)}


def extract_zip(zip_path: str, dest_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Estrae un archivio scaricato. Nessuna esecuzione di nulla.

    Gestisce sia .zip (release Windows) sia .tar.gz/.tgz (release Linux):
    con il solo zipfile gli archivi Linux restavano intatti e la directory
    risultava "installata" pur contenendo solo il .tar.gz.

    Gli archivi di llama.cpp si espandono in build/bin/: i file vengono
    appiattiti dentro dest_dir, che e' dove si cerca llama-server.
    """
    import tarfile
    import tempfile
    import zipfile

    dest_dir = dest_dir or os.path.dirname(zip_path)
    low = zip_path.lower()
    tmp = tempfile.mkdtemp(prefix="extract-", dir=dest_dir)
    try:
        if low.endswith(".zip"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
        elif low.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if low.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(zip_path, mode) as tf:
                tf.extractall(tmp)
        else:
            return {"ok": False,
                    "error": f"formato non riconosciuto: {os.path.basename(zip_path)}"}
        moved = 0
        for root, _dirs, files in os.walk(tmp):
            for fn in files:
                dst = os.path.join(dest_dir, fn)
                if os.path.exists(dst):
                    continue
                os.replace(os.path.join(root, fn), dst)
                moved += 1
        if not moved:
            return {"ok": False, "error": "archivio estratto ma nessun file trovato"}
        try:
            os.remove(zip_path)      # estratto: l'archivio non serve piu'
        except OSError:
            pass
        return {"ok": True, "dir": dest_dir, "files": moved}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
