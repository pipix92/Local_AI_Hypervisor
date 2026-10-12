"""
launch_presets.py - preset F1-F6 (Ornith / Qwen3.8 via RPC) e menu di pre-avvio.

Contenuto
---------
* PRESETS: i 6 preset dei tasti F1-F6. Ognuno porta il SUO modello (fisso,
  cambiabile solo dal menu di pre-avvio) e i flag consigliati dai benchmark
  (preset_bench.json, scritto dall'agente bench_miner il 2026-10-05):
    - NIENTE -ngl e NIENTE --tensor-split: disattivano il fitter automatico
      di llama.cpp (crash/OOM misurati). Si usa --fit on --fit-target.
    - Default aggiornati il 2026-10-05 con le misure dal vivo
      (agent_reports/live_tests.md, contesto PIENO, --fit-target 0,700):
    - Ornith: KV f16 + MTP draft-mtp n-max 2 a 16k/75k/120k = config migliore
      misurata: 30.15 tok/s a 81920 pieno, 26.84 a 122880 pieno
      (n-max 1: 21.95 / 25.61; KV q4_0: 22.09 / 18.67).
    - Qwen3.8 IsValorum: KV f16 senza MTP (26.47 a 81920, 22.40 a 122880
      pieni); con MTP e' sempre piu' lento a contesto pieno. Se lo si attiva,
      il file proposto e' mtp-*-Q4_0.gguf (misurato piu' veloce della Q8_0).
    - --fit-target 0,700: Miky (RPC0) al 100%, 3050 (CUDA0) con 700 MiB liberi
      (ordine dei device RPC0 poi CUDA0, verificato nel log).
* FLAG_CATALOG / PARAM_CATALOG: cosa si puo' attivare/modificare nel menu,
  con una riga di spiegazione e i flag di llama-server richiesti (si mostra
  solo cio' che la build scelta accetta davvero, letto da --help).
* OverrideStore: personalizzazioni per preset in ~/.ai_tui_preset_overrides.json.
* bench_for(): i tok/s MISURATI dal file dei benchmark; "n/m" se non c'e'
  una misura vicina. Mai numeri inventati.
* compose_argv(): la riga di comando completa (pura, nessun processo).
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(HERE, "preset_bench.json")
OVERRIDES_PATH = os.path.expanduser("~/.ai_tui_preset_overrides.json")

RPC_EP = "192.168.1.100:50052"
MODEL_ORNITH = ("/mnt/lm_models/models/gbuzhf/Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF/"
                "Ornith-1.5-35B-A3B-Abliterated-MTPv2-APEX-I-Mini-v2D-lite.gguf")
MODEL_QWEN38 = ("/mnt/lm_models/models/IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-"
                "Abliterated-GGUF/Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf")
MODEL_HERETIC = ("/mnt/lm_models/models/DavidAU/OpenAi-GPT-oss-20b-HERETIC-uncensored-NEO-Imatrix-gguf/"
                 "OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf")
# Qwen3.8: testa MTP in file separato (accanto al modello). Ornith: MTP
# integrata nel GGUF ("MTPv2" nel nome), quindi nessun -md.
MTP_QWEN38_Q8 = os.path.join(os.path.dirname(MODEL_QWEN38), "mtp-Qwen3.8-35B-A3B-Distill-Q8_0.gguf")
MTP_QWEN38_Q4 = os.path.join(os.path.dirname(MODEL_QWEN38), "mtp-Qwen3.8-35B-A3B-Distill-Q4_0.gguf")
# Proposto di default se l'utente attiva MTP: la Q4_0 e' risultata piu' veloce
# della Q8_0 (80k f16 n-max 2 pieno: 23.61 vs 20.45 tok/s). Se manca, Q8_0.
MTP_QWEN38 = MTP_QWEN38_Q4 if os.path.isfile(MTP_QWEN38_Q4) else MTP_QWEN38_Q8
# Default misurato di --fit-target (ordine --device RPC0,CUDA0).
# 512 MiB di margine su Miky (RPC0): con 0 l'avvio falliva a fine caricamento
# ("failed to allocate RPC0 buffer" per la cache KV) perche' su Windows/WDDM la
# VRAM libera di Miky cambia mentre si carica (es. Brave usa la GPU NVIDIA).
# Verificato il 2026-10-05 sera (Brave aperto su Miky): 0,700 e 512,700 ->
# errore a ~75 s; 1024,700 -> ONLINE in 70 s, 23,4 tok/s (prompt breve).
# Con Miky libera 0,700 rende di piu' (32,3 tok/s vuoto): resta nel ciclo.
# Default richiesto: Miky (RPC0) con 512 MiB di margine;
# 3050 (CUDA0) all'88%: su 7839 MiB restano ~940 MiB liberi.
FIT_TARGET_DEFAULT = "512,512"
PORT_DEFAULT = 9292             # 9205 se 9292 e' occupata (vedi porta_libera())
KV_DEFAULT = "q4_0"             # KV q4 in VRAM di default (richiesta utente)

CTX_16K, CTX_75K, CTX_120K, CTX_80K = 16384, 76800, 122880, 80000


def _base(**kw: Any) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "schema": 2,
        "recommended": False,
        "device": "RPC0,CUDA0",
        "rpc": RPC_EP,
        "parallel": 1,
        "ngl": "",              # vuoto = auto (fitter); un numero disattiva il fitter
        "threads": "",          # vuoto = default di llama-server
        "batch": "",
        "ubatch": "",
        "tensor_split": "",     # vuoto = nessuno (consigliato)
        "fit": "on",
        "fit_target": FIT_TARGET_DEFAULT,
        "port": PORT_DEFAULT,
        "kv_type": KV_DEFAULT,
        "kv_value": KV_DEFAULT,
        "extra": "",
        # n-max 2: misurato migliore di 1 a contesto pieno (Ornith 81920 f16:
        # 30.15 vs 21.95 tok/s; 122880: 26.84 vs 25.61). Vale solo con MTP attiva.
        "spec_n_max": 2,
        "n_cpu_moe": "",        # pesi MoE dei primi N layer in RAM (vuoto/0 = spento)
        "mtp_model": "",
        "chat_template_file": "",
        "reverse_prompt": "",
        "flags": {"flash_attn": True, "jinja": True, "nkvo": False, "kvu": False,
                  "mlock": False, "no_mmap": False, "metrics": True,
                  "context_shift": False, "mtp": False, "cpu_moe": False},
        "tok_s": 0.0,
    }
    flags = kw.pop("flags", {})
    p.update(kw)
    p["flags"] = dict(p["flags"], **flags)
    p["kv_value"] = p.get("kv_value") or p.get("kv_type")
    return p


PRESETS: Dict[str, Dict[str, Any]] = {
    "orn_16k": _base(name="ORN-16k", short="Ornith", family="ornith", model=MODEL_ORNITH,
                     ctx_size=CTX_16K, kv_type=KV_DEFAULT, flags={"mtp": True}, recommended=True,
                     description="Ornith 35B-A3B via RPC, 16k, KV f16, MTP integrata n-max 2."),
    "orn_75k": _base(name="ORN-75k", short="Ornith", family="ornith", model=MODEL_ORNITH,
                     ctx_size=CTX_75K, kv_type=KV_DEFAULT, flags={"mtp": True},
                     description="Ornith via RPC, 75k (76800), KV f16 + MTP n-max 2: misurato a 80k "
                                 "(81920) pieno 30.15 tok/s."),
    "orn_120k": _base(name="ORN-120k", short="Ornith", family="ornith", model=MODEL_ORNITH,
                      ctx_size=CTX_120K, kv_type=KV_DEFAULT, flags={"mtp": True},
                      description="Ornith via RPC, 120k, KV f16 + MTP n-max 2: misurato pieno 26.84 tok/s "
                                  "(q4_0 n-max 1: 17.23)."),
    "qw_16k": _base(name="QW38-16k", short="Qwen3.8", family="qwen38", model=MODEL_QWEN38,
                    ctx_size=CTX_16K, kv_type=KV_DEFAULT, mtp_model=MTP_QWEN38,
                    description="Qwen3.8 35B-A3B IsValorum via RPC, 16k, KV f16, senza draft."),
    "qw_75k": _base(name="QW38-75k", short="Qwen3.8", family="qwen38", model=MODEL_QWEN38,
                    ctx_size=CTX_75K, kv_type=KV_DEFAULT, mtp_model=MTP_QWEN38,
                    description="Qwen3.8 via RPC, 75k (76800), KV f16, senza MTP: misurato a 80k (81920) "
                                "pieno 26.47 tok/s."),
    "heretic_80k": _base(name="HERETIC-80k", short="Heretic", family="gpt_oss_20b", model=MODEL_HERETIC,
                         ctx_size=CTX_80K, kv_type="q4_1", kv_value="q4_1", fit_target="0,900",
                         flags={"flash_attn": True, "jinja": True, "metrics": True, "context_shift": False, "mtp": False},
                         description="OpenAI-GPT-oss-20B Heretic Uncensored NEO IQ4_NL, 80k contesto via RPC dual-GPU, KV q4_1."),
}
ORDER: List[str] = ["orn_16k", "orn_75k", "orn_120k", "qw_16k", "qw_75k", "heretic_80k"]
BENCH_FAMILY = {"ornith": "ornith", "qwen38": "qwen38_isvalorum", "gpt_oss_20b": "gpt_oss_20b"}


# ---------------------------------------------------------------------------
# Catalogo dei flag (caselle [x]/[ ]) e dei parametri
# ---------------------------------------------------------------------------
# needs: flag di --help di cui ne basta UNO perche' la casella sia mostrata.
FLAG_CATALOG: List[Dict[str, Any]] = [
    {"key": "flash_attn", "label": "flash attention (-fa on)", "needs": ["--flash-attn", "-fa"],
     "help": "Flash Attention: meno VRAM per la KV e piu' velocita'. Necessaria per KV quantizzata."},
    {"key": "jinja", "label": "--jinja (template chat)", "needs": ["--jinja"],
     "help": "Usa il template Jinja del modello (tool calling corretto). Gia' attivo di default in b11330."},
    {"key": "nkvo", "label": "-nkvo (cache KV in RAM)", "needs": ["--no-kv-offload", "-nkvo"],
     "help": "KV cache in RAM di sistema: via RPC misurato < 1 tok/s. Da usare solo per prove."},
    {"key": "cpu_moe", "label": "--cpu-moe (TUTTI i pesi MoE in RAM)", "needs": ["--cpu-moe", "-cmoe"],
     "help": "Tutti i pesi degli esperti MoE restano in RAM/CPU: libera molta VRAM per il contesto ma "
             "rallenta. Mai misurato su questi preset. Per spostarne solo una parte usa --n-cpu-moe."},
    {"key": "kvu", "label": "-kvu (KV unificata)", "needs": ["--kv-unified", "-kvu"],
     "help": "Un solo buffer KV condiviso fra gli slot: utile con -np > 1, inutile con 1 slot."},
    {"key": "mlock", "label": "mlock (tieni in RAM)", "needs": ["--load-mode", "--mlock"],
     "help": "Blocca il modello in RAM (niente swap). Con b11330 diventa --load-mode mmap+mlock."},
    {"key": "no_mmap", "label": "no mmap (--load-mode none)", "needs": ["--load-mode", "--no-mmap"],
     "help": "Carica il modello senza memory-map: avvio piu' lento, meno page-out."},
    {"key": "metrics", "label": "--metrics (Prometheus)", "needs": ["--metrics"],
     "help": "Espone /metrics: la TUI legge tok/s reali e richieste in coda dal pannello server."},
    {"key": "context_shift", "label": "--context-shift", "needs": ["--context-shift"],
     "help": "A contesto pieno scarta i token vecchi invece di fallire (sessioni lunghe)."},
    {"key": "mtp", "label": "speculative MTP (draft-mtp)", "needs": ["--spec-type"],
     "help": "Decodifica speculativa MTP. Ornith: testa integrata nel GGUF, n-max 2 (30.15 tok/s a 80k "
             "pieno, 26.84 a 120k). Qwen3.8: file mtp-*.gguf separato (-md, proposto Q4_0); a contesto "
             "pieno misurato PIU' LENTO (80k: 23.61 con Q4_0 vs 26.47 senza), spento di default."},
]

# kind: int | text | enum | model. empty_ok: vuoto = "auto"/non passato.
PARAM_CATALOG: List[Dict[str, Any]] = [
# La porta e' il PRIMO parametro: e' quella che si cerca per prima (es. localhost:8080).
    {"key": "port", "label": "PORTA (digita il numero)", "kind": "int", "typed": True,
     "cycle": [9292, 9205, 8080, 8087, 8088],
     "help": "Porta HTTP di llama-server. Premi INVIO e digita il numero (1024-65535). "
             "Il server ascolta su localhost: aprilo su http://localhost:<porta>."},
    {"key": "ctx_size", "label": "contesto (-c)", "kind": "int",
     "cycle": [8192, 16384, 32768, 49152, 65536, 76800, 98304, 122880, 131072, 163840, 196608, 262144],
     "help": "Token di contesto. -c esplicito: --fit non lo riduce, se non entra il server esce."},
    {"key": "ngl", "label": "-ngl", "kind": "int", "empty_ok": True,
     "cycle": ["", 0, 20, 30, 40, 99],
     "help": "VUOTO = auto (consigliato). Un numero DISATTIVA il fitter: misurati crash/OOM sulla KV."},
    {"key": "kv_type", "label": "-ctk (KV K)", "kind": "enum", "choices_from": "cache",
     "help": "Tipo della cache K. f16 = piu' veloce via RPC nei benchmark; q4_0 = meta' memoria. "
             "Cambiandolo cambia anche -ctv (se erano uguali): K e V diversi = FlashAttention su CPU."},
    {"key": "kv_value", "label": "-ctv (KV V)", "kind": "enum", "choices_from": "cache",
     "help": "Tipo della cache V. Segue -ctk finche' sono uguali; impostarlo diverso li SBLOCCA "
             "(K/V misti: FlashAttention CUDA puo' ripiegare su CPU)."},
    {"key": "threads", "label": "threads (-t)", "kind": "int", "empty_ok": True,
     "cycle": ["", 4, 6, 8, 10, 12], "help": "Thread CPU. Vuoto = default di llama-server."},
    {"key": "parallel", "label": "parallel (-np)", "kind": "int", "cycle": [1, 2, 3, 4],
     "help": "Slot paralleli: il contesto viene diviso fra gli slot."},
    {"key": "batch", "label": "batch (-b)", "kind": "int", "empty_ok": True,
     "cycle": ["", 512, 1024, 2048, 4096], "help": "Batch logico (default 2048). Vuoto = default."},
    {"key": "ubatch", "label": "ubatch (-ub)", "kind": "int", "empty_ok": True,
     "cycle": ["", 256, 512, 1024, 2048], "help": "Batch fisico (default 512). -ub 1024 usato in avvia-ornith-hermes."},
    {"key": "tensor_split", "label": "--tensor-split", "kind": "text", "empty_ok": True,
     "cycle": ["", "50,50", "45,55", "55,45"],
     "help": "VUOTO consigliato: con -ts il fitter si ferma ('tensor_split already set by user, abort')."},
    {"key": "fit", "label": "--fit (on/off)", "kind": "enum", "cycle": ["on", "off"],
     "needs": ["--fit"],
     "help": "on = llama.cpp regola da solo gli argomenti non fissati (-ngl, -c, ripartizione) per entrare "
             "in memoria, rispettando --fit-target. off = nessun aggiustamento: --fit-target viene ignorato."},
    {"key": "fit_target", "label": "--fit-target (MiB)", "kind": "text", "empty_ok": True,
     "cycle": ["1024,700", "512,700", "0,700", "200,200", "0,600", "50,50", "512,512", "1024,1024", ""],
     "help": "Margine VRAM per device nell'ordine di --device (RPC0,CUDA0). 1024,700 = default sicuro (Brave/desktop su Miky usano la GPU). 0,700 = misurato a Miky libera: Miky al "
             "100%, 3050 con 700 MiB liberi. Vuoto = niente --fit."},
    {"key": "n_cpu_moe", "label": "pesi MoE in RAM (--n-cpu-moe N)", "kind": "int", "empty_ok": True,
     "needs": ["--n-cpu-moe", "-ncmoe"],
     "cycle": ["", 4, 8, 12, 16, 20, 24, 32, 40],
     "help": "Pesi degli esperti MoE dei primi N layer in RAM invece che in VRAM (vuoto/0 = spento). "
             "Libera VRAM per contesto/KV a costo di velocita'. Mai misurato su questi preset."},
    {"key": "spec_n_max", "label": "--spec-draft-n-max", "kind": "int", "cycle": [1, 2, 3, 4],
     "needs": ["--spec-draft-n-max"],
     "help": "Token proposti per passo MTP (vale solo con MTP attiva). Default 2 = misurato migliore a "
             "contesto pieno (Ornith 80k f16: 30.15 tok/s con 2, 21.95 con 1)."},
    # Solo Qwen3.8 (testa MTP in file separato accanto al modello): Q8_0 o Q4_0.
    {"key": "mtp_model", "label": "file MTP (-md)", "kind": "enum", "choices_from": "mtp",
     "only_family": "qwen38", "needs": ["--spec-type"],
     "help": "Testa MTP separata di Qwen3.8 (vale solo con MTP attiva): mtp-*-Q4_0.gguf (1.1 GB, "
             "proposta: misurata piu' veloce) o mtp-*-Q8_0.gguf (1.9 GB). Ornith ha la MTP integrata."},
    {"key": "extra", "label": "flag extra", "kind": "text", "empty_ok": True,
     "help": "Flag liberi aggiunti IN FONDO alla riga (vince l'ultima occorrenza)."},
    # ULTIMO apposta: un [→] distratto sulla prima riga cambiava il modello.
    {"key": "model", "label": "modello", "kind": "model",
     "help": "GGUF del preset. Fisso per F1-F6: cambialo qui ([←/→] scorre la lista, INVIO = percorso). "
             "[R] torna al modello del preset."},
]


def mtp_files(model: str) -> List[str]:
    """Teste MTP separate (mtp-*.gguf) nella cartella del modello, ordinate."""
    d = os.path.dirname(str(model or ""))
    try:
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.startswith("mtp-") and f.endswith(".gguf"))
    except OSError:
        return []


# ---------------------------------------------------------------------------
# --help della build: quali flag accetta
# ---------------------------------------------------------------------------
_HELP_CACHE: Dict[str, Tuple[float, str]] = {}
_HELP_LOCK = threading.Lock()


def build_help(binary: str) -> str:
    """`llama-server --help` della build (cache per mtime). '' se non leggibile."""
    if not binary or not os.path.isfile(binary):
        return ""
    try:
        mt = os.path.getmtime(binary)
    except OSError:
        return ""
    with _HELP_LOCK:
        hit = _HELP_CACHE.get(binary)
        if hit and hit[0] == mt:
            return hit[1]
    try:
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(binary) + ":"
                   + os.environ.get("LD_LIBRARY_PATH", ""))
        out = subprocess.run([binary, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=10, env=env).stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        # timeout/errore: NON si mette in cache "" (help_has("") = True
        # spegnerebbe il filtro dei flag per sempre); si riprovera'.
        return ""
    if out.strip():
        with _HELP_LOCK:
            _HELP_CACHE[binary] = (mt, out)
    return out


def cached_help(binary: str) -> Optional[str]:
    """--help gia' in cache (None se non ancora letto). Non avvia processi:
    da usare nel disegno della UI, la lettura vera va fatta in un thread."""
    with _HELP_LOCK:
        hit = _HELP_CACHE.get(binary or "")
    if not hit:
        return None
    try:
        if os.path.getmtime(binary) != hit[0]:
            return None
    except OSError:
        return None
    return hit[1]


def help_has(help_text: str, flag: str) -> bool:
    """True se `flag` compare come opzione nell'help (non come sottostringa di un altro)."""
    if not help_text:
        return True          # help non leggibile: non si nasconde nulla
    return re.search(r"(^|[\s,])" + re.escape(flag) + r"($|[\s,=\[])", help_text, re.M) is not None


def cache_types(help_text: str) -> List[str]:
    m = re.search(r"--cache-type-k TYPE.*?allowed values: ([\w, ]+)", help_text or "", re.S)
    vals = [v.strip() for v in m.group(1).split(",") if v.strip()] if m else []
    return vals or ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "bf16", "f32"]


def flag_supported(spec: Dict[str, Any], help_text: str) -> bool:
    return any(help_has(help_text, f) for f in spec["needs"])


# ---------------------------------------------------------------------------
# Personalizzazioni per preset
# ---------------------------------------------------------------------------
class OverrideStore:
    def __init__(self, path: str = OVERRIDES_PATH) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        self.error = ""
        self.load()

    def load(self) -> None:
        import atomic_json as _AJ
        # File corrotto: rinominato in .corrupt-<ts> e avviso in self.error
        # (prima veniva azzerato in silenzio al primo salvataggio).
        try:
            mt = os.path.getmtime(self.path)
            if hasattr(self, "_mtime") and self._mtime == mt:
                return
            self._mtime = mt
        except OSError:
            pass
        d, err = _AJ.load_json(self.path, {})
        self.error = err or ""
        pr = d.get("preset") if isinstance(d, dict) else None
        self.data = {k: v for k, v in (pr or {}).items() if isinstance(v, dict)} \
            if isinstance(pr, dict) else {}

    def save(self) -> bool:
        import atomic_json as _AJ
        try:
            _AJ.save_json(self.path, {"_meta": "Personalizzazioni dei preset F1-F6 di ai_tui (menu di "
                                               "pre-avvio). [R] nel menu le cancella.",
                                      "preset": self.data})
            try:
                self._mtime = os.path.getmtime(self.path)
            except OSError:
                pass
            self.error = ""
            return True
        except (OSError, TypeError) as exc:
            self.error = str(exc)
            return False

    def get(self, key: str) -> Dict[str, Any]:
        self.load()
        return self.data.get(key, {})

    def set_from(self, key: str, eff: Dict[str, Any]) -> bool:
        """Salva SOLO le differenze rispetto al preset di base."""
        base = PRESETS.get(key)
        if base is None:
            return False
        diff: Dict[str, Any] = {}
        for k, v in eff.items():
            if k == "flags":
                fd = {fk: fv for fk, fv in v.items() if base["flags"].get(fk) != fv}
                if fd:
                    diff["flags"] = fd
            elif k in base and base.get(k) != v and k not in ("tok_s",):
                diff[k] = v
        if diff:
            self.data[key] = diff
        else:
            self.data.pop(key, None)
        return self.save()

    def reset(self, key: str) -> bool:
        self.data.pop(key, None)
        return self.save()


_STORE: Optional[OverrideStore] = None


def store() -> OverrideStore:
    global _STORE
    if _STORE is None:
        _STORE = OverrideStore()
    return _STORE


def effective(key: str) -> Optional[Dict[str, Any]]:
    """Preset di base + personalizzazioni salvate (copia indipendente)."""
    base = PRESETS.get(key)
    if base is None:
        return None
    eff = copy.deepcopy(base)
    ov = store().get(key)
    for k, v in ov.items():
        if k == "flags" and isinstance(v, dict):
            eff["flags"].update({fk: bool(fv) for fk, fv in v.items() if fk in eff["flags"]})
        elif k in eff:
            if k in _INT_KEYS and not _int_ok(v, empty_ok=k in _INT_EMPTY_OK):
                continue        # valore non valido scritto a mano nel JSON: si ignora
            eff[k] = v
    return eff


_INT_KEYS = ("ctx_size", "port", "parallel", "ngl", "threads", "batch", "ubatch", "spec_n_max", "n_cpu_moe")
_INT_EMPTY_OK = ("ngl", "threads", "batch", "ubatch", "n_cpu_moe")


def _int_ok(v: Any, empty_ok: bool = False) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    sv = str(v).strip()
    return (empty_ok and sv == "") or bool(re.fullmatch(r"-?\d+", sv))


def customized(key: str) -> bool:
    return bool(store().get(key))


# ---------------------------------------------------------------------------
# Misure da preset_bench.json
# ---------------------------------------------------------------------------
_BENCH: Dict[str, Any] = {"mtime": None, "data": {}}


def _bench_data() -> Dict[str, Any]:
    try:
        mt = os.path.getmtime(BENCH_PATH)
    except OSError:
        return {}
    if _BENCH["mtime"] != mt:
        try:
            with open(BENCH_PATH, "r", encoding="utf-8") as fh:
                _BENCH["data"] = json.load(fh)
            _BENCH["mtime"] = mt
        except (OSError, ValueError):
            return {}
    return _BENCH["data"]


def _ctx_label(ctx: int) -> str:
    return f"{ctx // 1024}k" if ctx % 1024 == 0 else f"{round(ctx / 1000)}k"


def _run_has_mtp(r: Dict[str, Any]) -> bool:
    spec = str((r.get("flags") or {}).get("spec") or "")
    return "mtp" in spec.lower() or ("mtp" in str(r.get("tag", "")).lower()
                                     and "no-mtp" not in str(r.get("tag", "")).lower()
                                     and "nomtp" not in str(r.get("tag", "")).lower())


def _run_md(r: Dict[str, Any]) -> str:
    """File MTP separato (-md) di una misura: 'Q4_0' / 'Q8_0' / ''."""
    fl = r.get("flags") or {}
    txt = " ".join(str(x) for x in (fl.get("spec"), fl.get("draft_model"), r.get("tag")) if x)
    m = re.search(r"Q[48]_0", txt)
    return m.group(0) if m else ""


def bench_for(preset: Dict[str, Any], max_ratio: float = 1.35) -> Optional[Dict[str, Any]]:
    """
    La misura reale piu' vicina al preset: stesso modello, via RPC, senza
    -nkvo, contesto entro un fattore `max_ratio`. Conta PRIMA la stessa
    configurazione (KV, MTP si/no, n-max, pesi MoE in RAM, file -md), poi il
    contesto pieno, poi lo stesso --fit-target, poi il contesto piu' vicino,
    poi la misura piu' recente. Cosi' i T/s accanto ai preset riflettono i
    default attuali (es. F2 76800 mostra la misura a 81920 con la STESSA
    config, non una a 74240 con config diversa). None = nessuna misura ("n/m").
    """
    fam = BENCH_FAMILY.get(str(preset.get("family", "")))
    runs = ((_bench_data().get(fam) or {}).get("runs") or []) if fam else []
    try:
        ctx = int(preset.get("ctx_size") or 0)
    except (TypeError, ValueError):
        ctx = 0
    if not runs or not ctx:
        return None
    want_rpc = bool(preset.get("rpc"))
    want_mtp = bool((preset.get("flags") or {}).get("mtp"))
    want_n = str(preset.get("spec_n_max") or 1)
    want_fit = str(preset.get("fit_target") or "").strip()
    want_md = ""
    if want_mtp and preset.get("mtp_model"):
        m_md = re.search(r"Q[48]_0", os.path.basename(str(preset["mtp_model"])))
        want_md = m_md.group(0) if m_md else ""
    best, best_score = None, None
    for r in runs:
        g, rc = r.get("gen_tok_s"), r.get("ctx")
        if not isinstance(g, (int, float)) or not (0 < g < 200) or not isinstance(rc, int) or rc <= 0:
            continue
        if not str(r.get("esito", "")).startswith("OK") or r.get("nkvo"):
            continue
        is_rpc = str(r.get("rpc") or "").startswith("192.")
        if is_rpc != want_rpc:
            continue
        ratio = max(rc, ctx) / min(rc, ctx)
        if ratio > max_ratio:
            continue
        pieno = str(r.get("contesto_riempito", "")).startswith("pieno")
        fl_r = r.get("flags") or {}
        m_n = re.search(r"n_max=(\d+)", str(fl_r.get("spec") or ""))
        r_n = m_n.group(1) if m_n else str(fl_r.get("spec_draft_n_max") or "")
        nmax_diff = bool(want_mtp and r_n != want_n)
        ncm_p = str(preset.get("n_cpu_moe") or "").strip()
        ncm_diff = str(fl_r.get("n_cpu_moe") or "") != ("" if ncm_p in ("", "0") else ncm_p)
        md_diff = bool(want_md and _run_has_mtp(r) and (_run_md(r) or "Q8_0") != want_md)
        fit_diff = str(fl_r.get("fit_target") or "") != want_fit
        score = (r.get("ctk") != preset.get("kv_type"), _run_has_mtp(r) != want_mtp, ncm_diff,
                 nmax_diff, md_diff, not pieno, fit_diff, round(abs(math.log(ratio)), 3),
                 -_ts(r.get("data")))
        if best_score is None or score < best_score:
            best, best_score = r, score
    if best is None:
        return None
    diffs = []
    if best.get("ctk") != preset.get("kv_type"):
        diffs.append(f"KV {best.get('ctk')}")
    if _run_has_mtp(best) != want_mtp:
        diffs.append("con MTP" if _run_has_mtp(best) else "senza MTP")
    if best.get("ctx") != ctx:
        diffs.append(f"ctx {best.get('ctx')}")
    fl_b = best.get("flags") or {}
    m_b = re.search(r"n_max=(\d+)", str(fl_b.get("spec") or ""))
    b_n = m_b.group(1) if m_b else str(fl_b.get("spec_draft_n_max") or "")
    if want_mtp and b_n and b_n != want_n:
        diffs.append(f"n-max {b_n}")
    if want_md and _run_has_mtp(best) and (_run_md(best) or "Q8_0") != want_md:
        diffs.append(f"-md {_run_md(best) or 'Q8_0'}")
    if str(fl_b.get("fit_target") or "") != want_fit:
        diffs.append(f"fit-target {fl_b.get('fit_target') or '?'}")
    return {"tok_s": float(best["gen_tok_s"]), "ctx": int(best["ctx"]), "tag": best.get("tag", ""),
            "ctk": best.get("ctk"), "filled": best.get("contesto_riempito", ""),
            "source": best.get("source", ""), "diffs": diffs, "exact": not diffs}


def _ts(s: Any) -> float:
    try:
        import time as _t
        return _t.mktime(_t.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S"))
    except (ValueError, TypeError):
        return 0.0


def tok_label(preset: Dict[str, Any]) -> str:
    """Es. '29.8T/s@74k' ('*' = misurato con una config diversa), 'n/m' se non misurato."""
    b = bench_for(preset)
    if not b:
        return "n/m"
    return f"{b['tok_s']:.1f}T/s@{_ctx_label(b['ctx'])}" + ("" if b["exact"] or b["diffs"] == [f"ctx {b['ctx']}"]
                                                             else "*")


def ctx_label(ctx: int) -> str:
    return _ctx_label(int(ctx))


# ---------------------------------------------------------------------------
# Riga di comando
# ---------------------------------------------------------------------------
def ignored_reason(p: Dict[str, Any], key: str) -> str:
    """Perche' un parametro e' IGNORATO da llama.cpp con questa configurazione ('' = attivo)."""
    if key != "fit_target":
        return ""
    if str(p.get("fit", "on")).strip() == "off":
        return "--fit off: il fitter e' spento"
    if str(p.get("ngl", "")).strip() != "":
        return "-ngl fisso: il fitter non regola gli strati"
    if str(p.get("tensor_split", "")).strip():
        return "--tensor-split fisso: il fitter si ferma"
    if (p.get("flags", {}).get("cpu_moe") or str(p.get("n_cpu_moe", "")).strip() not in ("", "0")):
        return "pesi MoE in RAM: il fitter si ferma (tensor overrides)"
    return ""


def warnings_for(p: Dict[str, Any]) -> List[str]:
    w = []
    if str(p.get("ngl", "")).strip() != "":
        w.append("-ngl esplicito: il fitter (--fit) non regola gli strati, rischio OOM")
    if str(p.get("tensor_split", "")).strip():
        w.append("--tensor-split: il fitter si ferma ('tensor_split already set by user, abort')")
    if p.get("flags", {}).get("cpu_moe") or str(p.get("n_cpu_moe", "")).strip() not in ("", "0"):
        # verificato dal vivo (agent_reports/live_tests.md): con --n-cpu-moe/--cpu-moe
        # il log dice "tensor_buft_overrides already set by user, abort" e --fit-target
        # non viene piu' rispettato (ripartizione di default fra i device).
        w.append("pesi MoE in RAM: il fitter si ferma (tensor overrides dell'utente), --fit-target ignorato")
    if p.get("flags", {}).get("nkvo") and p.get("rpc"):
        w.append("-nkvo via RPC: misurato < 1 tok/s")
    kv_k = str(p.get("kv_type") or "f16")
    kv_v = str(p.get("kv_value") or kv_k)
    if kv_k != kv_v:
        w.append(f"-ctk {kv_k} e -ctv {kv_v} diversi: FlashAttention CUDA puo' ripiegare su CPU (molto lento)")
    if not p.get("flags", {}).get("flash_attn", True) and kv_v not in ("f16", "bf16", "f32"):
        w.append(f"-fa off con cache V quantizzata ({kv_v}): llama-server rifiuta l'avvio")
    if p.get("flags", {}).get("mtp") and p.get("family") == "qwen38":
        w.append("MTP su Qwen3.8: a contesto pieno misurato piu' lento (80k: 23.61 con -md Q4_0 "
                 "vs 26.47 senza)")
    if p.get("model") and not os.path.isfile(str(p["model"])):
        w.append(f"modello non trovato: {p['model']}")
    if p.get("flags", {}).get("mtp") and p.get("family") == "gpt_oss_20b":
        w.append("MTP non supportato da GPT-OSS / Heretic: il modello non contiene layer MTP (crash all'avvio)")
    return w


def compose_argv(binary: str, model: str, p: Dict[str, Any], port: Optional[int] = None,
                 host: str = "127.0.0.1", help_text: Optional[str] = None) -> List[str]:
    """
    argv completo per un preset schema 2. Puro: nessun processo.
    help_text = --help della build: i flag che la build non conosce vengono
    omessi (None = lo si legge dal binario, '' = non filtrare).
    """
    if help_text is None:
        help_text = build_help(binary)
    has = lambda f: help_has(help_text, f)          # noqa: E731
    fl = p.get("flags") or {}
    cmd = [binary, "-m", model]
    # --rpc PRIMA di --device: i device RPCn esistono solo dopo --rpc
    # (ordine inverso = "invalid device: RPC0", verificato su b11330).
    if p.get("rpc"):
        cmd += ["--rpc", str(p["rpc"])]
    if p.get("device"):
        cmd += ["--device", str(p["device"])]
    try:
        ctx = int(p.get("ctx_size") or 16384)
    except (TypeError, ValueError):
        ctx = 16384
    cmd += ["-c", str(ctx)]
    if str(p.get("parallel", "")).strip():
        cmd += ["-np", str(p["parallel"])]
    if str(p.get("ngl", "")).strip():
        cmd += ["-ngl", str(p["ngl"])]
    cmd += ["-fa", "on" if fl.get("flash_attn", True) else "off"]
    cmd += ["-ctk", str(p.get("kv_type") or "f16"), "-ctv", str(p.get("kv_value") or p.get("kv_type") or "f16")]
    for key, flag in (("threads", "-t"), ("batch", "-b"), ("ubatch", "-ub"), ("tensor_split", "-ts")):
        if str(p.get(key, "")).strip():
            cmd += [flag, str(p[key]).strip()]
    # Pesi MoE in RAM: --cpu-moe (tutti) ha la precedenza su --n-cpu-moe N.
    if fl.get("cpu_moe") and has("--cpu-moe"):
        cmd.append("--cpu-moe")
    elif str(p.get("n_cpu_moe", "")).strip() not in ("", "0") and has("--n-cpu-moe"):
        cmd += ["--n-cpu-moe", str(p["n_cpu_moe"]).strip()]
    if has("--fit") and str(p.get("fit", "on")).strip() == "off":
        cmd += ["--fit", "off"]
    elif str(p.get("fit_target", "")).strip() and has("--fit"):
        cmd += ["--fit", "on", "--fit-target", str(p["fit_target"]).strip()]
    if fl.get("jinja") and has("--jinja"):
        cmd.append("--jinja")
    elif not fl.get("jinja") and has("--no-jinja"):
        cmd.append("--no-jinja")
    if fl.get("nkvo") and (has("--no-kv-offload") or has("-nkvo")):
        cmd.append("-nkvo")
    if fl.get("kvu") and (has("--kv-unified") or has("-kvu")):
        cmd.append("-kvu")
    if has("--load-mode"):
        mode = None
        if fl.get("no_mmap"):
            mode = "mlock" if fl.get("mlock") else "none"
        elif fl.get("mlock"):
            mode = "mmap+mlock"
        if mode:
            cmd += ["--load-mode", mode]
    else:
        if fl.get("mlock") and has("--mlock"):
            cmd.append("--mlock")
        if fl.get("no_mmap") and has("--no-mmap"):
            cmd.append("--no-mmap")
    if fl.get("metrics") and has("--metrics"):
        cmd.append("--metrics")
    if fl.get("context_shift") and has("--context-shift"):
        cmd.append("--context-shift")
    if fl.get("mtp") and has("--spec-type") and p.get("family") != "gpt_oss_20b":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(p.get("spec_n_max") or 1)]
        if p.get("mtp_model"):
            cmd += ["-md", str(p["mtp_model"])]
    if p.get("chat_template_file") and has("--chat-template-file"):
        cmd += ["--chat-template-file", str(p["chat_template_file"])]
    if p.get("reverse_prompt") and (has("--reverse-prompt") or has("-r")):
        rev = p["reverse_prompt"]
        if isinstance(rev, list):
            for r in rev:
                cmd += ["-r", str(r)]
        else:
            cmd += ["-r", str(rev)]
    cmd += ["--host", host, "--port", str(port if port is not None else p.get("port", 8087))]
    extra = p.get("extra") or ""
    if isinstance(extra, str):
        try:
            extra = shlex.split(extra)
        except ValueError:
            extra = extra.split()
    cmd += [str(a) for a in extra]
    ext2 = p.get("extra_args") or []
    if isinstance(ext2, str):
        try:
            ext2 = shlex.split(ext2)
        except ValueError:          # virgolette non chiuse: niente eccezione nel disegno
            ext2 = ext2.split()
    cmd += [str(a) for a in ext2]
    return cmd


__all__ = ["PRESETS", "ORDER", "FLAG_CATALOG", "PARAM_CATALOG", "OverrideStore", "store",
           "effective", "customized", "bench_for", "tok_label", "compose_argv", "build_help",
           "help_has", "cache_types", "flag_supported", "warnings_for", "ctx_label", "mtp_files"]


PORT_USER_SET: Dict[str, bool] = {}   # preset per cui la porta e' stata scelta a mano


def porta_libera(preferite: Tuple[int, ...] = (9292, 9205)) -> int:
    """Prima porta libera tra quelle preferite (9292, poi 9205). Se sono
    occupate entrambe usa la prima libera a partire da 9292."""
    import socket as _s
    def libera(p: int) -> bool:
        with _s.socket(_s.AF_INET6, _s.SOCK_STREAM) as x:
            try:
                x.bind(("::1", p))
                return True
            except OSError:
                return False
    for p in preferite:
        if libera(p):
            return p
    for p in range(9292, 9400):
        if libera(p):
            return p
    return preferite[0]
