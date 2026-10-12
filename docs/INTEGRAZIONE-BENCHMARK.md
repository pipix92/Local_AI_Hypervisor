# Integrazione benchmark → TUI

Documento per **sa-1-29dcdc3c**, che sta modificando `tui_app.py`, `backend.py`
e `llama_config.py`.

**Cosa ho fatto io (finito, non serve fare niente):**
- creato `/home/user/benchmark/` con `raw/`, `reports/`, `modeli/`, `logs/`;
- creato `/home/user/ai_tui/benchmark_preset.py` — **solo lettura**, non importa
  `tui_app.py` né `backend.py` (niente conflitti, niente dipendenze circolari);
- scritto `/home/user/benchmark/PRESET-CONSIGLIATI.md`.

**Cosa serve a te:** 4 righe di import + 1 funzione in `tui_app.py`. Il modulo
è già pronto e testato, non devi toccare `backend.py` né `llama_config.py`.

---

## 1. L'import

In cima a `tui_app.py`, con gli altri import:

```python
# benchmark reali: /home/user/benchmark/modeli/*.json (solo lettura)
try:
    import benchmark_preset
except Exception:  # non far mai cadere la TUI se manca il modulo
    benchmark_preset = None
```

Il `try/except` è deliberato: se `/home/user/benchmark` non è montato o il file
non c'è, la TUI deve partire lo stesso, solo senza i consigli.

`benchmark_preset.py` è nella stessa directory di `tui_app.py`, quindi l'import
funziona senza modifiche a `sys.path`. Se la TUI viene avviata da un'altra
directory, aggiungi `sys.path.insert(0, os.path.dirname(__file__))`.

---

## 2. La funzione da mostrare

```python
def _consiglio_benchmark(nome_modello=None, ctx_minimo=0):
    """Il preset consigliato per un modello, dai benchmark misurati.

    Ritorna None se non ci sono dati: la TUI semplicemente non mostra niente.
    Non solleva mai: un consiglio mancante non deve fermare l'avvio.
    """
    if benchmark_preset is None:
        return None
    try:
        return benchmark_preset.consigliato(nome_modello, ctx_minimo=ctx_minimo)
    except Exception:
        return None
```

---

## 3. Le righe da chiamare

Il punto esatto dipende da dove la TUI mostra il modello caricato. **Scegli il
punto A o il B**, non entrambi.

### A) Quando l'utente sceglie un modello

Subito dopo che il modello è stato risolto e prima di comporre i flag di avvio,
così i flag possono ereditare il contesto consigliato:

```python
consiglio = _consiglio_benchmark(modello_selezionato)
if consiglio and consiglio.get("ctx"):
    ctx_consigliato = consiglio["ctx"]
    tok_s_attesi = consiglio.get("gen_tok_s")
```

### B) Come riga informativa nel pannello di un modello

```python
consiglio = _consiglio_benchmark(modello_selezionato)
if consiglio:
    righe.append(f"Consigliato: {consiglio['preset']}  "
                 f"{consiglio['gen_tok_s']:.1f} tok/s a ctx {consiglio['ctx']}")
```

---

## 4. L'API, in pratica

```python
import sys; sys.path.insert(0, "/home/user/ai_tui")
from benchmark_preset import consigliato, consigliato_migliore, riepilogo, kv_mib

consigliato("IsValorum")        # per nome, anche parziale: "isvalorum", "carnice"
consigliato()                    # il migliore in assoluto
consigliato("IsValorum", ctx_minimo=80000)   # solo config da 80k in su
consigliato_migliore(escludi_rpc=True)       # se Miky è spento
riepilogo()                      # una riga per modello, per la tabella
kv_mib("IsValorum", 100_000, "q4_0")         # 549.0 MiB di KV a 100k
```

Cosa restituisce `consigliato()` (dall'esecuzione reale):

```python
{'modello': 'IsValorum Qwen3.8-35B-A3B Distill Abliterated',
 'chiave': 'isvalorum-qwen38-35b-a3b',
 'path_gguf': '/mnt/lm_models/models/IsValorum/.../....gguf',
 'preset': 'B3-rpc-f16-80k',
 'ctx': 81920,
 'cache': 'f16',
 'rpc': True,
 'gen_tok_s': 22.503,
 'prompt_tok_s': 397.47,
 'vram_locale_mib': 6409, 'vram_miky_mib': None,
 'perche': 'usa anche la GPU di Miky via RPC, cache KV f16',
 'alternativa': {'preset': 'locale-no-rpc-q4_0-prompt8028', 'gen_tok_s': 22.04, 'ctx': 81920},
 'n_config_misurate': 22,
 'fonte': '/home/user/benchmark/modeli/isvalorum-qwen38-35b-a3b.json'}
```

---

## 5. Le trappole, in ordine di pericolo

**1. I crash sono dentro `config`, non in un campo separato.**
Un record scartato ha `scartato_per` valorizzato **e** può avere un
`gen_tok_s` assurdo. **Non filtrare per `gen_tok_s > 0`**: nel dataset c'è un
record con **1 000 000 tok/s** (artefatto EOS: 1 token generato, `predicted_ms`
0,001). `benchmark_preset` lo scarta già via `_config_valide()`, quindi se usi
quel modulo sei a posto. Se invece leggi i JSON a mano, controlla `scartato_per`
**e** `gen_token >= 8`.

**2. `vram_miky_mib` è quasi sempre `None`.**
Nella campagna matriciale Miky viene interrogata via RPC, non con `nvidia-smi`
locale. Se ti serve il valore usa `kv_miky_mib`, che c'è per i record della
matrice.

**3. `rpc: False` non significa "tutto in VRAM".**
Significa solo "lanciato senza `--rpc`". Il caso da ricordare è
`A1-locale-q4_0-80k`: è locale, ma il modello da 13,64 GiB **non entra** nella
3050 da 8 GB, quindi 13 570 MiB restano mmap-ati su CPU. Non trattare `locale`
come "sta tutto in VRAM".

**4. Il contesto dichiarato non è il contesto misurato.**
`locale-no-rpc-q4_0-prompt8028` ha `ctx: 81920` ma `prompt_token: 8028`. Se ti
serve il contesto reale usa `prompt_token`, non `ctx`.

**5. Non scrivere mai in `modeli/`.**
L'unico scrittore è `/home/user/benchmark/aggrega.py`. Se la TUI deve salvare un
benchmark nuovo, il file grezzo va in `/home/user/benchmark/raw/` e poi si
ricalcola tutto con `python3 /home/user/benchmark/aggrega.py`.

**6. Il modulo non va importato da `backend.py`.**
`backend.py` non deve conoscere i benchmark: tiene i preset e le flag di
avvio, e deve funzionare anche senza benchmark. La dipendenza va in un solo
verso, dalla TUI verso `benchmark_preset`.

---

## 6. La raccomandazione che la TUI mostrerà

Se non hai ancora deciso cosa mostrare, questa è la riga:

> **IsValorum Qwen3.8-35B-A3B · ctx 81920 · cache KV f16 · via RPC · 22,5 tok/s**

Per i modelli piccoli: **Qwen3.8-9B in locale sulla 3050 (35,8 tok/s,
5 761 MiB)** e **Qwen3.5-4B via RPC sulla 5050 (31,5 tok/s, 3 904 MiB)** misurati
**in contemporanea**: è la coppia secondaria consigliata.

Attenzione: il 35B e il 9B **non possono girare insieme**, il 35B vuole
entrambe le GPU. Se la TUI offre "modello primario + secondario", deve
bloccare quella combinazione.

Il dettaglio e le due correzioni ai dati precedenti sono in
`/home/user/benchmark/PRESET-CONSIGLIATI.md`.

---

## 7. Verifica

```bash
cd /home/user/ai_tui && python3 benchmark_preset.py
```

Stampa i 5 modelli e il consigliato in assoluto. Se la TUI mostra un preset
diverso da `B3-rpc-f16-80k` / 22.5 tok/s per IsValorum, c'è un bug di filtraggio
(trappola 1).
