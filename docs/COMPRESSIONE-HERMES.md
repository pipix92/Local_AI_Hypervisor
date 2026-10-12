# Compressione Hermes — parametri reali (verificati nel codice 2026-09-29)

Fonte: `C:\Users\User\AppData\Local\hermes\hermes-agent\agent\context_compressor.py`
Verificato contro il `SessionDB` reale, non con stub.
Report completo: `session-audit-work\VERIFICATION-COMPRESSION-REPORT.md`

## Config attiva (dopo le correzioni)

```yaml
compression:
  enabled: true
  threshold: 0.45                 # INERTE, vedi sotto
  target_ratio: 0.4               # MAI LETTO, vedi sotto
  protect_last_n: 20
  protect_first_n: 3
  proactive_prune_tokens: 25000   # era 4000 -> troppo aggressivo
  proactive_prune_min_result_chars: 3000
  proactive_prune_min_reclaim_tokens: 4096   # era 2048
```

Backup: `config.yaml.bak-compr-20260929T000337`, `config.yaml.bak-compr2-20260929T003829`

## Due parametri sono inerti — non toccarli più

### threshold → forzato al 75%

```python
_SMALL_CTX_WINDOW_LIMIT      = 512_000   # context_compressor.py:1007
_SMALL_CTX_THRESHOLD_PERCENT = 0.75      # :1008

def _effective_threshold_percent(context_length, threshold_percent):
    if context_length and context_length < _SMALL_CTX_WINDOW_LIMIT:
        return max(threshold_percent, _SMALL_CTX_THRESHOLD_PERCENT)
    return threshold_percent
```

Il floor è **raise-only**. Su Space Bunny Free (256K) i valori 0.45 / 0.60 / 0.74
producono lo stesso trigger a 192.000 token (75%).

**Peggio:** l'attributo `threshold_percent` continua a *riportare* 0.45 nei log,
quindi la config sembra applicata mentre il trigger è al 75%.

### target_ratio → mai letto con tail_mode='lean'

```python
@property
def tail_token_budget(self):                     # context_compressor.py:1942
    if self._tail_token_budget is None:
        if getattr(self, "tail_mode", "lean") == "lean":
            self._tail_token_budget = max(LEAN_TAIL_FLOOR_TOKENS,
                                          min(LEAN_TAIL_CAP_TOKENS,
                                              int(self.context_length * 0.025)))
        else:
            self._tail_token_budget = int(self.threshold_tokens * self.summary_target_ratio)
    return self._tail_token_budget
```

Solo il ramo `legacy` consuma `summary_target_ratio`.
A 256K: lean = 10.000 · legacy = 60.000.

## Cosa FUNZIONA

**Nessuna perdita di contenuto** — verificato contro il DB reale:

```
prune committed: 8 righe, live 44 -> 44
canary nel transcript live : False
canary presente nel DB     : True     # archiviata, recuperabile
active=44  archived(inactive)=44
```

`archive_and_compact()` archivia l'integrale con `active=0`, recuperabile via
`session_search`. È la proprietà che mancava al plugin `groq-context-compress`.

Funzionano anche: dedup lossless (10 copie → 1, zero contenuto unico perso),
isteresi di cache (2ª chiamata immediata = no-op pulito), `protect_last_n=20` /
`protect_first_n=3` rispettati, `min_result_chars: 3000` (con 8000 non
selezionava **zero** righe).

## Perché proactive_prune_tokens era il vero problema

È **indipendente** dal trigger di compattazione. Con 4000 il primo commit
cache-breaking arrivava a **8 turni ≈ 10.487 token = 4,1%** della finestra.
`min_reclaim: 2048` (metà del default 4096 del codice) lo rendeva ancora più
facile da superare: a 2.476 token recuperati l'8-turn prune passava il gate.

Portati a 25000 / 4096 il primo intervento arriva ~50 turni.

## Limite noto residuo

Il path LLM di `compress()` tronca head/tail 6000 char per messaggio prima del
summarizer: è l'unico punto dove una canaria può sparire senza archive trail.
Preesistente, non introdotto dai nuovi parametri.

## Come disattivare il plugin Groq (fatto 2026-09-29)

Root cause: OmniRoute era in **self-deadlock SQLite** (WAL congelato), quindi la
disattivazione via API falliva e lo stato restava `active` per sempre.
Riavvio → `POST /api/plugins/groq-context-compress/deactivate` → 200 →
DB `status=inactive, enabled=0`. File archiviati in
`C:\Users\User\.omniroute\plugins-archive\groq-context-compress_20260929_003315\`.
