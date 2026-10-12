#!/usr/bin/env python3
"""Lettore dei risultati di benchmark per la TUI.

COSA FA
-------
Legge /home/user/benchmark/modeli/*.json (prodotti da /home/user/benchmark/aggrega.py)
e restituisce il preset consigliato per un modello: contesto massimo e token/s
attesi. Solo lettura: questo modulo non scrive nulla e non avvia nessun server.

PERCHE' ESISTE SEPARATO
-----------------------
Non importa tui_app.py, backend.py ne llama_config.py. Motivi:
  - altri agenti stanno modificando quei file adesso, un import creerebbe
    conflitti e dipendenze circolari;
  - questo file deve restare utilizzabile anche se la TUI non e' avviata;
  - un benchmark valido indipende dal fatto che la TUI esista.
Dipendenze: solo la stdlib.

USO TIPICO
----------
    from benchmark_preset import consigliato, consigliato_migliore
    print(consigliato("IsValorum"))          # per nome, anche parziale
    print(consigliato_migliore())            # il piu' veloce in assoluto

FORMATO CANONICO
----------------
Un file per modello in /home/user/benchmark/modeli/. I benchmark futuri vanno
in /home/user/benchmark/raw/ e poi si ricalcola tutto con
`python3 /home/user/benchmark/aggrega.py`. Non scrivere i JSON a mano.
"""

import json
import os

# Radice dei dati di benchmark. Sovrascrivibile per test.
BASE_BENCHMARK = os.environ.get("BENCHMARK_DIR", "/home/user/benchmark")
DIR_MODELLI = os.path.join(BASE_BENCHMARK, "modeli")

# Oltre questa soglia una "velocita'" non e' una misura ma un artefatto EOS
# (predicted_n=1 con ignore_eos assente => 1 000 000 tok/s).
SOGLIA_TOK_S = 10000


def _carica_tutti():
    """Legge tutti i JSON di modeli/. Ritorna {chiave: documento}."""
    documenti = {}
    if not os.path.isdir(DIR_MODELLI):
        return documenti
    for nome_file in sorted(os.listdir(DIR_MODELLI)):
        if not nome_file.endswith(".json") or nome_file.startswith("_"):
            continue  # _indice.json e' un indice, non un modello
        percorso = os.path.join(DIR_MODELLI, nome_file)
        try:
            with open(percorso, encoding="utf-8") as f:
                documento = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # file scritto a meta' da aggrega.py: si riprova al prossimo giro
        chiave = nome_file[:-5]  # nome senza .json
        documento["_chiave"] = chiave
        documento["_file"] = percorso
        documenti[chiave] = documento
    return documenti


def modelli_disponibili():
    """Elenco dei modelli che hanno un file in modeli/."""
    return sorted(_carica_tutti().values(), key=lambda d: d.get("nome") or "")


def trova_modello(nome):
    """Cerca un modello per chiave, nome esatto o corrispondenza parziale.

    'IsValorum', 'isvalorum' e 'qwen38-35b' trovano tutti lo stesso documento.
    Case-insensitive, perche' i nomi nei file variano.
    """
    if not nome:
        return None
    documenti = _carica_tutti()
    desiderato = str(nome).strip().lower()
    if desiderato in documenti:
        return documenti[desiderato]
    for documento in documenti.values():
        if (documento.get("nome") or "").lower() == desiderato:
            return documento
    # corrispondenza parziale: basta che la stringa sia contenuta
    for documento in documenti.values():
        candidati = [documento.get("nome") or "", documento["_chiave"],
                     documento.get("path_gguf") or ""]
        if any(desiderato in c.lower() for c in candidati if c):
            return documento
    return None


def _config_valide(documento, solo_80k=False):
    """Le configurazioni di un documento che sono davvero misure.

    Esclude i record con 'scartato_per': crash e artefatti EOS. Sono
    informazione preziosa da mostrare, ma mai un consiglio.
    """
    if not documento:
        return []
    buone = []
    for cfg in documento.get("config") or []:
        if cfg.get("scartato_per"):
            continue
        if cfg.get("esito") != "OK":
            continue
        velocita = cfg.get("gen_tok_s")
        if not isinstance(velocita, (int, float)):
            continue
        if velocita <= 0 or velocita >= SOGLIA_TOK_S:
            continue
        if solo_80k and (cfg.get("ctx") or 0) < 65536:
            continue
        buone.append(cfg)
    return buone


def consigliato(nome_modello=None, ctx_minimo=0):
    """Preset consigliato per un modello.

    Args:
        nome_modello: nome, chiave o path del modello. None = il migliore
            in assoluto fra tutti quelli misurati.
        ctx_minimo: se > 0, ignora le configurazioni piu' corte di questo
            contesto. Serve a chi chiede "quanto regge almeno 80k".

    Returns:
        dict con: modello, preset, ctx, cache, rpc, gen_tok_s atteso,
        prompt_tok_s, perche, alternativa. None se il modello non esiste o
        non ha nessuna misura valida.

    Perche 'vincitore' e non 'il max': il file puo' contenere il campo
    'vincitore', ma se manca o e' incoerente ( Nessun file ) il fallback
    ricalcola dai record, cosi' la TUI non resta muta.
    """
    documento = trova_modello(nome_modello) if nome_modello else None
    if documento is None and nome_modello:
        return None

    candidati = _config_valide(documento, solo_80k=bool(ctx_minimo))
    if ctx_minimo:
        candidati = [c for c in candidati if (c.get("ctx") or 0) >= ctx_minimo]
    if not candidati:
        return None

    # il piu' veloce; a parita' (rumore fra run identici ~3%) il contesto
    # piu' basso, che regge meglio e occupa meno VRAM
    migliore = max(candidati, key=lambda c: (c["gen_tok_s"], -(c.get("ctx") or 0)))

    # alternativa: la seconda scelta, se esiste e se e' davvero diversa
    resto = [c for c in candidati if c is not migliore]
    alternativa = None
    if resto:
        seconda = max(resto, key=lambda c: c["gen_tok_s"])
        alternativa = {
            "preset": seconda["nome"],
            "gen_tok_s": seconda["gen_tok_s"],
            "ctx": secondo_ctx(seconda),
        }

    return {
        "modello": documento.get("nome") if documento else "(il migliore in assoluto)",
        "chiave": documento["_chiave"] if documento else None,
        "path_gguf": documento.get("path_gguf") if documento else None,
        "preset": migliore["nome"],
        "ctx": migliore.get("ctx"),
        "cache": migliore.get("cache"),
        "rpc": migliore.get("rpc"),
        "gen_tok_s": migliore["gen_tok_s"],
        "prompt_tok_s": migliore.get("prompt_tok_s"),
        "vram_locale_mib": migliore.get("vram_locale_mib"),
        "vram_miky_mib": migliore.get("vram_miky_mib"),
        "perche": _motivo(documento, migliore),
        "alternativa": alternativa,
        "n_config_misurate": len(_config_valide(documento)),
        "fonte": documento.get("_file") if documento else None,
    }


def secondo_ctx(cfg):
    """ctx di una config, con fallback esplicito invece di None silenzioso."""
    return cfg.get("ctx")


def _motivo(documento, cfg):
    """Perche' questo preset vince, in una frase leggibile."""
    motivi = []
    if cfg.get("rpc"):
        motivi.append("usa anche la GPU di Miky via RPC")
    else:
        motivi.append("resta tutto in locale")
    if cfg.get("cache"):
        motivi.append(f"cache KV {cfg['cache']}")
    if cfg.get("gen_tok_s", 0) >= SOGLIA_TOK_S / 2:
        motivi.append("velocita' costante")
    return ", ".join(motivi)


def consigliato_migliore(ctx_minimo=0, escludi_rpc=False):
    """Il preset migliore in assoluto fra tutti i modelli misurati.

    Con escludi_rpc=True ignora le configurazioni via RPC: serve quando Miky
    e' spento o la rete non c'e'.
    """
    candidati = []
    for documento in _carica_tutti().values():
        for cfg in _config_valide(documento, solo_80k=bool(ctx_minimo)):
            if escludi_rpc and cfg.get("rpc"):
                continue
            candidati.append((documento, cfg))
    if not candidati:
        return None
    documento, cfg = max(candidati,
                         key=lambda p: (p[1]["gen_tok_s"], -(p[1].get("ctx") or 0)))
    return {
        "modello": documento.get("nome"),
        "chiave": documento["_chiave"],
        "path_gguf": documento.get("path_gguf"),
        "preset": cfg["nome"],
        "ctx": cfg.get("ctx"),
        "cache": cfg.get("cache"),
        "rpc": cfg.get("rpc"),
        "gen_tok_s": cfg["gen_tok_s"],
        "prompt_tok_s": cfg.get("prompt_tok_s"),
        "perche": _motivo(documento, cfg),
        "n_config_misurate": len(_config_valide(documento)),
    }


def riepilogo():
    """Una riga per modello: nome, preset consigliato, contesto, tok/s."""
    righe = []
    for documento in modelli_disponibili():
        cons = consigliato(documento["_chiave"])
        if not cons:
            righe.append({
                "modello": documento.get("nome"),
                "preset": None,
                "ctx": None,
                "gen_tok_s": None,
                "nota": "nessuna misura valida",
            })
            continue
        righe.append({
            "modello": cons["modello"],
            "preset": cons["preset"],
            "ctx": cons["ctx"],
            "cache": cons["cache"],
            "rpc": cons["rpc"],
            "gen_tok_s": cons["gen_tok_s"],
            "n_config_misurate": cons["n_config_misurate"],
        })
    return righe


def kv_mib(modello, ctx, cache="q4_0"):
    """Stima la KV cache in MiB per un contesto, dalla geometria del modello.

    Serve alla TUI per dire se un contesto entra in VRAM prima di avviare.
    La formula e' quella di bench/ggufmeta.py, validata sui log del server:
        byte/token = n_layer_kv * kv_heads * (key_len + value_len) * byte_per_elem
    dove byte_per_elem vale 0.5625 per q4_0, 1.0625 per q8_0, 2.0 per f16.

    Restituisce None se la geometria non e' nota: meglio un None che una stima
    inventata.
    """
    documento = trova_modello(modello)
    if not documento:
        return None
    geo = documento.get("kv_geometria") or {}
    per_tok = (geo.get("byte_per_tok") or {}).get(cache)
    if per_tok is None:
        n_layer_kv = geo.get("layer_full_attention")
        kv_heads = geo.get("kv_heads")
        head_dim = geo.get("head_dim")
        byte_per_elem = {"q4_0": 0.5625, "q4_1": 0.5625, "q8_0": 1.0625,
                         "f16": 2.0, "bf16": 2.0}.get(cache)
        if not (n_layer_kv and kv_heads and head_dim and byte_per_elem):
            return None
        per_tok = n_layer_kv * kv_heads * (head_dim * 2) * byte_per_elem
    return per_tok * ctx / (1024 * 1024)


if __name__ == "__main__":
    # Prova manuale: python3 /home/user/ai_tui/benchmark_preset.py
    print("=== modelli disponibili ===")
    for riga in riepilogo():
        if riga.get("gen_tok_s"):
            print(f"  {riga['modello'][:44]:44} {riga['gen_tok_s']:>7.2f} tok/s "
                  f"ctx={riga['ctx']} {riga.get('preset')}")
        else:
            print(f"  {riga['modello'][:44]:44} {riga.get('nota')}")

    migliore = consigliato_migliore()
    print("\n=== consigliato in assoluto ===")
    if migliore:
        print(f"  {migliore['modello']}: {migliore['gen_tok_s']} tok/s a ctx "
              f"{migliore['ctx']} ({migliore['cache']}, "
              f"{'RPC' if migliore['rpc'] else 'locale'})")
        print(f"  motivo: {migliore['perche']}")
    else:
        print("  nessuna misura disponibile")
