# Nerd Font per la TUI ai_tui

## Situazione attuale (verificata il 2026-09-29)

Windows Terminal e' installato e ha gia' un campo Nerd Font nei **default dei
profili**, ma punta a un font **che non e' installato** sul sistema:

| Dove | Valore | Stato |
|---|---|---|
| `settings.json` → `profiles.defaults.font.face` | `RecMonoCasual Nerd Font` | ❌ **NON installato** (assente da `C:\Windows\Fonts` e dal registro) |
| Registro font Nerd Font di sistema | 96 famiglie registrate | ✅ quasi tutte **JetBrainsMono Nerd Font** |
| `C:\Windows\Fonts\JetBrainsMonoNerdFont*.ttf` | 28 file (Regular, Bold, Italic, Mono…) | ✅ **installata** |

Conclusione: il terminale **non** sta usando una Nerd Font funzionante. Sta
facendo fallback su un font generico, quindi i glifi Nerd Font (Powerline,
`\ue0b0`, icone Material/Seti) e i box-drawing Unicode possono rendersi come
quadratini vuoti o sbagliati. La TUI usa principalmente caratteri ASCII e
Unicode base (box-drawing, `★`, `🔗`, `⚡`), quindi resta leggibile anche senza
Nerd Font, ma le icone si vedono meglio dopo la configurazione qui sotto.

## Si puo' cambiare il font da codice?

**No.** Il font e' una proprieta' del *terminale*, non del processo applicativo.
Un programma in esecuzione dentro il terminale puo' solo scrivere caratteri e
sequenze di escape ANSI/VT:

- escape VT possono cambiare colori, attributi, posizione del cursore,
  title della finestra, ma **non il tipo di carattere**;
- esiste la sequenza OSC 52 (copia negli appunti) e OSC 777 (notifiche), ma
  nessuna sequenza standard per "carica questo TTF";
- Windows Terminal non espone un protocollo di font-switch via escape.

L'unica eccezione sarebbe scrivere direttamente nel profilo del terminale da un
processo esterno, cosa che la TUI non deve fare di nascosto.

Quindi: **configurazione manuale**, 30 secondi.

## Configurazione manuale (Windows Terminal)

1. Apri Windows Terminal.
2. `Ctrl+Shift+,` oppure **Impostazioni** (ingranaggio) → **Profili**.
3. Seleziona il profilo che usi per la TUI (quello con la scheda `xTermLex`,
   o **PowerShell 7.7.0-preview.3**).
4. **Aspetto** → **Face del carattere** → digita:

   ```
   JetBrainsMono Nerd Font
   ```

   Windows Terminal fa il completamento automatico: scegli la voce senza
   suffisso di peso (`JetBrainsMono Nerd Font`, non `... Bold`).
5. Salva con **Applica**. La TUI si ridisegna subito.

### Terminale alternativo (Se preferisci scaricarla)

- Scarica **JetBrainsMono Nerd Font** da
  <https://www.nerdfonts.com/font-downloads> (variante `JetBrainsMono.zip`).
- Estrai e installa tutti i `.ttf` con doppio clic → **Installa per tutti gli utenti**.
- Poi ripeti i passi 1-5 sopra.

## Verifica che funzioni

Dopo il cambio, in un terminale con la TUI attiva:

```
python -c "print('\ue0b0 \ue0b1 \uf013  \uF2A8  \uE0A0  box: \u2500\u2502\u250c\u2510\u2514\u2518')"
```

- Con la Nerd Font: il Powerline e le icone di Material Design appaiono.
- Senza: appaiono quadratini vuoti (tofu) o caratteri sbagliati.

Oppure, dentro la TUI, apri il **Tab 9 (RPC & CUDA13)** e guarda la riga dei
preset: i titoli `★ 🔗 ⚡ ●` e le box `─│┌┐└┘` devono apparire puliti.

## Per la TUI: fallback ASCII

La TUI e' stata scritta per non dipendere dai glifi Nerd Font:

- i bordi dei pannelli usano i box-drawing Unicode standard (`─│┌┐└┘`), che
  esistono in qualsiasi font monospaziato;
- le icone usate sono poche e tutte di fallback Unicode ampio (`★` U+2605,
  `🔗` U+1F517, `⚡` U+26A1, `●` U+25CF, `▊` U+258A);
- se anche questi non renderizzassero, il pannello resta comprensibile: il
  contenuto informative e' tutto testo (`RPC-128k-FAST`, `-c 131072`, `33.32`).

Non usare i codici privati Nerd Font (U+E0B0, U+F013, ...) direttamente nel
codice della TUI: a seconda del font del terminale diventano quadratini.

## Note

- La Nerd Font conta solo per l'aspetto: nessuna impostazione della TUI
  dipende dal font, e `settings.json` non viene mai scritto dalla TUI.
- Se un giorno cambi terminale (WezTerm, Alacritty, conhost classico), la stessa
  procedura vale con nomi diversi: in WezTerm si imposta `font.family` in
  `wezterm.lua`, in conhost il font si sceglie dalle Proprieta' della finestra.
