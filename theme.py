"""
theme.py - temi colore, densita' e proporzioni della TUI (tab "Aspetto").

Tre controlli INDIPENDENTI, salvati in ~/.ai_tui_appearance.json:

* theme       : palette colori (5 temi reali + l'originale cyberpunk)
* density     : compatto / normale / ampio (padding e spaziatura righe)
* proportions : quanto spazio relativo prendono i pannelli affiancati

Il FONT non e' fra questi: lo decide l'emulatore di terminale (kitty,
konsole...), non un'app Python che gira dentro. Qui si cambia la densita'
delle informazioni, non il carattere.

Come funziona il colore
-----------------------
Il codice della TUI usa RUOLI semantici negli stili Rich ("bold ok",
"[warn]...", border_style="border"), mai nomi di colore. `ThemedConsole`
risolve ogni ruolo nell'hex del tema attivo AL MOMENTO DEL RENDER: cambiare
tema (anche con Ctrl+T) vale dal frame successivo, senza riavvio.

Per sicurezza anche i vecchi nomi Rich (bright_green, cyan, grey62...),
che possono arrivare da stringhe generate in backend.py, vengono tradotti
nel ruolo equivalente: cosi' nessuna parte dello schermo resta con la
palette vecchia.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.style import Style

APPEARANCE_PATH = os.path.expanduser("~/.ai_tui_appearance.json")

# ---------------------------------------------------------------------------
# Palette: valori ESATTI richiesti (non approssimati)
# ---------------------------------------------------------------------------
PALETTES: Dict[str, Dict[str, str]] = {
    "dracula": {
        "label": "Dracula",
        "background": "#282a36", "foreground": "#f8f8f2", "selection": "#44475a",
        "comment": "#6272a4", "red": "#ff5555", "orange": "#ffb86c",
        "yellow": "#f1fa8c", "green": "#50fa7b", "purple": "#bd93f9",
        "cyan": "#8be9fd", "pink": "#ff79c6",
    },
    "nord": {
        "label": "Nord",
        "background": "#2e3440", "foreground": "#d8dee9", "selection": "#434c5e",
        "comment": "#4c566a", "red": "#bf616a", "orange": "#d08770",
        "yellow": "#ebcb8b", "green": "#a3be8c", "blue": "#5e81ac",
        "cyan": "#88c0d0", "purple": "#b48ead",
    },
    "gruvbox": {
        "label": "Gruvbox Dark",
        "background": "#282828", "foreground": "#ebdbb2", "selection": "#3c3836",
        "comment": "#928374", "red": "#fb4934", "orange": "#fe8019",
        "yellow": "#fabd2f", "green": "#b8bb26", "blue": "#83a598",
        "purple": "#d3869b", "aqua": "#8ec07c",
    },
    "solarized": {
        "label": "Solarized Dark",
        "background": "#002b36", "foreground": "#839496", "selection": "#073642",
        "comment": "#586e75", "red": "#dc322f", "orange": "#cb4b16",
        "yellow": "#b58900", "green": "#859900", "blue": "#268bd2",
        "cyan": "#2aa198", "magenta": "#d33682",
    },
    "onedark": {
        "label": "One Dark",
        "background": "#282c34", "foreground": "#abb2bf", "selection": "#3e4451",
        "comment": "#5c6370", "red": "#e06c75", "orange": "#d19a66",
        "yellow": "#e5c07b", "green": "#98c379", "blue": "#61afef",
        "cyan": "#56b6c2", "purple": "#c678dd",
    },
    # Verde fosforo su nero, stile terminale (aggiunto 2026-10-05).
    "hacker": {
        "label": "Hacker",
        "background": "#0a0e0a", "foreground": "#33ff66", "selection": "#0f2a17",
        "comment": "#1f6b33", "red": "#ff3b3b", "orange": "#ffb000",
        "yellow": "#d4ff00", "green": "#39ff14", "blue": "#00b3ff",
        "cyan": "#00ffcc", "purple": "#b967ff",
    },
    # Scuro neutro moderno (aggiunto 2026-10-05).
    "dark": {
        "label": "Dark",
        "background": "#121212", "foreground": "#e6e6e6", "selection": "#2a2a2a",
        "comment": "#6b6b6b", "red": "#ef5350", "orange": "#ffa726",
        "yellow": "#ffee58", "green": "#66bb6a", "blue": "#42a5f5",
        "cyan": "#4dd0e1", "purple": "#ab47bc",
    },
}
THEME_ORDER: List[str] = ["dracula", "nord", "gruvbox", "solarized", "onedark", "hacker", "dark",
                          "cyberpunk"]

# ---------------------------------------------------------------------------
# Ruoli semantici
# ---------------------------------------------------------------------------
#   fg      testo primario              -> foreground
#   bg      sfondo pannelli             -> background
#   border  bordi/separatori            -> comment
#   muted   testo secondario/etichette  -> comment
#   sel     elemento selezionato/focus  -> selection
#   ok      OK / server online          -> green
#   err     errore / server offline     -> red
#   warn    warning / VRAM quasi piena  -> yellow
#   warn2   warning forte               -> orange
#   metric  valori numerici (tok/s,MiB) -> cyan (aqua su Gruvbox)
#   metric2 valori secondari / info     -> blue
#   accent  titoli, tasti, evidenze     -> purple/magenta/pink
ROLES = ("fg", "bg", "border", "muted", "sel", "ok", "err", "warn", "warn2",
         "metric", "metric2", "accent")


def _roles_for(p: Dict[str, str]) -> Dict[str, str]:
    accent = p.get("pink") or p.get("magenta") or p.get("purple")
    metric = p.get("cyan") or p.get("aqua") or p.get("blue")
    metric2 = p.get("blue") or p.get("purple") or metric
    return {
        "fg": p["foreground"], "bg": p["background"], "border": p["comment"],
        "muted": p["comment"], "sel": p["selection"], "ok": p["green"],
        "err": p["red"], "warn": p["yellow"], "warn2": p["orange"],
        "metric": metric, "metric2": metric2, "accent": accent,
    }


# Il tema originale: nomi Rich del terminale (16/256 colori), nessuno sfondo
# imposto. Utile se il terminale ha gia' un suo tema che si vuole tenere.
CYBERPUNK_ROLES: Dict[str, str] = {
    "fg": "bright_white", "bg": "default", "border": "bright_cyan",
    "muted": "grey62", "sel": "grey19", "ok": "bright_green", "err": "bright_red",
    "warn": "bright_yellow", "warn2": "dark_orange", "metric": "bright_cyan",
    "metric2": "bright_blue", "accent": "bright_magenta",
}

# Vecchi nomi Rich -> ruolo (rete di sicurezza per stringhe non migrate,
# es. markup prodotto in backend.py).
COLOR_TO_ROLE: Dict[str, str] = {
    "green": "ok", "bright_green": "ok", "green3": "ok", "spring_green3": "ok",
    "red": "err", "bright_red": "err", "red1": "err", "dark_red": "err",
    "yellow": "warn", "bright_yellow": "warn", "gold1": "warn",
    "orange1": "warn2", "dark_orange": "warn2", "orange3": "warn2",
    "cyan": "metric", "bright_cyan": "metric", "dark_cyan": "metric",
    "blue": "metric2", "bright_blue": "metric2", "dark_blue": "metric2",
    "magenta": "accent", "bright_magenta": "accent", "purple": "accent",
    "white": "fg", "bright_white": "fg",
    "black": "bg",
    "bright_black": "muted", "grey62": "muted", "grey70": "muted",
    "grey50": "muted", "grey37": "muted", "grey23": "muted", "gray62": "muted",
    "grey19": "sel", "grey15": "sel",
}

DENSITIES: Dict[str, Dict[str, Any]] = {
    # padding pannelli (verticale, orizzontale), righe vuote fra righe di
    # tabella, padding celle (orizzontale)
    "compatto": {"label": "Compatto", "panel_pad": (0, 0), "table_leading": 0, "cell_pad": 0},
    "normale": {"label": "Normale", "panel_pad": None, "table_leading": 0, "cell_pad": 1},
    "ampio": {"label": "Ampio", "panel_pad": (1, 2), "table_leading": 1, "cell_pad": 2},
}
DENSITY_ORDER = ["compatto", "normale", "ampio"]

# (sinistra, destra) per gli split affiancati; "originale" = quelli del
# codice, tab per tab. "server" = colonna del pannello server F6.
PROPORTIONS: Dict[str, Dict[str, Any]] = {
    "originale": {"label": "Originale (per tab)", "split": None, "server": 76},
    "30/70": {"label": "30 / 70", "split": (3, 7), "server": (7, 3)},
    "40/60": {"label": "40 / 60", "split": (4, 6), "server": (6, 4)},
    "50/50": {"label": "50 / 50", "split": (1, 1), "server": (1, 1)},
    "60/40": {"label": "60 / 40", "split": (6, 4), "server": (4, 6)},
    "70/30": {"label": "70 / 30", "split": (7, 3), "server": (3, 7)},
}
PROPORTION_ORDER = ["originale", "30/70", "40/60", "50/50", "60/40", "70/30"]

DEFAULT_APPEARANCE = {"theme": "dracula", "density": "normale", "proportions": "originale"}


class Appearance:
    """Stato dei tre controlli + THEME (dizionario ruolo -> colore) attivo."""

    def __init__(self, path: str = APPEARANCE_PATH) -> None:
        self.path = path
        self.theme = DEFAULT_APPEARANCE["theme"]
        self.density = DEFAULT_APPEARANCE["density"]
        self.proportions = DEFAULT_APPEARANCE["proportions"]
        self.THEME: Dict[str, str] = {}
        self.load()
        self._apply()

    # -- persistenza -------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        if d.get("theme") in THEME_ORDER:
            self.theme = d["theme"]
        if d.get("density") in DENSITIES:
            self.density = d["density"]
        if d.get("proportions") in PROPORTIONS:
            self.proportions = d["proportions"]

    def save(self) -> bool:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"theme": self.theme, "density": self.density,
                           "proportions": self.proportions}, fh, indent=2)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False

    # -- i tre controlli: ognuno tocca SOLO il suo campo ---------------------
    def _apply(self) -> None:
        if self.theme == "cyberpunk":
            self.THEME = dict(CYBERPUNK_ROLES)
        else:
            self.THEME = _roles_for(PALETTES[self.theme])
        _CACHE.clear()

    def set_theme(self, key: str) -> str:
        if key in THEME_ORDER:
            self.theme = key
            self._apply()
            self.save()
        return self.theme

    def cycle_theme(self, step: int = 1) -> str:
        i = THEME_ORDER.index(self.theme)
        return self.set_theme(THEME_ORDER[(i + step) % len(THEME_ORDER)])

    def cycle_density(self, step: int = 1) -> str:
        i = DENSITY_ORDER.index(self.density)
        self.density = DENSITY_ORDER[(i + step) % len(DENSITY_ORDER)]
        self.save()
        return self.density

    def cycle_proportions(self, step: int = 1) -> str:
        i = PROPORTION_ORDER.index(self.proportions)
        self.proportions = PROPORTION_ORDER[(i + step) % len(PROPORTION_ORDER)]
        self.save()
        return self.proportions

    # -- letture comode ----------------------------------------------------
    @property
    def theme_label(self) -> str:
        return "Cyberpunk (originale)" if self.theme == "cyberpunk" else PALETTES[self.theme]["label"]

    @property
    def density_cfg(self) -> Dict[str, Any]:
        return DENSITIES[self.density]

    @property
    def proportion_cfg(self) -> Dict[str, Any]:
        return PROPORTIONS[self.proportions]

    def palette_hex(self) -> List[str]:
        """Tutti i colori ammessi dal tema attivo (per il test di copertura)."""
        return sorted({v.lower() for v in self.THEME.values() if v.startswith("#")})


# Cache stili e istanza unica
_CACHE: Dict[str, Style] = {}
APPEARANCE = Appearance()
_MODIFIERS = {"bold", "b", "dim", "d", "italic", "i", "underline", "u", "blink",
              "blink2", "reverse", "r", "conceal", "c", "strike", "s",
              "underline2", "uu", "frame", "encircle", "overline", "o",
              "not", "on", "none", "default", "link"}


def init_appearance(path: str = APPEARANCE_PATH) -> Appearance:
    APPEARANCE.__init__(path)        # type: ignore[misc]
    return APPEARANCE


def translate(style: str) -> str:
    """'bold ok on sel' -> 'bold #50fa7b on #44475a' secondo il tema attivo."""
    theme = APPEARANCE.THEME
    out: List[str] = []
    for w in style.split():
        lw = w.lower()
        if lw in theme:
            out.append(theme[lw])
        elif lw in COLOR_TO_ROLE:
            out.append(theme[COLOR_TO_ROLE[lw]])
        elif lw.startswith("grey") or lw.startswith("gray"):
            # grey0..grey100 non mappati: scuri -> sel, chiari -> muted
            n = re.sub(r"\D", "", lw)
            out.append(theme["sel"] if n and int(n) < 30 else theme["muted"])
        else:
            out.append(w)
    return " ".join(out)


class ThemedConsole(Console):
    """Console Rich che risolve i ruoli semantici col tema attivo."""

    def get_style(self, name, *, default=None):        # type: ignore[override]
        if isinstance(name, Style):
            return name
        if not isinstance(name, str):
            return super().get_style(name, default=default)
        key = f"{APPEARANCE.theme}|{name}"
        st = _CACHE.get(key)
        if st is not None:
            return st
        try:
            st = self._theme_stack.get(name)
        except Exception:
            st = None
        if st is None:
            try:
                st = Style.parse(translate(name))
            except Exception:
                try:
                    return super().get_style(name, default=default)
                except Exception:
                    if default is not None:
                        return Style.parse(default) if isinstance(default, str) else default
                    raise
        _CACHE[key] = st
        return st


def base_style() -> Style:
    """Stile di fondo dello schermo: testo fg su sfondo bg del tema."""
    t = APPEARANCE.THEME
    if t.get("bg") in (None, "", "default"):
        return Style()
    return Style(color=t["fg"], bgcolor=t["bg"])


def T(role: str) -> str:
    """Lookup esplicito: THEME[role] del tema attivo (hex o nome Rich)."""
    return APPEARANCE.THEME.get(role, role)
