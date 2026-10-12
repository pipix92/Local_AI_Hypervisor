#!/usr/bin/env python3
"""Verifica REALE della lettura dei tasti: si simulano le sequenze ANSI
invece di fidarsi del codice a occhio.

Il test che conta e' 'NESSUN BYTE NON CONSUMATO': e' esattamente li' che
il bug rompeva. Con la condizione di break sbagliata, _kbd_compat tornava
'\x1b[' e lasciava 'A'/'B' nel buffer, che al giro dopo venivano letti
come tasti autonomi -> frecce ferme e cambio pagina a caso.
"""
import io, sys, termios, tty, os, select

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CASI = [
    ("freccia giu' CSI",   "\x1b[B", "DOWN"),
    ("freccia giu' SS3",   "\x1bOB", "DOWN"),
    ("freccia su' CSI",    "\x1b[A", "UP"),
    ("freccia su' SS3",    "\x1bOA", "UP"),
    ("destra CSI",         "\x1b[C", "RIGHT"),
    ("sinistra SS3",       "\x1bOD", "LEFT"),
    ("Home CSI",           "\x1b[H", "HOME"),
    ("End SS3",            "\x1bOF", "END"),
    ("PgUp",               "\x1b[5~", "PGUP"),
    ("PgDn",               "\x1b[6~", "PGDN"),
    ("F6",                 "\x1b[17~", "F6"),
    ("F7",                 "\x1b[18~", "F7"),
    ("ESC da solo",        "\x1b",   "ESC"),
]

def mangia(seq, risposta_attesa):
    """Alimenta _kbd_compat con seq e restituisce (ottenuto, residuo).

    Stub necessario: _ensure() mette in raw il tty e su una pipe non c'e'.
    tutto il resto gira sul serio, ma con dup2 su fd 0: serve perche'
    sia sys.stdin che il fd grezzo usato da kbhit() puntino alla pipe.
    Sostituire solo sys.stdin non basta: kbhit() interroga il fd 0.
    """
    import _kbd_compat
    import tempfile
    vecchio_stdin = sys.stdin
    vecchio_fd0 = os.dup(0)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(seq.encode())
    tmp.flush()
    os.lseek(tmp.fileno(), 0, os.SEEK_SET)
    try:
        os.dup2(tmp.fileno(), 0)          # fd 0 = pipe/temp del test
        sys.stdin = io.TextIOWrapper(io.FileIO(0, "r"))
        orig = _kbd_compat._ensure
        _kbd_compat._ensure = lambda: True
        try:
            ottenuto = _kbd_compat.getwch()
        finally:
            _kbd_compat._ensure = orig
            sys.stdin = vecchio_stdin
    finally:
        os.dup2(vecchio_fd0, 0)           # ripristina fd 0
        os.close(vecchio_fd0)
        tmp.close()
        os.unlink(tmp.name)
    residuo = ottenuto[len(seq):] if ottenuto.startswith(seq) else ottenuto
    return ottenuto, residuo

fatti = falliti = 0
print("=" * 72)
print("LETTURA TASTI: sequenza consumata per intero e nome corretto")
print("=" * 72)
for nome, seq, atteso in CASI:
    ottenuto, residuo = mangia(seq, atteso)
    # il nome atteso lo decide tui_app: stessa chiave in TUI_FKEY_CODES
    from tui_app import TUI_FKEY_CODES
    tradotto = TUI_FKEY_CODES.get(ottenuto, ottenuto if ottenuto in ("UP","DOWN","LEFT","RIGHT","HOME","END","PGUP","PGDN","ESC") else "?")
    ok_seq = ottenuto == seq
    ok_res = residuo == ""
    ok_nom = tradotto == atteso
    if ok_seq and ok_res and ok_nom:
        print(f"  [PASS] {nome:16} {seq!r:12} -> {tradotto:7} buffer VUOTO")
        fatti += 1
    else:
        problema = []
        if not ok_seq: problema.append(f"sequenza troncata: {ottenuto!r}")
        if not ok_res: problema.append(f"RESIDUO nel buffer: {residuo!r}")
        if not ok_nom: problema.append(f"nome {tradotto!r} != {atteso!r}")
        print(f"  [FAIL] {nome:16} {seq!r:12} -> {'; '.join(problema)}")
        falliti += 1

print("=" * 72)
print(f"  {fatti} PASS, {falliti} FAIL")
sys.exit(1 if falliti else 0)