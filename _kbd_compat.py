"""Shim per la tastiera: msvcrt su Windows, termios/termios su Linux.

La TUI importa `msvcrt` per leggere i tasti senza bloccare (kbhit/getwch).
Su Linux msvcrt non esiste, ma l'equivalente si fa con termios in modalita'
raw non canonica: tcgetattr salva le impostazioni, il carattere letto e' subito
disponibile, tcsetattr le ripristina.

Le frecce arrivano come sequenza ESC [ A/B/C/D: va letta la terza parte per
ricavare la direzione, altrimenti l'utente vedrebbe caratteri spazzatura.

I tasti FUNZIONE, invece, sono la trappola: su Linux F6 arriva come
ESC [ 1 7 ~ (cinque caratteri). Se si leggono solo ESC + 2 caratteri, le
cifre 1 e 7 restano nel buffer di input e vengono lette al giro dopo come
tasti normali: F6 "preme" il Tab 1 e il Tab 7. E' il motivo per cui F6
sembrava aprire a volte i preset e a volte le skill. Qui la sequenza viene
consumata TUTTA, fino al byte finale, e restituita per intero a chi chiama.
"""

try:                                    # Windows
    import msvcrt as _kbd

    def kbhit() -> bool:
        return _kbd.kbhit()

    def getwch() -> str:
        return _kbd.getwch()

    def restore() -> None:
        """No-op su Windows: msvcrt non altera lo stato del terminale come
        fa termios, quindi non c'e' nulla da ripristinare. Serve comunque
        che la funzione esista: tui_app.py la chiama sempre prima di ogni
        input() a schermo nero (es. incolla URL GitHub), su qualsiasi OS."""
        pass

    IS_WINDOWS = True

except ImportError:                     # Linux / Unix
    import os
    import sys
    import select
    import termios
    import tty

    import codecs

    IS_WINDOWS = False
    _fd = None
    _old = None
    # Decoder UTF-8 incrementale: si legge un BYTE alla volta dal file
    # descriptor (os.read), quindi le lettere accentate (2 byte) vanno
    # ricomposte qui.
    _dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _read_char() -> str:
        """
        Un carattere letto DIRETTAMENTE dal fd, senza il buffer di sys.stdin.

        Il bug di prima: sys.stdin.read(1) e' bufferizzato e preleva dal
        terminale TUTTI i byte disponibili (es. l'intera sequenza di F11,
        ESC [ 2 3 ~). select() sul fd vedeva poi "vuoto", quindi la sequenza
        veniva troncata a ESC e i byte '[', '2', '3', '~' restavano nel buffer
        di Python, letti al giro dopo come tasti normali ('2'/'3' = cambio
        tab). os.read(fd, 1) non bufferizza: select() resta affidabile.
        """
        while True:
            b = os.read(_fd, 1)
            if not b:
                return ""
            ch = _dec.decode(b)
            if ch:
                return ch

    def _ensure():
        """Porta lo stdin in raw non canonico, una volta sola."""
        global _fd, _old
        if _old is not None:
            return
        _fd = sys.stdin.fileno()
        _old = termios.tcgetattr(_fd)
        new = termios.tcgetattr(_fd)
        # -echo: non mostrare i caratteri premuti
        # ICANON/ECHO off: restituisce subito, anche un singolo tasto
        new[3] &= ~(termios.ICANON | termios.ECHO)
        # IXON off: Ctrl+S/Ctrl+Q arrivano alla TUI invece di essere presi dal
        # driver tty come XOFF/XON (Ctrl+S "congelava" lo schermo).
        # restore() rimette le impostazioni originali all'uscita.
        new[0] &= ~termios.IXON
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        termios.tcsetattr(_fd, termios.TCSADRAIN, new)

    def restore():
        """Rimette la tastiera come l'avevamo trovata. Va SEMPRE chiamata."""
        global _old
        if _old is not None and _fd is not None:
            try:
                termios.tcsetattr(_fd, termios.TCSADRAIN, _old)
            except termios.error:
                pass
            _old = None

    def kbhit() -> bool:
        """True se c'e' un tasto in attesa, senza bloccare."""
        try:
            _ensure()
            r, _, _ = select.select([sys.stdin], [], [], 0)
            return bool(r)
        except (ValueError, OSError, termios.error):
            return False

    def _more(timeout: float = 0.03) -> bool:
        """C'e' un altro byte entro `timeout`? Serve dopo ESC: su SSH o con
        terminali lenti il resto della sequenza puo' arrivare qualche ms
        dopo; senza attesa una freccia diventerebbe un ESC isolato (che
        chiude le Impostazioni) piu' caratteri spuri ('1' = Tab 1)."""
        try:
            r, _, _ = select.select([_fd], [], [], timeout)
            return bool(r)
        except (ValueError, OSError):
            return False

    def getwch() -> str:
        """Legge un carattere. Se e' ESC, legge TUTTA la sequenza che segue."""
        if not kbhit():
            return ""
        try:
            _ensure()
            ch = _read_char()
        except (ValueError, OSError):
            return ""
        if ch == "\x1b" and _more():          # ESC [ ... / ESC O ...
            # Si legge la coda finche' non arriva il byte finale di una
            # sequenza ANSI: '@'..'~'. Solo cosi' le cifre di F6 (ESC [ 1 7 ~)
            # NON restano nel buffer e non vengono lette come Tab 1 / Tab 7.
            seq = ""
            while _more():
                nxt = _read_char()
                seq += nxt
                # '[' (0x5B) e 'O' (0x4F) sono gli INTRODUTTORI CSI/SS3: stanno
                # nell'intervallo 0x40-0x7E ma NON sono il byte finale della
                # sequenza. Fermandosi su di loro la sequenza torna troncata e
                # il resto resta nel buffer: 'A'/'B' delle frecce diventano
                # tasti autonomi (e 'a' = seleziona tutto, 'r', 'n', 'd', 'c'...),
                # mentre '1','7','~' di F6 vengono letti come Tab 1 e Tab 7.
                # Il sintomo era: frecce che non scorrono la lista + cambio
                # pagina a casuale con qualsiasi tasto.
                if len(seq) == 1 and seq in ("[", "O"):
                    continue
                if len(seq) == 1:
                    # ESC + carattere normale = Alt+tasto (es. Alt+1): finito
                    # qui, il tasto successivo non va mangiato.
                    break
                if "\x40" <= nxt <= "\x7e":
                    break
            # Si restituisce la sequenza INTERA: la traduce tui_app.py,
            # che conosce i codici nel TUI_FKEY_CODES.
            return ch + seq
        return ch


__all__ = ["kbhit", "getwch", "IS_WINDOWS", "restore"]