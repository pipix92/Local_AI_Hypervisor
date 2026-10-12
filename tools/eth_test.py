#!/usr/bin/env python3
"""
eth_test.py - verifica la scheda di rete e il cavo verso Miky.

  1  test: velocita' di link (negoziata), duplex, latenza e throughput reale
  2  forza la negoziazione a 1000 Mbps full duplex (serve ethtool + sudo)
  3  ripristina tutte le velocita' supportate dalla scheda
  q  esci

Uso: cambi cavo, premi 1. Con 2 il link viene forzato al massimo: se il cavo
non regge 1000 Mbps il link cade o scende, e il test lo mostra.

Nota: la scheda e' una Intel I211 Gigabit: il massimo reale e' 1000 Mbps
(non 1000 Gbps). Il throughput passa da SSH verso Miky: e' un valore
indicativo, limitato anche dalla CPU per la cifratura.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

IFACE = os.environ.get("ETH_IFACE", "enp3s0")
TARGET = os.environ.get("ETH_TARGET", "192.168.1.100")
SSH_KEY = os.path.expanduser("~/.ssh/id_rsa_cluster")
SSH_USER = os.environ.get("ETH_SSH_USER", "root@192.168.1.100")
PAYLOAD_MB = 200
ADVERTISE_1G_FULL = "0x020"   # 1000baseT/Full
ADVERTISE_ALL = None          # ripristino


def sysfs(name: str) -> str:
    try:
        with open(f"/sys/class/net/{IFACE}/{name}", "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def link_state() -> dict:
    return {"carrier": sysfs("carrier") == "1", "speed": sysfs("speed"),
            "duplex": sysfs("duplex"), "operstate": sysfs("operstate")}


def ping(n: int = 20) -> str:
    try:
        out = subprocess.run(["ping", "-c", str(n), "-q", TARGET], capture_output=True, text=True, timeout=n + 10)
        m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)", out.stdout)
        loss = re.search(r"(\d+)% packet loss", out.stdout)
        if m:
            return f"min {m.group(1)} / media {m.group(2)} / max {m.group(3)} ms, perdita {loss.group(1) if loss else '?'}%"
        return "nessuna risposta"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"errore: {exc}"


def throughput() -> str:
    """Invia PAYLOAD_MB di dati a Miky via SSH e misura Mbps effettivi."""
    cmd = ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", SSH_USER,
           "cat > /dev/null 2>/dev/null || powershell -NoProfile -Command \"$null = [Console]::OpenStandardInput().CopyTo([IO.Stream]::Null)\""]
    data = os.urandom(1024 * 1024)
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for _ in range(PAYLOAD_MB):
                p.stdin.write(data)
            p.stdin.close()
        except BrokenPipeError:
            pass                              # SSH ha chiuso: il motivo e' in stderr
        err = p.stderr.read().decode("utf-8", "replace")
        p.wait(timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"errore: {exc}"
    if "HOST IDENTIFICATION HAS CHANGED" in err or "Host key verification failed" in err:
        return ("SSH bloccato: la chiave di Miky e' cambiata (reinstallazione?). "
                "Verifica l'impronta sulla macchina, poi: ssh-keygen -R 192.168.1.100")
    dt = time.time() - t0
    if p.returncode != 0:
        return f"errore SSH ({p.returncode}): {err.strip()[:120]}"
    mbps = PAYLOAD_MB * 8 / dt
    return f"{mbps:,.0f} Mbps effettivi ({PAYLOAD_MB} MB in {dt:.1f} s)"


def report() -> None:
    st = link_state()
    print(f"\n== {IFACE}  ({time.strftime('%H:%M:%S')})")
    print(f"  link     : {'SI' if st['carrier'] else 'NO (cavo scollegato?)'}  stato {st['operstate']}")
    print(f"  velocita : {st['speed']} Mbps  duplex {st['duplex']}")
    if st["carrier"]:
        print(f"  latenza  : {ping()}")
        print(f"  throughput: {throughput()}")
    if st["speed"] not in ("?", "-1") and st["speed"].isdigit() and int(st["speed"]) < 1000:
        print("  ATTENZIONE: negoziato sotto 1000 Mbps: cavo, porta dello switch o cavo usurato.")


def set_advertise(mask) -> None:
    if mask is None:
        cmd = ["sudo", "ethtool", "-s", IFACE, "autoneg", "on"]
        what = "tutte le velocita'"
    else:
        cmd = ["sudo", "ethtool", "-s", IFACE, "autoneg", "on", "advertise", mask]
        what = "solo 1000 Mbps full duplex"
    print(f"  eseguo: {' '.join(cmd)}  ({what}) - sudo puo' chiedere la password")
    subprocess.run(cmd)
    time.sleep(6)                 # la negoziazione richiede qualche secondo
    report()


def main() -> None:
    print("eth_test - 1 test | 2 forza 1000 Mbps | 3 ripristina | q esci")
    report()
    while True:
        try:
            k = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if k == "1":
            report()
        elif k == "2":
            set_advertise(ADVERTISE_1G_FULL)
        elif k == "3":
            set_advertise(ADVERTISE_ALL)
        elif k == "q":
            break
        else:
            print("  tasti: 1 test, 2 forza 1000 Mbps, 3 ripristina, q esci")


if __name__ == "__main__":
    sys.exit(main())
