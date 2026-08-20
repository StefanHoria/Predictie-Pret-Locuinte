"""Salvarea si incarcarea datelor brute aduse de pe OLX."""


import json
from datetime import datetime
from pathlib import Path

RAW_DIR = Path("data/raw")

def save_offers(offers, prefix = "olx_apartamente"):
    """Salveaza lista de oferte intr-un fisier JSON cu data in nume.

    Returneaza calea fisierului scris.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{prefix}_{timestamp}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(offers, f, ensure_ascii=False, indent=2)

    print (f"Salvat : {path}({len(offers)} oferte)")
    return path

def load_offers(path):
    """Incarca o lista de oferte dintr-un fisier JSON salvat anterior."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)