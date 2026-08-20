"""Scriptul principal: aduce anunturi de pe OLX si le salveaza in data/raw/."""

import sys
import time

from src.scraper.regions import REGIUNI, CATEGORII_CAMERE
from src.scraper.fetch import fetch_all_offers
from src.scraper.storage import save_offers

MAX_OFFSET = 400
DELAY = 1.5

def colecteaza_tot(regiuni = REGIUNI, max_offset = MAX_OFFSET):
    """Parcurge fiecare combinatie judet x nr. camere si aduna ofertele unice."""
    toate = []
    seen_ids = set()
    total_partitii = len(regiuni) * len(CATEGORII_CAMERE)
    i = 0

    for region_id, judet in regiuni.items():
        for category_id, camere in CATEGORII_CAMERE.items():
            i +=1
            print(f"[{i:3d}/{total_partitii}] {judet:20s} {camere:10s}", end = "", flush = True)

            try:
                oferte = fetch_all_offers(
                    category_id,
                    region_id = region_id,
                    max_offset = max_offset,
                    delay = DELAY,
                )
            except Exception as e:
                print(f"ESUAT ({e})")
                continue

            noi = 0
            for oferta in oferte:
                if oferta["id"] not in seen_ids:
                    seen_ids.add(oferta["id"])
                    oferta["_judet"] = judet
                    oferta["_camere"] = camere
                    toate.append(oferta)
                    noi += 1
            print(f"{len(oferte):4d} aduse, {noi:4d} noi | total : {len(toate)}")
    return toate

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    start = time.time()
    try:
        oferte = colecteaza_tot()
    except KeyboardInterrupt:
        print("\n\nIntrerupt de utilizator, salvarea partiala a datelor.")
        oferte = []

    durata =(time.time() - start) / 60
    print(f"\n=== {len(oferte)} anunturi in {durata:.1f} minute ===")

    if oferte:
        save_offers(oferte, prefix = "olx_apartamente_full")