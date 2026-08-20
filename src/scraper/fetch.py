"""Scraper OLX: aduce anunturi de apartamente prin API-ul public JSON."""

import json 
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://www.olx.ro/api/v1/offers/"
USER_AGENT = "practica-student-scraper/0.1 (proiect educational)"

def fetch_offers(category_id, offset = 0, limit = 40, region_id = None, retries = 3):
    """Aduce o pagina de oferte din categoria data.

    Returneaza dict-ul JSON complet, cu cheile 'data', 'metadata' si 'links'.
    """
    params = {
        "category_id": category_id,
        "offset": offset,
        "limit": limit,
    }

    if region_id is not None:
        params["region_id"] = region_id

    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for incercare in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout = 25) as response:
                return json.loads(response.read().decode("utf-8"))
            
        except urllib.error.HTTPError as e:
            # 4xx = cererea noastra e gresita; reincercarea nu ajuta.
            # Exceptie: 429 = am cerut prea repede, merita sa asteptam.
            if e.code < 500 and e.code != 429:
                raise
            eroare = f"HTTP{e.code}"

        except Exception as e:
            eroare = f"{type(e).__name__}: {e}"

        if incercare == retries:
            raise RuntimeError(f"Esec dupa {retries} incercari: {eroare}")

        pauza = 2 ** incercare
        print(f"    !{eroare} -> reincerc in {pauza}s ({incercare}/{retries})")
        time.sleep(pauza)
            


def fetch_all_offers(category_id, limit = 40, delay = 1.5, max_offset = 1000, region_id = None):
    """Parcurge paginile unei interogari si returneaza ofertele unice.
    """
    results = []
    seen_ids = set()
    offset = 0

    while offset < max_offset:
        data = fetch_offers(category_id, offset = offset, limit = limit, region_id = region_id)
        offers = data["data"]

        for offer in offers:
            if offer["id"] not in seen_ids:
                seen_ids.add(offer["id"])
                results.append(offer)

        print(f" offset = {offset:4d} | primite = {len(offers):3d} | unice acumulate = {len(results)} ")

        if "next" not in data.get("links", {}):
            print(" -> API-ul nu mai are pagina urmaoare, oprire.")
            break


        offset += limit
        time.sleep(delay)
    return results

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    oferte = fetch_all_offers(907, max_offset=80)
    print(f"Test OK: {len(oferte)} oferte")