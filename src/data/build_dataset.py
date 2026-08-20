"""Transforma JSON-ul brut de pe OLX intr-un tabel curat, gata de modelare."""

import glob
import os
import sys

import pandas as pd

from src.scraper.storage import load_offers

PROCESSED_DIR = "data/processed"

ETAJE = {
    "demisol": -1, "parter": 0,
    "fl_1": 1, "fl_2": 2, "fl_3": 3, "fl_4": 4, "fl_5": 5,
    "fl_6": 6, "fl_7": 7, "fl_8": 8, "fl_9": 9, "fl_10": 10,
    "mansarda": None,
}

CONSTRUCTIE_ORDINAL = {
    "inainte-de-1977": 0,
    "1977-1990": 1,
    "1990-2000": 2,
    "dupa-2000": 3,
}

CAMERE = {"1 camera": 1, "2 camere": 2, "3 camere": 3, "4+ camere": 4}


def extrage_param(oferta, cheie):
    """Cauta un parametru dupa cheie in lista 'params' a unei oferte."""
    for p in oferta.get("params", []):
        if p["key"] == cheie:
            return p["value"]
    return None


def numar_sau_none(text):
    """Converteste un text in float; returneaza None daca nu se poate."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def oferta_la_rand(oferta):
    """Transforma o oferta JSON intr-un dictionar plat, cu o cheie per coloana."""
    pret = extrage_param(oferta, "price") or {}
    suprafata = extrage_param(oferta, "m") or {}
    etaj = extrage_param(oferta, "floor") or {}
    constructie = extrage_param(oferta, "constructie") or {}
    harta = oferta.get("map") or {}

    return {
        "id": oferta["id"],
        "pret": pret.get("value"),
        "moneda": pret.get("currency"),
        "negociabil": bool(pret.get("negotiable")),
        "suprafata_mp": numar_sau_none(suprafata.get("key")),
        "camere": CAMERE.get(oferta.get("_camere")),
        "etaj": ETAJE.get(etaj.get("key")),
        "etaj_eticheta": etaj.get("label"),
        "an_constructie": constructie.get("key"),
        "an_constructie_ord": CONSTRUCTIE_ORDINAL.get(constructie.get("key")),
        "judet": oferta.get("_judet"),
        "oras": oferta.get("location", {}).get("city", {}).get("name"),
        "lat": harta.get("lat"),
        "lon": harta.get("lon"),
        "agentie": bool(oferta.get("business")),
        "data_postare": oferta.get("created_time", "")[:10],
        "url": oferta.get("url"),
    }


def construieste_dataframe(oferte):
    """Creeaza un DataFrame din lista de oferte brute."""
    return pd.DataFrame([oferta_la_rand(o) for o in oferte])


def curata(df):
    """Aplica filtrele de calitate si adauga pretul pe metru patrat."""
    print(f"Pornire:                        {len(df):6d} randuri")

    df = df[df["moneda"] == "EUR"].copy()
    print(f"Dupa pastrarea doar a EUR:      {len(df):6d}")

    df = df.dropna(subset=["pret", "suprafata_mp", "camere"])
    print(f"Dupa eliminarea valorilor lipsa:{len(df):6d}")

    df = df[(df["suprafata_mp"] >= 15) & (df["suprafata_mp"] <= 400)]
    print(f"Dupa filtrarea suprafetei:      {len(df):6d}")

    df = df[(df["pret"] >= 5000) & (df["pret"] <= 2_000_000)]
    print(f"Dupa filtrarea pretului:        {len(df):6d}")

    # Coordonate in afara Romaniei = date gresite (am gasit si (0,0)).
    in_tara = (df["lat"].between(43.5, 48.5)) & (df["lon"].between(20.0, 30.0))
    df = df[in_tara]
    print(f"Dupa filtrarea coordonatelor: {len(df):6d}")

    df["pret_per_mp"] = df["pret"] / df["suprafata_mp"]

    jos = df["pret_per_mp"].quantile(0.01)
    sus = df["pret_per_mp"].quantile(0.99)
    df = df[(df["pret_per_mp"] >= jos) & (df["pret_per_mp"] <= sus)]
    print(f"Dupa taierea outlierilor EUR/mp:{len(df):6d}  (pastrat {jos:.0f}-{sus:.0f} EUR/mp)")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    fisiere = glob.glob("data/raw/olx_apartamente_full_*.json")
    if not fisiere:
        print("Nu am gasit date brute in data/raw/.")
        print("Ruleaza mai intai colectarea:  python run_scraper.py")
        sys.exit(1)

    cale_raw = max(fisiere, key=os.path.getmtime)   # cel mai recent snapshot
    print(f"Citesc: {cale_raw}\n")

    oferte = load_offers(cale_raw)
    df = construieste_dataframe(oferte)
    df = curata(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cale_out = os.path.join(PROCESSED_DIR, "apartamente.csv")
    df.to_csv(cale_out, index=False, encoding="utf-8")

    print(f"\nSalvat: {cale_out}")
    print(f"\nDimensiune finala: {df.shape[0]} randuri x {df.shape[1]} coloane")
    print("\n--- primele randuri ---")
    print(df[["pret", "suprafata_mp", "camere", "etaj", "an_constructie", "oras", "pret_per_mp"]].head(8).to_string(index=False))
    print("\n--- statistici ---")
    print(df[["pret", "suprafata_mp", "camere", "etaj", "pret_per_mp"]].describe().round(0).to_string())
    print("\n--- valori lipsa per coloana ---")
    lipsa = df.isna().sum()
    print(lipsa[lipsa > 0].to_string() if lipsa.sum() else "  niciuna")
