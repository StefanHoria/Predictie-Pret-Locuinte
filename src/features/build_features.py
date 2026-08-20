"""Feature engineering: transforma tabelul curat in matrici gata de antrenare.

Ordinea e esentiala: intai impartim train/test, abia apoi invatam orice
transformare care foloseste pretul. Altfel apare data leakage.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_IN = "data/processed/apartamente.csv"
PROCESSED_DIR = "data/processed"
CALE_ENCODER = "data/processed/encoder.json"

# Cele mai mari centre economice - polii care determina preturile.
ORASE_MARI = {
    "Bucuresti": (44.4268, 26.1025),
    "Cluj-Napoca": (46.7712, 23.6236),
    "Timisoara": (45.7489, 21.2087),
    "Iasi": (47.1585, 27.6014),
    "Constanta": (44.1598, 28.6348),
    "Brasov": (45.6427, 25.5887),
    "Craiova": (44.3302, 23.7949),
    "Galati": (45.4353, 28.0080),
    "Oradea": (47.0465, 21.9189),
    "Sibiu": (45.7983, 24.1256),
}

RAZA_PAMANT_KM = 6371.0
NETEZIRE = 10  # cate anunturi "virtuale" adaugam la media globala


def haversine(lat1, lon1, lat2, lon2):
    """Distanta in km intre doua puncte de pe glob, pe suprafata sferei."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RAZA_PAMANT_KM * np.arcsin(np.sqrt(a))


def adauga_distanta_oras_mare(df):
    """Adauga distanta pana la cel mai apropiat mare centru urban."""
    distante = pd.DataFrame(index=df.index)
    for nume, (lat, lon) in ORASE_MARI.items():
        distante[nume] = haversine(df["lat"], df["lon"], lat, lon)

    df = df.copy()
    df["dist_oras_mare"] = distante.min(axis=1)
    df["cel_mai_apropiat"] = distante.idxmin(axis=1)
    return df


def invata_encoding_oras(train, netezire=NETEZIRE):
    """Invata pretul mediu pe mp pentru fiecare oras, DOAR din setul de train.

    Foloseste netezire bayesiana: orasele cu putine anunturi sunt trase
    spre media judetului, iar judetele mici spre media nationala.
    """
    media_globala = train["pret_per_mp"].mean()

    pe_judet = train.groupby("judet")["pret_per_mp"].agg(["mean", "count"])
    enc_judet = (
        (pe_judet["count"] * pe_judet["mean"] + netezire * media_globala)
        / (pe_judet["count"] + netezire)
    ).to_dict()

    pe_oras = train.groupby(["judet", "oras"])["pret_per_mp"].agg(["mean", "count"])
    enc_oras = {}
    for (judet, oras), rand in pe_oras.iterrows():
        baza = enc_judet.get(judet, media_globala)
        enc_oras[(judet, oras)] = (
            (rand["count"] * rand["mean"] + netezire * baza) / (rand["count"] + netezire)
        )

    return {"oras": enc_oras, "judet": enc_judet, "global": media_globala}


def aplica_encoding_oras(df, enc):
    """Aplica encoding-ul invatat; oras necunoscut -> judet -> media globala."""
    def cauta(rand):
        cheie = (rand["judet"], rand["oras"])
        if cheie in enc["oras"]:
            return enc["oras"][cheie]
        return enc["judet"].get(rand["judet"], enc["global"])

    df = df.copy()
    df["oras_encoded"] = df.apply(cauta, axis=1)
    return df


def construieste_matrice(df, coloane_judet=None, mediane=None):
    """Construieste matricea de features X.

    Returneaza (X, coloane_judet, mediane) ca sa poata fi reutilizate pe test.
    """
    X = pd.DataFrame(index=df.index)

    # Logaritmam marimile care actioneaza multiplicativ asupra pretului.
    # Cu tinta log(pret), asta transforma modelul liniar intr-o lege de putere:
    #   pret = constanta * suprafata^b * (EUR/mp al orasului)^c
    # care este forma corecta pentru imobiliare. Vezi discutia din raport.
    X["log_suprafata"] = np.log(df["suprafata_mp"])
    X["log_oras_eur_mp"] = np.log(df["oras_encoded"])
    X["log_dist_oras_mare"] = np.log1p(df["dist_oras_mare"])
    X["camere"] = df["camere"]
    X["agentie"] = df["agentie"].astype(int)
    X["negociabil"] = df["negociabil"].astype(int)

    # Valorile lipsa: le completam, dar pastram si urma faptului ca lipseau.
    X["etaj_lipsa"] = df["etaj"].isna().astype(int)
    X["an_lipsa"] = df["an_constructie_ord"].isna().astype(int)

    if mediane is None:
        mediane = {
            "etaj": df["etaj"].median(),
            "an_constructie_ord": df["an_constructie_ord"].median(),
        }
    X["etaj"] = df["etaj"].fillna(mediane["etaj"])
    X["an_constructie_ord"] = df["an_constructie_ord"].fillna(mediane["an_constructie_ord"])

    judete = pd.get_dummies(df["judet"], prefix="jud").astype(int)
    if coloane_judet is None:
        coloane_judet = judete.columns.tolist()
    judete = judete.reindex(columns=coloane_judet, fill_value=0)

    X = pd.concat([X, judete], axis=1)
    return X, coloane_judet, mediane


def salveaza_encoder(enc, coloane_judet, mediane, coloane_X, train, cale=CALE_ENCODER):
    """Persista tot ce s-a invatat din train, ca sa poata fi refolosit.

    Fara asta, nu poti construi un apartament nou (notebook, Streamlit)
    pentru ca nu ai de unde sti nici encoding-ul orasului, nici unde e pe harta.
    """
    coord = train.groupby(["judet", "oras"])[["lat", "lon"]].median()

    payload = {
        "oras": [{"judet": j, "oras": o, "valoare": v,
                  "lat": float(coord.loc[(j, o), "lat"]),
                  "lon": float(coord.loc[(j, o), "lon"])}
                 for (j, o), v in enc["oras"].items()],
        "judet": enc["judet"],
        "global": enc["global"],
        "coloane_judet": coloane_judet,
        "mediane": {k: float(v) for k, v in mediane.items()},
        "coloane_X": coloane_X,
        "orase_mari": ORASE_MARI,
    }
    with open(cale, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"Encoder salvat: {cale}")


def incarca_encoder(cale=CALE_ENCODER):
    """Reconstruieste encoder-ul salvat de build_features."""
    with open(cale, encoding="utf-8") as f:
        p = json.load(f)
    p["enc"] = {
        "oras": {(r["judet"], r["oras"]): r["valoare"] for r in p["oras"]},
        "judet": p["judet"],
        "global": p["global"],
    }
    p["coord"] = {(r["judet"], r["oras"]): (r["lat"], r["lon"]) for r in p["oras"]}
    p["orase_disponibile"] = sorted({r["oras"] for r in p["oras"]})
    return p


def creeaza_apartament(oras, suprafata_mp, camere, etaj=3, an_ord=3,
                       judet=None, agentie=False, negociabil=False, encoder=None):
    """Construieste randul de features pentru un apartament inventat.

    Cauta orasul in encoder ca sa afle si pretul mediu pe mp, si coordonatele.
    Returneaza un DataFrame cu o linie, cu exact coloanele asteptate de modele.
    """
    if encoder is None:
        encoder = incarca_encoder()

    potriviri = [(j, o) for (j, o) in encoder["enc"]["oras"]
                 if o == oras and (judet is None or j == judet)]
    if not potriviri:
        raise ValueError(f"Oras necunoscut: {oras!r}. "
                         f"Exemple valide: {encoder['orase_disponibile'][:5]}")
    cheie = potriviri[0]
    judet = cheie[0]

    eur_mp = encoder["enc"]["oras"][cheie]
    lat, lon = encoder["coord"][cheie]
    dist = min(haversine(lat, lon, la, lo)
               for la, lo in encoder["orase_mari"].values())

    rand = {
        "log_suprafata": np.log(suprafata_mp),
        "log_oras_eur_mp": np.log(eur_mp),
        "log_dist_oras_mare": np.log1p(dist),
        "camere": camere,
        "agentie": int(agentie),
        "negociabil": int(negociabil),
        "etaj_lipsa": 0,
        "an_lipsa": 0,
        "etaj": etaj,
        "an_constructie_ord": an_ord,
    }
    for col in encoder["coloane_judet"]:
        rand[col] = int(col == f"jud_{judet}")

    return pd.DataFrame([rand])[encoder["coloane_X"]]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(CSV_IN)
    print(f"Incarcat: {len(df)} randuri")

    df = adauga_distanta_oras_mare(df)
    print(f"Distanta mediana pana la un oras mare: {df['dist_oras_mare'].median():.1f} km")

    # PASUL CRITIC: impartim INAINTE de orice transformare care vede pretul.
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(train)} | Test: {len(test)}")

    enc = invata_encoding_oras(train)
    print(f"Orase invatate din train: {len(enc['oras'])} | judete: {len(enc['judet'])}")

    train = aplica_encoding_oras(train, enc)
    test = aplica_encoding_oras(test, enc)

    X_train, coloane_judet, mediane = construieste_matrice(train)
    X_test, _, _ = construieste_matrice(test, coloane_judet, mediane)

    y_train = np.log(train["pret"])
    y_test = np.log(test["pret"])

    print(f"\nX_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Tinta: log(pret), interval {y_train.min():.2f} - {y_train.max():.2f}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.assign(log_pret=y_train, pret=train["pret"].values).to_csv(
        f"{PROCESSED_DIR}/train.csv", index=False, encoding="utf-8")
    X_test.assign(log_pret=y_test, pret=test["pret"].values).to_csv(
        f"{PROCESSED_DIR}/test.csv", index=False, encoding="utf-8")
    print(f"\nSalvat: {PROCESSED_DIR}/train.csv si test.csv")
    salveaza_encoder(enc, coloane_judet, mediane, X_train.columns.tolist(), train)

    print("\n--- coloane (primele 12) ---")
    print(X_train.columns.tolist()[:12], "...", f"({len(X_train.columns)} total)")
    print("\n--- verificare valori lipsa in X ---")
    print("train:", int(X_train.isna().sum().sum()), "| test:", int(X_test.isna().sum().sum()))
    print("\n--- exemple de distanta ---")
    for oras in ["Bucuresti", "Floresti", "Cluj-Napoca", "Vaslui", "Roman"]:
        sub = df[df["oras"] == oras]
        if len(sub):
            print(f"  {oras:14s} {sub['dist_oras_mare'].median():6.1f} km "
                  f"-> {sub['cel_mai_apropiat'].mode()[0]}")
