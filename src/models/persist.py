"""Salvarea si incarcarea modelelor antrenate.

Modelele sklearn si cele PyTorch se salveaza diferit:
  - sklearn: joblib serializeaza obiectul intreg (structura + parametri)
  - PyTorch: se salveaza doar greutatile (state_dict), plus arhitectura,
    pentru ca obiectul in sine depinde de definitia clasei din cod

De aceea exista doua perechi de functii, nu una singura.
"""

import json
import os

import joblib
import torch

MODELE_DIR = "models"


def _asigura_folder():
    os.makedirs(MODELE_DIR, exist_ok=True)


def salveaza_sklearn(model, nume, metadate=None):
    """Salveaza un model sklearn (sau un Pipeline) intr-un fisier .joblib.

    compress=3 e important pentru Random Forest: 300 de arbori cu adancime
    nelimitata ocupa 255 MB necomprimati si 78 MB comprimati, fara nicio
    pierdere de precizie. Costa ~2 secunde in plus la salvare.
    """
    _asigura_folder()
    cale = os.path.join(MODELE_DIR, f"{nume}.joblib")
    joblib.dump({"model": model, "metadate": metadate or {}}, cale, compress=3)
    marime_mb = os.path.getsize(cale) / 1024 / 1024
    print(f"  model salvat: {cale} ({marime_mb:.1f} MB)")
    return cale


def incarca_sklearn(nume):
    """Incarca un model sklearn salvat anterior. Returneaza (model, metadate)."""
    cale = os.path.join(MODELE_DIR, f"{nume}.joblib")
    pachet = joblib.load(cale)
    return pachet["model"], pachet["metadate"]


def salveaza_torch(model, nume, straturi, n_features, metadate=None):
    """Salveaza greutatile unei retele PyTorch, plus ce trebuie ca s-o reconstruim.

    Nu salvam obiectul model direct: ar depinde de calea exacta a clasei si
    s-ar strica la orice refactorizare. state_dict-ul e doar un dictionar de
    tensori - stabil si portabil.
    """
    _asigura_folder()
    cale = os.path.join(MODELE_DIR, f"{nume}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "straturi": list(straturi),
        "n_features": n_features,
        "metadate": metadate or {},
    }, cale)
    print(f"  model salvat: {cale}")
    return cale


def incarca_torch(nume, clasa_model):
    """Reconstruieste reteaua din greutatile salvate.

    clasa_model e clasa RetaPreturi - o primim ca parametru ca sa evitam
    un import circular intre persist.py si mlp_torch.py.
    """
    cale = os.path.join(MODELE_DIR, f"{nume}.pt")
    pachet = torch.load(cale, weights_only=False)

    model = clasa_model(pachet["n_features"], straturi=tuple(pachet["straturi"]))
    model.load_state_dict(pachet["state_dict"])
    model.eval()               # modelele incarcate sunt pentru inferenta
    return model, pachet["metadate"]


def listeaza_modele():
    """Ce modele exista pe disc, cu metadatele lor."""
    if not os.path.isdir(MODELE_DIR):
        return []

    gasite = []
    for fisier in sorted(os.listdir(MODELE_DIR)):
        nume, ext = os.path.splitext(fisier)
        if ext == ".joblib":
            _, meta = incarca_sklearn(nume)
        elif ext == ".pt":
            pachet = torch.load(os.path.join(MODELE_DIR, fisier), weights_only=False)
            meta = pachet["metadate"]
        else:
            continue
        gasite.append({"nume": nume, "tip": ext.lstrip("."), **meta})
    return gasite
