"""Aceeasi retea neuronala, scrisa de la zero in PyTorch.

Fata de MLPRegressor, aici bucla de antrenare e explicita: se vede fiecare
pas al gradient descent-ului. Tinta de performanta e data de versiunea
sklearn (MAE ~16.100 EUR); daca nu ajungem acolo, avem un bug.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.evaluate import (grafic_diagnostic, incarca_date, metrici,
                                 salveaza_rezultate, tabel_rezultate)
from src.models.persist import salveaza_torch

SEED = 42
STRATURI = (64, 32)
RATA_INVATARE = 1e-3
MARIME_BATCH = 256
EPOCI_MAX = 800
RABDARE = 15            # cate epoci fara imbunatatire pana ne oprim
FRACTIUNE_VALIDARE = 0.1
WEIGHT_DECAY = 1e-4


def fixeaza_seed(seed=SEED):
    """Face rularea reproductibila: acelasi seed -> aceleasi rezultate."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class RetaPreturi(nn.Module):
    """Perceptron multistrat: n_features -> 64 -> 32 -> 1."""

    def __init__(self, n_features, straturi=STRATURI):
        super().__init__()

        blocuri = []
        dim_intrare = n_features
        for dim_iesire in straturi:
            blocuri.append(nn.Linear(dim_intrare, dim_iesire))
            blocuri.append(nn.ReLU())
            dim_intrare = dim_iesire
        blocuri.append(nn.Linear(dim_intrare, 1))   # stratul de iesire, fara activare

        self.retea = nn.Sequential(*blocuri)

    def forward(self, x):
        """Trece datele prin retea. Returneaza un vector, nu o matrice Nx1."""
        return self.retea(x).squeeze(-1)


def pregateste_tensori(X_train, y_train, fractiune_validare=FRACTIUNE_VALIDARE):
    """Imparte train in train/validare si converteste totul in tensori."""
    n_val = int(len(X_train) * fractiune_validare)
    indici = np.random.permutation(len(X_train))
    idx_val, idx_tr = indici[:n_val], indici[n_val:]

    def tensor(date):
        return torch.tensor(np.asarray(date, dtype=np.float32))

    return (tensor(X_train.iloc[idx_tr]), tensor(y_train.iloc[idx_tr]),
            tensor(X_train.iloc[idx_val]), tensor(y_train.iloc[idx_val]))


def antreneaza(model, X_tr, y_tr, X_val, y_val, verbose=True):
    """Bucla de antrenare, cu early stopping pe setul de validare."""
    incarcator = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=MARIME_BATCH, shuffle=True)
    optimizator = torch.optim.Adam(model.parameters(), lr=RATA_INVATARE,
                                   weight_decay=WEIGHT_DECAY)
    criteriu = nn.MSELoss()

    istoric = {"train": [], "validare": []}
    cea_mai_buna = float("inf")
    cele_mai_bune_greutati = None
    epoci_fara_progres = 0

    for epoca in range(1, EPOCI_MAX + 1):
        model.train()                      # activeaza modul de antrenare
        suma_loss = 0.0

        for batch_X, batch_y in incarcator:
            optimizator.zero_grad()        # 1. sterge gradientii de la pasul anterior
            predictii = model(batch_X)     # 2. forward: calculeaza predictiile
            loss = criteriu(predictii, batch_y)   # 3. cat de gresit am fost
            loss.backward()                # 4. backward: calculeaza gradientii
            optimizator.step()             # 5. ajusteaza greutatile

            suma_loss += loss.item() * len(batch_X)

        loss_train = suma_loss / len(X_tr)

        model.eval()                       # dezactiveaza dropout/batchnorm
        with torch.no_grad():              # nu urmarim gradientii la evaluare
            loss_val = criteriu(model(X_val), y_val).item()

        istoric["train"].append(loss_train)
        istoric["validare"].append(loss_val)

        if loss_val < cea_mai_buna - 1e-6:
            cea_mai_buna = loss_val
            cele_mai_bune_greutati = {k: v.clone() for k, v in model.state_dict().items()}
            epoci_fara_progres = 0
        else:
            epoci_fara_progres += 1
            if epoci_fara_progres >= RABDARE:
                if verbose:
                    print(f"  early stopping la epoca {epoca}")
                break

        if verbose and epoca % 20 == 0:
            print(f"  epoca {epoca:3d} | train {loss_train:.5f} | validare {loss_val:.5f}")

    model.load_state_dict(cele_mai_bune_greutati)   # revenim la cel mai bun model
    return istoric


def prezice(model, X):
    """Predictii pentru un DataFrame; returneaza numpy, in spatiul log."""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype=np.float32))
        return model(t).numpy()


def sensibilitate_etaj(model, X_test, etaje=(0, 2, 4, 6, 8, 10)):
    """Cat misca modelul predictia cand schimbam DOAR etajul?"""
    esantion = X_test.sample(300, random_state=SEED)
    rezultat = {}
    for et in etaje:
        v = esantion.copy()
        v["etaj"] = et
        v["etaj_lipsa"] = 0
        rezultat[et] = np.exp(prezice(model, v)).mean()
    return pd.Series(rezultat)


def grafic_invatare(istoric, cale="reports/figures/08_curba_invatare.png"):
    """Curba de invatare: cum scade eroarea, epoca dupa epoca."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(istoric["train"], label="antrenare", lw=1.8)
    ax.plot(istoric["validare"], label="validare", lw=1.8)
    cel_mai_bun = int(np.argmin(istoric["validare"]))
    ax.axvline(cel_mai_bun, color="crimson", ls="--", lw=1,
               label=f"cel mai bun model (epoca {cel_mai_bun+1})")
    ax.set_yscale("log")   # prima epoca are MSE ~60, ar strivi tot restul
    ax.set_xlabel("Epoca"); ax.set_ylabel("MSE pe log(pret), scara log")
    ax.set_title("Curba de invatare - PyTorch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(cale, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  grafic salvat: {cale}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fixeaza_seed()

    X_train, X_test, y_train, y_test = incarca_date()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    X_tr, y_tr, X_val, y_val = pregateste_tensori(X_train, y_train)
    print(f"Antrenare efectiva: {len(X_tr)} | Validare: {len(X_val)}\n")

    model = RetaPreturi(X_train.shape[1])
    n_param = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Parametri antrenabili: {n_param:,}\n")

    t0 = time.time()
    istoric = antreneaza(model, X_tr, y_tr, X_val, y_val)
    print(f"\nAntrenat in {time.time()-t0:.0f}s, {len(istoric['train'])} epoci")

    rezultate = {"MLP PyTorch (64, 32)": metrici(y_test, prezice(model, X_test))}
    print()
    print(tabel_rezultate(rezultate).to_string())

    salveaza_rezultate(rezultate)
    grafic_diagnostic(y_test, prezice(model, X_test),
                      "MLP PyTorch (64, 32)", "09_diagnostic_torch.png")
    grafic_invatare(istoric)
    salveaza_torch(model, "mlp_torch", STRATURI, X_train.shape[1],
                   {"tip": "RetaPreturi", "epoci": len(istoric["train"]),
                    "MAE_EUR": round(rezultate["MLP PyTorch (64, 32)"]["MAE (EUR)"], 1)})

    print()
    print("=== SENSIBILITATE LA ETAJ ===")
    s = sensibilitate_etaj(model, X_test)
    tabel = pd.DataFrame({"pret prezis (EUR)": s.round(0),
                          "diferenta (%)": (100 * (s / s.iloc[0] - 1)).round(1)})
    tabel.index.name = "etaj"
    print(tabel.to_string())

    print()
    print("=== VERIFICARE OVERFITTING ===")
    m_tr = metrici(y_train, prezice(model, X_train))
    m_te = metrici(y_test, prezice(model, X_test))
    print(f"  MAE train: {m_tr['MAE (EUR)']:,.0f} EUR | MAE test: {m_te['MAE (EUR)']:,.0f} EUR")
    print(f"  R2 train:  {m_tr['R2 (log)']:.3f}      | R2 test:  {m_te['R2 (log)']:.3f}")
