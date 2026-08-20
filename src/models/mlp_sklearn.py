"""Retea neuronala cu scikit-learn - baseline pentru implementarea PyTorch.

Scopul nu e performanta maxima, ci un numar de referinta: daca versiunea
PyTorch nu ajunge aici, are un bug.

Include si un mic experiment despre standardizare, pentru ca rezultatul
contrazice regula generala si merita documentat in raport.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (grafic_diagnostic, incarca_date, metrici,
                                 salveaza_rezultate, tabel_rezultate)
from src.models.persist import salveaza_sklearn

warnings.filterwarnings("ignore", category=UserWarning)

ARHITECTURA = (64, 32)
SEED = 42


def creeaza_retea(seed=SEED, **kwargs):
    """MLP-ul propriu-zis, fara niciun fel de preprocesare."""
    return MLPRegressor(
        hidden_layer_sizes=ARHITECTURA,
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # regularizare L2
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,        # opreste cand validarea nu se mai imbunatateste
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed,
        **kwargs,
    )


def construieste_model(scalare, coloane_continue, seed=SEED):
    """scalare: 'tot', 'continue' sau 'nimic'."""
    retea = creeaza_retea(seed)
    if scalare == "tot":
        return make_pipeline(StandardScaler(), retea), retea
    if scalare == "continue":
        ct = ColumnTransformer([("num", StandardScaler(), coloane_continue)],
                               remainder="passthrough")
        return make_pipeline(ct, retea), retea
    return retea, retea


def sensibilitate_etaj(model, X_test, etaje=(0, 2, 4, 6, 8, 10)):
    """Cat misca modelul predictia cand schimbam DOAR etajul?

    Mediem peste 300 de apartamente reale, ca sa nu depindem de un caz izolat.
    """
    esantion = X_test.sample(300, random_state=SEED)
    rezultat = {}
    for et in etaje:
        v = esantion.copy()
        v["etaj"] = et
        v["etaj_lipsa"] = 0
        rezultat[et] = np.exp(model.predict(v)).mean()
    return pd.Series(rezultat)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    X_train, X_test, y_train, y_test = incarca_date()
    continue_ = [c for c in X_train.columns if not c.startswith("jud_")]
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Coloane continue: {len(continue_)} | dummy-uri judet: {X_train.shape[1]-len(continue_)}\n")

    print("=== EXPERIMENT: cum afecteaza standardizarea reteaua ===")
    modele, rezultate = {}, {}
    for scalare, eticheta in [("tot", "MLP (scalat tot)"),
                              ("continue", "MLP (scalat doar continue)"),
                              ("nimic", "MLP (nescalat)")]:
        t0 = time.time()
        model, retea = construieste_model(scalare, continue_)
        model.fit(X_train, y_train)
        modele[eticheta] = model
        rezultate[eticheta] = metrici(y_test, model.predict(X_test))
        print(f"  {eticheta:30s} {retea.n_iter_:3d} epoci, {time.time()-t0:4.1f}s, "
              f"MAE {rezultate[eticheta]['MAE (EUR)']:,.0f} EUR")

    liniar = LinearRegression().fit(X_train, y_train)
    rezultate["Regresie liniara"] = metrici(y_test, liniar.predict(X_test))

    cel_mai_bun = min(modele, key=lambda k: rezultate[k]["MAE (EUR)"])
    print(f"\nCel mai bun: {cel_mai_bun}")

    print()
    print(tabel_rezultate(rezultate).to_string())

    salveaza_rezultate({k: v for k, v in rezultate.items() if k.startswith("MLP")})
    grafic_diagnostic(y_test, modele[cel_mai_bun].predict(X_test),
                      f"MLP sklearn {ARHITECTURA}", "07_diagnostic_mlp.png")
    salveaza_sklearn(modele[cel_mai_bun], "mlp_sklearn",
                     {"tip": "MLPRegressor", "arhitectura": list(ARHITECTURA),
                      "varianta": cel_mai_bun,
                      "MAE_EUR": round(rezultate[cel_mai_bun]["MAE (EUR)"], 1)})

    print()
    print("=== TESTUL DECISIV: invata reteaua forma in U a etajului? ===")
    s_lin = sensibilitate_etaj(liniar, X_test)
    s_mlp = sensibilitate_etaj(modele[cel_mai_bun], X_test)
    comp = pd.DataFrame({
        "liniar (EUR)": s_lin.round(0),
        "liniar (%)": (100 * (s_lin / s_lin.iloc[0] - 1)).round(1),
        "MLP (EUR)": s_mlp.round(0),
        "MLP (%)": (100 * (s_mlp / s_mlp.iloc[0] - 1)).round(1),
    })
    comp.index.name = "etaj"
    print(comp.to_string())

    print()
    print("=== VERIFICARE OVERFITTING ===")
    m_tr = metrici(y_train, modele[cel_mai_bun].predict(X_train))
    m_te = metrici(y_test, modele[cel_mai_bun].predict(X_test))
    print(f"  MAE train: {m_tr['MAE (EUR)']:,.0f} EUR | MAE test: {m_te['MAE (EUR)']:,.0f} EUR")
    print(f"  R2 train:  {m_tr['R2 (log)']:.3f}      | R2 test:  {m_te['R2 (log)']:.3f}")
