"""Al treilea model: Random Forest.

Pe date tabulare de dimensiunea asta, metodele bazate pe arbori bat de obicei
si regresia liniara, si retelele neuronale. Il includem tocmai ca sa verificam
daca afirmatia se confirma pe datele noastre.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.evaluate import (grafic_diagnostic, incarca_date, metrici,
                                 salveaza_rezultate, tabel_rezultate)
from src.models.persist import salveaza_sklearn

N_ARBORI = 300
MIN_FRUNZA = 2
SEED = 42


def antreneaza_padure(X_train, y_train, n_arbori=N_ARBORI, adancime=None,
                      min_frunza=MIN_FRUNZA, seed=SEED):
    """Antreneaza o padure de arbori de regresie."""
    model = RandomForestRegressor(
        n_estimators=n_arbori,
        max_depth=adancime,
        min_samples_leaf=min_frunza,
        n_jobs=-1,              # foloseste toate nucleele
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def sensibilitate_etaj(model, X_test, etaje=(0, 2, 4, 6, 8, 10)):
    """Cat misca modelul predictia cand schimbam DOAR etajul?"""
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
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    print(f"Antrenez Random Forest cu {N_ARBORI} arbori...")
    t0 = time.time()
    rf = antreneaza_padure(X_train, y_train)
    print(f"  gata in {time.time()-t0:.0f}s")

    eticheta = f"Random Forest ({N_ARBORI})"
    rezultate = {eticheta: metrici(y_test, rf.predict(X_test))}

    print()
    print(tabel_rezultate(rezultate).to_string())

    salveaza_rezultate(rezultate)
    salveaza_sklearn(rf, "random_forest",
                     {"tip": "RandomForestRegressor", "arbori": N_ARBORI,
                      "MAE_EUR": round(rezultate[eticheta]["MAE (EUR)"], 1)})
    grafic_diagnostic(y_test, rf.predict(X_test),
                      eticheta, "10_diagnostic_rf.png")

    print()
    print("=== CELE MAI IMPORTANTE FEATURES ===")
    imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    for nume, val in imp.sort_values(ascending=False).head(10).items():
        print(f"  {nume:22s} {val:.4f}  {'#' * int(val * 60)}")

    print()
    print("=== SENSIBILITATE LA ETAJ ===")
    s = sensibilitate_etaj(rf, X_test)
    tabel = pd.DataFrame({"pret prezis (EUR)": s.round(0),
                          "diferenta (%)": (100 * (s / s.iloc[0] - 1)).round(1)})
    tabel.index.name = "etaj"
    print(tabel.to_string())

    print()
    print("=== VERIFICARE OVERFITTING ===")
    m_tr = metrici(y_train, rf.predict(X_train))
    m_te = metrici(y_test, rf.predict(X_test))
    print(f"  MAE train: {m_tr['MAE (EUR)']:,.0f} EUR | MAE test: {m_te['MAE (EUR)']:,.0f} EUR")
    print(f"  R2 train:  {m_tr['R2 (log)']:.3f}      | R2 test:  {m_te['R2 (log)']:.3f}")
