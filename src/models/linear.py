"""Modelul de referinta: regresie liniara, plus doua baseline-uri simple."""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (grafic_diagnostic, incarca_date, metrici,
                                 salveaza_rezultate, tabel_rezultate)
from src.models.persist import salveaza_sklearn


def baseline_mediana(y_train, y_test):
    """Prezice mereu aceeasi valoare: mediana din train. Pragul minim absolut."""
    return np.full(len(y_test), y_train.median())


def baseline_expert(X_test):
    """Regula empirica a unui agent imobiliar: pret = EUR/mp al orasului x suprafata."""
    return X_test["log_oras_eur_mp"] + X_test["log_suprafata"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    X_train, X_test, y_train, y_test = incarca_date()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    rezultate = {}

    rezultate["Baseline: mediana"] = metrici(y_test, baseline_mediana(y_train, y_test))
    rezultate["Baseline: EUR/mp x suprafata"] = metrici(y_test, baseline_expert(X_test))

    liniar = LinearRegression()
    liniar.fit(X_train, y_train)
    rezultate["Regresie liniara"] = metrici(y_test, liniar.predict(X_test))

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(X_train, y_train)
    rezultate["Ridge (alpha=1)"] = metrici(y_test, ridge.predict(X_test))

    print(tabel_rezultate(rezultate).to_string())

    salveaza_rezultate(rezultate)
    grafic_diagnostic(y_test, liniar.predict(X_test),
                      "Regresie liniara", "06_diagnostic_liniar.png")
    salveaza_sklearn(liniar, "regresie_liniara",
                     {"tip": "LinearRegression",
                      "MAE_EUR": round(rezultate["Regresie liniara"]["MAE (EUR)"], 1)})

    print("\n=== COEFICIENTII REGRESIEI LINIARE ===")
    coef = pd.Series(liniar.coef_, index=X_train.columns)
    principale = coef.drop([c for c in coef.index if c.startswith("jud_")])
    print("\nVariabile principale (efect pe log-pret):")
    for nume, val in principale.sort_values(key=abs, ascending=False).items():
        print(f"  {nume:22s} {val:+.5f}   -> {100*(np.exp(val)-1):+7.2f}% per unitate")

    print("\nTop 5 judete cu efect pozitiv:")
    jud = coef[[c for c in coef.index if c.startswith("jud_")]].sort_values(ascending=False)
    for nume, val in jud.head(5).items():
        print(f"  {nume:22s} {val:+.4f}   -> {100*(np.exp(val)-1):+6.1f}%")
    print("Top 5 judete cu efect negativ:")
    for nume, val in jud.tail(5).items():
        print(f"  {nume:22s} {val:+.4f}   -> {100*(np.exp(val)-1):+6.1f}%")

    print("\n=== VERIFICARE OVERFITTING ===")
    m_tr = metrici(y_train, liniar.predict(X_train))
    m_te = metrici(y_test, liniar.predict(X_test))
    print(f"  MAE train: {m_tr['MAE (EUR)']:,.0f} EUR | MAE test: {m_te['MAE (EUR)']:,.0f} EUR")
    print(f"  R2 train:  {m_tr['R2 (log)']:.3f}      | R2 test:  {m_te['R2 (log)']:.3f}")

    reziduuri = y_train - liniar.predict(X_train)
    print(f"\n  Factor de corectie Duan (bias la exp): {np.mean(np.exp(reziduuri)):.4f}")
