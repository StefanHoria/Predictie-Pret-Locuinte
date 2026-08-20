"""Metrici de evaluare, exprimate in euro ca sa fie interpretabile."""

import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)

CAI_DATE = {"train": "data/processed/train.csv", "test": "data/processed/test.csv"}
COLOANE_TINTA = ["log_pret", "pret"]


def incarca_date():
    """Incarca train/test si separa features de tinta.

    Scoate din X atat 'log_pret' cat si 'pret' - a doua ar fi leakage direct.
    """
    train = pd.read_csv(CAI_DATE["train"])
    test = pd.read_csv(CAI_DATE["test"])

    X_train = train.drop(columns=COLOANE_TINTA)
    X_test = test.drop(columns=COLOANE_TINTA)
    y_train = train["log_pret"]
    y_test = test["log_pret"]

    return X_train, X_test, y_train, y_test


def metrici(y_log_adevarat, y_log_prezis):
    """Calculeaza metricile in spatiul euro, pornind de la predictii logaritmice."""
    y = np.exp(np.asarray(y_log_adevarat))
    y_prezis = np.exp(np.asarray(y_log_prezis))

    return {
        "MAE (EUR)": mean_absolute_error(y, y_prezis),
        "MedAE (EUR)": median_absolute_error(y, y_prezis),
        "RMSE (EUR)": np.sqrt(mean_squared_error(y, y_prezis)),
        "MAPE (%)": np.mean(np.abs((y - y_prezis) / y)) * 100,
        "R2 (log)": r2_score(y_log_adevarat, y_log_prezis),
        "R2 (EUR)": r2_score(y, y_prezis),
    }


def tabel_rezultate(rezultate):
    """Transforma un dict {nume_model: metrici} intr-un tabel citibil."""
    df = pd.DataFrame(rezultate).T
    formatare = {
        "MAE (EUR)": "{:,.0f}", "MedAE (EUR)": "{:,.0f}", "RMSE (EUR)": "{:,.0f}",
        "MAPE (%)": "{:.1f}", "R2 (log)": "{:.3f}", "R2 (EUR)": "{:.3f}",
    }
    for col, fmt in formatare.items():
        df[col] = df[col].map(fmt.format)
    return df


REZULTATE_CSV = "reports/rezultate_modele.csv"
FIGURI_DIR = "reports/figures"


def salveaza_rezultate(rezultate, cale=REZULTATE_CSV):
    """Adauga metricile in tabelul cumulativ de rezultate.

    Rulari repetate ale aceluiasi model ii actualizeaza randul, nu il dubleaza,
    ca sa poti compara toate modelele intr-un singur fisier.
    """
    os.makedirs(os.path.dirname(cale), exist_ok=True)
    nou = pd.DataFrame(rezultate).T

    if os.path.exists(cale):
        vechi = pd.read_csv(cale, index_col=0)
        combinat = nou.combine_first(vechi)
        combinat.loc[nou.index] = nou          # rulare noua are prioritate
    else:
        combinat = nou

    combinat.index.name = "model"
    combinat = combinat.sort_values("MAE (EUR)")
    combinat.round(4).to_csv(cale, encoding="utf-8")
    print()
    print(f"Rezultate salvate in {cale} ({len(combinat)} modele)")
    return combinat


def grafic_diagnostic(y_log_adevarat, y_log_prezis, nume_model, nume_fisier):
    """Predictii vs. realitate si distributia erorilor - pentru raport."""
    os.makedirs(FIGURI_DIR, exist_ok=True)
    y = np.exp(np.asarray(y_log_adevarat))
    y_prezis = np.exp(np.asarray(y_log_prezis))
    eroare_procent = 100 * (y_prezis - y) / y

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].scatter(y, y_prezis, s=6, alpha=0.2, color="#4C72B0")
    lim = [y.min(), np.percentile(y, 99.5)]
    axes[0].plot(lim, lim, "r--", lw=1.5, label="predictie perfecta")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Pret real (EUR)"); axes[0].set_ylabel("Pret prezis (EUR)")
    axes[0].set_title("Prezis vs. real"); axes[0].legend()

    axes[1].hist(np.clip(eroare_procent, -100, 100), bins=60,
                 color="#55A868", edgecolor="white")
    axes[1].axvline(0, color="crimson", ls="--")
    axes[1].set_xlabel("Eroare (%)"); axes[1].set_ylabel("Numar anunturi")
    axes[1].set_title(f"Distributia erorilor (mediana {np.median(eroare_procent):+.1f}%)")

    axes[2].scatter(y, np.clip(eroare_procent, -150, 150), s=6, alpha=0.2, color="#C44E52")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Pret real (EUR, log)"); axes[2].set_ylabel("Eroare (%)")
    axes[2].set_title("Unde greseste modelul")
    axes[2].set_xticks([20_000, 50_000, 100_000, 250_000, 500_000])
    axes[2].get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    axes[2].minorticks_off()

    fig.suptitle(nume_model, fontsize=13)
    fig.tight_layout()
    cale = os.path.join(FIGURI_DIR, nume_fisier)
    fig.savefig(cale, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  grafic salvat: {cale}")
    return cale
