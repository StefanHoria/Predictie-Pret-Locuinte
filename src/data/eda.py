"""Analiza exploratorie: genereaza graficele care ajung in raport."""

import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURI_DIR = "reports/figures"
CSV = "data/processed/apartamente.csv"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def salveaza(fig, nume):
    """Salveaza o figura in reports/figures si o inchide."""
    os.makedirs(FIGURI_DIR, exist_ok=True)
    cale = os.path.join(FIGURI_DIR, nume)
    fig.savefig(cale)
    plt.close(fig)
    print(f"  salvat: {cale}")
    return cale


def fig_distributie_pret(df):
    """Pretul brut vs. logaritmat - argumentul pentru target-ul logaritmic."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df["pret"], bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Distributia pretului (brut)")
    axes[0].set_xlabel("Pret (EUR)")
    axes[0].set_ylabel("Numar anunturi")
    axes[0].axvline(df["pret"].median(), color="crimson", ls="--",
                    label=f"mediana {df['pret'].median():,.0f} EUR")
    axes[0].axvline(df["pret"].mean(), color="darkorange", ls="--",
                    label=f"media {df['pret'].mean():,.0f} EUR")
    axes[0].legend()

    axes[1].hist(np.log(df["pret"]), bins=60, color="#55A868", edgecolor="white")
    axes[1].set_title("Distributia log(pretului)")
    axes[1].set_xlabel("log(Pret)")
    axes[1].set_ylabel("Numar anunturi")

    fig.suptitle("De ce modelam log(pret): asimetria dispare", fontsize=13)
    return salveaza(fig, "01_distributie_pret.png")


def fig_pret_suprafata(df):
    """Relatia dintre suprafata si pret, pe numar de camere."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    esantion = df.sample(min(6000, len(df)), random_state=42)
    axes[0].scatter(esantion["suprafata_mp"], esantion["pret"],
                    s=6, alpha=0.25, color="#4C72B0")
    axes[0].set_xlabel("Suprafata utila (mp)")
    axes[0].set_ylabel("Pret (EUR)")
    axes[0].set_title("Scara liniara")

    axes[1].scatter(esantion["suprafata_mp"], esantion["pret"],
                    s=6, alpha=0.25, color="#55A868")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Suprafata utila (mp, log)")
    axes[1].set_ylabel("Pret (EUR, log)")
    axes[1].set_title("Scara log-log: relatia devine liniara")

    # pe scara logaritmica, ticks-urile automate se suprapun; le fixam manual
    axes[1].set_xticks([20, 30, 50, 75, 100, 150, 250, 400])
    axes[1].set_yticks([20_000, 50_000, 100_000, 250_000, 500_000])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    axes[1].minorticks_off()

    fig.suptitle("Pret vs. suprafata", fontsize=13)
    return salveaza(fig, "02_pret_vs_suprafata.png")


def fig_judete(df, top=16):
    """Cat de mult conteaza locatia: EUR/mp pe judete."""
    ordine = (df.groupby("judet")["pret_per_mp"].median()
                .sort_values(ascending=False).head(top).index)
    sub = df[df["judet"].isin(ordine)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(data=sub, x="judet", y="pret_per_mp", order=ordine,
                ax=ax, showfliers=False)
    ax.set_xlabel("")
    ax.set_ylabel("Pret pe metru patrat (EUR)")
    ax.set_title(f"Pret pe mp - top {top} judete dupa mediana")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    return salveaza(fig, "03_pret_per_mp_judete.png")


def fig_corelatii(df):
    """Matricea de corelatie intre variabilele numerice."""
    numerice = ["pret", "suprafata_mp", "camere", "etaj",
                "an_constructie_ord", "lat", "lon", "pret_per_mp"]
    corr = df[numerice].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Corelatii intre variabilele numerice")
    return salveaza(fig, "04_corelatii.png")


def fig_factori(df):
    """Efectul fiecarui factor categorial asupra pretului pe mp."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    sns.boxplot(data=df, x="camere", y="pret", ax=axes[0], showfliers=False)
    axes[0].set_title("Pret dupa numarul de camere")
    axes[0].set_xlabel("Camere")
    axes[0].set_ylabel("Pret (EUR)")

    etaje = df.dropna(subset=["etaj"]).copy()
    etaje["etaj"] = etaje["etaj"].astype(int)  # altfel apar etichete "3.0"
    sns.boxplot(data=etaje, x="etaj", y="pret_per_mp", ax=axes[1], showfliers=False)
    axes[1].set_title("Pret pe mp dupa etaj")
    axes[1].set_xlabel("Etaj (-1 = demisol, 0 = parter)")
    axes[1].set_ylabel("EUR/mp")

    ordine_ani = ["inainte-de-1977", "1977-1990", "1990-2000", "dupa-2000"]
    ani = df.dropna(subset=["an_constructie"])
    sns.boxplot(data=ani, x="an_constructie", y="pret_per_mp",
                order=ordine_ani, ax=axes[2], showfliers=False)
    axes[2].set_title("Pret pe mp dupa perioada constructiei")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("EUR/mp")
    axes[2].tick_params(axis="x", rotation=30)
    for tick in axes[2].get_xticklabels():
        tick.set_ha("right")

    return salveaza(fig, "05_factori.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(CSV)
    print(f"Incarcat: {len(df)} randuri\n")

    fig_distributie_pret(df)
    fig_pret_suprafata(df)
    fig_judete(df)
    fig_corelatii(df)
    fig_factori(df)

    print("\n=== CORELATII CU PRETUL ===")
    numerice = ["suprafata_mp", "camere", "etaj", "an_constructie_ord", "lat", "lon"]
    c = df[numerice + ["pret"]].corr()["pret"].drop("pret").sort_values(key=abs, ascending=False)
    for nume, val in c.items():
        print(f"  {nume:20s} {val:+.3f}")

    print("\n=== ASIMETRIE (skewness) ===")
    print(f"  pret        : {df['pret'].skew():.2f}")
    print(f"  log(pret)   : {np.log(df['pret']).skew():.2f}")
