"""Ruleaza intreg pipeline-ul, de la datele brute pana la modelele salvate.

Nu include scraperul: acela se ruleaza separat (run_scraper.py), dureaza ~20
de minute si depinde de ce e pe OLX in ziua respectiva. Pipeline-ul de aici
porneste de la JSON-ul deja colectat si e complet reproductibil.

    python train_all.py
"""

import subprocess
import sys
import time

PASI = [
    ("Curatare date          ", "src.data.build_dataset"),
    ("Analiza exploratorie   ", "src.data.eda"),
    ("Feature engineering    ", "src.features.build_features"),
    ("Regresie liniara       ", "src.models.linear"),
    ("MLP scikit-learn       ", "src.models.mlp_sklearn"),
    ("MLP PyTorch            ", "src.models.mlp_torch"),
    ("Random Forest          ", "src.models.random_forest"),
]


def ruleaza(modul):
    """Ruleaza un modul ca subproces si intoarce (reusit, output)."""
    rezultat = subprocess.run(
        [sys.executable, "-m", modul],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return rezultat.returncode == 0, rezultat.stdout + rezultat.stderr


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 62)
    print("  PIPELINE COMPLET - predictia pretului locuintelor")
    print("=" * 62)
    print()

    start = time.time()

    for eticheta, modul in PASI:
        t0 = time.time()
        print(f"  {eticheta} ", end="", flush=True)

        reusit, output = ruleaza(modul)
        durata = time.time() - t0

        if reusit:
            print(f"OK   ({durata:5.1f}s)")
            continue

        # Fiecare pas depinde de rezultatul celui dinainte, deci nu are rost
        # sa continuam: am produce inca sase erori derivate din prima.
        print(f"ESEC ({durata:5.1f}s)")
        print()
        print("=" * 62)
        print(f"  Pipeline oprit la: {modul}")
        print()
        print("  " + "\n  ".join(output.strip().splitlines()[-15:]))
        print("=" * 62)
        sys.exit(1)

    print()
    print("=" * 62)

    print(f"  Gata in {time.time()-start:.0f}s. Rezultate:")
    print("    reports/rezultate_modele.csv   - tabelul comparativ")
    print("    reports/figures/               - graficele")
    print("    models/                        - modelele antrenate")
    print("=" * 62)
