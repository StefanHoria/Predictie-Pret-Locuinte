# Predicția prețului unei locuințe folosind IA

Proiect de practică — An III.

Estimarea prețului apartamentelor din România pe baza caracteristicilor lor
(suprafață, număr de camere, an de construcție, etaj, localizare), cu o
**comparație între regresia liniară, o rețea neuronală și Random Forest**.

Datele nu provin dintr-un set public, ci sunt colectate direct de pe OLX
printr-un scraper scris pentru acest proiect: **30.128 de anunțuri**, din care
**28.637** rămân după curățare.

## Rezultate

| Model | MAE | MedAE | MAPE | R² (log) |
|---|---:|---:|---:|---:|
| Random Forest (300 arbori) | **14.820 €** | 8.690 € | **14,9%** | 0,850 |
| MLP scikit-learn (64, 32) | 16.144 € | 9.851 € | 16,2% | 0,833 |
| MLP PyTorch (64, 32) | 16.272 € | 9.950 € | 16,5% | 0,830 |
| Regresie liniară | 17.197 € | 10.622 € | 17,4% | 0,815 |
| *Baseline:* €/m² × suprafață | 18.326 € | 11.905 € | 19,2% | 0,792 |
| *Baseline:* mediana | 38.513 € | 26.500 € | 43,1% | −0,001 |

Evaluare pe 5.728 de anunțuri ținute deoparte (20%), neatinse în timpul
antrenării. Tabelul se regenerează în `reports/rezultate_modele.csv`.

### Concluzia comparației

Rețeaua neuronală bate regresia liniară cu ~6% la MAE, dar câștigul cel mai
interesant nu e în cifră, ci în **ce anume învață**. Analiza de sensibilitate
arată că relația dintre etaj și preț este neliniară (în formă de U): etajul 4
este cel mai ieftin, iar etajele 6–10 cele mai scumpe — pentru că blocurile
comuniste sunt P+4, iar peste etajul 5 ești automat într-o clădire modernă cu
lift.

| Etaj | Regresie liniară | Rețea neuronală |
|---:|---:|---:|
| 0 | 0,0% | 0,0% |
| 4 | −1,0% | **−6,1%** |
| 8 | −2,1% | **+3,6%** |

Modelul liniar poate produce doar o pantă monotonă. Rețeaua reconstruiește
forma reală. Random Forest, deși cel mai precis, e cel mai greu de interpretat.

## Instalare

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

`torch` se instalează separat pentru că varianta CPU (~200 MB) stă pe alt
server decât cea cu CUDA (~2,5 GB). Pentru acest proiect CPU e suficient:
rețeaua are 5.441 de parametri și se antrenează în ~20 de secunde.

## Utilizare

### Colectarea datelor (opțional, ~22 minute)

```bash
python run_scraper.py
```

Parcurge 41 de județe × 4 categorii de camere = 164 de interogări către API-ul
public JSON al OLX, cu o pauză de 1,5 secunde între cereri. Salvează JSON-ul
brut în `data/raw/`.

Nu e necesar dacă ai deja un fișier în `data/raw/` — pipeline-ul îl folosește
pe cel mai recent.

### Pipeline-ul complet (~60 secunde)

```bash
python train_all.py
```

Rulează pe rând: curățarea datelor, analiza exploratorie, feature engineering,
și antrenarea celor patru modele. Produce tabelul de rezultate, graficele și
modelele salvate.

### Pași individuali

```bash
python -m src.data.build_dataset      # JSON brut -> CSV curat
python -m src.data.eda                # graficele de analiza exploratorie
python -m src.features.build_features # features + split train/test + encoder
python -m src.models.linear           # regresie liniara + baseline-uri
python -m src.models.mlp_sklearn      # MLP scikit-learn (+ experiment scalare)
python -m src.models.mlp_torch        # MLP PyTorch
python -m src.models.random_forest    # Random Forest
```

Modulele din `src/` se rulează cu `python -m` și cale cu puncte, **din
rădăcina proiectului**. Rulate direct ca fișier (`python src/...py`) dau
`ModuleNotFoundError`, pentru că rădăcina nu ajunge în `sys.path`.

### Notebook-uri

```bash
jupyter lab
```

- `notebooks/01_explorare.ipynb` — explorarea interactivă a datelor
- `notebooks/02_modele.ipynb` — antrenare cu parametri modificabili, comparații,
  analiză de sensibilitate, estimarea prețului pentru un apartament inventat

În VS Code, selectează kernel-ul din `.venv`.

### Folosirea unui model salvat

```python
from src.models.persist import incarca_sklearn
from src.features.build_features import creeaza_apartament
import numpy as np

model, metadate = incarca_sklearn("random_forest")
ap = creeaza_apartament("Cluj-Napoca", suprafata_mp=65, camere=2, etaj=3, an_ord=3)
print(f"{np.exp(model.predict(ap))[0]:,.0f} EUR")
```

## Structura proiectului

```
├── run_scraper.py              colectarea datelor de pe OLX
├── train_all.py                pipeline-ul complet
├── src/
│   ├── scraper/
│   │   ├── fetch.py            cereri HTTP + paginare + retry
│   │   ├── storage.py          salvare/incarcare JSON brut
│   │   └── regions.py          ID-urile judetelor si categoriilor OLX
│   ├── data/
│   │   ├── build_dataset.py    curatare, filtrare outlieri -> CSV
│   │   └── eda.py              graficele de analiza exploratorie
│   ├── features/
│   │   └── build_features.py   feature engineering, split, encoder
│   └── models/
│       ├── evaluate.py         metrici, tabel cumulativ, grafice diagnostic
│       ├── persist.py          salvarea/incarcarea modelelor
│       ├── linear.py           regresie liniara + Ridge + baseline-uri
│       ├── mlp_sklearn.py      MLPRegressor (referinta pentru PyTorch)
│       ├── mlp_torch.py        reteaua in PyTorch, bucla explicita
│       └── random_forest.py    Random Forest
├── notebooks/                  explorare interactiva
├── data/
│   ├── raw/                    JSON brut de la OLX (nu se modifica niciodata)
│   └── processed/              CSV curat, train/test, encoder
├── models/                     modelele antrenate
└── reports/
    ├── rezultate_modele.csv    tabelul comparativ
    └── figures/                graficele
```

## Decizii metodologice

**Ținta este `log(preț)`.** Distribuția prețului are asimetrie 3,49; după
logaritmare scade la 0,17. Regresia liniară presupune erori normal distribuite —
pe prețul brut presupunerea e falsă.

**Suprafața și prețul pe m² al orașului intră tot logaritmate.** Cu ținta deja
logaritmată, asta transformă modelul liniar într-o lege de putere
(`preț ≈ c · suprafață^0,87 · (€/mp)^1,19`), care e forma corectă pentru
imobiliare. Cu suprafața brută, modelul devine exponențial în suprafață și
prezice 3,7 milioane € pentru un apartament de 400 m².

**Împărțirea train/test se face înainte de orice transformare care vede
prețul.** Encoding-ul pe oraș folosește prețul mediu pe m², deci calculat pe
tot setul ar fi data leakage.

**Encoding-ul pe oraș folosește netezire bayesiană.** Cele 604 orașe au între 1
și 1.400 de anunțuri; media unui oraș cu 2 anunțuri e zgomot. Formula
`(n·media_oraș + 10·media_județ) / (n + 10)` trage orașele mici spre județ, iar
județele mici spre media națională.

**Valorile lipsă se imputează, dar se păstrează un indicator.** 24,5% dintre
anunțuri nu au anul construcției. Absența informației e ea însăși informație.

**Etajul rămâne un simplu număr.** Deliberat: nu am construit feature-uri care
să codifice forma de U, ca să putem măsura dacă modelele o descoperă singure.

## Limitări

- **Prețul din anunț ≠ prețul de tranzacționare.** În România se negociază
  tipic 5–10%. Modelul prezice prețul *cerut*, nu valoarea de piață.
- **Dataset-ul e un snapshot** din 20 august 2026. Anunțurile apar și dispar
  continuu; o nouă rulare a scraperului nu va reproduce exact aceleași date.
- **API-ul OLX limitează paginarea la 1000 de rezultate** per interogare. De
  aceea colectarea e partiționată pe județ × număr de camere. Județele mari
  (București, Cluj) rămân totuși plafonate.
- **Anul construcției e o categorie, nu un an** (`înainte de 1977`,
  `1977–1990`, `1990–2000`, `după 2000`), așa cum îl expune OLX.
- **Compartimentarea a fost eliminată**: prezentă în doar 15% din anunțuri.
- **81,9% dintre anunțuri sunt postate de agenții**, ceea ce poate introduce
  bias față de piața reală.

## Notă privind colectarea datelor

Scraperul folosește API-ul JSON public al OLX, cu un User-Agent care se
identifică onest și cu o pauză de 1,5 secunde între cereri (~1,3 cereri pe
secundă în medie). Datele sunt folosite exclusiv în scop educațional, în cadrul
acestui proiect de practică, și **nu sunt redistribuite** — de aceea
`data/raw/` este exclus din controlul de versiuni.
