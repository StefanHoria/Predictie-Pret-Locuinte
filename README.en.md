*[Română](README.md) · **English***

# House Price Prediction with Machine Learning

Internship project — 3rd year.

Estimating apartment prices in Romania from their characteristics (floor area,
number of rooms, construction period, floor level, location), with a
**comparison between linear regression, a neural network, and Random Forest**.

The data does not come from a public dataset. It was collected directly from
OLX — Romania's largest classifieds platform — using a scraper written for this
project: **30,128 listings**, of which **28,637** survive cleaning.

## Results

| Model | MAE | MedAE | MAPE | R² (log) |
|---|---:|---:|---:|---:|
| Random Forest (300 trees) | **€14,820** | €8,690 | **14.9%** | 0.850 |
| MLP scikit-learn (64, 32) | €16,144 | €9,851 | 16.2% | 0.833 |
| MLP PyTorch (64, 32) | €16,272 | €9,950 | 16.5% | 0.830 |
| Linear regression | €17,197 | €10,622 | 17.4% | 0.815 |
| *Baseline:* €/m² × area | €18,326 | €11,905 | 19.2% | 0.792 |
| *Baseline:* median | €38,513 | €26,500 | 43.1% | −0.001 |

Evaluated on 5,728 held-out listings (20%), untouched during training. The table
regenerates into `reports/rezultate_modele.csv`.

### What the comparison shows

The neural network beats linear regression by ~6% on MAE. The more interesting
gain is not in the number, but in **what each model learns**.

A sensitivity analysis — hold every feature fixed, vary one, observe the
prediction — reveals that the relationship between floor level and price is
non-linear and U-shaped: floor 4 is the cheapest, floors 7–9 the most expensive.

| Floor | Linear regression | Neural network |
|---:|---:|---:|
| 0 (ground) | 0.0% | 0.0% |
| 4 | −1.0% | **−6.1%** |
| 8 | −2.1% | **+3.6%** |

The explanation lies in Romania's housing stock. Apartment blocks built between
1960 and 1990 are predominantly ground-floor-plus-four, making floor 4 the top
floor — with the associated roof problems and, frequently, no elevator. Anything
above floor 5 necessarily sits in a newer building with an elevator.

A linear model can only produce a monotone slope. The network reconstructs the
real shape. All three non-linear models find the same pattern independently,
which rules out a library artifact.

### The counter-intuitive result

Random Forest wins on test-set accuracy — but the overfitting analysis
complicates that verdict:

| Model | Train MAE | Test MAE | Gap |
|---|---:|---:|---:|
| Linear regression | €17,024 | €17,197 | +1.0% |
| MLP PyTorch | €16,009 | €16,272 | **+1.6%** |
| Random Forest | €7,620 | €14,820 | **+94.5%** |

Random Forest largely memorizes the training set. The neural network offers the
best balance between accuracy and generalization.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

`torch` installs separately because the CPU build (~200 MB) lives on a different
index than the CUDA build (~2.5 GB). CPU is enough here: the network has 5,441
parameters and trains in about 20 seconds.

## Usage

### Collecting the data (optional, ~22 minutes)

```bash
python run_scraper.py
```

Issues 164 requests to OLX's public JSON API — 41 counties × 4 room-count
categories — with a 1.5-second pause between them. Writes raw JSON to
`data/raw/`.

Skip this if `data/raw/` already holds a snapshot; the pipeline picks the most
recent one.

### Full pipeline (~70 seconds)

```bash
python train_all.py
```

Runs cleaning, exploratory analysis, feature engineering, and trains all four
models. Produces the results table, the figures, and the saved models.

### Individual steps

```bash
python -m src.data.build_dataset      # raw JSON -> clean CSV
python -m src.data.eda                # exploratory figures
python -m src.features.build_features # features + train/test split + encoder
python -m src.models.linear           # linear regression + baselines
python -m src.models.mlp_sklearn      # MLPRegressor (+ scaling experiment)
python -m src.models.mlp_torch        # PyTorch network
python -m src.models.random_forest    # Random Forest
```

Modules under `src/` run with `python -m` and dotted paths, **from the project
root**. Running them as plain files (`python src/...py`) raises
`ModuleNotFoundError`, because the project root never enters `sys.path`.

### Notebooks

```bash
jupyter lab
```

- `notebooks/01_explorare.ipynb` — interactive data exploration
- `notebooks/02_modele.ipynb` — training with adjustable hyperparameters, model
  comparison, sensitivity analysis, price estimation for a hypothetical apartment

In VS Code, select the `.venv` kernel.

### Using a saved model

```python
from src.models.persist import incarca_sklearn
from src.features.build_features import creeaza_apartament
import numpy as np

model, metadata = incarca_sklearn("random_forest")
apartment = creeaza_apartament("Cluj-Napoca", suprafata_mp=65, camere=2,
                               etaj=3, an_ord=3)
print(f"{np.exp(model.predict(apartment))[0]:,.0f} EUR")
```

## Project layout

```
├── run_scraper.py              data collection from OLX
├── train_all.py                full pipeline
├── src/
│   ├── scraper/
│   │   ├── fetch.py            HTTP requests, pagination, retries
│   │   ├── storage.py          raw JSON save/load
│   │   └── regions.py          OLX county and category identifiers
│   ├── data/
│   │   ├── build_dataset.py    cleaning, outlier filtering -> CSV
│   │   └── eda.py              exploratory figures
│   ├── features/
│   │   └── build_features.py   feature engineering, split, encoder
│   └── models/
│       ├── evaluate.py         metrics, cumulative table, diagnostic plots
│       ├── persist.py          model saving/loading
│       ├── linear.py           linear regression + Ridge + baselines
│       ├── mlp_sklearn.py      MLPRegressor (reference for PyTorch)
│       ├── mlp_torch.py        the network in PyTorch, explicit training loop
│       └── random_forest.py    Random Forest
├── notebooks/                  interactive exploration
├── data/
│   ├── raw/                    raw OLX JSON (never modified)
│   └── processed/              clean CSV, train/test, encoder
├── models/                     trained models
└── reports/
    ├── lucrare.md / .docx      the written report (Romanian)
    ├── rezultate_modele.csv    comparison table
    └── figures/                figures
```

Note: identifiers and comments in the source are in Romanian, matching the
report. The structure above maps them to English.

## Methodological decisions

**The target is `log(price)`.** Price skewness is 3.30; after the log transform
it drops to 0.16. Linear regression assumes normally distributed errors — on raw
prices that assumption is false.

**Floor area and the city's price-per-m² are logged as well.** With an already
logged target, this turns the linear model into a power law
(`price ≈ c · area^0.87 · (€/m²)^1.19`), which is the right functional form for
real estate. With raw area, the model becomes exponential in area and predicts
€3.7 million for a 400 m² apartment. See section 8.1 of the report.

**The train/test split happens before any transformation that sees the target.**
City encoding uses mean price per m²; computing it over the full dataset would
leak test information into training.

**City encoding uses Bayesian smoothing.** The 632 cities hold between 1 and
1,445 listings each; the mean of a city with 2 listings is noise. The formula
`(n·city_mean + 10·county_mean) / (n + 10)` pulls small cities toward their
county, and small counties toward the national mean.

**Missing values are imputed, but an indicator is kept.** 24.5% of listings lack
a construction period. The absence of information is itself information.

**Floor level stays a plain number.** Deliberately: no engineered features encode
the U shape, so the comparison measures whether each model discovers it alone.

## Limitations

- **Asking price ≠ transaction price.** Romanian buyers typically negotiate
  5–10%. The model predicts the *asking* price, not market value.
- **The dataset is a snapshot** from 20 August 2026. Listings appear and vanish
  continuously; re-running the scraper will not reproduce it exactly.
- **OLX caps pagination at 1,000 results** per query. Collection is therefore
  partitioned by county × room count, but high-volume counties (Bucharest, Cluj)
  still hit the cap.
- **Construction year is a bucket, not a year** (`before 1977`, `1977–1990`,
  `1990–2000`, `after 2000`), as OLX exposes it.
- **Room layout was dropped**: present in only 15% of listings.
- **83.5% of listings come from agencies**, which may bias the sample relative to
  the whole market.

## A note on data collection

The scraper uses OLX's public JSON API with a User-Agent that identifies itself
honestly and a 1.5-second delay between requests (~1.3 requests per second on
average). The data serves this educational project only and is **not
redistributed** — hence `data/raw/` is excluded from version control.
