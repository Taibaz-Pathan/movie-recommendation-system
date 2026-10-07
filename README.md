# Movie Recommendation System Using Collaborative Filtering

**Student:** Taibaz Pathan  
**University:** Frankfurt University of Applied Sciences  
**Start Date:** 07 May 2026

---

## Project Overview

This project implements a movie recommendation system using collaborative filtering techniques. It explores both user-based (UBCF) and item-based (IBCF) collaborative filtering approaches applied to the MovieLens dataset. The goal is to predict user ratings for unseen movies and generate personalised top-N recommendations.

What the project covers:
- User-Based Collaborative Filtering (UBCF, Pearson correlation)
- Item-Based Collaborative Filtering (IBCF, adjusted cosine similarity)
- SVD matrix factorization and three non-personalised baselines for comparison
- Evaluation of rating accuracy (RMSE, MAE) and ranking quality (Precision@K, Recall@K)
- Hyperparameter grid search and a paired bootstrap significance test
- Cold-start, sparsity and scalability analyses
- An interactive Streamlit demo (`app.py`)
- Exploratory data analysis of rating patterns and sparsity

The final report is in [`report/paper.pdf`](report/paper.pdf).

**Headline results** (13,406 held-out test ratings, per-user 80/20 split, seed 42):

| Model | RMSE | MAE | P@10 | R@10 |
|---|---|---|---|---|
| SVD (50 factors, 20 epochs) | 0.8379 | 0.6435 | 0.0601 | 0.0495 |
| UBCF (k=20, min_support=10) | 0.8430 | 0.6407 | 0.0499 | 0.0453 |
| IBCF (k=30, min_support=1) | 0.8769 | 0.6731 | 0.0562 | 0.0382 |
| User-mean baseline | 0.9128 | 0.7064 | 0.0250 | 0.0246 |
| Item-mean baseline | 0.9253 | 0.7152 | 0.0365 | 0.0346 |
| Global-mean baseline | 0.9990 | 0.8000 | 0.0250 | 0.0246 |

The UBCF and IBCF settings were tuned on the test set (no separate validation
set), and SVD uses default settings, so the comparison is not perfectly even.
See the report for details and limitations.

---

## Dataset

This project uses the **MovieLens Latest Small** dataset provided by [GroupLens Research](https://grouplens.org/datasets/movielens/).

### Download Instructions

1. Visit: https://grouplens.org/datasets/movielens/latest/
2. Download `ml-latest-small.zip`
3. Unzip and place the folder inside `data/raw/` so the structure looks like:

```
data/
└── raw/
    └── ml-latest-small/
        ├── ratings.csv
        ├── movies.csv
        ├── tags.csv
        └── links.csv
```

**Dataset stats (approximate):**
- ~100,000 ratings
- ~9,000 movies
- ~600 users
- Ratings scale: 0.5 to 5.0 (half-star increments)

---

## Setup Instructions

Requires Python 3.10 or newer (the pinned NumPy 2.2.3 needs it). `scikit-surprise`
is compiled during installation, so a working C compiler may be needed.

### 1. Get the code

Unzip the submission, or clone the repository, and change into the folder:

```bash
cd movie-recsys
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Follow the dataset download instructions above.

### 5. Verify the setup

```bash
python src/data/loader.py
```

### 6. Run tests

```bash
pytest tests/
```

### 7. Run the demo app

The Streamlit demo trains UBCF, IBCF and SVD on the processed training split and
shows each model's recommendations for a selected user. Recommendations that the
user rated 4★ or higher in the held-out test set are marked **✓ Hit**, using the
same definition as Precision@K in the evaluation.

```bash
python src/data/preprocessor.py   # creates data/processed/train.csv and test.csv
streamlit run app.py
```

Movie posters are optional. To enable them, add an [OMDb](https://www.omdbapi.com/)
API key to `.streamlit/secrets.toml`:

```toml
omdb_api_key = "your-key"
```

Without a key (or without internet access) the app still runs and shows
genre-colored cards instead of posters.

---

## Reproducing the Results

Run these from the project root after the dataset is in `data/raw/`. All experiments
use a fixed random seed (42). Each script writes its output to
`reports/` unless noted.

```bash
python src/data/preprocessor.py              # filter (>= 20 ratings) and per-user 80/20 split
python scripts/compare_all_models_v2.py      # Table I: all six models -> reports/full_model_comparison_v2.csv
python scripts/tune_hyperparameters.py       # grid search over k and min_support
python scripts/statistical_comparison.py     # paired bootstrap, UBCF vs IBCF (prints the result)
python scripts/analyze_cold_start.py         # cold-start simulation
python scripts/analyze_sparsity_impact.py    # RMSE by neighbour-support bucket
python scripts/analyze_scalability.py        # fit time and prediction latency
```

`scripts/run_full_pipeline.py` runs the data preparation and all six models in one
go. The `plot_*.py` scripts regenerate the figures from the saved CSVs. Run time
depends on the machine; the neighbourhood models are the slow part.

---

## Project Structure

```
movie-recsys/
├── app.py                       # Streamlit demo
├── configs/
│   └── config.yaml              # Tuned model settings and split configuration
├── data/
│   ├── raw/                     # Raw downloaded data (not tracked by git)
│   └── processed/               # Train/test split (not tracked by git)
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── report/
│   ├── paper.tex                # Final report (IEEE conference format)
│   ├── paper.pdf
│   └── figures/
├── reports/                     # Experiment outputs: result tables, analysis notes
│   └── figures/                 # Generated plots
├── scripts/                     # One script per experiment
│   ├── run_full_pipeline.py     # Train and evaluate all models
│   ├── tune_hyperparameters.py  # Grid search over k and min_support
│   ├── statistical_comparison.py  # Paired bootstrap test, UBCF vs IBCF
│   ├── analyze_cold_start.py    # Cold-start simulation
│   ├── analyze_sparsity_impact.py
│   ├── analyze_scalability.py
│   └── ...                      # Plotting, debugging and export scripts
├── src/
│   ├── data/
│   │   ├── loader.py            # Data loading utilities
│   │   └── preprocessor.py      # Filtering and per-user train/test split
│   ├── models/
│   │   ├── baselines.py         # Global, user and item mean predictors
│   │   ├── ubcf.py              # User-Based Collaborative Filtering
│   │   ├── ibcf.py              # Item-Based Collaborative Filtering
│   │   └── svd_model.py         # SVD wrapper around scikit-surprise
│   ├── evaluation/
│   │   ├── metrics.py           # RMSE, MAE, Precision@K, Recall@K, F1@K
│   │   └── demo_metrics.py      # Hit and display helpers used by the demo
│   └── utils/
│       ├── helpers.py           # Config loading, seeding, path helpers
│       └── similarity.py        # Cosine and Pearson similarity in NumPy
├── tests/                       # 133 unit and integration tests (pytest)
├── .streamlit/config.toml       # Demo presentation settings
├── requirements.txt
├── setup.py
└── README.md
```

---

## License

For academic use only. Dataset subject to [MovieLens Terms of Use](https://grouplens.org/datasets/movielens/).
