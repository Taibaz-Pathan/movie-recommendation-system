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

### 1. Clone the repository

```bash
git clone <your-repo-url>
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
