# Movie Recommendation System Using Collaborative Filtering

**Student:** Taibaz Pathan  
**University:** Frankfurt University of Applied Sciences  
**Start Date:** 07 May 2026

---

## Project Overview

This project implements a movie recommendation system using collaborative filtering techniques. It explores both user-based (UBCF) and item-based (IBCF) collaborative filtering approaches applied to the MovieLens dataset. The goal is to predict user ratings for unseen movies and generate personalised top-N recommendations.

Key techniques covered:
- User-Based Collaborative Filtering (UBCF)
- Item-Based Collaborative Filtering (IBCF)
- Evaluation via RMSE and MAE
- Exploratory data analysis of rating patterns and sparsity

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

---

## Project Structure

```
movie-recsys/
├── configs/
│   └── config.yaml          # Centralised configuration
├── data/
│   ├── raw/                 # Raw downloaded data (not tracked by git)
│   └── processed/           # Processed/split data (not tracked by git)
├── notebooks/               # Jupyter notebooks for exploration
├── reports/
│   └── figures/             # Generated plots and figures
├── src/
│   ├── data/
│   │   └── loader.py        # Data loading utilities
│   ├── models/
│   │   ├── ubcf.py          # User-Based Collaborative Filtering
│   │   └── ibcf.py          # Item-Based Collaborative Filtering
│   ├── evaluation/
│   │   └── metrics.py       # RMSE, MAE evaluation metrics
│   └── utils/
│       └── helpers.py       # Config loading, seeding, path helpers
├── tests/
│   ├── test_loader.py       # Unit tests for data loader
│   └── test_metrics.py      # Unit tests for evaluation metrics
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

## License

For academic use only. Dataset subject to [MovieLens Terms of Use](https://grouplens.org/datasets/movielens/).
