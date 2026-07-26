# Project Progress Report — Weeks 1 to 4

**Project Title:** Movie Recommendation System Using Collaborative Filtering  
**Student:** Taibaz Pathan  
**Student ID:** 284085  
**University:** Frankfurt University of Applied Sciences  
**Supervisor:** Prof. Dr. Andreas Pech
**Report Date:** 02 June 2026  
**Project Start:** 07 May 2026  

---

## Overview

This report summarises the technical progress made during the first four weeks of the project. The goal of the project is to build and evaluate a Movie Recommendation System using User-Based and Item-Based Collaborative Filtering on the MovieLens dataset. Weeks 1 through 4 cover the foundation of the project: environment setup, exploratory data analysis, data preprocessing, and implementation of core similarity metrics from scratch.

---

## Week 1 — Project Setup and Environment

The project repository was initialised with a structured folder layout following standard Python packaging conventions. The key components built this week were:

- A virtual environment with all required dependencies pinned to specific versions (NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Jupyter, pytest)
- A centralised configuration file (`configs/config.yaml`) for dataset paths, split parameters, and logging settings
- A data loading module (`src/data/loader.py`) that reads the MovieLens CSV files and converts the Unix timestamp column to a readable datetime format
- Skeleton classes for User-Based CF and Item-Based CF (`src/models/ubcf.py`, `src/models/ibcf.py`)
- An evaluation metrics module (`src/evaluation/metrics.py`) with RMSE and MAE functions
- A test suite in `tests/` with 17 passing unit tests covering the loader and metrics modules
- Git repository initialised with the first commit on branch `main`

The dataset used throughout the project is the **MovieLens ml-latest-small** dataset from GroupLens Research (100,836 ratings, 610 users, 9,742 movies).

---

## Week 2 — Exploratory Data Analysis

A Jupyter notebook (`notebooks/eda.ipynb`) was produced covering eight analysis sections. Seven figures were generated and saved to `reports/figures/`. The key findings are summarised below.

**Rating Distribution**  
The mean rating is 3.50, above the theoretical midpoint of 2.75. Users show a clear positive bias — they tend to rate movies they chose to watch and enjoyed. Whole-star ratings (1.0, 2.0, 3.0, 4.0, 5.0) are used significantly more than half-star ratings. This bias will be addressed through mean-centring in the CF models.

**User Activity**  
The distribution of ratings per user is heavily skewed. One user has over 2,000 ratings while the median user has only 68. This long-tail behaviour means that Pearson correlation will be unreliable for low-activity users due to insufficient co-rated items.

**Movie Popularity**  
A small number of blockbuster films (Forrest Gump, The Shawshank Redemption, Pulp Fiction) account for a disproportionate share of ratings. The median movie has only 3 ratings, confirming a strong long-tail effect. Movies with fewer than 20 ratings will be filtered out before model training.

**Matrix Sparsity**  
Only 1.7% of the 610 × 9,724 user-item matrix is filled. At 98.3% sparsity, most user pairs share very few co-rated movies. This directly motivates the use of a minimum co-rating threshold (min\_support) when computing similarity.

**Temporal Patterns**  
Ratings span 1996 to 2018. Activity peaked around 2000 and has been more evenly distributed since. Average rating per year remains stable at approximately 3.5.

---

## Week 3 — Data Preprocessing

A preprocessing module (`src/data/preprocessor.py`) was built with four functions:

**filter\_ratings()**  
Removes users and movies that fall below a minimum rating count threshold. Filtering is applied iteratively until convergence to handle cases where removing sparse movies makes some users drop below the threshold. With thresholds of 20 ratings per user and 20 per movie, the dataset reduces from 100,836 to 67,020 ratings, covering 566 users and 1,286 movies.

**build\_user\_item\_matrix()**  
Constructs a pivot table with users as rows and movies as columns. Missing entries are represented as NaN. After filtering, the matrix has 566 × 1,286 dimensions with 92.6% sparsity — notably lower than the raw dataset because rare movies have been removed.

**train\_test\_split\_per\_user()**  
Splits ratings on a per-user basis: for each user, 20% of their ratings are randomly held out for the test set and 80% are kept for training. This guarantees that every user appears in both splits, which is essential for evaluating CF models. The result is 53,614 training ratings and 13,406 test ratings. The random seed is fixed at 42 for reproducibility.

**save\_splits()**  
Saves the train and test DataFrames as CSV files to `data/processed/`.

A test suite of 18 unit tests was written for this module, all passing.

---

## Week 4 — Similarity Metrics from Scratch

A dedicated similarity module (`src/utils/similarity.py`) was implemented using NumPy only — no pandas `.corr()` or scikit-learn functions were used.

**cosine\_similarity(u, v)**  
Computes cosine similarity between two rating vectors. Only positions where both vectors are non-NaN (i.e. both users rated the same item) are included. Returns 0.0 if there are no co-rated items.

**pearson\_similarity(u, v, min\_support)**  
Computes the exact Pearson correlation, mean-centring each vector over the co-rated items only (not each user's full rating history). Returns 0.0 if the number of co-rated items is below min\_support. This is the formulation described in the original collaborative filtering literature (Resnick et al., 1994).

**cosine\_similarity\_matrix(matrix)**  
Efficiently computes the full pairwise cosine similarity matrix using vectorised NumPy operations. NaN values are replaced with 0 before L2 normalisation.

**pearson\_similarity\_matrix(matrix, min\_support)**  
Computes the full pairwise Pearson similarity matrix. Each row is mean-centred using its own overall mean. Pairs with fewer co-rated items than min\_support are set to 0.0. This vectorised implementation avoids nested Python loops and runs efficiently on the full 566 × 1,286 matrix.

22 unit tests were written and all pass.

---

## Current Test Suite Summary

| Module | Tests | Status |
|---|---|---|
| src/data/loader.py | 7 | Passing |
| src/evaluation/metrics.py | 10 | Passing |
| src/data/preprocessor.py | 18 | Passing |
| src/utils/similarity.py | 22 | Passing |
| **Total** | **57** | **All passing** |

---

## Repository Structure (current)

```
movie-recsys/
├── configs/config.yaml
├── data/processed/          train.csv, test.csv
├── notebooks/eda.ipynb      Week 2 EDA
├── reports/figures/         7 EDA figures (PNG)
├── src/
│   ├── data/loader.py       dataset loading
│   ├── data/preprocessor.py filtering, matrix, split
│   ├── models/ubcf.py       User-Based CF skeleton
│   ├── models/ibcf.py       Item-Based CF skeleton
│   ├── evaluation/metrics.py RMSE, MAE
│   └── utils/similarity.py  cosine and Pearson from scratch
└── tests/                   57 unit tests
```

---

## Plan for Weeks 5 and 6

**Week 5** — Complete the User-Based CF implementation by wiring the custom similarity module into the UBCF class (replacing the current pandas-based approach) and evaluating predictions on the held-out test set using RMSE and MAE.

**Week 6** — Complete the Item-Based CF implementation in the same way and run initial comparisons between UBCF and IBCF.

---

*All code is version-controlled in a local Git repository. Four commits have been made, one per week milestone.*
