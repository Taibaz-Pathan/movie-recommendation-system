"""Streamlit demo app: live recommendations from UBCF, IBCF, and SVD."""

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_movies
from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.svd_model import SVDModel
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
COMPARISON_PATH = "reports/full_model_comparison_v2.csv"

MIN_TRAIN_RATINGS = 15
MAX_TRAIN_RATINGS = 30
N_DROPDOWN_USERS = 25
N_TOP_RATED = 5
N_RECOMMENDATIONS = 5
SEED = 42


@st.cache_resource
def load_and_train():
    """Load data and fit all 3 models once; cached across reruns."""
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    movies = load_movies()
    train_matrix = build_user_item_matrix(train)

    ubcf = UserBasedCF(
        k=ubcf_cfg["k"], similarity=ubcf_cfg["similarity"], min_support=ubcf_cfg["min_support"]
    )
    ubcf.fit(train_matrix)

    ibcf = ItemBasedCF(k=ibcf_cfg["k"], min_support=ibcf_cfg["min_support"])
    ibcf.fit(train)

    svd = SVDModel(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(train)

    return train, movies, ubcf, ibcf, svd


def get_dropdown_users(train: pd.DataFrame, n: int, seed: int) -> list:
    """Pick n userIds with a moderate rating count (a meaningful taste profile)."""
    rating_counts = train.groupby("userId").size()
    eligible = rating_counts[
        (rating_counts >= MIN_TRAIN_RATINGS) & (rating_counts <= MAX_TRAIN_RATINGS)
    ].index.to_numpy()

    rng = np.random.default_rng(seed)
    n = min(n, len(eligible))
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def top_rated_movies(train: pd.DataFrame, user_id: int, movies: pd.DataFrame, n: int) -> pd.DataFrame:
    user_ratings = (
        train[train["userId"] == user_id]
        .sort_values(["rating", "movieId"], ascending=[False, True])
        .head(n)
    )
    result = user_ratings.merge(movies, on="movieId")[["title", "rating"]]
    result["rating"] = result["rating"].round(2)
    return result


def recommendations_table(recs: list, movies: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(recs, columns=["movieId", "predicted_score"])
    result = df.merge(movies, on="movieId")[["title", "predicted_score"]]
    result["predicted_score"] = result["predicted_score"].round(2)
    return result


st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("Movie Recommendation System — Collaborative Filtering Demo")
st.caption("MovieLens dataset | UBCF, IBCF, and SVD compared")

with st.spinner("Loading and training models..."):
    train, movies, ubcf, ibcf, svd = load_and_train()

dropdown_users = get_dropdown_users(train, N_DROPDOWN_USERS, SEED)
user_id = st.selectbox("Select a user", dropdown_users)

if user_id is not None:
    top_rated = top_rated_movies(train, user_id, movies, N_TOP_RATED)
    ubcf_recs = recommendations_table(ubcf.recommend(user_id, n=N_RECOMMENDATIONS), movies)
    ibcf_recs = recommendations_table(ibcf.recommend(user_id, n=N_RECOMMENDATIONS), movies)
    svd_recs = recommendations_table(svd.recommend(user_id, n=N_RECOMMENDATIONS), movies)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("This user's top-rated movies")
        st.table(top_rated.set_index("title"))

    with col2:
        st.subheader("UBCF Recommendations")
        st.table(ubcf_recs.set_index("title"))

    with col3:
        st.subheader("IBCF Recommendations")
        st.table(ibcf_recs.set_index("title"))

    st.subheader("SVD Recommendations")
    st.dataframe(svd_recs, hide_index=True, use_container_width=True)

with st.expander("Model Performance Comparison"):
    comparison_df = pd.read_csv(COMPARISON_PATH)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
