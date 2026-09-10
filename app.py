"""Streamlit demo app: Netflix-style genre-card recommendations from UBCF, IBCF, and SVD."""

import html

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.data.loader import load_movies
from src.data.preprocessor import build_user_item_matrix
from src.models.ibcf import ItemBasedCF
from src.models.svd_model import SVDModel
from src.models.ubcf import UserBasedCF
from src.utils.helpers import load_config

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
LINKS_PATH = "data/raw/ml-latest-small/links.csv"
COMPARISON_PATH = "reports/full_model_comparison_v2.csv"
OMDB_URL = "http://www.omdbapi.com/"

MIN_TRAIN_RATINGS = 15
MAX_TRAIN_RATINGS = 30
N_DROPDOWN_USERS = 25
N_TOP_RATED = 5
N_RECOMMENDATIONS = 5
SEED = 42
N_CARD_COLUMNS = 5

GENRE_GRADIENTS = {
    "Action": "linear-gradient(135deg, #ff512f, #f09819)",
    "Comedy": "linear-gradient(135deg, #f7971e, #ffd200)",
    "Drama": "linear-gradient(135deg, #1e3c72, #2a5298)",
    "Horror": "linear-gradient(135deg, #232526, #4b134f)",
    "Romance": "linear-gradient(135deg, #f857a6, #c0356e)",
    "Sci-Fi": "linear-gradient(135deg, #00c9ff, #0c6478)",
    "Animation": "linear-gradient(135deg, #ff6a00, #ee0979, #00c3ff)",
    "Documentary": "linear-gradient(135deg, #757f9a, #3b4045)",
    "Thriller": "linear-gradient(135deg, #4a0000, #1a0000)",
    "default": "linear-gradient(135deg, #606c88, #3f4c6b)",
}

CARD_CSS = """
<style>
.movie-card {
    position: relative;
    width: 100%;
    height: 220px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    padding: 10px 10px 46px 10px;
    margin-bottom: 14px;
    box-sizing: border-box;
}
.genre-tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    color: rgba(255,255,255,0.9);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: rgba(0,0,0,0.28);
    padding: 3px 7px;
    border-radius: 4px;
}
.movie-title {
    margin-top: 10px;
    text-align: center;
    color: white;
    font-weight: 700;
    font-size: 14px;
    line-height: 1.25;
    text-shadow: 0 1px 3px rgba(0,0,0,0.55);
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.score-badge {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background: rgba(0,0,0,0.65);
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    padding: 4px 9px;
    border-radius: 6px;
}
.poster-card {
    background-size: cover;
    background-position: center;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.poster-title {
    color: white;
    font-weight: 700;
    font-size: 13px;
    text-align: center;
    line-height: 1.2;
    text-shadow: 0 1px 3px rgba(0,0,0,0.85);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 6px;
}
</style>
"""


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

    links = pd.read_csv(LINKS_PATH)
    imdb_lookup = {
        int(row["movieId"]): f"tt{int(row['imdbId']):07d}" for _, row in links.iterrows()
    }

    return train, movies, ubcf, ibcf, svd, imdb_lookup


@st.cache_data(show_spinner=False)
def get_poster_url(imdb_id: str) -> str | None:
    """Fetch a movie's OMDb poster URL. Returns None if unavailable or the call fails."""
    try:
        response = requests.get(
            OMDB_URL,
            params={"i": imdb_id, "apikey": st.secrets["omdb_api_key"]},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    poster = data.get("Poster")
    if not poster or poster == "N/A":
        return None
    return poster


def get_dropdown_users(train: pd.DataFrame, n: int, seed: int) -> list:
    """Pick n userIds with a moderate rating count (a meaningful taste profile)."""
    rating_counts = train.groupby("userId").size()
    eligible = rating_counts[
        (rating_counts >= MIN_TRAIN_RATINGS) & (rating_counts <= MAX_TRAIN_RATINGS)
    ].index.to_numpy()

    rng = np.random.default_rng(seed)
    n = min(n, len(eligible))
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def top_rated_movies(train: pd.DataFrame, user_id: int, movies: pd.DataFrame, n: int) -> list:
    """Return [(movieId, title, genres, rating), ...] for a user's top-n rated movies."""
    user_ratings = (
        train[train["userId"] == user_id]
        .sort_values(["rating", "movieId"], ascending=[False, True])
        .head(n)
    )
    merged = user_ratings.merge(movies, on="movieId")
    return list(zip(merged["movieId"], merged["title"], merged["genres"], merged["rating"]))


def recommendations_table(recs: list, movies: pd.DataFrame) -> list:
    """Return [(movieId, title, genres, predicted_score), ...] for a model's recommend() output."""
    df = pd.DataFrame(recs, columns=["movieId", "predicted_score"])
    merged = df.merge(movies, on="movieId")
    return list(
        zip(merged["movieId"], merged["title"], merged["genres"], merged["predicted_score"])
    )


def primary_genre_gradient(genres: str) -> tuple:
    """Pick the first listed genre and its CSS gradient (falls back to a neutral default)."""
    primary = genres.split("|")[0]
    return primary, GENRE_GRADIENTS.get(primary, GENRE_GRADIENTS["default"])


def render_card_row(items: list, imdb_lookup: dict) -> None:
    """Render a horizontal row of cards for [(movieId, title, genres, score), ...].

    Uses a real OMDb poster when available, falling back to the genre-gradient
    card style otherwise.
    """
    cols = st.columns(N_CARD_COLUMNS)
    for col, (movie_id, title, genres, score) in zip(cols, items):
        safe_title = html.escape(str(title))
        imdb_id = imdb_lookup.get(int(movie_id))
        poster_url = get_poster_url(imdb_id) if imdb_id else None

        with col:
            if poster_url:
                st.markdown(
                    f"""
                    <div class="movie-card poster-card" style="background-image:
                        linear-gradient(to bottom, rgba(0,0,0,0) 35%, rgba(0,0,0,0.55) 65%, rgba(0,0,0,0.92) 100%),
                        url('{poster_url}');">
                        <div class="poster-title">{safe_title}</div>
                        <div class="score-badge">★ {score:.1f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                genre, gradient = primary_genre_gradient(genres)
                st.markdown(
                    f"""
                    <div class="movie-card" style="background:{gradient};">
                        <div class="genre-tag">{html.escape(genre)}</div>
                        <div class="movie-title">{safe_title}</div>
                        <div class="score-badge">★ {score:.1f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

st.title("Movie Recommendation System — Collaborative Filtering Demo")
st.caption("MovieLens dataset | UBCF, IBCF, and SVD compared")

with st.spinner("Loading and training models..."):
    train, movies, ubcf, ibcf, svd, imdb_lookup = load_and_train()

dropdown_users = get_dropdown_users(train, N_DROPDOWN_USERS, SEED)
user_id = st.selectbox("Select a user", dropdown_users)

if user_id is not None:
    top_rated = top_rated_movies(train, user_id, movies, N_TOP_RATED)
    ubcf_recs = recommendations_table(ubcf.recommend(user_id, n=N_RECOMMENDATIONS), movies)
    ibcf_recs = recommendations_table(ibcf.recommend(user_id, n=N_RECOMMENDATIONS), movies)
    svd_recs = recommendations_table(svd.recommend(user_id, n=N_RECOMMENDATIONS), movies)

    st.subheader(f"🎬 User {user_id}'s Top-Rated Movies")
    render_card_row(top_rated, imdb_lookup)

    st.subheader("🤝 Recommended for You (User-Based CF)")
    render_card_row(ubcf_recs, imdb_lookup)

    st.subheader("🎯 Recommended for You (Item-Based CF)")
    render_card_row(ibcf_recs, imdb_lookup)

    st.subheader("🧠 Recommended for You (SVD)")
    render_card_row(svd_recs, imdb_lookup)

with st.expander("Model Performance Comparison"):
    comparison_df = pd.read_csv(COMPARISON_PATH)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
