"""Streamlit demo app: sidebar-navigated Popular / Search / Profile views over UBCF, IBCF, SVD."""

import html

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.data.loader import load_movies
from src.data.preprocessor import build_user_item_matrix
from src.evaluation.demo_metrics import (
    RELEVANCE_THRESHOLD,
    expected_hits_by_model,
    experienced_rating_range,
    format_comparison_table,
    held_out_liked,
    hit_summary,
    relevant_test_items,
    star_histogram,
    top_contributors,
)
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
N_SIMILAR = 5
N_WHY_ITEMS = 2  # rated movies named in an IBCF explanation
WHY_TITLE_MAX_CHARS = 26
SEED = 42
N_CARD_COLUMNS = 5

POPULAR_MIN_RATINGS = 20  # same threshold as src/data/preprocessor.py's min_movie_ratings

# Purely cosmetic demo labels -- MovieLens is fully anonymized and has no real
# names. Mapped onto userIds deterministically in build_display_names().
DISPLAY_NAMES = [
    "Alex M.", "Priya K.", "Jordan T.", "Sam R.", "Taylor B.",
    "Morgan L.", "Casey W.", "Riley S.", "Jamie H.", "Avery D.",
    "Chris P.", "Dana F.", "Quinn G.", "Skyler N.", "Reese V.",
    "Emerson J.", "Rowan C.", "Hayden Z.", "Kai O.", "Noor A.",
    "Leo Q.", "Mia X.", "Theo Y.", "Zara I.", "Finn E.",
    "Ivy U.", "Owen T.", "Luca R.", "Nina W.", "Max B.",
]
# Separate cosmetic labels for the experienced-user group, so no name appears
# in both dropdowns. Same caveat: MovieLens users are anonymous.
EXPERIENCED_DISPLAY_NAMES = [
    "Elena V.", "Marcus D.", "Sofia L.", "Arjun P.", "Hannah K.",
    "Diego M.", "Lena S.", "Omar F.", "Clara B.", "Yusuf T.",
    "Maya R.", "Jonas H.", "Aisha N.", "Felix W.", "Grace O.",
    "Ravi C.", "Lucia G.", "Ben A.", "Ines Z.", "Tomas J.",
    "Anya E.", "Henrik U.", "Selin Y.", "Victor Q.", "Olivia X.",
]
POPULAR_N_MOVIES = 20

NAV_POPULAR = "Popular"
NAV_SEARCH = "Search"
NAV_PROFILE = "Profile"
SEARCH_MAX_RESULTS = 10

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
    aspect-ratio: 2 / 3;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    padding: 10px 10px 46px 10px;
    margin-bottom: 6px;
    box-sizing: border-box;
    overflow: hidden;
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
.similarity-badge {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background: rgba(124, 58, 237, 0.88);
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    padding: 4px 9px;
    border-radius: 6px;
}
.hit-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(22, 163, 74, 0.95);
    color: #ffffff;
    font-weight: 700;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
.why {
    min-height: 68px;
    font-size: 12px;
    line-height: 1.35;
    opacity: 0.9;
    margin: 0 2px 6px 2px;
}
.why-hist {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 22px;
    margin-top: 5px;
}
.why-bar {
    flex: 1;
    min-height: 2px;
    background: currentColor;
    opacity: 0.5;
    border-radius: 2px 2px 0 0;
}
.why-axis {
    display: flex;
    gap: 3px;
    font-size: 9px;
    opacity: 0.65;
}
.why-axis span {
    flex: 1;
    text-align: center;
}
.poster-card {
    background-size: cover;
    background-position: center top;
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
.detail-fallback {
    width: 220px;
    height: 320px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    text-align: center;
    padding: 16px;
    box-sizing: border-box;
}
/* ---- page layout ---- */
div[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 2.5rem;
    max-width: 1300px;
}
/* ---- sidebar ---- */
.side-brand {
    font-size: 20px;
    font-weight: 700;
    line-height: 1.2;
}
.side-tagline {
    font-size: 12px;
    opacity: 0.7;
    margin: 2px 0 14px 0;
}
.side-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 4px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
    width: 100%;
    align-items: stretch;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%;
    box-sizing: border-box;
    padding: 9px 12px;
    border-radius: 8px;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(128, 128, 128, 0.16);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected] {
    background: rgba(255, 75, 75, 0.2);
    font-weight: 600;
}
/* Hide the radio circle, never an element that holds the label text.
   Current Streamlit: the circle is the div just before the label's markdown.
   Older Streamlit: the circle is the label's first div. */
section[data-testid="stSidebar"] div[role="radiogroup"] > label
    div:not(:has(p)):has(+ [data-testid="stMarkdownContainer"]),
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-of-type:not(:has(p)) {
    display: none !important;
}
</style>
"""


# ===== cached data / model loading (unchanged from before) =====


@st.cache_resource
def load_and_train():
    """Load data and fit all 3 models once; cached across reruns and across all sections."""
    config = load_config()
    ubcf_cfg = config["model"]["ubcf"]
    ibcf_cfg = config["model"]["ibcf"]

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
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

    return train, test, movies, ubcf, ibcf, svd, imdb_lookup


def get_omdb_api_key() -> str | None:
    """Return the OMDb API key from Streamlit secrets, or None if it isn't configured.

    A missing .streamlit/secrets.toml or a missing key must not crash the demo;
    cards then fall back to the genre-gradient style.
    """
    try:
        return st.secrets["omdb_api_key"]
    except Exception:  # Streamlit raises different errors for missing file vs. missing key
        return None


@st.cache_data(show_spinner=False)
def get_poster_url(imdb_id: str) -> str | None:
    """Fetch a movie's OMDb poster URL. Returns None if unavailable or the call fails."""
    api_key = get_omdb_api_key()
    if not api_key:
        return None
    try:
        response = requests.get(
            OMDB_URL,
            params={"i": imdb_id, "apikey": api_key},
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


# ===== shared data-shaping helpers (unchanged logic, reused across sections) =====


def get_dropdown_users(
    train: pd.DataFrame,
    n: int,
    seed: int,
    min_ratings: int = MIN_TRAIN_RATINGS,
    max_ratings: int = MAX_TRAIN_RATINGS,
) -> list:
    """Pick n userIds whose training-rating count lies in [min_ratings, max_ratings]."""
    rating_counts = train.groupby("userId").size()
    eligible = rating_counts[
        (rating_counts >= min_ratings) & (rating_counts <= max_ratings)
    ].index.to_numpy()

    rng = np.random.default_rng(seed)
    n = min(n, len(eligible))
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def build_display_names(user_ids: list, seed: int, names: list = DISPLAY_NAMES) -> dict:
    """Deterministically map each userId to a friendly cosmetic display name.

    Purely cosmetic demo labeling -- MovieLens is fully anonymized and has no
    real names, and the underlying userId is unaffected everywhere else in
    the app. Shuffles DISPLAY_NAMES with a seeded RNG and zips it with the
    sorted userId list for a fixed, reproducible mapping.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(names).tolist()
    sorted_ids = sorted(user_ids)
    return {uid: shuffled[i] for i, uid in enumerate(sorted_ids)}


def compute_movie_stats(train: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by movieId with columns avg_rating, n_ratings."""
    return train.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")


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


def get_similar_movies(movie_id: int, ibcf: ItemBasedCF, movies: pd.DataFrame, n: int) -> list:
    """Return [(movieId, title, genres, similarity), ...] using IBCF's fitted item-item matrix.

    Reuses IBCF's already-fitted _sim_matrix / _movies internals directly -- no new
    modeling, just a read-only lookup into what fit() already computed.
    """
    try:
        idx = ibcf._get_movie_index(int(movie_id))
    except ValueError:
        return []

    sims = ibcf._sim_matrix[idx].copy()
    sims[idx] = 0.0
    top_idx = np.argsort(sims)[::-1][:n]

    movies_by_id = movies.set_index("movieId")
    results = []
    for i in top_idx:
        similar_id = int(ibcf._movies[i])
        if similar_id not in movies_by_id.index:
            continue
        row = movies_by_id.loc[similar_id]
        results.append((similar_id, row["title"], row["genres"], float(sims[i])))
    return results


def primary_genre_gradient(genres: str) -> tuple:
    """Pick the first listed genre and its CSS gradient (falls back to a neutral default)."""
    primary = genres.split("|")[0]
    return primary, GENRE_GRADIENTS.get(primary, GENRE_GRADIENTS["default"])


# ===== "Why this?" explanations =====


def shorten_title(title: str, max_chars: int = WHY_TITLE_MAX_CHARS) -> str:
    """Trim a long movie title so an explanation stays on a few lines."""
    title = str(title)
    return title if len(title) <= max_chars else title[: max_chars - 1].rstrip() + "…"


def ubcf_why_html(explanation: dict) -> str:
    """Explain a UBCF prediction: how many similar users rated the movie, and how.

    Shows the neighbours' average and a small 1-5 star histogram of their ratings.
    """
    if explanation["fallback"]:
        return (
            '<div class="why">Too few similar users rated this movie, '
            "so the score is this user's own average.</div>"
        )

    counts = star_histogram(explanation["neighbour_ratings"])
    peak = max(counts)
    bars = "".join(
        f'<div class="why-bar" style="height:{round(100 * count / peak)}%" '
        f'title="{count} rated it {star}★"></div>'
        for star, count in enumerate(counts, start=1)
    )
    axis = "".join(f"<span>{star}</span>" for star in range(1, 6))
    return (
        f'<div class="why"><b>{explanation["n_neighbours"]}</b> similar users · '
        f'average <b>{explanation["mean_neighbour_rating"]:.1f}★</b>'
        f'<div class="why-hist">{bars}</div><div class="why-axis">{axis}</div></div>'
    )


def ibcf_why_html(explanation: dict, titles: dict) -> str:
    """Explain an IBCF prediction: the rated movies that raised the score the most."""
    if explanation["fallback"]:
        return (
            '<div class="why">None of the movies this user rated is similar to this one. '
            "The score is this user's own average rating.</div>"
        )

    top = top_contributors(explanation["neighbours"], n=N_WHY_ITEMS)
    if not top:
        count = explanation["n_neighbours"]
        noun = "movie" if count == 1 else "movies"
        return (
            f'<div class="why">Based on <b>{count}</b> {noun} this user rated; '
            "none raised the score.</div>"
        )

    parts = [
        f"<i>{html.escape(shorten_title(titles.get(item['movieId'], item['movieId'])))}</i> "
        f"<b>{item['rating']:g}★</b>"
        for item in top
    ]
    return f'<div class="why">Because this user rated {" and ".join(parts)}</div>'


# ===== card rendering (unchanged card look; now with a Details button per card) =====


def full_width_button(label: str, key: str) -> bool:
    """A button that spans its column, on both current and older Streamlit versions."""
    try:
        return st.button(label, key=key, width="stretch")
    except TypeError:  # Streamlit versions before the `width` parameter
        return st.button(label, key=key, use_container_width=True)



def render_card_row(
    items: list,
    imdb_lookup: dict,
    context: str,
    badge_icon: str = "★",
    badge_class: str = "score-badge",
    hit_ids: set | None = None,
    why_html: dict | None = None,
) -> None:
    """Render a horizontal row of cards for [(movieId, title, genres, score), ...].

    Uses a real OMDb poster when available, falling back to the genre-gradient card
    style otherwise. score may be None (e.g. a search result with no training data),
    in which case the badge is omitted. Each card gets a "Details" button that sets
    st.session_state.selected_movie and reruns to show the movie detail view.

    badge_icon/badge_class let callers visually distinguish score types -- e.g.
    predicted ratings ("★", score-badge) vs IBCF similarity scores ("🔗",
    similarity-badge) -- without changing anything else about the card.

    hit_ids, if given, marks cards whose movieId the user rated at or above the
    relevance threshold in the held-out test set (a "hit" in Precision@K terms).

    why_html, if given, maps movieId to a short HTML explanation shown under the card.
    """
    cols = st.columns(N_CARD_COLUMNS)
    for i, (col, (movie_id, title, genres, score)) in enumerate(zip(cols, items)):
        safe_title = html.escape(str(title))
        imdb_id = imdb_lookup.get(int(movie_id))
        poster_url = get_poster_url(imdb_id) if imdb_id else None
        badge_html = (
            f'<div class="{badge_class}">{badge_icon} {score:.1f}</div>'
            if score is not None
            else ""
        )
        if hit_ids and int(movie_id) in hit_ids:
            badge_html += (
                '<div class="hit-badge" title="Rated '
                f'{RELEVANCE_THRESHOLD:g}+ by this user in the held-out test set">✓ Hit</div>'
            )

        with col:
            if poster_url:
                st.markdown(
                    f"""
                    <div class="movie-card poster-card" style="background-image:
                        linear-gradient(to bottom, rgba(0,0,0,0) 45%, rgba(0,0,0,0.82) 68%, rgba(0,0,0,0.96) 100%),
                        url('{poster_url}');">
                        <div class="poster-title">{safe_title}</div>
                        {badge_html}
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
                        {badge_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if why_html and int(movie_id) in why_html:
                st.markdown(why_html[int(movie_id)], unsafe_allow_html=True)

            if full_width_button("Details", key=f"details_{context}_{int(movie_id)}_{i}"):
                st.session_state.selected_movie = int(movie_id)
                st.rerun()


def render_card_rows(items: list, imdb_lookup: dict, context: str) -> None:
    """Chunk items into rows of N_CARD_COLUMNS and render each via render_card_row."""
    for start in range(0, len(items), N_CARD_COLUMNS):
        chunk = items[start : start + N_CARD_COLUMNS]
        render_card_row(chunk, imdb_lookup, context=f"{context}_{start}")


# ===== movie detail view =====


def render_movie_detail(
    movie_id: int,
    movies: pd.DataFrame,
    stats: pd.DataFrame,
    ibcf: ItemBasedCF,
    imdb_lookup: dict,
) -> None:
    """Render the detail panel for a single movie: poster, stats, and similar movies."""
    if st.button("← Back", key="back_button"):
        st.session_state.selected_movie = None
        st.rerun()

    movie_row = movies[movies["movieId"] == movie_id]
    if movie_row.empty:
        st.warning("Movie not found.")
        return

    title = movie_row["title"].values[0]
    genres = movie_row["genres"].values[0]

    st.subheader(title)
    col_poster, col_info = st.columns([1, 2])

    with col_poster:
        imdb_id = imdb_lookup.get(int(movie_id))
        poster_url = get_poster_url(imdb_id) if imdb_id else None
        if poster_url:
            st.image(poster_url, width=220)
        else:
            genre, gradient = primary_genre_gradient(genres)
            st.markdown(
                f"""<div class="detail-fallback" style="background:{gradient};">
                    {html.escape(str(title))}</div>""",
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown(f"**Genres:** {genres.replace('|', ', ')}")
        if movie_id in stats.index:
            row = stats.loc[movie_id]
            st.markdown(f"**Average rating:** ★ {row['avg_rating']:.2f} ({int(row['n_ratings'])} ratings)")
        else:
            st.markdown("**Average rating:** No ratings in the training data.")

    st.subheader("Similar Movies")
    similar = get_similar_movies(movie_id, ibcf, movies, n=N_SIMILAR)
    if similar:
        render_card_row(
            similar, imdb_lookup, context="detail_similar",
            badge_icon="🔗", badge_class="similarity-badge",
        )
    else:
        st.info("No similar movies found for this title (not enough co-rating data).")


# ===== the 3 sidebar sections =====


def render_popular_section(train: pd.DataFrame, movies: pd.DataFrame, imdb_lookup: dict) -> None:
    st.subheader("🏠 Popular Movies")
    st.caption(
        f"Highest average rating among movies with at least {POPULAR_MIN_RATINGS} ratings. "
        "Not personalized: every user sees the same list."
    )

    stats = compute_movie_stats(train)
    popular = (
        stats[stats["n_ratings"] >= POPULAR_MIN_RATINGS]
        .sort_values("avg_rating", ascending=False)
        .head(POPULAR_N_MOVIES)
    )
    merged = popular.reset_index().merge(movies, on="movieId")
    items = list(zip(merged["movieId"], merged["title"], merged["genres"], merged["avg_rating"]))

    render_card_rows(items, imdb_lookup, context="popular")


def render_search_section(
    movies: pd.DataFrame, stats: pd.DataFrame, imdb_lookup: dict
) -> None:
    st.subheader("🔍 Search")
    st.caption("Find a movie, then open Details to see similar movies (Item-Based CF).")
    query = st.text_input("Search for a movie by title")

    if not query:
        st.info("Type a movie title above to search.")
        return

    matches = movies[movies["title"].str.contains(query, case=False, na=False, regex=False)]
    matches = matches.head(SEARCH_MAX_RESULTS)

    if matches.empty:
        st.warning(f"No movies found matching '{query}'.")
        return

    st.caption(f"{len(matches)} result(s)")
    items = []
    for _, row in matches.iterrows():
        movie_id = row["movieId"]
        avg_rating = stats["avg_rating"].get(movie_id)
        items.append((movie_id, row["title"], row["genres"], avg_rating))

    render_card_rows(items, imdb_lookup, context="search")


def render_model_recommendations(
    heading: str,
    model_key: str,
    recs: list,
    relevant: set,
    expected_hits: dict,
    imdb_lookup: dict,
    context: str,
    why_note: str,
    why_html: dict | None = None,
) -> None:
    """Render one model's recommendation row with its hit count and explanations."""
    st.markdown(f"#### {heading}")
    lines = []
    if relevant:
        summary = hit_summary([movie_id for movie_id, *_ in recs], relevant)
        expected = expected_hits_by_model_text(expected_hits, model_key)
        lines.append(
            f"Hits for this user: **{summary['n_hits']} / {summary['n_recommended']}**{expected}"
        )
    lines.append(why_note)
    st.caption("  \n".join(lines))
    render_card_row(recs, imdb_lookup, context=context, hit_ids=relevant, why_html=why_html)


def expected_hits_by_model_text(expected_hits: dict, model_key: str) -> str:
    """Format the average-hits reference shown next to a user's hit count."""
    if model_key not in expected_hits:
        return ""
    return f" · average over all test users ≈ {expected_hits[model_key]:.2f}"


def render_profile_section(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    ubcf: UserBasedCF,
    ibcf: ItemBasedCF,
    svd: SVDModel,
    imdb_lookup: dict,
) -> None:
    st.subheader("👤 Profile")

    exp_min, exp_max = experienced_rating_range(train, light_max=MAX_TRAIN_RATINGS)
    light_label = f"Light users ({MIN_TRAIN_RATINGS}–{MAX_TRAIN_RATINGS} ratings)"
    experienced_label = f"Experienced users ({exp_min}–{exp_max} ratings)"
    group = st.radio("User group", [light_label, experienced_label], horizontal=True)

    if group == experienced_label:
        dropdown_users = get_dropdown_users(
            train, N_DROPDOWN_USERS, SEED, min_ratings=exp_min, max_ratings=exp_max
        )
        display_names = build_display_names(dropdown_users, SEED, EXPERIENCED_DISPLAY_NAMES)
        st.caption(
            "Users between the 75th and 95th percentile of training ratings: "
            "neighborhood models have much more history to compare against."
        )
    else:
        dropdown_users = get_dropdown_users(train, N_DROPDOWN_USERS, SEED)
        display_names = build_display_names(dropdown_users, SEED)
        st.caption(
            "The least active users in the filtered data: the hardest case for "
            "neighborhood models, close to the cold-start setting in Section IV-D."
        )

    if not dropdown_users:
        st.warning("No users fall in this range in the current data.")
        return

    user_id = st.selectbox(
        "Select a user",
        dropdown_users,
        format_func=lambda uid: display_names[uid],
        key=f"user_select_{group}",
    )

    if user_id is None:
        return

    display_name = display_names[user_id]

    user_ratings = train[train["userId"] == user_id]["rating"]
    n_held_out = int((test["userId"] == user_id).sum())
    relevant = relevant_test_items(test, user_id)
    st.markdown(f"**{display_name}** _(User #{user_id})_")
    st.markdown(
        f"**{len(user_ratings)} ratings given · {user_ratings.mean():.2f} average rating**  \n"
        f"{n_held_out} further ratings held out for testing, "
        f"{len(relevant)} of them rated {RELEVANCE_THRESHOLD:g}★ or higher"
    )

    top_rated = top_rated_movies(train, user_id, movies, N_TOP_RATED)
    ubcf_recs = recommendations_table(ubcf.recommend(user_id, n=N_RECOMMENDATIONS), movies)
    ibcf_recs = recommendations_table(ibcf.recommend(user_id, n=N_RECOMMENDATIONS), movies)
    svd_recs = recommendations_table(svd.recommend(user_id, n=N_RECOMMENDATIONS), movies)

    comparison_df = pd.read_csv(COMPARISON_PATH)
    expected_hits = expected_hits_by_model(comparison_df, N_RECOMMENDATIONS)

    st.markdown(f"#### 🎬 {display_name}'s Top-Rated Movies")
    st.caption("★ = the user's own rating in the training data.")
    render_card_row(top_rated, imdb_lookup, context="profile_top_rated")

    st.divider()
    st.caption(
        "★ on recommendation cards = the model's predicted rating. "
        f"**✓ Hit** = a recommended movie this user rated {RELEVANCE_THRESHOLD:g}★ or higher "
        "in the held-out test set, which the models never saw during training. "
        "This is the same definition as Precision@K in the report. With Precision@10 "
        "around 0.05, a 5-movie list contains about 0.25 hits on average, so zero hits "
        "is the usual outcome for a single user."
    )
    if not relevant:
        st.info(
            f"This user rated none of their held-out movies {RELEVANCE_THRESHOLD:g}★ or "
            "higher, so no hit is possible for any model. The offline evaluation skips "
            "such users as well."
        )

    titles = movies.set_index("movieId")["title"].to_dict()
    ubcf_why = {
        int(movie_id): ubcf_why_html(ubcf.explain(user_id, int(movie_id)))
        for movie_id, *_ in ubcf_recs
    }
    ibcf_why = {
        int(movie_id): ibcf_why_html(ibcf.explain(user_id, int(movie_id)), titles)
        for movie_id, *_ in ibcf_recs
    }

    render_model_recommendations(
        "🤝 Recommended for You (User-Based CF)", "UBCF", ubcf_recs,
        relevant, expected_hits, imdb_lookup, context="profile_ubcf",
        why_note="Why: how many similar users rated each movie, and how "
        "(bars show their ratings from 1★ to 5★).",
        why_html=ubcf_why,
    )
    render_model_recommendations(
        "🎯 Recommended for You (Item-Based CF)", "IBCF", ibcf_recs,
        relevant, expected_hits, imdb_lookup, context="profile_ibcf",
        why_note="Why: the movies this user rated that raised each prediction the most.",
        why_html=ibcf_why,
    )
    render_model_recommendations(
        "🧠 Recommended for You (SVD)", "SVD", svd_recs,
        relevant, expected_hits, imdb_lookup, context="profile_svd",
        why_note="No per-movie explanation: SVD predicts from 50 learned factors "
        "that have no individual meaning.",
    )

    if relevant:
        with st.expander(
            f"🔑 Answer key: held-out movies {display_name} rated "
            f"{RELEVANCE_THRESHOLD:g}★ or higher"
        ):
            st.caption("★ = the user's actual rating, hidden from all models during training.")
            liked = held_out_liked(test, user_id, movies)
            render_card_rows(liked, imdb_lookup, context="profile_answer_key")

    with st.expander("📊 Model Performance Comparison (Table I of the report)"):
        st.caption("All six models on the same held-out test set of 13,406 ratings.")
        st.dataframe(
            format_comparison_table(comparison_df), hide_index=True, use_container_width=True
        )


# ===== app entry point =====

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

st.title("Movie Recommendation System")
st.caption(
    "Collaborative filtering on MovieLens · User-Based, Item-Based and SVD compared"
)

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

with st.spinner("Loading and training models..."):
    train, test, movies, ubcf, ibcf, svd, imdb_lookup = load_and_train()

movie_stats = compute_movie_stats(train)

st.sidebar.markdown(
    '<div class="side-brand">Movie Recommender</div>'
    '<div class="side-tagline">Collaborative filtering demo</div>'
    '<div class="side-label">Pages</div>',
    unsafe_allow_html=True,
)
section = st.sidebar.radio(
    "Pages", [NAV_POPULAR, NAV_SEARCH, NAV_PROFILE], label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown("**Model settings**")
st.sidebar.caption(
    f"UBCF: Pearson, k = {ubcf.k}, min_support = {ubcf.min_support}  \n"
    f"IBCF: adjusted cosine, k = {ibcf.k}, min_support = {ibcf.min_support}  \n"
    f"SVD: {svd.n_factors} factors, {svd.n_epochs} epochs  \n"
    f"Data: {train['userId'].nunique()} users, {train['movieId'].nunique():,} movies, "
    f"{len(train):,} train / {len(test):,} test ratings (per-user 80/20 split)"
)

if st.session_state.selected_movie is not None:
    render_movie_detail(st.session_state.selected_movie, movies, movie_stats, ibcf, imdb_lookup)
elif section == NAV_POPULAR:
    render_popular_section(train, movies, imdb_lookup)
elif section == NAV_SEARCH:
    render_search_section(movies, movie_stats, imdb_lookup)
elif section == NAV_PROFILE:
    render_profile_section(train, test, movies, ubcf, ibcf, svd, imdb_lookup)
