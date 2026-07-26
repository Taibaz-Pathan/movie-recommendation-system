"""Smoke test for UserBasedCF against the real MovieLens dataset."""

import numpy as np
from src.data.loader import load_all
from src.models.ubcf import UserBasedCF

print("Loading data...")
data = load_all()
ratings = data["ratings"]
movies  = data["movies"]
movie_titles = movies.set_index("movieId")["title"]

print("Building ratings matrix...")
matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
print(f"Matrix shape: {matrix.shape}")

print("Fitting UserBasedCF (k=20, pearson)...")
model = UserBasedCF(k=20, similarity="pearson", min_support=5, min_k=2)
model.fit(matrix)
print("Fit complete.\n")

for user_id in [1, 42, 200]:
    mean = model._user_means[user_id]
    n_rated = int(matrix.loc[user_id].count())
    recs = model.recommend(user_id, n=5, min_movie_ratings=20)
    pred_vals = [r[1] for r in recs]

    print(f"User {user_id:3d}  |  mean rating: {mean:.2f}  |  movies rated: {n_rated}")
    print(f"  Top-5 recommendations:")
    for rank, (mid, score) in enumerate(recs, 1):
        title = movie_titles.get(mid, "Unknown")
        print(f"    {rank}. {title:<50} predicted: {score:.3f}")
    print()
