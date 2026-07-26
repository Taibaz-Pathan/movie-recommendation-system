"""Diagnostic script to inspect UserBasedCF internals for user 1."""

import numpy as np
from src.data.loader import load_all
from src.models.ubcf import UserBasedCF

data = load_all()
ratings = data["ratings"]
movies  = data["movies"]
movie_titles = movies.set_index("movieId")["title"]

matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
model = UserBasedCF(k=20, similarity="pearson", min_support=5, min_k=2)
model.fit(matrix)

user_id = 1
print(f"User {user_id} mean rating : {model._user_means[user_id]:.3f}")
print(f"User {user_id} total ratings: {matrix.loc[user_id].count()}")

# Check similarity scores for user 1
sim_row = model._sim_matrix[user_id].drop(index=user_id).dropna()
print(f"\nValid Pearson neighbours (min_support=5): {len(sim_row)}")
print("Top-5 most similar users:")
print(sim_row.nlargest(5).to_string())

# Inspect a specific popular movie: Toy Story (1)
movie_id = 1
print(f"\n--- Inspecting movie {movie_id}: {movie_titles.get(movie_id)} ---")
print(f"User 1 actual rating: {matrix.loc[user_id, movie_id]}")
print(f"Total users who rated it: {matrix[movie_id].count()}")

rated_by = matrix[movie_id].dropna().index
common   = sim_row.index.intersection(rated_by)
print(f"Neighbours who rated this movie: {len(common)}")
if len(common) > 0:
    top_k   = sim_row[common].abs().nlargest(5).index
    for uid in top_k:
        mv   = matrix.loc[uid, movie_id]
        mu   = model._user_means[uid]
        sim  = sim_row[uid]
        print(f"  user {uid:4d}  sim={sim:+.3f}  rating={mv}  mean={mu:.3f}  deviation={mv-mu:+.3f}")

# Show rating distribution for user 1
u1_ratings = matrix.loc[user_id].dropna()
print(f"\nUser 1 rating distribution:")
print(u1_ratings.value_counts().sort_index())

# Show how many popular movies get predicted below 5.0
popular = matrix.count()[matrix.count() >= 20].index
unrated = matrix.loc[user_id][matrix.loc[user_id].isna()].index.intersection(popular)
preds = [(mid, model.predict(user_id, mid)) for mid in unrated[:200]]
pred_values = [p[1] for p in preds]
print(f"\nOf first 200 popular unrated movies:")
print(f"  Predicted exactly 5.0 : {sum(p == 5.0 for p in pred_values)}")
print(f"  Predicted < 5.0       : {sum(p < 5.0 for p in pred_values)}")
print(f"  Min prediction        : {min(pred_values):.3f}")
print(f"  Max prediction        : {max(pred_values):.3f}")
print(f"  Mean prediction       : {np.mean(pred_values):.3f}")
