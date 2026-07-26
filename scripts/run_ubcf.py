"""Train and evaluate the User-Based CF model on the processed MovieLens split."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.loader import load_movies
from src.data.preprocessor import build_user_item_matrix
from src.models.ubcf import UserBasedCF

TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH = os.path.join("data", "processed", "test.csv")
USER_ID = 1


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    movie_titles = load_movies().set_index("movieId")["title"]

    train_matrix = build_user_item_matrix(train)

    model = UserBasedCF(k=20, similarity="pearson", min_support=5)
    model.fit(train_matrix)

    metrics = model.evaluate(test)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")

    print(f"\nTop 10 recommendations for user {USER_ID}:")
    for rank, (movie_id, score) in enumerate(model.recommend(USER_ID, n=10), 1):
        title = movie_titles.get(movie_id, "Unknown")
        print(f"  {rank:>2}. {title:<50} predicted: {score:.3f}")


if __name__ == "__main__":
    main()
