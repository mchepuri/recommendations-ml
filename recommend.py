"""
Real Estate Listing Recommender
================================
Strategy: content-weighted hybrid (content-based 80% + ALS collaborative 20%).

Why not pure CF?
  - Only 200 users with avg 0.68 shared likes between any two users.
  - 58% of items were liked by exactly 1 user → almost no co-occurrence signal.
  - ALS learns weak item embeddings in this regime.

Why content-based works here?
  - Each user has ~133 liked listings → strong preference profile over
    price, beds, baths, size, and city.
  - Cosine similarity against item features gives interpretable, sensible recs.

Usage:
    python recommend.py --user-id 1
    python recommend.py --user-id 5 --top-n 5 --alpha 0.9
    python recommend.py --user-id 1 --show-profile
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import implicit
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    normalize,
)
from sklearn.pipeline import Pipeline


DATA_DIR = Path("data-refined")
TOP_CITIES = 25   # one-hot encode top N cities; rest → "Other"
ALS_FACTORS = 64
ALS_ITERATIONS = 30
ALS_REGULARIZATION = 0.02
ALS_ALPHA = 40.0  # confidence scaling for implicit feedback


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    users = pd.read_csv(data_dir / "users.csv")
    listings = pd.read_csv(data_dir / "cleaned_listings.csv")
    likes = pd.read_csv(data_dir / "user_likes.csv")
    return users, listings, likes


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _top_cities(listings: pd.DataFrame, n: int) -> list[str]:
    return listings["city"].value_counts().head(n).index.tolist()


def build_item_features(listings: pd.DataFrame, top_cities: list[str]) -> csr_matrix:
    df = listings.copy()

    # Log-scale skewed numerics to reduce outlier influence
    df["log_price"] = np.log1p(df["price"])
    df["log_house_size"] = np.log1p(df["house_size"])
    df["log_acre_lot"] = np.log1p(df["acre_lot"])

    # Bucket city into top-N + "Other"
    df["city_bucket"] = df["city"].where(df["city"].isin(top_cities), other="Other")

    num_cols = ["log_price", "bed", "bath", "log_house_size", "log_acre_lot"]
    cat_cols = ["city_bucket", "status"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler(with_mean=False)),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    X = preprocessor.fit_transform(df)
    X = normalize(X, norm="l2", axis=1)
    return csr_matrix(X), preprocessor


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------

def build_interaction_matrix(likes: pd.DataFrame, user_ids, item_ids):
    user_to_idx = {int(u): i for i, u in enumerate(user_ids)}
    item_to_idx = {int(it): i for i, it in enumerate(item_ids)}

    rows = likes["user_id"].map(user_to_idx)
    cols = likes["listing_id"].map(item_to_idx)
    mask = rows.notna() & cols.notna()
    rows = rows[mask].astype(int).to_numpy()
    cols = cols[mask].astype(int).to_numpy()

    mat = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    ).tocsr()
    mat.sum_duplicates()
    mat.data[:] = 1.0
    return mat, user_to_idx, item_to_idx


# ---------------------------------------------------------------------------
# ALS model
# ---------------------------------------------------------------------------

def train_als(user_items: csr_matrix, seed: int = 42):
    model = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        alpha=ALS_ALPHA,
        iterations=ALS_ITERATIONS,
        random_state=seed,
    )
    model.fit(user_items, show_progress=False)
    return model


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def recommend(
    user_id: int,
    users: pd.DataFrame,
    listings: pd.DataFrame,
    likes: pd.DataFrame,
    item_features: csr_matrix,
    als_model,
    user_items: csr_matrix,
    user_to_idx: dict,
    item_to_idx: dict,
    item_ids: np.ndarray,
    top_n: int = 10,
    alpha: float = 0.8,   # weight for content; (1-alpha) for ALS
) -> pd.DataFrame:
    """
    Returns a DataFrame of top_n recommended listings for user_id.
    alpha=1.0 → pure content-based; alpha=0.0 → pure ALS.
    """
    if user_id not in user_to_idx:
        raise ValueError(f"user_id {user_id} not found in interaction data.")

    u_idx = user_to_idx[user_id]

    # Liked items for this user
    start, end = user_items.indptr[u_idx], user_items.indptr[u_idx + 1]
    liked_item_indices = set(user_items.indices[start:end])

    # --- Content score ---
    # User profile = mean of liked item feature vectors
    liked_features = item_features[sorted(liked_item_indices)]  # shape (n_liked, d)
    user_profile = np.asarray(liked_features.mean(axis=0))      # (1, d)
    user_profile = normalize(user_profile, norm="l2")            # unit vector
    scores_raw = item_features.dot(user_profile.T)
    content_scores = (scores_raw.toarray() if hasattr(scores_raw, "toarray") else np.asarray(scores_raw)).ravel()

    # --- ALS score ---
    als_ids, als_raw = als_model.recommend(
        u_idx,
        user_items[u_idx],
        N=len(item_ids),
        filter_already_liked_items=False,
    )
    als_scores_full = np.zeros(len(item_ids), dtype=np.float32)
    als_scores_full[np.asarray(als_ids, dtype=int)] = np.asarray(als_raw, dtype=np.float32)

    # --- Blend ---
    content_norm = _minmax(content_scores)
    als_norm = _minmax(als_scores_full)
    hybrid = alpha * content_norm + (1.0 - alpha) * als_norm

    # Mask out already-liked items
    hybrid[sorted(liked_item_indices)] = -np.inf

    top_indices = np.argsort(-hybrid)[:top_n]
    top_listing_ids = item_ids[top_indices]
    top_scores = hybrid[top_indices]

    recs = listings[listings["listing_id"].isin(top_listing_ids)].copy()
    score_map = dict(zip(top_listing_ids, top_scores))
    recs["rec_score"] = recs["listing_id"].map(score_map)
    recs = recs.sort_values("rec_score", ascending=False).reset_index(drop=True)
    recs.index += 1
    return recs


def user_profile_summary(user_id: int, listings: pd.DataFrame, likes: pd.DataFrame) -> dict:
    liked_ids = likes[likes["user_id"] == user_id]["listing_id"]
    liked = listings[listings["listing_id"].isin(liked_ids)]
    top_cities = liked["city"].value_counts().head(3).index.tolist()
    return {
        "n_liked": len(liked_ids),
        "avg_price": liked["price"].mean(),
        "price_range": (liked["price"].min(), liked["price"].max()),
        "avg_beds": liked["bed"].mean(),
        "avg_baths": liked["bath"].mean(),
        "avg_sqft": liked["house_size"].mean(),
        "top_cities": top_cities,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_profile(user_id: int, users: pd.DataFrame, profile: dict):
    row = users[users["user_id"] == user_id].iloc[0]
    print(f"\nUser {user_id}  |  Age: {row.get('age', '?')}  |  Gender: {row.get('gender', '?')}  |  Education: {row.get('education', '?')}")
    print(f"Liked {profile['n_liked']} listings  |  "
          f"Avg price: ${profile['avg_price']:,.0f}  |  "
          f"Price range: ${profile['price_range'][0]:,.0f} – ${profile['price_range'][1]:,.0f}")
    print(f"Avg beds: {profile['avg_beds']:.1f}  |  Avg baths: {profile['avg_baths']:.1f}  |  Avg sqft: {profile['avg_sqft']:.0f}")
    print(f"Top cities liked: {', '.join(profile['top_cities'])}")


def print_recs(recs: pd.DataFrame, top_n: int):
    cols = ["listing_id", "price", "bed", "bath", "house_size", "acre_lot", "city", "status", "rec_score"]
    display = recs[[c for c in cols if c in recs.columns]].head(top_n)
    display = display.rename(columns={
        "listing_id": "ID",
        "price": "Price ($)",
        "bed": "Beds",
        "bath": "Baths",
        "house_size": "SqFt",
        "acre_lot": "Acres",
        "city": "City",
        "status": "Status",
        "rec_score": "Score",
    })
    display["Price ($)"] = display["Price ($)"].apply(lambda x: f"${x:,.0f}")
    display["Score"] = display["Score"].apply(lambda x: f"{x:.4f}")
    print(f"\n{'Rank':<6}" + "  ".join(f"{c:<14}" for c in display.columns))
    print("-" * 110)
    for rank, (_, row) in enumerate(display.iterrows(), start=1):
        print(f"{rank:<6}" + "  ".join(f"{str(v):<14}" for v in row.values))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Real estate listing recommender")
    ap.add_argument("--user-id", type=int, required=True, help="User ID to recommend for")
    ap.add_argument("--top-n", type=int, default=10, help="Number of recommendations")
    ap.add_argument("--alpha", type=float, default=0.8, help="Content weight (0=pure ALS, 1=pure content)")
    ap.add_argument("--data-dir", default="data-refined")
    ap.add_argument("--show-profile", action="store_true", help="Print user preference profile")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print("Loading data...", end=" ", flush=True)
    users, listings, likes = load_data(data_dir)
    print(f"done. ({users.shape[0]} users, {listings.shape[0]} listings, {likes.shape[0]} likes)")

    # Build feature matrix
    print("Building item features...", end=" ", flush=True)
    top_cities = _top_cities(listings, TOP_CITIES)
    item_features, _ = build_item_features(listings, top_cities)
    item_ids = listings["listing_id"].to_numpy()
    print(f"done. Feature dim: {item_features.shape[1]}")

    # Build interaction matrix
    user_ids = np.sort(likes["user_id"].unique())
    user_items, user_to_idx, item_to_idx = build_interaction_matrix(
        likes, user_ids, item_ids
    )

    # Train ALS
    print("Training ALS model...", end=" ", flush=True)
    als_model = train_als(user_items)
    print("done.")

    # Validate user
    if args.user_id not in user_to_idx:
        print(f"\nError: user_id {args.user_id} has no interactions in the data.")
        print(f"Valid user IDs: {sorted(user_to_idx.keys())[:10]} ...")
        sys.exit(1)

    # User profile
    profile = user_profile_summary(args.user_id, listings, likes)
    if args.show_profile:
        print_profile(args.user_id, users, profile)
    else:
        row = users[users["user_id"] == args.user_id].iloc[0]
        print(f"\nUser {args.user_id}  |  Age: {row.get('age','?')}  |  Gender: {row.get('gender','?')}  |  "
              f"Liked {profile['n_liked']} listings  |  Avg price: ${profile['avg_price']:,.0f}  |  "
              f"Top cities: {', '.join(profile['top_cities'])}")

    # Recommend
    recs = recommend(
        user_id=args.user_id,
        users=users,
        listings=listings,
        likes=likes,
        item_features=item_features,
        als_model=als_model,
        user_items=user_items,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        item_ids=item_ids,
        top_n=args.top_n,
        alpha=args.alpha,
    )

    print(f"\nTop {args.top_n} recommendations  (content weight={args.alpha:.0%}, ALS weight={(1-args.alpha):.0%}):")
    print_recs(recs, args.top_n)


if __name__ == "__main__":
    main()
