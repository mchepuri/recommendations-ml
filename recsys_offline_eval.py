import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_mappings(users: np.ndarray, items: np.ndarray):
    users = np.asarray(users)
    items = np.asarray(items)
    user_to_index = {int(u): i for i, u in enumerate(users)}
    item_to_index = {int(it): i for i, it in enumerate(items)}
    index_to_user = {i: int(u) for i, u in enumerate(users)}
    index_to_item = {i: int(it) for i, it in enumerate(items)}
    return user_to_index, item_to_index, index_to_user, index_to_item


def _build_user_items(likes: pd.DataFrame, user_to_index, item_to_index, n_users: int, n_items: int):
    rows = likes["user_id"].map(user_to_index).astype("Int64")
    cols = likes["listing_id"].map(item_to_index).astype("Int64")
    mask = rows.notna() & cols.notna()
    rows = rows[mask].astype(int).to_numpy()
    cols = cols[mask].astype(int).to_numpy()
    data = np.ones_like(rows, dtype=np.float32)
    user_items = sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()
    user_items.sum_duplicates()
    user_items.data[:] = 1.0
    return user_items, int((~mask).sum())


def _leave_one_out_split(user_items: sparse.csr_matrix, seed: int):
    rng = _seeded_rng(seed)
    user_items = user_items.tocsr()
    user_items.sort_indices()

    train = user_items.copy().tolil()
    test_pairs = []

    for u in range(user_items.shape[0]):
        start, end = user_items.indptr[u], user_items.indptr[u + 1]
        items = user_items.indices[start:end]
        if len(items) < 2:
            continue
        test_item = int(rng.choice(items))
        train[u, test_item] = 0.0
        test_pairs.append((u, test_item))

    train = train.tocsr()
    train.eliminate_zeros()
    return train, test_pairs


def _fraction_split(user_items: sparse.csr_matrix, test_fraction: float, seed: int):
    rng = _seeded_rng(seed)
    user_items = user_items.tocsr()
    user_items.sort_indices()

    train = user_items.copy().tolil()
    test = sparse.lil_matrix(user_items.shape, dtype=np.float32)

    for u in range(user_items.shape[0]):
        start, end = user_items.indptr[u], user_items.indptr[u + 1]
        items = user_items.indices[start:end]
        if len(items) < 2:
            continue
        n_test = max(1, int(round(len(items) * test_fraction)))
        n_test = min(n_test, len(items) - 1)
        test_items = rng.choice(items, size=n_test, replace=False)
        for it in test_items:
            train[u, int(it)] = 0.0
            test[u, int(it)] = 1.0

    train = train.tocsr()
    train.eliminate_zeros()
    test = test.tocsr()
    test.eliminate_zeros()
    return train, test


@dataclass(frozen=True)
class Metrics:
    users_eval: int
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float
    coverage_at_k: float


def _metrics_at_k(recs_by_user: dict[int, list[int]], test_user_items: sparse.csr_matrix, n_items: int, k: int):
    precision = 0.0
    recall = 0.0
    ap = 0.0
    ndcg = 0.0
    recommended_items = set()

    users_eval = 0
    for u in range(test_user_items.shape[0]):
        start, end = test_user_items.indptr[u], test_user_items.indptr[u + 1]
        if start == end:
            continue
        users_eval += 1
        test_items = set(test_user_items.indices[start:end])

        recs = recs_by_user.get(u, [])[:k]
        recommended_items.update(recs)

        hits = 0
        dcg = 0.0
        running_hits = 0
        user_ap = 0.0
        for rank, item in enumerate(recs, start=1):
            if item in test_items:
                hits += 1
                running_hits += 1
                dcg += 1.0 / np.log2(rank + 1)
                user_ap += running_hits / rank

        precision += hits / max(1, k)
        recall += hits / max(1, len(test_items))

        denom = min(k, len(test_items))
        if denom:
            ap += user_ap / denom

        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, denom + 1))
        if idcg > 0:
            ndcg += dcg / idcg

    if users_eval == 0:
        return Metrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    return Metrics(
        users_eval=users_eval,
        precision_at_k=precision / users_eval,
        recall_at_k=recall / users_eval,
        map_at_k=ap / users_eval,
        ndcg_at_k=ndcg / users_eval,
        coverage_at_k=len(recommended_items) / max(1, n_items),
    )


def _build_item_features(listings: pd.DataFrame):
    df = listings.copy()
    # Treat these as categorical IDs rather than continuous variables.
    for col in ["street", "zip_code", "brokered_by"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str)

    num_cols = [c for c in ["price", "bed", "bath", "acre_lot", "house_size"] if c in df.columns]
    cat_cols = [c for c in ["status", "city", "state", "zip_code", "brokered_by", "street"] if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = pre.fit_transform(df)
    X = normalize(X, norm="l2", axis=1, copy=False)
    X = sparse.csr_matrix(X)
    return X, pre


def _build_user_features(users: pd.DataFrame):
    df = users.copy()
    num_cols = [c for c in ["age"] if c in df.columns]
    cat_cols = [c for c in ["gender", "education"] if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = pre.fit_transform(df)
    X = normalize(X, norm="l2", axis=1, copy=False)
    X = sparse.csr_matrix(X)
    return X, pre


def _train_reranker_sgd(
    train_user_items: sparse.csr_matrix,
    user_features: sparse.spmatrix,
    item_features: sparse.spmatrix,
    seed: int,
    neg_ratio: int = 5,
):
    rng = _seeded_rng(seed)

    pos_users, pos_items = train_user_items.nonzero()
    pos_users = pos_users.astype(np.int32)
    pos_items = pos_items.astype(np.int32)

    # Negative sampling per user for speed.
    neg_users = []
    neg_items = []
    n_items = train_user_items.shape[1]
    for u in range(train_user_items.shape[0]):
        start, end = train_user_items.indptr[u], train_user_items.indptr[u + 1]
        liked = set(train_user_items.indices[start:end])
        n_pos_u = end - start
        if n_pos_u == 0:
            continue
        n_neg_u = n_pos_u * neg_ratio
        # Sample until we have enough negatives not in liked.
        picked = set()
        while len(picked) < n_neg_u:
            cand = int(rng.integers(0, n_items))
            if cand in liked or cand in picked:
                continue
            picked.add(cand)
        neg_users.extend([u] * n_neg_u)
        neg_items.extend(list(picked))

    neg_users = np.asarray(neg_users, dtype=np.int32)
    neg_items = np.asarray(neg_items, dtype=np.int32)

    X_pos = sparse.hstack([user_features[pos_users], item_features[pos_items]], format="csr")
    y_pos = np.ones(X_pos.shape[0], dtype=np.int8)
    X_neg = sparse.hstack([user_features[neg_users], item_features[neg_items]], format="csr")
    y_neg = np.zeros(X_neg.shape[0], dtype=np.int8)

    X = sparse.vstack([X_pos, X_neg], format="csr")
    y = np.concatenate([y_pos, y_neg])

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=25,
        tol=1e-3,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _recommend_rerank(
    base_model,
    reranker,
    train_user_items: sparse.csr_matrix,
    user_features: sparse.spmatrix,
    item_features: sparse.spmatrix,
    k: int,
    candidates: int = 1000,
):
    recs = {}
    for u in range(train_user_items.shape[0]):
        cand_ids, _scores = base_model.recommend(
            userid=u,
            user_items=train_user_items[u],
            N=candidates,
            filter_already_liked_items=True,
        )
        cand_ids = np.asarray(cand_ids, dtype=np.int32)
        row = user_features[u : u + 1]
        uf = sparse.vstack([row] * len(cand_ids), format="csr")
        X = sparse.hstack([uf, item_features[cand_ids]], format="csr")
        # decision_function works even if classes are imbalanced; higher = more positive.
        s = reranker.decision_function(X)
        top = np.argpartition(-s, kth=min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        recs[u] = [int(cand_ids[i]) for i in top]
    return recs


def _recommend_popular(train_user_items: sparse.csr_matrix, n_items: int, k: int):
    item_pop = np.asarray(train_user_items.sum(axis=0)).ravel().astype(np.float32)
    popular = np.argsort(-item_pop)
    recs = {}
    for u in range(train_user_items.shape[0]):
        start, end = train_user_items.indptr[u], train_user_items.indptr[u + 1]
        seen = set(train_user_items.indices[start:end])
        out = []
        for it in popular:
            if it not in seen:
                out.append(int(it))
                if len(out) >= k:
                    break
        recs[u] = out
    return recs


def _recommend_content(train_user_items: sparse.csr_matrix, item_features: sparse.spmatrix, k: int):
    # User profile = sum of liked item vectors; cosine works since features are L2-normalized.
    recs = {}
    for u in range(train_user_items.shape[0]):
        profile = train_user_items[u].dot(item_features)  # 1 x d
        profile = normalize(profile, norm="l2", axis=1, copy=False)
        scores = item_features.dot(profile.T).toarray().ravel()
        scores[np.asarray(train_user_items[u].todense()).ravel() > 0] = -np.inf
        top = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        recs[u] = [int(i) for i in top]
    return recs


def _fit_implicit_models(user_items: sparse.csr_matrix, seed: int):
    import implicit

    als = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.02,
        alpha=40.0,
        iterations=30,
        random_state=seed,
    )
    als.fit(user_items)

    bpr = implicit.bpr.BayesianPersonalizedRanking(
        factors=64,
        regularization=0.01,
        learning_rate=0.05,
        iterations=80,
        random_state=seed,
    )
    bpr.fit(user_items)

    return als, bpr


def _recommend_implicit(model, train_user_items: sparse.csr_matrix, k: int):
    recs = {}
    for u in range(train_user_items.shape[0]):
        ids, _scores = model.recommend(
            userid=u,
            user_items=train_user_items[u],
            N=k,
            filter_already_liked_items=True,
        )
        recs[u] = [int(i) for i in ids]
    return recs


def _recommend_hybrid_als_content(als_model, train_user_items: sparse.csr_matrix, item_features: sparse.spmatrix, k: int, alpha: float):
    recs = {}
    for u in range(train_user_items.shape[0]):
        als_ids, als_scores = als_model.recommend(
            userid=u,
            user_items=train_user_items[u],
            N=500,
            filter_already_liked_items=True,
        )
        als_ids = np.asarray(als_ids, dtype=int)
        als_scores = np.asarray(als_scores, dtype=np.float32)

        profile = train_user_items[u].dot(item_features)
        profile = normalize(profile, norm="l2", axis=1, copy=False)
        # Content candidates: score all items, then take top 500.
        content_scores_all = item_features.dot(profile.T).toarray().ravel().astype(np.float32)
        seen = train_user_items[u].indices
        content_scores_all[seen] = -np.inf
        content_top = np.argpartition(-content_scores_all, kth=min(500, len(content_scores_all) - 1))[:500]
        content_top = content_top[np.argsort(-content_scores_all[content_top])]

        cand = np.unique(np.concatenate([als_ids, content_top.astype(int)]))

        # Normalize within candidate set.
        als_map = {int(i): float(s) for i, s in zip(als_ids, als_scores)}
        als_c = np.array([als_map.get(int(i), 0.0) for i in cand], dtype=np.float32)
        cont_c = content_scores_all[cand]

        def _minmax(x: np.ndarray):
            finite = np.isfinite(x)
            if not finite.any():
                return np.zeros_like(x)
            lo = x[finite].min()
            hi = x[finite].max()
            if hi - lo < 1e-8:
                return np.zeros_like(x)
            out = np.zeros_like(x)
            out[finite] = (x[finite] - lo) / (hi - lo)
            return out

        als_n = _minmax(als_c)
        cont_n = _minmax(cont_c)
        hybrid = alpha * als_n + (1.0 - alpha) * cont_n

        top = np.argpartition(-hybrid, kth=min(k, len(hybrid) - 1))[:k]
        top = top[np.argsort(-hybrid[top])]
        recs[u] = [int(cand[i]) for i in top]
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data-refined", help="Directory containing users.csv, cleaned_listings.csv, user_likes.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.7, help="Hybrid blend weight for ALS vs content")
    ap.add_argument("--split", choices=["frac", "loo"], default="frac", help="How to create the offline test set")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Per-user test fraction when split=frac")
    ap.add_argument("--out", default="reports/offline_metrics.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    users_df = pd.read_csv(data_dir / "users.csv")
    listings_df = pd.read_csv(data_dir / "cleaned_listings.csv")
    likes_df = pd.read_csv(data_dir / "user_likes.csv")

    all_user_ids = np.sort(likes_df["user_id"].unique())
    all_item_ids = np.sort(listings_df["listing_id"].unique())
    user_to_index, item_to_index, index_to_user, index_to_item = _build_mappings(all_user_ids, all_item_ids)

    user_items, skipped = _build_user_items(likes_df, user_to_index, item_to_index, len(all_user_ids), len(all_item_ids))
    if skipped:
        print(f"Warning: skipped {skipped} likes with listing_id not in listings.")

    print(f"Users: {len(all_user_ids)} | Items: {len(all_item_ids)} | Interactions: {int(user_items.nnz)}")
    if args.split == "loo":
        train_user_items, test_pairs = _leave_one_out_split(user_items, seed=args.seed)
        test_user_items = sparse.csr_matrix(user_items.shape, dtype=np.float32)
        for u, it in test_pairs:
            test_user_items[u, it] = 1.0
        print(f"Split: leave-one-out | Eval users (>=2 likes): {len(test_pairs)}")
    else:
        train_user_items, test_user_items = _fraction_split(user_items, test_fraction=args.test_frac, seed=args.seed)
        eval_users = int((test_user_items.getnnz(axis=1) > 0).sum())
        print(f"Split: per-user {args.test_frac:.2f} holdout | Eval users: {eval_users}")

    item_features, _pre = _build_item_features(listings_df.set_index("listing_id").loc[all_item_ids].reset_index())
    user_features, _upre = _build_user_features(users_df.set_index("user_id").loc[all_user_ids].reset_index())

    als, bpr = _fit_implicit_models(user_items=train_user_items, seed=args.seed)
    reranker = _train_reranker_sgd(train_user_items, user_features=user_features, item_features=item_features, seed=args.seed, neg_ratio=5)

    results = {}
    for name, recs in [
        ("popular", _recommend_popular(train_user_items, n_items=len(all_item_ids), k=args.k)),
        ("content", _recommend_content(train_user_items, item_features=item_features, k=args.k)),
        ("als", _recommend_implicit(als, train_user_items, k=args.k)),
        ("bpr", _recommend_implicit(bpr, train_user_items, k=args.k)),
        ("hybrid_als_content", _recommend_hybrid_als_content(als, train_user_items, item_features=item_features, k=args.k, alpha=args.alpha)),
        ("als_rerank_sgd", _recommend_rerank(als, reranker, train_user_items, user_features, item_features, k=args.k, candidates=1000)),
    ]:
        m = _metrics_at_k(recs, test_user_items, n_items=len(all_item_ids), k=args.k)
        results[name] = {
            "users_eval": m.users_eval,
            f"precision@{args.k}": m.precision_at_k,
            f"recall@{args.k}": m.recall_at_k,
            f"map@{args.k}": m.map_at_k,
            f"ndcg@{args.k}": m.ndcg_at_k,
            f"coverage@{args.k}": m.coverage_at_k,
        }
        print(
            f"{name:18s} "
            f"P@{args.k}={m.precision_at_k:.4f} "
            f"R@{args.k}={m.recall_at_k:.4f} "
            f"MAP@{args.k}={m.map_at_k:.4f} "
            f"NDCG@{args.k}={m.ndcg_at_k:.4f} "
            f"Cov@{args.k}={m.coverage_at_k:.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"k": args.k, "seed": args.seed, "alpha": args.alpha, "split": args.split, "test_frac": args.test_frac, "results": results},
            indent=2,
        )
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
