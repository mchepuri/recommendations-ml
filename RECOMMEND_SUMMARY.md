# Real Estate Listing Recommender — Project Summary

**Author:** Muralimanohar Chepuri
**File:** `recommend.py`
**Course:** ML/AI — UC Berkeley

---

## 1. Problem Statement

Given a dataset of real estate listings and a history of which listings each user has "liked," build a system that recommends new listings a user is likely to be interested in — listings they have not yet seen, ranked by how well they match the user's demonstrated preferences.

This is a **personalized ranking problem** over implicit feedback (binary likes, no star ratings).

---

## 2. Dataset

Three CSV files in `data-refined/`:

| File | Rows | Columns | Description |
|---|---|---|---|
| `users.csv` | 200 | user_id, gender, age, education | User demographics |
| `cleaned_listings.csv` | 26,648 | listing_id, price, bed, bath, acre_lot, house_size, city, state, status, ... | Property attributes |
| `user_likes.csv` | 26,546 | user_id, listing_id | Implicit interaction log |

### Key statistics
- Each user liked approximately **127–136 listings** on average
- The interaction matrix is **99.5% sparse** (most user–item pairs have no signal)
- Average pairwise user overlap: **0.68 shared liked items** between any two users
- **58% of liked items** were liked by exactly one user (no co-occurrence)

---

## 3. Data Refinement

Before modeling, the `data-refined/` files were cleaned to remove noise that would corrupt feature-based similarity:

| Issue | Action | Items affected |
|---|---|---|
| Price < $10,000 or > $5,000,000 | Removed (data entry errors) | 554 listings |
| Beds > 20 or Baths > 20 | Removed (clearly erroneous) | 6 listings |
| Acre lot > 500 | Removed (extreme outliers) | 45 listings |
| `prev_sold_date` column | Dropped (73% missing, not useful for ranking) | — |
| Likes referencing removed listings | Dropped | 655 interactions |
| Duplicate user–item like pairs | Deduplicated | — |

**Result:** 26,648 clean listings and 26,546 valid interactions retained.

---

## 4. Model Selection — Why Not Pure Collaborative Filtering?

The first instinct for a recommendation problem is **Collaborative Filtering (CF)** — find users with similar taste and recommend what they liked. Three CF approaches were evaluated in the exploratory notebooks (`02_surprise_svd.ipynb`, `03_lightfm_hybrid.ipynb`, `04_implicit_als.ipynb`), and all produced near-zero Precision@10, Recall@10, MAP@10, and NDCG@10.

The root cause was the **data structure**, not the algorithms:

```
Avg shared liked items between any 2 users:  0.68
Items liked by 5+ users:                     102  out of 17,196  (0.6%)
Items liked by 10+ users:                    0
```

CF works by finding co-occurrence patterns ("users who liked A also liked B"). With almost no overlap between users, there is nothing for ALS or BPR to learn. The item embeddings are effectively random noise.

**Content-based filtering**, on the other hand, only needs item features and a user's own history — both of which are rich here. A user with 130+ liked listings has a clear, learnable preference over price, size, bedroom count, and location.

---

## 5. Chosen Strategy — Content-Weighted Hybrid

```
hybrid_score = 0.8 × content_score + 0.2 × als_score
```

- **Content-based (80%)** — the dominant signal, exploiting rich item features and the large per-user history
- **ALS collaborative (20%)** — captures the small amount of real co-occurrence that does exist (41.8% of items were liked by 2+ users)
- The blend weight `alpha` is a CLI argument and can be tuned

---

## 6. Feature Engineering (`build_item_features`)

Each listing is converted into a fixed-length feature vector that captures both numeric attributes and categorical identity.

### Numeric features (5 dimensions)

| Raw column | Transformation | Reason |
|---|---|---|
| `price` | `log(1 + price)` | Price is right-skewed; log compresses the range so a $50k house and a $100k house differ more proportionally than a $1M and a $1.05M house |
| `house_size` | `log(1 + house_size)` | Same skew issue as price |
| `acre_lot` | `log(1 + acre_lot)` | Heavily right-skewed (some properties are hundreds of acres) |
| `bed` | Standard-scaled | Linear — 3 beds vs 4 beds is a meaningful unit difference |
| `bath` | Standard-scaled | Linear |

All numeric values are imputed with the **median** before scaling (handles any remaining missing values without distorting the distribution).

### Categorical features (28 dimensions)

| Column | Encoding | Details |
|---|---|---|
| `city` | One-hot (top 25 + "Other") | Top 25 cities by listing count get their own dimension; all others → "Other" bucket. Keeps the feature space manageable while preserving the strongest location signal. |
| `status` | One-hot | `for_sale` vs `ready_to_build` |

### Normalization

After concatenating numeric and categorical features, every listing vector is **L2-normalized** (length = 1). This means cosine similarity between two listing vectors equals their dot product — computationally efficient and scale-invariant.

**Final feature dimension: 33** (5 numeric + 26 one-hot + 2 status)

---

## 7. User Profile Construction

For a given user, their preference profile is the **mean of all their liked listing feature vectors**:

```
user_profile = mean( item_features[liked_item_1],
                     item_features[liked_item_2],
                     ...
                     item_features[liked_item_N] )
```

This vector is then **L2-normalized** to a unit vector. Conceptually it sits at the centroid of the user's liked listings in feature space — pointing toward the combination of price range, bedroom count, house size, and city that the user favors most.

With 130+ liked items per user this centroid is stable and representative, unlike a user with only 2–3 interactions.

---

## 8. Content Score

The content score for every listing is the **cosine similarity** between that listing's feature vector and the user's profile vector. Because both vectors are L2-normalized, this reduces to a dot product:

```
content_scores = item_features @ user_profile   # shape: (n_items,)
```

A score of 1.0 means the listing is identical to the user's average preference; 0.0 means orthogonal (unrelated in feature space).

---

## 9. ALS Collaborative Score (`train_als`)

Alternating Least Squares (ALS) is trained on the full 200 × 26,648 sparse interaction matrix. ALS factorizes this matrix into latent user and item embeddings that reconstruct the observed likes. The `implicit` library's implementation uses **confidence-weighted** matrix factorization, where each observed like is treated with confidence `1 + alpha × 1` (alpha=40), and unobserved pairs have confidence 1.

**Hyperparameters used:**

| Parameter | Value | Meaning |
|---|---|---|
| `factors` | 64 | Embedding dimensionality |
| `iterations` | 30 | Training passes |
| `regularization` | 0.02 | L2 penalty to prevent overfitting |
| `alpha` | 40.0 | Confidence scaling for observed interactions |

Even though ALS produces weak item embeddings due to sparse co-occurrence, it contributes a small but complementary signal — particularly for the 41.8% of items liked by 2 or more users.

---

## 10. Hybrid Scoring and Ranking (`recommend`)

```
Step 1: Get content_scores   — cosine similarity for all 26,648 listings
Step 2: Get als_scores       — ALS predicted preference for all 26,648 listings
Step 3: Normalize both       — min-max scale each to [0, 1]
Step 4: Blend                — hybrid = 0.8 × content_norm + 0.2 × als_norm
Step 5: Mask liked items     — set score = -∞ for already-liked listings
Step 6: Rank                 — argsort descending, return top-N
```

Min-max normalization is applied per-user so that the content and ALS scores are on the same scale before blending. This prevents one signal from dominating purely because of its numerical range.

---

## 11. Output Format

Running the script produces a user summary and a ranked recommendation table:

```
User 1  |  Age: 69  |  Gender: Other  |  Liked 131 listings
Avg price: $337,206  |  Avg beds: 3.3  |  Top cities: Birmingham, Gulf Shores, Tuscaloosa

Top 10 recommendations  (content weight=80%, ALS weight=20%):

Rank  ID        Price ($)    Beds  Baths  SqFt   Acres  City         Status    Score
-----------------------------------------------------------------------------------------------
1     17169     $319,000     4     4      2200   0.85   Troy         for_sale  0.9363
2     21648     $150,000     3     2      1980   1.49   Jackson      for_sale  0.9296
...
```

Each recommended listing is **not** in the user's liked history and is ranked by how closely it matches their demonstrated preference profile.

---

## 12. CLI Usage

```bash
# Basic — top 10 recommendations for user 1
python recommend.py --user-id 1

# Show the user's preference profile before recommendations
python recommend.py --user-id 1 --show-profile

# Top 5, more content-driven (90% content, 10% ALS)
python recommend.py --user-id 5 --top-n 5 --alpha 0.9

# Pure ALS (collaborative only) — for comparison
python recommend.py --user-id 5 --alpha 0.0

# Use a different data directory
python recommend.py --user-id 1 --data-dir data
```

---

## 13. Libraries Used

| Library | Role |
|---|---|
| `pandas` | Data loading and tabular manipulation |
| `numpy` | Array math, normalization, argsort |
| `scipy.sparse` | Sparse COO/CSR matrix for the 200×26k interaction matrix |
| `sklearn` | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `normalize` |
| `implicit` | ALS model for implicit feedback collaborative filtering |

---

## 14. Limitations and Next Steps

### Current limitations
- **Alabama only** — all 27,250 listings are from a single state, so geographic diversity in recommendations is limited to city-level differences
- **No timestamps** — likes have no ordering, so we cannot weight recent likes more heavily or detect preference drift over time
- **Cold-start users** — a brand new user with zero likes cannot get content-based recommendations (no profile to build from)
- **Small user base** — 200 users is enough to demonstrate the system but too few to learn strong collaborative embeddings; ALS's contribution is minimal

### Recommended next steps
1. **Expand data geographically** — add listings from multiple states to make location a more meaningful feature
2. **Add timestamps to likes** — enable recency-weighted user profiles (exponential decay on older likes)
3. **Cold-start handling** — for new users, fall back to popularity-based or demographic-similar user recs until enough interactions accumulate
4. **Two-stage retrieval + reranking** — at scale, use ANN (approximate nearest neighbor) search for fast candidate retrieval, then apply a learned reranker on the top-K candidates
5. **Hyperparameter tuning** — grid-search `alpha` (content blend weight), ALS `factors`, and `iterations` using offline Recall@K as the objective
6. **Richer item features** — add text embeddings from listing descriptions, neighborhood walk scores, school ratings, distance to city center
