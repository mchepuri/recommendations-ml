### Project Title
Real Estate Listing Recommender

**Author**
Muralimanohar Chepuri

#### Executive summary
Built a recommendation baseline for real-estate listings using user-like interactions. Prepared data, explored pricing vs. house size, tried Surprise SVD for a matrix-factorization baseline (saw poor relevance because interactions are implicit likes), then switched to LightFM with a WARP loss for ranking. Reported RMSE/MAE from SVD only as a sanity check and focused on LightFM ranking metrics (AUC, Precision@K, Recall@K) and example top-N recommendations.

#### Rationale
Help users discover relevant listings faster by leveraging historical likes instead of only search filters.

#### Research Question
Given user-like interactions, which listings should we recommend to each user to maximize relevance?

#### Data Sources
- `data/users.csv` (demographics)
- `data/cleaned_listings.csv` (listing attributes, price, house_size)
- `data/user_likes.csv` (implicit interactions)

#### Methodology
- EDA on listings (price vs. square footage correlation/visualization).
- Built an implicit-only dataset from user likes.
- Trained implicit-feedback recommenders (matrix factorization and baselines); evaluated ranking metrics at K.

#### Results
- For implicit likes (binary interactions), ranking metrics such as Precision@K/Recall@K/NDCG@K are more meaningful than RMSE/MAE.
- The script `recsys_offline_eval.py` runs a reproducible offline evaluation using the `implicit` library (works in Python 3.13).

#### Reproducible evaluation
Run:
`python recsys_offline_eval.py --data-dir data-refined --split frac --test-frac 0.2 --k 10`

#### Next steps
- Add better side-information (text embeddings, geospatial features, recency) and consider a two-stage system (retrieval + reranking).
- Tune hyperparameters and evaluate Recall/NDCG at multiple K.
- Add holdout/temporal splits and cold-start handling for new listings/users.
- Consider production-serving approach (batch scoring, ANN retrieval).

#### Outline of project
- [Recommendation notebook](recommendations-ml.ipynb)

##### Contact and Further Information
For questions or collaboration, please open an issue or reach out to the author.
