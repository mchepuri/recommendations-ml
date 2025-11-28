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
- Built an implicit-only dataset for Surprise SVD; train/test split and 5-fold CV (kept RMSE/MAE just for comparison, not as a ranking target).
- Trained LightFM WARP on user likes (implicit feedback); evaluated AUC, Precision@10, Recall@10; generated example top-N recommendations per user.

#### Results
- Surprise SVD: ran but produced weak relevance because implicit likes collapsed to a single rating; RMSE/MAE kept only as a baseline sanity check.
- LightFM (WARP): primary model; delivers ranking metrics (AUC, Precision@10, Recall@10) and sample top-5 recommendations; better suited for implicit interactions and used going forward. The notebook uses `lightfm.evaluation.{auc_score, precision_at_k, recall_at_k}` on an 80/20 train/test split—rerun the LightFM section to view the current metric values.

#### Next steps
- Incorporate user/listing side features into LightFM (demographics, price, home size).
- Tune LightFM hyperparameters (components, learning rate, loss) and evaluate Recall/NDCG at multiple K.
- Add holdout/temporal splits and cold-start handling for new listings/users.
- Consider production-serving approach (batch scoring or approximate nearest neighbors on embeddings).

#### Outline of project
- [Recommendation notebook](recommendations-ml.ipynb)

##### Contact and Further Information
For questions or collaboration, please open an issue or reach out to the author.
