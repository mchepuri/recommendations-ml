### Project Title
Real Estate Listing Recommender

**Author**
Muralimanohar Chepuri

#### Executive summary
Built a recommendation baseline for real-estate listings using user-like interactions. Prepared data, explored pricing vs. house size, trained an SVD model with Surprise, and built an implicit-feedback ranking model with LightFM (WARP), reporting RMSE/MAE (for comparison only) and ranking metrics (AUC, Precision@K, Recall@K).

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
- Build implicit-like dataset for Surprise; train/test split and 5-fold CV with SVD to compare models.
- Train LightFM WARP model on user-likes; evaluate AUC, Precision@10, Recall@10; generate sample top-N recommendations.

#### Results
- Surprise SVD: provides latent factors; RMSE/MAE on implicit 1.0 ratings are for relative model comparison only.
- LightFM (WARP): produces ranking metrics (AUC, Precision@10, Recall@10) and sample top-5 recommendations per user; performs better-suited ranking for implicit data.

#### Next steps
- Incorporate user/listing side features into LightFM (demographics, price, home size).
- Tune LightFM hyperparameters (components, learning rate, loss) and evaluate Recall/NDCG at multiple K.
- Add holdout/temporal splits and cold-start handling for new listings/users.
- Consider production-serving approach (batch scoring or approximate nearest neighbors on embeddings).

#### Outline of project
- [Recommendation notebook](recommendations-ml.ipynb)

##### Contact and Further Information
For questions or collaboration, please open an issue or reach out to the author.
