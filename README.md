# Movie Recommendation System

## Dataset
MovieLens files supplied with the project:
- `data/movies.csv`: movie IDs, titles and genres.
- `data/ratings.csv`: user IDs, movie IDs, ratings and timestamps.

## Methods
- Content-Based Filtering: recommends movies with genres similar to movies the user liked.
- User-Based Collaborative Filtering: uses ratings from similar users.
- Item-Based Collaborative Filtering: uses movies with similar user-rating patterns.
- Hybrid: combines content and item-based scores equally.

## Evaluation
Precision@10, Recall@10 and F1-score@10 compare top-10 recommendations. MAE and RMSE are also reported for a user-mean rating-prediction baseline. The baseline metrics are not the same as the recommendation-ranking metrics.

## Run
```bash
pip install -r requirements.txt
python movie_recommendation.py
```
Enter a User ID present in the ratings dataset when prompted.
