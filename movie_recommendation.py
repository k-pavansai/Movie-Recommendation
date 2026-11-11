import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

movies = movies.drop_duplicates("movieId").reset_index(drop=True)
ratings = ratings.drop_duplicates(["userId", "movieId"]).dropna(
    subset=["userId", "movieId", "rating"]
)
movies["genres"] = movies["genres"].fillna("")
ratings = ratings[ratings["movieId"].isin(movies["movieId"])].copy()
train, test = train_test_split(ratings, test_size=0.20, random_state=42)

user_item = train.pivot_table(index="userId", columns="movieId", values="rating")
filled = user_item.fillna(0)
movie_to_idx = {mid: i for i, mid in enumerate(movies["movieId"])}
user_ids = list(user_item.index)
user_similarity = cosine_similarity(filled)
user_sim = pd.DataFrame(user_similarity, index=user_ids, columns=user_ids)

tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genres"])
item_user = filled.T
knn = NearestNeighbors(metric="cosine", algorithm="brute",
                       n_neighbors=min(51, len(item_user)))
knn.fit(item_user.values)
distances, indices = knn.kneighbors(item_user.values)
cf_ids = list(item_user.index)
cf_to_idx = {mid: i for i, mid in enumerate(cf_ids)}

def normalize(values):
    values = np.asarray(values, dtype=float)
    low, high = values.min(), values.max()
    return np.zeros_like(values) if high == low else (values-low)/(high-low)

def content_scores(user_id):
    scores = np.zeros(len(movies))
    if user_id not in user_item.index:
        return scores
    liked = user_item.loc[user_id].dropna()
    liked_ids = liked[liked >= 4].index.tolist()
    if not liked_ids:
        liked_ids = liked.index.tolist()
    liked_idx = [movie_to_idx[x] for x in liked_ids if x in movie_to_idx]
    if liked_idx:
        profile = np.asarray(genre_matrix[liked_idx].mean(axis=0)).ravel()
        scores = np.asarray(genre_matrix @ profile).ravel()
    return scores

def user_scores(user_id):
    scores = np.zeros(len(movies))
    if user_id not in user_item.index:
        return scores
    sims = user_sim.loc[user_id].drop(user_id)
    for mid in user_item.columns:
        movie_ratings = user_item[mid].dropna()
        common = sims.index.intersection(movie_ratings.index)
        s = sims.loc[common]
        r = movie_ratings.loc[common]
        valid = s > 0
        s, r = s[valid], r[valid]
        if len(s):
            top = s.nlargest(20).index
            scores[movie_to_idx[mid]] = np.average(r.loc[top], weights=s.loc[top])
    return scores

def item_scores(user_id):
    scores = np.zeros(len(movies))
    weights = np.zeros(len(movies))
    if user_id not in user_item.index:
        return scores
    for mid, rating in user_item.loc[user_id].dropna().items():
        if mid not in cf_to_idx:
            continue
        idx = cf_to_idx[mid]
        for dist, neighbor in zip(distances[idx], indices[idx]):
            other_id = cf_ids[neighbor]
            if other_id == mid or other_id not in movie_to_idx:
                continue
            sim = 1 - dist
            j = movie_to_idx[other_id]
            scores[j] += sim * rating
            weights[j] += sim
    valid = weights > 0
    scores[valid] /= weights[valid]
    return scores

def recommend(user_id, method="Hybrid", top_n=10):
    if user_id not in user_item.index:
        return pd.DataFrame()
    seen = set(user_item.loc[user_id].dropna().index)
    candidates = np.array([i for i, mid in enumerate(movies.movieId) if mid not in seen])
    if len(candidates) == 0:
        return pd.DataFrame()
    if method == "Content-Based":
        scores = content_scores(user_id)
    elif method == "User-Based":
        scores = user_scores(user_id)
    elif method == "Item-Based":
        scores = item_scores(user_id)
    elif method == "Hybrid":
        c, it = content_scores(user_id), item_scores(user_id)
        scores = np.zeros(len(movies))
        scores[candidates] = 0.5*normalize(c[candidates]) + 0.5*normalize(it[candidates])
    else:
        raise ValueError("Unknown method")
    chosen = candidates[np.argsort(scores[candidates])[::-1][:top_n]]
    result = movies.iloc[chosen][["movieId", "title", "genres"]].copy()
    result["Recommendation Score"] = scores[chosen]
    return result

def evaluate():
    relevant = test[test.rating >= 4].groupby("userId").movieId.apply(set)
    rows = []
    for method in ["Content-Based", "User-Based", "Item-Based", "Hybrid"]:
        ps, rs, fs = [], [], []
        for uid, actual in relevant.items():
            rec = recommend(uid, method, 10)
            if rec.empty:
                continue
            hits = len(actual.intersection(set(rec.movieId)))
            precision = hits / 10
            recall = hits / len(actual) if actual else 0
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
            ps.append(precision); rs.append(recall); fs.append(f1)
        rows.append({"Algorithm": method,
                     "Precision@10": np.mean(ps) if ps else 0,
                     "Recall@10": np.mean(rs) if rs else 0,
                     "F1-score@10": np.mean(fs) if fs else 0})
    result = pd.DataFrame(rows).sort_values("F1-score@10", ascending=False)
    print("\nRECOMMENDATION MODEL COMPARISON")
    print(result.round(4).to_string(index=False))
    result.set_index("Algorithm").plot(kind="bar", ylim=(0, 1), figsize=(9, 5))
    plt.title("Movie Recommendation Algorithm Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "recommendation_comparison.png"), dpi=200)
    plt.show()

    # Rating prediction metrics for a simple user-mean baseline.
    user_means = train.groupby("userId").rating.mean()
    global_mean = train.rating.mean()
    predicted = [user_means.get(uid, global_mean) for uid in test.userId]
    print("\nUser-mean rating baseline")
    print("MAE:", round(mean_absolute_error(test.rating, predicted), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(test.rating, predicted)), 4))

if __name__ == "__main__":
    print("Movies:", movies.movieId.nunique(), "| Users:", ratings.userId.nunique(),
          "| Ratings:", len(ratings))
    evaluate()
    uid = int(input("\nEnter a User ID for recommendations: "))
    print("\nHYBRID RECOMMENDATIONS")
    print(recommend(uid, "Hybrid", 10).to_string(index=False))
