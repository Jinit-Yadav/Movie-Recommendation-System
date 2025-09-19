import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Config ----------
RATINGS_CSV = "ratings.csv"
MOVIES_CSV = "movies.csv"
OUTDIR = "movie_outputs2"
K = 20  # latent factors
os.makedirs(OUTDIR, exist_ok=True)

# ---------- Load Data ----------
ratings = pd.read_csv(RATINGS_CSV)
movies = pd.read_csv(MOVIES_CSV)

# ---------- Map IDs ----------
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}
ratings['user_idx'] = ratings['userId'].map(user_map)
ratings['movie_idx'] = ratings['movieId'].map(movie_map)

# ---------- Sparse Matrix ----------
R_sparse = csr_matrix((ratings['rating'], (ratings['user_idx'], ratings['movie_idx'])))

# ---------- Truncated SVD ----------
svd = TruncatedSVD(n_components=K, random_state=42)
R_reduced = svd.fit_transform(R_sparse)
print("TruncatedSVD fit complete.")

# ---------- Save necessary objects ----------
np.save(os.path.join(OUTDIR, "R_reduced.npy"), R_reduced)
np.save(os.path.join(OUTDIR, "components.npy"), svd.components_)
pd.Series(user_map).to_csv(os.path.join(OUTDIR, "user_map.csv"))
pd.Series(movie_map).to_csv(os.path.join(OUTDIR, "movie_map.csv"))

# ---------- Content-Based Genre Matrix ----------
movies['genres_array'] = movies['genres'].str.split('|')
movies['genres_text'] = movies['genres_array'].apply(lambda x: ' '.join(x))
cv = CountVectorizer(max_features=1000)
X_genres = cv.fit_transform(movies['genres_text'])
pd.DataFrame(X_genres.toarray(), columns=cv.get_feature_names_out(), index=movies['movieId']).to_csv(
    os.path.join(OUTDIR, "movies_genres_features.csv")
)

movies.to_csv(os.path.join(OUTDIR, "movies.csv"), index=False)
print("Pipeline finished. Outputs saved to:", OUTDIR)
