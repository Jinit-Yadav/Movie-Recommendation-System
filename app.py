from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---------- Load precomputed data ----------
R_reduced = np.load("movie_outputs2/R_reduced.npy")
components = np.load("movie_outputs2/components.npy")
user_map = pd.read_csv("movie_outputs2/user_map.csv", index_col=0).to_dict()['0']
movie_map = pd.read_csv("movie_outputs2/movie_map.csv", index_col=0).to_dict()['0']
movies = pd.read_csv("movie_outputs2/movies.csv")
genres_matrix = pd.read_csv("movie_outputs2/movies_genres_features.csv", index_col=0)

# ---------- Helper functions ----------
def recommend_by_user(user_id, top_n=10):
    """Collaborative filtering recommendation for known users"""
    if user_id not in user_map:
        return pd.DataFrame()
    
    u_idx = user_map[user_id]
    user_vec = R_reduced[u_idx, :]
    preds = components.T @ user_vec
    top_idx = np.argsort(-preds)[:top_n]
    
    # Map back to movie IDs
    top_movie_ids = [mid for mid, idx in movie_map.items() if idx in top_idx]
    top_ratings = preds[top_idx]
    
    return pd.DataFrame({'movieId': top_movie_ids, 'predicted_rating': top_ratings}).merge(movies, on='movieId')


def recommend_by_genre(preferred_genres, top_n=10):
    """Content-based recommendation for new users"""
    selected_cols = [g.lower() for g in preferred_genres if g.lower() in genres_matrix.columns]
    if not selected_cols:
        return pd.DataFrame()
    
    # Sum scores of selected genres
    genre_scores = genres_matrix[selected_cols].sum(axis=1)
    top_idx = genre_scores.sort_values(ascending=False).head(top_n).index
    
    return movies[movies['movieId'].isin(top_idx)]


# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = pd.DataFrame()
    message = ""
    
    if request.method == "POST":
        user_id = request.form.get("user_id")
        genres = request.form.get("genres")
        
        if user_id:  # Existing user
            try:
                user_id = int(user_id)
                recommendations = recommend_by_user(user_id)
                if recommendations.empty:
                    message = "User ID not found. Showing recommendations by genre instead."
            except ValueError:
                message = "Invalid User ID. Showing recommendations by genre instead."
        
        if recommendations.empty and genres:  # First-time user
            genre_list = [g.strip() for g in genres.split(',')]
            recommendations = recommend_by_genre(genre_list)
            if recommendations.empty:
                message = "No matching movies found for the selected genres."
        
        if recommendations.empty and not genres:
            # Cold-start fallback: show popular movies
            top_movies = movies.head(10)
            recommendations = top_movies
            message = "Showing top popular movies."

    return render_template("index.html", recommendations=recommendations.to_dict(orient="records"), message=message)


if __name__ == "__main__":
    app.run(debug=True)
