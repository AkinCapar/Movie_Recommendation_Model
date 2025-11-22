import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, model_path="artifacts/movie_recommender.pkl", movies_path=None):
        self.model = joblib.load(model_path)

        self.pivot = self.model["pivot"]
        self.sparse_matrix = self.model["sparse_matrix"]
        self.movie_ids = self.model["movie_ids"]
        self.similarity_matrix = self.model["similarity_matrix"]

        if movies_path:
            self.movies_df = pd.read_csv(movies_path)
        else:
            raise ValueError("No movies_path found.")

    
    def get_similar_movies(self, title, top_n=10):
        movie_id = self.movies_df.loc[self.movies_df['title'] == title, 'movieId'].values[0]

        if movie_id not in self.movie_ids:
            raise ValueError("This movie has fewer than 10 ratings. Try another.")

        idx = self.movie_ids.index(movie_id)
        sim_row = self.similarity_matrix[idx]

        sim_row[idx] = -1

        top_indices = sim_row.argsort()[-top_n:][::-1]
        top_movie_ids = [self.movie_ids[i] for i in top_indices]

        return self.movies_df[self.movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
