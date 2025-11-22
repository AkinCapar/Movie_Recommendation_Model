from scipy.sparse import csr_matrix
from src.exception import CustomException
from src.logger import logging
import sys
import os


class DataTransformation:
    def filter_popular_movies(self, ratings_df, movies_df, min_ratings=10):
        try:
            movie_counts = ratings_df['movieId'].value_counts()
            popular_movies = movie_counts[movie_counts >= min_ratings].index

            filtered = ratings_df[ratings_df['movieId'].isin(popular_movies)].copy()
            filtered_movies = movies_df[movies_df["movieId"].isin(popular_movies)].copy()
            logging.info("Popular movies are filtered.")

            os.makedirs("data", exist_ok=True)
            filtered_movies.to_csv("data/filtered_movies.csv", index=False)

            logging.info("Filtered movies saved to data/filtered_movies.csv")
            return filtered
        
        except Exception as e:
            raise CustomException(e, sys)

    def create_pivot(self, ratings_df):
        try:
            pivot = ratings_df.pivot(index="userId",
                                 columns="movieId",
                                 values="rating").fillna(0)
            
            logging.info("Pivot is created.")
            return pivot
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def to_sparse_matrix(self, pivot):
        try:
            sparse = csr_matrix(pivot.values)
            movie_ids = list(pivot.columns)

            logging.info("Sparse matrix is constructed.")
            return sparse, movie_ids
        
        except Exception as e:
            raise CustomException(e, sys)