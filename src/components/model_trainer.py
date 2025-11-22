from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from src.exception import CustomException
from src.logger import logging
import sys

class ModelTrainer:
    def compute_similarity(self, sparse_matrix):
        try:
            similarity = cosine_similarity(sparse_matrix.T)


            logging.info("Cosine similarity calculated.")
            return similarity
        
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, pivot, sparse_matrix, movie_ids, similarity_matrix, path="artifacts/movie_recommender.pkl"):
        try:
            os.makedirs("artifacts", exist_ok=True)

            model_data = {
                "pivot": pivot,
                "sparse_matrix": sparse_matrix,
                "movie_ids": movie_ids,
                "similarity_matrix": similarity_matrix
            }

            joblib.dump(model_data, path)
            logging.info(f"Model is saved to {path}.")
            
        
        except Exception as e:
            raise CustomException(e, sys)
