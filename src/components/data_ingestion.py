import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys


class DataIngestion:
    def __init__(self, ratings_path, movies_path=None):
        try:
            self.ratings_path = ratings_path
            self.movies_path = movies_path
        
        except Exception as e:
            raise CustomException(e, sys)

    def load_ratings(self):
        logging.info("Ratings data ingestion is started.")
        try:
            return pd.read_csv(self.ratings_path)
        
        except Exception as e:
            raise CustomException(e, sys) 

    def load_movies(self):
        logging.info("Movies data ingestion is started.")
        try:     
            if self.movies_path:
                return pd.read_csv(self.movies_path)
            
        except Exception as e:
            raise CustomException(e, sys)