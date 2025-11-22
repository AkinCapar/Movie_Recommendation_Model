from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    ingestion = DataIngestion(
        ratings_path="data/ratings.csv",
        movies_path="data/movies.csv"
    )
    ratings_df = ingestion.load_ratings()
    movies_df = ingestion.load_movies()

    transformer = DataTransformation()
    filtered = transformer.filter_popular_movies(ratings_df, movies_df)
    pivot = transformer.create_pivot(filtered)
    sparse_matrix, movie_ids = transformer.to_sparse_matrix(pivot)

    trainer = ModelTrainer()
    similarity_matrix = trainer.compute_similarity(sparse_matrix)
    trainer.save_model(
        pivot,
        sparse_matrix,
        movie_ids,
        similarity_matrix
    )


if __name__ == "__main__":
    main()