import gradio as gr
import pandas as pd
from src.pipeline.predict_pipeline import MovieRecommender

recommender = MovieRecommender(
    model_path="artifacts/movie_recommender.pkl",
    movies_path="data/filtered_movies.csv"
)

movies_df = pd.read_csv("data/filtered_movies.csv")
movie_titles = sorted(movies_df["title"].unique().tolist())


def recommend_movie(selected_movie):
    try:
        recommendations = recommender.get_similar_movies(selected_movie, top_n=10)
        return "\n".join(recommendations)
    except Exception as e:
        return str(e)


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🎬 Movie recommendation system.")

    movie_input = gr.Dropdown(
        choices=movie_titles,
        label="Choose a movie that you like.",
        interactive=True
    )

    output = gr.Textbox(
        label="Recommended movies.",
        lines=10
    )

    btn = gr.Button("Show recommendations.")
    btn.click(fn=recommend_movie, inputs=movie_input, outputs=output)


if __name__ == "__main__":
    app.launch()
