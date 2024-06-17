import streamlit as st
import pandas as pd
import pickle
import zipfile
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the movie data
columns = ["movieid", "title", "genre"]
movies_df = pd.read_csv("movies.csv")
movies_df.columns = columns

# Extract movie titles and create a lookup for movie IDs
movie_titles = movies_df['title'].tolist()
movie_id_lookup = dict(zip(movies_df['title'], movies_df['movieid']))

# Initialize session state for storing movie ratings
if 'movie_ratings' not in st.session_state:
    st.session_state['movie_ratings'] = {}

# Unzip and load the model
with zipfile.ZipFile("model.pkl.zip", "r") as zip_ref:
    zip_ref.extractall("model")
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to get top N recommendations
def _get_top_n_recommendations(user_ratings, algo, n=10):
    # Create a dictionary to store the sum of similarity scores and weighted ratings
    scores = defaultdict(float)
    total_sim = defaultdict(float)

    # Iterate over each rated item
    for movie_title, rating in user_ratings.items():
        movie_id = movie_id_lookup[movie_title]
        try:
            # Get the inner id of the item
            inner_id = algo.trainset.to_inner_iid(movie_id)
            
            # Get the k-nearest neighbors of this item
            neighbors = algo.get_neighbors(inner_id, k=n)

            # Calculate weighted ratings
            for neighbor in neighbors:
                neighbor_id = algo.trainset.to_raw_iid(neighbor)
                sim_score = algo.sim[inner_id][neighbor]
                scores[neighbor_id] += sim_score * rating
                total_sim[neighbor_id] += sim_score
        except Exception as e:
            logger.error(f"Error processing movie_id {movie_id}: {e}")

    # Normalize the scores by the total similarity
    for movie_id in scores:
        if total_sim[movie_id] != 0:
            scores[movie_id] /= total_sim[movie_id]

    # Sort and get the top n recommendations
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_n_movies = sorted_scores[:n]

    return top_n_movies

# Function to handle recommendations
def recommend_movies(movie_ratings):
    recommendations = _get_top_n_recommendations(movie_ratings, model, n=5)
    recommended_movie_titles = [movies_df[movies_df['movieid'] == movie_id].iloc[0]['title'] for movie_id, _ in recommendations]
    return recommended_movie_titles

def main():
    st.title("Movie Search")

    # Multiselect for choosing multiple movies
    selected_movies = st.multiselect("Search for movies:", movie_titles)

    if selected_movies:
        for movie in selected_movies:
            # Display the selected movie
            st.write(f"You selected: {movie}")

            # Allow user to rate the movie
            rating = st.slider(f"Rate {movie}", 0, 5, 0, key=movie)

            # Store the rating in session state
            if rating > 0:
                st.session_state['movie_ratings'][movie] = rating

    # Display the movie ratings dictionary
    st.write("Movie Ratings:")
    st.write(st.session_state['movie_ratings'])

    # Recommend button
    if st.button("Recommend"):
        recommendations = recommend_movies(st.session_state['movie_ratings'])
        st.write("Recommendations:")
        for recommendation in recommendations:
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; margin: 10px; border-radius: 5px; font-size: 20px;">
                {recommendation}
            </div>
            """, unsafe_allow_html=True)

    # Clear button
    if st.button("Clear"):
        st.session_state['movie_ratings'] = {}
        st.experimental_rerun()

if __name__ == "__main__":
    main()
