from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from toignore import SECRETS
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict
import pickle
from collections import defaultdict
import logging
import sys
sys.path.append("/Users/ibragimzhussup/Desktop/MovieRecommendation")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def _get_top_n_recommendations(user_ratings, algo, n=10):
    # Create a dictionary to store the sum of similarity scores and weighted ratings
    scores = defaultdict(float)
    total_sim = defaultdict(float)

    # Iterate over each rated item
    for movie_id, rating in user_ratings.items():
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

# Database connection details
connection_link = f"postgresql://{SECRETS.USERNAME}:{SECRETS.PASSWORD}@{SECRETS.HOSTNAME}:{SECRETS.PORT}/{SECRETS.DATABASE_NAME}"
engine = create_engine(connection_link)

app = FastAPI()

class Item(BaseModel):
    movie_ratings: Dict[int, float]

@app.get("/")
def read_root():
    return {"Health": "OK"}

@app.get("/movies/by_name/{movie_name}")
def get_movie_id_by_name(movie_name: str):
    query = text("SELECT movieid FROM movies WHERE title = :movie_name")
    try:
        with engine.connect() as connection:
            result = connection.execute(query, {"movie_name": movie_name}).fetchone()
            if result:
                return {"movieid": result[0]}
            else:
                raise HTTPException(status_code=404, detail="Movie not found")
    except SQLAlchemyError as e:
        logger.error(f"Error querying database: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/movies/by_id/{movie_id}")
def get_movie_name_by_id(movie_id: int):
    query = text("SELECT title FROM movies WHERE movieid = :movie_id")
    try:
        with engine.connect() as connection:
            result = connection.execute(query, {"movie_id": movie_id}).fetchone()
            if result:
                return {"title": result[0]}
            else:
                raise HTTPException(status_code=404, detail="Movie not found")
    except SQLAlchemyError as e:
        logger.error(f"Error querying database: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    
@app.post("/predict")
def predict(item: Item):
    try:
        logger.info(f"Received ratings: {item.movie_ratings}")
        preds = _get_top_n_recommendations(item.movie_ratings, model, n=5)
        movie_ids = [x[0] for x in preds]
        movie_names = []
        query = text("SELECT title FROM movies WHERE movieid = :movie_id")
        with engine.connect() as connection:
            for movie_id in movie_ids:
                result = connection.execute(query, {"movie_id": movie_id}).fetchone()
                if result:
                    movie_names.append(result[0])
                else:
                    logger.warning(f"No movie name found for movie ID: {movie_id}")
        return {"recommended_movies": movie_names}
    except SQLAlchemyError as e:
        logger.error(f"Error querying database: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
