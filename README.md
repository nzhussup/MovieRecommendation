# What-to-Watch: Movie Recommendation System

Welcome to **What-to-Watch**, a movie recommendation system that helps you find movies you'll love based on your preferences! This application allows you to pick a movie you have watched, rate it, and then receive top 5 movie recommendations tailored to your taste.


![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-9B59B6?style=for-the-badge&logo=python&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Dataset](#dataset)
- [Model](#model)
- [License](#license)

## Overview

The What-to-Watch application is available at [what-to-watch.streamlit.app](https://what-to-watch.streamlit.app). It is deployed using Streamlit Community Cloud.

## Features

- **Movie Search:** Easily search for movies you have watched.
- **Rating System:** Rate the movies you have watched.
- **Personalized Recommendations:** Get top 5 movie recommendations based on your ratings.
- **API Access:** Ready-to-use but not yet deployed REST API for retrieving data and making predictions.

## Local Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/nzhussup/MovieRecommendation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd what-to-watch
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the Streamlit application locally:
```bash
streamlit run app.py
```

Visit the link provided in your terminal.

## API

Under the `API` folder, you will find the ready-to-use but not yet deployed REST API for retrieving data from a PostgreSQL database and making predictions. However, this API is not meant to use locally.

## Dataset

The dataset used for this project is the MovieLens dataset, which includes a wide range of recent movies. You can find more information about the dataset [here](https://grouplens.org/datasets/movielens/).

## Model

The recommendation model is built using item-item collaborative filtering with cosine similarity. This approach helps in finding similar items (movies) based on user ratings and provides personalized recommendations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.