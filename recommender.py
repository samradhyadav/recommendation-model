import pickle
import pandas as pd
import os
import gdown

# Ensure artifacts folder exists
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Download movie_dict.pkl if not present
movie_dict_path = os.path.join(artifacts_dir, "movie_dict.pkl")
if not os.path.exists(movie_dict_path):
    url_movie_dict = "https://drive.google.com/uc?id=1s1oJmMvmxKLnDsHvqJvwc10662fK7RJF"
    gdown.download(url_movie_dict, movie_dict_path, quiet=False)

# Download similarity.pkl if not present
similarity_path = os.path.join(artifacts_dir, "similarity.pkl")
if not os.path.exists(similarity_path):
    url_similarity = "https://drive.google.com/uc?id=17wpMUAyhyGOACQEbS9C16zuZBQlJtFnV"
    gdown.download(url_similarity, similarity_path, quiet=False)

# Load preprocessed data
movies_dict = pickle.load(open(movie_dict_path, "rb"))
similarity = pickle.load(open(similarity_path, "rb"))
movies = pd.DataFrame(movies_dict)


def recommend(movie):
    """
    Recommend top 5 movies similar to the given movie
    Returns: list of titles only
    """
    if movie not in movies['title'].values:
        return []
    
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)), key=lambda x: x[1], reverse=True
    )[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies
