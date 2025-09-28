import pickle
import pandas as pd
import requests

# Load preprocessed data
movies_dict = pickle.load(open("artifacts/movie_dict.pkl", "rb"))
similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

# TMDB API Key
API_KEY = "c883692b8d60d6edc197faab212c7fe1"


def fetch_poster(movie_id):
    """
    Fetch poster URL for a given movie_id from TMDB
    with retry + timeout handling
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)  # ⏱ timeout to avoid hanging
        response.raise_for_status()  # raise error for 4xx/5xx
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.Timeout:
        print(f"⏳ Timeout fetching poster for movie_id {movie_id}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Request error: {e}")
    
    # Fallback poster if API fails
    return "https://via.placeholder.com/500x750.png?text=No+Image"

def recommend(movie):
    """
    Recommend top 5 movies similar to the given movie
    Returns: list of (title, poster_url)
    """
    if movie not in movies['title'].values:
        return []
    
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return list(zip(recommended_movies, recommended_posters))
