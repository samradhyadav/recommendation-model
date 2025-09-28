import numpy as np
import pandas as pd

movies = pd.read_csv('/content/tmdb_5000_movies.csv')
credits = pd.read_csv('/content/tmdb_5000_credits.csv')

movies.head(5)

credits.head(5)

movies.shape

credits.shape

movies = movies.merge(credits,on='title')

movies.head(2)

movies.shape

# Keeping important columns for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew', 'release_date', 'vote_average']]


# This handles any missing data and creates the 'year' column
movies.dropna(inplace=True)
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year

movies.head(2)

movies.shape

movies.isnull().sum()

movies.dropna(inplace=True)

movies.duplicated().sum()

movies.iloc[0]['genres']

import ast #for converting str to list

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

movies.head()

movies.iloc[0]['keywords']

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

movies.iloc[0]['cast']

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)
movies.head()

movies.iloc[0]['crew']

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies.head()

movies.iloc[0]['overview']

movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(4)

movies.iloc[0]['overview']

# now removing space like that
'Snow White'
'SnowWhite'

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

movies.iloc[0]['tags']

new_df = movies[['movie_id', 'title', 'tags', 'year', 'vote_average']]

new_df.head()

# Converting list to str
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()

new_df.iloc[0]['tags']

# Converting to lower case
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df.head()

new_df.iloc[0]['tags']

%pip install nltk

import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stems(text):
    T = []

    for i in text.split():
        T.append(ps.stem(i))

    return " ".join(T)

new_df['tags'] = new_df['tags'].apply(stems)

new_df.iloc[0]['tags']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new_df['tags']).toarray()

vector[0]

vector.shape

len(cv.get_feature_names_out())

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

similarity.shape

new_df[new_df['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)

recommend('Spider-Man 2')

import pickle

import os

# Create the directory if it doesn't exist
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')

# Now save the files
pickle.dump(new_df.to_dict(), open('artifacts/movie_dict.pkl','wb'))
pickle.dump(similarity,open('artifacts/similarity.pkl','wb'))
