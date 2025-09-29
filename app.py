import streamlit as st
import pickle
import pandas as pd
from recommender import recommend
import os
import gdown

# Google Drive file ID
file_id = "17wpMUAyhyGOACQEbS9C16zuZBQlJtFnV"
output_path = "artifacts/similarity.pkl"

# Create artifacts folder if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# Download the file if it doesn't exist
if not os.path.exists(output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Load movie dictionary
movies_dict = pickle.load(open("artifacts/movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Header
st.markdown(
    "<h1 style='text-align: center; color: #FF6347;'>🎬 Movie Recommender Model 🍿</h1>",
    unsafe_allow_html=True
)

# Movie selection dropdown
selected_movie = st.selectbox("🔍 Select a Movie", movies['title'].values)

# Recommend button
if st.button("Recommend"):
    # Show loading spinner while recommendations are being computed
    with st.spinner('⏳ Fetching recommendations, please wait...'):
        recommendations = recommend(selected_movie)

    if recommendations:
        st.markdown("### 🎯 Top Recommendations")
        for idx, title in enumerate(recommendations, start=1):
            st.write(f"{idx}. {title}")
    else:
        st.warning("⚠️ No recommendations found!")

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>Built with ❤️ by Samradh using Streamlit</p>",
    unsafe_allow_html=True
)
