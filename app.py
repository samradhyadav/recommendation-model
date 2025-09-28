import streamlit as st
import pickle
import pandas as pd
from recommender import recommend

# Load movie dictionary
movies_dict = pickle.load(open("artifacts/movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Header
st.markdown(
    "<h1 style='text-align: center; color: #FF6347;'>🎬 Movie Recommender System 🍿</h1>",
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

        # Display recommendations in 5 columns
        cols = st.columns(5)
        for idx, (title, poster) in enumerate(recommendations):
            with cols[idx % 5]:
                st.image(poster, use_column_width=True)
                st.markdown(f"<p style='text-align:center'><b>{title}</b></p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No recommendations found!")

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>Built with ❤️ by Samradh using Streamlit</p>",
    unsafe_allow_html=True
)
