# 🎬 Movie Recommendation System

[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://recommendation-model-jdnhm5yqzrb8bvje3eghvk.streamlit.app/)  
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)

Interactive movie recommendation web app built with **Streamlit**. Get top 5 recommended movies based on your favorite movie, complete with posters fetched from TMDB.

---

## 🚀 Live Demo

[Open Movie Recommendation System](https://recommendation-model-jdnhm5yqzrb8bvje3eghvk.streamlit.app/)

---

## ✨ Features

- Recommend top 5 similar movies.
- Fetch movie posters using TMDB API.
- Lightweight and fast.
- User-friendly web interface.

---

## 🛠 Installation (Local)

```bash
git clone https://github.com/samradhyadav/recommendation-model.git
cd recommendation-model
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run app.py
```

📂 Project Structure
recommendation-model/
│
├─ app.py                # Streamlit frontend
├─ recommender.py        # Recommendation logic
├─ artifacts/            # Pickle files (movie_dict.pkl, similarity.pkl)
├─ requirements.txt      # Python dependencies
└─ README.md

⚡ Technologies

* Python 3.x
* Streamlit
* Pandas, Requests
* TMDB API

📄 License

MIT License

