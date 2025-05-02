# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv(r"C:\Users\rishi\Downloads\archive\tmdb_5000_movies.csv")
    credits = pd.read_csv(r"C:\Users\rishi\Downloads\archive\tmdb_5000_credits.csv")
    
    # Merge both datasets on title
    df = movies.merge(credits, left_on='title', right_on='title')

    # Use genres, keywords, cast, director, and overview for better recommendations
    df['cast'] = df['cast'].fillna('')
    df['crew'] = df['crew'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')

    # Combine features into a single string
    df['combined_features'] = (
        df['genres'] + ' ' +
        df['keywords'] + ' ' +
        df['overview'] + ' ' +
        df['tagline'] + ' ' +
        df['cast'] + ' ' +
        df['crew']
    )
    return df

@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def recommend(title, df, similarity_matrix, num=10):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit App UI
st.title("🎬 Movie Recommender")
st.markdown("Content-based system using TMDB 5000 Movies + Credits")

df = load_data()
similarity_matrix = compute_similarity(df)

# Dropdown for movie selection
movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Pick a movie:", sorted(movie_list))

if st.button("Recommend"):
    recommendations = recommend(selected_movie, df, similarity_matrix)
    if recommendations:
        st.subheader("Top Recommendations:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("No recommendations found.")
