import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ================================
# 🔹 Load and Prepare Data
# ================================
DATA_PATH = 'tmdb_data/cleaned_movies.csv'
movies_df = pd.read_csv(DATA_PATH)

# Fill missing values
movies_df['genre_names'] = movies_df['genre_names'].fillna('')
movies_df['vote_average'] = movies_df['vote_average'].fillna(movies_df['vote_average'].mean())
movies_df['popularity'] = movies_df['popularity'].fillna(movies_df['popularity'].mean())
movies_df['release_year'] = movies_df['release_year'].fillna(movies_df['release_year'].median())

# Vectorize genres
tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies_df['genre_names'])

# Normalize numeric features
scaler = MinMaxScaler()
numeric_features = scaler.fit_transform(movies_df[['vote_average', 'popularity', 'release_year']])

# Combine all features
feature_matrix = hstack([genre_matrix, numeric_features])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# ================================
# 🔹 Recommendation Logic
# ================================
def recommend_movies(title, cosine_sim, df, top_n=10):
    idx = df.index[df['title'].str.lower() == title.lower()]
    if len(idx) == 0:
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genre_names', 'vote_average', 'popularity', 'release_year']]

# ================================
# 🔹 Streamlit UI
# ================================
st.title("🎥 CineMatik")

# Filters
st.sidebar.header("🔍 Filter Preferences")

selected_genre = st.sidebar.selectbox("Select Genre", sorted(movies_df['genre_names'].unique()))
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 5.0, 0.1)
year_range = st.sidebar.slider("Release Year Range", int(movies_df['release_year'].min()), int(movies_df['release_year'].max()), (2000, 2025))

# Filter dataset
filtered_df = movies_df[
    (movies_df['genre_names'].str.contains(selected_genre)) &
    (movies_df['vote_average'] >= min_rating) &
    (movies_df['release_year'].between(year_range[0], year_range[1]))
]

# Movie selection
st.subheader("🎬 Choose a Movie You Like")
movie_list = filtered_df['title'].tolist()

if not movie_list:
    st.warning("No movies match your filter settings.")
else:
    selected_movie = st.selectbox("Pick a movie", movie_list)

    if st.button("🎯 Recommend Similar Movies"):
        recommendations = recommend_movies(selected_movie, cosine_sim, movies_df)
        if recommendations is not None:
            st.success(f"Recommendations similar to **{selected_movie}**:")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.error("Movie not found in dataset.")
