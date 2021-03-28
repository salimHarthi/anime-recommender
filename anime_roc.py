import streamlit as st
import pandas as pd
from PIL import Image
from joblib import load
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Load from model

 
st.title('Anime Recommender')

st.markdown("""
This app **Recommends Animes** based on other Animes you liked
""")

# loading files 

@st.cache
def load_Data():
    anime = pd.read_csv('anime_clean.csv', error_bad_lines=False)
    anime_in = pd.read_csv('anime_in.csv', error_bad_lines=False)
    knn_genre_score =load('knn_genre_score.joblib')
    knn_synopsis =load('knn_synopsis.joblib')
    tfidf_vector = TfidfVectorizer(stop_words='english')
    anime['synopsis'] = anime['synopsis'].fillna('')
    tfidf_matrix = tfidf_vector.fit_transform(anime['synopsis'].values)
    indices = pd.Series(anime.index, index=anime['title']).drop_duplicates()
    return anime,anime_in,knn_genre_score,knn_synopsis,indices,tfidf_matrix
    
anime,anime_in,knn_genre_score,knn_synopsis,indices,tfidf_matrix = load_Data()


option = st.selectbox("Select an anime you like", np.insert(anime['title'].values,0,'None') )

def synopsis_based_recommender(title):
    idx = indices[title]
    distances , anime_index = knn_synopsis.kneighbors(tfidf_matrix[idx] ,n_neighbors=25)
    return anime[['title','score','img_url'	,'link']].iloc[anime_index[0]]

def genre_score_based_recommender(title):
    idx = indices[title]
    distances , anime_index = knn_genre_score.kneighbors(anime_in.iloc[idx].values.reshape(1, -1) ,n_neighbors=25)
    return anime[['title','score','img_url'	,'link']].iloc[anime_index[0]]

if option!='None':
    # out = content_based_recommender(option)
    # out2 =  all_based_recommender(option)
    out = pd.concat([genre_score_based_recommender(option),synopsis_based_recommender(option)]).drop_duplicates()
    my_images = ''
    for  index, row in out.iterrows():
        my_images += """[![this is an image link](""" +row['img_url']+""")]("""+row['link']+""") """


    st.markdown(my_images)
