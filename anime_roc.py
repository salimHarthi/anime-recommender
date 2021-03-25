import streamlit as st
import pandas as pd
from PIL import Image
from joblib import load
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# Load from model

 
st.title('Anime Recommender')

st.markdown("""
This app **Recommends Animes** based on other Animes you liked
""")

# loading files 

#@st.cache
def load_Data():
    anime = pd.read_csv('anime_clean.csv', error_bad_lines=False)
    anime_in = pd.read_csv('anime_in.csv', error_bad_lines=False)
    tfidf_vector = TfidfVectorizer(stop_words='english')
    anime['synopsis'] = anime['synopsis'].fillna('')
    tfidf_matrix = tfidf_vector.fit_transform(anime['synopsis'].values)
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    knn =load('anime_roc.joblib')
    return anime,anime_in,sim_matrix,knn
    
anime,anime_in,sim_matrix,knn = load_Data()
indices = pd.Series(anime.index, index=anime['title']).drop_duplicates()

option = st.selectbox("Select an anime you like", np.insert(anime['title'].values,0,'None') )

def content_based_recommender(title, sim_scores=sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime[['title','score'	,'img_url'	,'link']].iloc[anime_indices]

def all_based_recommender(title):
    idx = indices[title]
    distances , anime_index = knn.kneighbors(anime_in.iloc[idx].values.reshape(1, -1) ,n_neighbors=10)
    return anime[['title','score','img_url'	,'link']].iloc[anime_index[0]]

if option!='None':
    # out = content_based_recommender(option)
    # out2 =  all_based_recommender(option)
    out = pd.concat([content_based_recommender(option),all_based_recommender(option)]).drop_duplicates()
    my_images = ''
    for  index, row in out.iterrows():
        my_images += """[![this is an image link](""" +row['img_url']+""")]("""+row['link']+""") """


    st.markdown(my_images)
