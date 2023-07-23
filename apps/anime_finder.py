import streamlit as st
import pandas as pd
from PIL import Image
from joblib import load
import numpy as np
# Load from model

def app():

    st.markdown("""
    This app **Findes Animes** based on anime description.
    """)

    # loading files 

    def load_Data():
        anime = pd.read_csv('data/anime_clean.csv')
        knn_synopsis =load('data/knn_synopsis.joblib')
        tfidf_vector =load('data/user_desc_finder.joblib')
        return anime,knn_synopsis,tfidf_vector

    anime,knn_synopsis,tfidf_vector = load_Data()

    def user_desc_finder(desc):
        distances , anime_index = knn_synopsis.kneighbors(tfidf_vector.transform([desc]) ,n_neighbors=20)   
        return anime[['title','score','img_url'	,'link']].iloc[anime_index[0]]

    user_input = st.text_area('Enter your description', height=16, max_chars=None, key=None)
    find_button = st.button('Find')
    if user_input or (find_button and user_input):
        out = user_desc_finder(user_input)
        my_images = ''
        for  index, row in out.iterrows():
            try:
                my_images += """[![this is an image link](""" +row['img_url']+""")]("""+row['link']+""") """
            except:
                my_images += """[! anime list link]("""+row['link']+""") """


        st.markdown(my_images)
