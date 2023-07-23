import streamlit as st
import pandas as pd
from PIL import Image
from joblib import load
import numpy as np
# Load from model

def app():
    st.markdown("""
    This app **Recommends Animes** based on other Animes you liked.
    """)
    # loading files 
    def load_Data():
        anime = pd.read_csv('data/anime_clean.csv')
        anime_in = pd.read_csv('data/anime_in.csv')
        knn_genre_score =load('data/knn_genre_score.joblib')
        knn_synopsis =load('data/knn_synopsis.joblib')
        tfidf_vector =load('data/user_desc_finder.joblib')
        anime['synopsis'] = anime['synopsis'].fillna('')
        tfidf_matrix = tfidf_vector.transform(anime['synopsis'].values)
        indices = pd.Series(anime.index, index=anime['title']).drop_duplicates()
        return anime,anime_in,knn_genre_score,knn_synopsis,indices,tfidf_matrix,tfidf_vector

    anime,anime_in,knn_genre_score,knn_synopsis,indices,tfidf_matrix,tfidf_vector = load_Data()


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
            try:
                my_images += """[![this is an image link](""" +row['img_url']+""")]("""+row['link']+""") """
            except:
                my_images += """[! anime list link]("""+row['link']+""") """


        st.markdown(my_images)
