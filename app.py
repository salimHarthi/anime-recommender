import streamlit as st
from multiapp import MultiApp
from apps import anime_roc, anime_finder # import your app modules here
import pandas as pd
from joblib import load
import numpy as np
app = MultiApp()

st.title('Anime Recommender/Finder')



# Add all your application here
app.add_app("Anime Recommender", anime_roc.app)
app.add_app("Anime Finder", anime_finder.app)
# The main app
app.run()