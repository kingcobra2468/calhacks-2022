import streamlit as st
from PIL import Image
import calorie_counter
import pandas as pd

uploaded_file = st.file_uploader("Choose a file")
weight = st.number_input('Weight of the meal')

if uploaded_file and weight:
    _, prediction = calorie_counter.count_calories(uploaded_file, weight)
    st.dataframe(pd.DataFrame([prediction]))
    