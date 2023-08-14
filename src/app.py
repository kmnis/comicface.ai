import streamlit as st

import os

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from .data_loader import preprocess_test_image

import warnings

warnings.filterwarnings("ignore")

with st.spinner('Loading the models...'):
    pix2pix = load_model("../saved_models/pix2pix/pix2pix.keras")
    vae = load_model("../saved_models/vae/vae.keras")

st.markdown("<center><h1>ComicBooks.AI</h1></center>", unsafe_allow_html=True)
st.caption("<center>Upload your photo to see how a comic book version of yourself would look!</center>", unsafe_allow_html=True)

model_option = st.selectbox(
    'Select a model',
    ('Pix2Pix', 'CVAE')
)
uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    st.image(uploaded_file)
    
    img = preprocess_test_image(uploaded_file)
    
    if model_option == "Pix2Pix":
        model = pix2pix
    else:
        model = vae
    
    pred = model.predict(img)[0] * 0.5 + 0.5
    st.image(pred)
