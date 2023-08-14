import streamlit as st

import os

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import array_to_img

import sys
sys.path.append(".")
from data_loader import preprocess_test_image

import warnings

warnings.filterwarnings("ignore")

@st.cache_data(show_spinner="Loading the model...")
def get_model():
    model_path = "models/pix2pix.keras"
    if not os.path.exists(model_path):
        model_path = "../saved_models/pix2pix/pix2pix.keras"

    pix2pix = load_model(model_path)
    return pix2pix

st.markdown("<center><h1>ComicBooks.AI</h1></center>", unsafe_allow_html=True)
st.caption("<center>Upload your photo to see how a comic book version of yourself would look!</center>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img.save("uploaded_image.png")
    st.image(uploaded_file)
    
    img = preprocess_test_image("uploaded_image.png")
    img = tf.expand_dims(img, axis=0)
    
    pix2pix = get_model()
    with st.spinner('Processing the image...'):
        pred = array_to_img(pix2pix.predict(img)[0] * 0.5 + 0.5)
        st.image(pred)
        _ = os.system("rm uploaded_image.png")
