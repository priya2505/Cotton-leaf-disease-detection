import streamlit as st

# EDA Pkgs

import numpy as np
import cv2


from PIL import Image, ImageOps

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'C:/Users/Priya/Downloads/model.h5'
model = load_model(MODEL_PATH)


def model_predict(img, model):
    # Create the array of the right shape to feed into the keras model
    """" data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.float32)

      # turn the image into a numpy array
      image_array = np.asarray(img)
      # Normalize the image
      normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

      # Load the image into the array
      data[0] = normalized_image_array

      # run the inference
      prediction = model.predict(data)
      print(np.argmax(prediction, axis=1))
      return np.argmax(prediction, axis=1)"""

    # def model_predict(img, model):
    loaded_image_in_array = img
    loaded_image_in_array = (loaded_image_in_array.astype(np.float32) / 127.0) - 1

    # add additional dim such as to match input dim of the model architecture
    x = np.expand_dims(loaded_image_in_array, axis=0)

    # prediction
    prediction = model.predict(x)

    results = np.argmax(prediction, axis=1)

    return results

    # return position of the highest probability


st.title("Cotton Leaf Disease Prediction")
st.header("Transfer Learning Using RESNET51V2")
st.text("Upload a Cotton Leaf Disease or Non-Diseased Image")

uploaded_file = st.file_uploader("Choose a Cotton Leaf Image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cotton Leaf Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    open_cv_image = np.array(image)
    image = open_cv_image[:, :, ::-1].copy()
    imageCpy = cv2.resize(image, (100, 100))

    label = model_predict(image, model)
    if label == 0:
        st.write("The leaf is a diseased cotton leaf.")
    elif label == 1:
        st.write("The leaf is a diseased cotton plant.")
    elif label == 2:
        st.write("The leaf is a fresh cotton leaf.")
    else:
        st.write("The leaf is a fresh cotton plant.")



