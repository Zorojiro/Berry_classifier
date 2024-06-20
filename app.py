import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
import tensorflow.keras as keras

# Load the model
model = keras.models.load_model('berryClf.keras')

def color_resize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    prediction = model.predict(img)
    return prediction[0][0]


st.title("Berry Classifier 🫐🍓")
image = st.file_uploader("Choose an image...", type="jpg")
if image is not None:
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    col1, col2 = st.columns(2)
    col1.image(opencv_image, channels="BGR", width=300, caption="Uploaded Image")
    prediction = color_resize(opencv_image)
    prob = float(prediction)*100
    
    col2.write("Black Currant")
    col2.write(f"Probability: {100-prob:.2f}%")
    

    col2.write("Raspberry")
    col2.write(f"Probability: {prob:.2f}")
    





    