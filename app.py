import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("model/hcr_model.keras")

st.title("🧠 Handwritten Digit Recognizer")

file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    pred = model.predict(img_array)
    result = np.argmax(pred)
    st.image(img, caption="Input Image")
    st.write(f"### Predicted Digit: {result}")
