import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.title("🧠 Handwritten Digit Recognizer")

@st.cache_resource
def load_model():
    return ort.InferenceSession("hcr_model.onnx")

session = load_model()

file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img_array})[0]
    predicted = np.argmax(result)

    st.image(img, caption="Input Image")
    st.write(f"### Predicted Digit: {predicted}")
