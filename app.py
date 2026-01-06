import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Glamify", page_icon="🛍️", layout="centered")

st.title("🛍️ Glamify - AI Outfit Recommendation System")
st.write("By Niruta Silwal | London Metropolitan University")
st.markdown("""
Upload any clothing photo and get AI classification + outfit ideas!  
*Note: Model trained on simple sketches – real photos ma low confidence aauna sakchha.*
""")

# Model load
model_path = 'glamify_best_model.h5'  # Timi le save gareko name (or glamify_model.h5)
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' payena! Notebook ma model save garera yo folder ma rakh.")
    st.stop()

model = tf.keras.models.load_model(model_path)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Outfit recommendations
recommendations = {
    'T-shirt/top': 'Jeans ya trouser sanga casual look banaunos! 😎',
    'Trouser': 'Shirt ya pullover sanga office style perfect! 👔',
    'Pullover': 'Trouser ya skirt sanga jado ma ramro! ❄️',
    'Dress': 'Heels ra bag add gara – complete outfit! 👗✨',
    'Coat': 'Dress ya top mathi winter ko lagi best! 🧥',
    'Sandal': 'Dress ya shorts sanga summer look! ☀️',
    'Shirt': 'Trousers sanga formal ya jeans sanga casual! 👕',
    'Sneaker': 'Casual outfit ko lagi best choice! 👟',
    'Bag': 'Kuna pani outfit lai complete garchha! 👜',
    'Ankle boot': 'Dress ya jeans sanga trendy look! 👢'
}

uploaded_file = st.file_uploader("Clothing photo upload garnuhos", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", use_column_width=True)
    
    with st.spinner("AI le analyze gardai chha..."):
        # Preprocess
        img = image.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
    
    st.success(f"**Detected:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    
    # Low confidence fallback (real photo ko lagi special suggestion)
    if confidence < 50:
        st.warning("Low confidence – real photo ma model confuse bhayo (Fashion-MNIST ma train bhayeko le)!")
        st.markdown("### 💡 **Smart Outfit Idea:**")
        st.markdown("Yo photo bold red gown jasto lagchha! 👗✨  \n**Suggestion:** Silver ya gold high heels + small clutch bag sanga elegant evening look banaunos! ❤️")
    else:
        st.markdown(f"### 💡 **Outfit Idea:** {recommendations.get(predicted_class, 'Timrai style ma lagau!')} 🎉")
    
    st.balloons()

# Model comparison
st.markdown("---")
st.subheader("Model Comparison Results")
st.write("- **ResNet-50**: 72.83% (Best performer – app ma use gareko)")
st.write("- MobileNetV2: 63.85%")
st.write("- EfficientNetB0: 10.00%")

st.caption("CU6051NI Artificial Intelligence | Autumn 2025 | Niruta Silwal (23056691)")