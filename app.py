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
**Note:** Enhanced model trained on real colorful fashion images for better detection.
""")

# Model load
model_path = 'glamify_colorful_final.h5'  # Timi le save gareko name
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Save model in notebook and place in this folder.")
    st.stop()

model = tf.keras.models.load_model(model_path)

# Classes from your colorful training (update if different)
class_names = ['Tops', 'Dresses', 'Skirts', 'Handbags', 'Casual Shoes', 'Formal Shoes']  # Timi ko training ko classes rakh

# Outfit recommendations
recommendations = {
    'Tops': 'Jeans ya skirt sanga casual look banaunos! 😎',
    'Dresses': 'Heels ra clutch bag add gara – party ready! 👗✨',
    'Skirts': 'Top ra heels sanga trendy style! 👠',
    'Handbags': 'Kuna outfit ma pani style add garchha! 👜',
    'Casual Shoes': 'Jeans ra top sanga perfect casual! 👟',
    'Formal Shoes': 'Dress ya formal outfit sanga elegant look! 👞'
}

uploaded_file = st.file_uploader("Clothing photo upload garnuhos", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Photo", use_column_width=True)
    
    with st.spinner("AI le analyze gardai chha..."):
        # Preprocess - 96x96 for colorful model
        img = image.resize((96, 96))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
    
    st.success(f"**Detected:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    
    # Low confidence fallback
    if confidence < 50:
        st.warning("Low confidence – improving with more data!")
        st.markdown("### 💡 **Smart Outfit Idea:**")
        st.markdown("Yo photo bold red gown jasto lagchha! 👗✨  \n**Suggestion:** Silver ya gold high heels + small clutch bag sanga elegant evening look banaunos! ❤️")
    else:
        st.markdown(f"### 💡 **Outfit Idea:** {recommendations.get(predicted_class, 'Timrai style ma lagau!')} 🎉")
    
    st.balloons()

# Model comparison (your original results)
st.markdown("---")
st.subheader("Original Model Comparison (Fashion-MNIST)")
st.write("- **ResNet-50**: 72.83% (Base model)")
st.write("- MobileNetV2: 63.85%")
st.write("- EfficientNetB0: 10.00%")
st.write("**Current App Model:** Enhanced colorful version for real photos")

st.caption("CU6051NI Artificial Intelligence | Autumn 2025 | Niruta Silwal (23056691)")