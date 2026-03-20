import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# إعداد واجهة الموقع
st.set_page_config(page_title="Car AI Detector", page_icon="🏎️")
st.title("🏎️ كاشف السيارات الذكي")
st.write("ارفع صورة السيارة من موبايلك وهسة أكولك شنو نوعها!")

# تحميل موديل الذكاء الاصطناعي
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# زر رفع الصورة (هذا اللي راح يشوفه صديقك)
uploaded_file = st.file_uploader("اختار صورة...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='الصورة المرفوعة', use_container_width=True)
    
    # معالجة الصورة ليفهمها الذكاء الاصطناعي
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # التوقع وإظهار النتيجة
    preds = model.predict(x)
    results = decode_predictions(preds, top=3)[0]

    st.success("✅ تم التحليل بنجاح!")
    for i, (id, label, prob) in enumerate(results):
        st.info(f"**{label.replace('_', ' ')}**: {prob*100:.2f}%")