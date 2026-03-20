import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🏎️ كاشف السيارات الذكي")
st.write("ارفع صورة السيارة وهسة أكلك شنو نوعها")

file = st.file_uploader("اختار صورة...", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    
    # تحضير الصورة
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    # تصحيح سطر 18 (الاسم الصحيح للدالة)
    img_array = np.expand_dims(img_array, axis=0)
    
    try:
        model = tf.keras.models.load_model("car_model.h5", compile=False)
        prediction = model.predict(img_array)
        st.success("🎉 مبروك يا وحش! اشتغل الموديل.")
    except Exception as e:
        st.error(f"خطأ تقني: {e}")
