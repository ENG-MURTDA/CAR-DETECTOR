import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🏎️ كاشف السيارات الذكي")
st.write("ارفع صورة السيارة من موبايلك وهسة أكلك شنو نوعها")

file = st.file_uploader("...اختار صورة", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    
    # تحضير الصورة للموديل
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_as_dims(img_array, axis=0)
    
    # تحميل الموديل والتوقع بطريقة آمنة
    try:
        model = tf.keras.models.load_model("car_model.h5", compile=False)
        prediction = model.predict(img_array)
        
        # هنا تقدر تضيف شروط بناءً على نتيجة الـ prediction مالتك
        st.success("🎉 اكتمل التحليل! الموديل تعرف على الصورة.")
    except Exception as e:
        st.error(f"حدث خطأ في قراءة الموديل: {e}")
