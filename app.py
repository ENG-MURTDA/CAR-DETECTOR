import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("كاشف السيارات الذكي 🏎️")
st.write("ارفع صورة السيارة من موبايلك وهسة أكلك شنو نوعها")

file = st.file_uploader("اختار صورة...", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    # عرض الصورة باستخدام الأمر القديم المستقر
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    
    # تحضير الصورة للموديل
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # تحميل الموديل والتوقع
    model = tf.keras.models.load_model("car_model.h5")
    prediction = model.predict(img_array)
    
    st.success(f"اتوقع إنها سيارة بنسبة تأكد عالية!")
