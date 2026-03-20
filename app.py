import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# خدعة برمجية لتجاوز خطأ batch_shape في النسخ الجديدة
from tensorflow.keras.layers import InputLayer
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

st.title("🏎️ كاشف السيارات الذكي")
st.write("ارفع صورة السيارة وهسة أكلك شنو نوعها")

file = st.file_uploader("اختار صورة...", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    
    # تحضير الصورة
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    try:
        # تحميل الموديل مع تعريف الطبقة المخصصة لتجنب الخطأ
        model = tf.keras.models.load_model(
            "car_model.h5", 
            custom_objects={'InputLayer': CustomInputLayer},
            compile=False
        )
        prediction = model.predict(img_array)
        st.success("🎉 مبروك! الموديل اشتغل وتجاوزنا خطأ النسخ.")
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
