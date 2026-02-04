import pandas as pd
import joblib
import streamlit as st

st.sidebar.title("Machine Learning")
st.sidebar.success("Dibuat Oleh Natasya Destiana Lestari")

st.title("Regresi Penjualan Tomat")
st.markdown("Aplikasi Machine Learning regression untuk menghitung total penjualan tomat berdasarkan fitur `Harga, hari, Cuaca, dan Promo`")

model_random = joblib.load("model_random.joblib")

Harga = st.slider("Harga", 6000, 13000, 9000)
Hari = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
Cuaca = st.selectbox("Cuaca", ["Cerah", "Berawan", "Mendung", "Hujan"])
Promo = st.pills("Promo", ["Ya", "Tidak"], default="Ya")

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[Harga, Hari, Cuaca, Promo]],
                         columns=["Harga", "Hari", "Cuaca", "Promo"])
	prediksi = model_random.predict(data_baru)[0]
	st.success(f"Model memprediksi total penjualan {prediksi:.0f}")

	st.balloons()
