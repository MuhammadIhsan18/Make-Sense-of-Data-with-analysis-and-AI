import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="PT Maju Bersama - Analytics & AI", layout="wide")

st.title("📊 PT Maju Bersama: Dasbor Prediksi & Manajemen Piutang")
st.caption("Studi Kasus: Make Sense of Data with Analysis and AI (Andria Arisal - BRIN)")

# Load Model dan Data
@st.cache_resource
def load_model():
    return joblib.load('model_late_payment.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('invoices_pt_maju_bersama.csv')

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error("Model atau data belum ditemukan.")
    st.stop()

# 1. Analisis Deskriptif
st.subheader("1. Ringkasan Deskriptif Piutang")
col1, col2, col3 = st.columns(3)
col1.metric("Total Invoice", len(df))
col2.metric("Total Piutang (Juta Rp)", f"Rp {df['invoice_amount_mil'].sum():,.2f}")
col3.metric("Rasio Terlambat (>30 Hari)", f"{(df['is_late'].mean()*100):.1f}%")

fig_segment = px.histogram(
    df, x="customer_segment", color="is_late",
    barmode="group", title="Distribusi Keterlambatan per Segmen Pelanggan",
    labels={'customer_segment': 'Segmen Pelanggan', 'is_late': 'Terlambat (>30 Hari)'}
)
st.plotly_chart(fig_segment, use_container_width=True)

# 2. Simulator Prediksi AI
st.markdown("---")
st.subheader("2. Simulator Prediksi Keterlambatan Tagihan Baru")

with st.form("prediction_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        segment = st.selectbox("Segmen Pelanggan", ['Retail', 'Wholesale', 'Corporate'])
        credit_limit = st.selectbox("Batas Kredit (Juta Rp)", [50, 100, 250, 500])
        payment_terms = st.selectbox("Jangka Waktu Pembayaran (Hari)", [14, 30, 60])
    with col_b:
        amount = st.number_input("Nilai Tagihan (Juta Rp)", min_value=1.0, max_value=500.0, value=25.0)
        past_late = st.number_input("Riwayat Terlambat Sebelumnya (Kali)", min_value=0, max_value=20, value=1)
    
    submit_button = st.form_submit_button("Jalankan Prediksi AI")

if submit_button:
    input_data = pd.DataFrame({
        'customer_segment': [segment],
        'credit_limit_mil': [credit_limit],
        'payment_terms_days': [payment_terms],
        'invoice_amount_mil': [amount],
        'past_late_count': [past_late]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("### Hasil Analisis AI")
    if prediction == 1:
        st.error(f"⚠️ **Tinggi Risiko Terlambat!** Probabilitas: **{probability*100:.1f}%**")
        st.warning("Tindakan Preskriptif: Batasi pembelian tambahan, kirim pengingat rutin, atau minta DP.")
    else:
        st.success(f"✅ **Risiko Rendah / Tepat Waktu.** Probabilitas: **{probability*100:.1f}%**")
        st.info("Tindakan Preskriptif: Tagihan dapat diproses sesuai prosedur standar.")
