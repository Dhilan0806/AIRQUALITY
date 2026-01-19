import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
import os
warnings.filterwarnings("ignore")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Dashboard ISPU DKI Jakarta",
    page_icon="🌿",
    layout="wide"
)

# ======================================================
# CSS GREEN - BLUE THEME
# ======================================================
st.markdown("""
<style>

/* Background utama */
.stApp {
    background: linear-gradient(180deg, #ecfeff 0%, #f0fdf4 100%);
}

/* Container */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f766e, #0284c7);
    color: white;
    padding-top: 1rem;
}

/* Sidebar text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h2 {
    color: white !important;
}

/* Widget sidebar */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stDateInput {
    background-color: white;
    border-radius: 8px;
    padding: 4px;
    margin-bottom: 0.8rem;
}

/* Metric card */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #22c55e, #0ea5e9);
    border-radius: 14px;
    padding: 16px;
    color: white;
    box-shadow: 0 6px 15px rgba(0,0,0,0.12);
}

/* Metric label & value */
div[data-testid="metric-container"] label {
    color: #ecfeff !important;
}

div[data-testid="metric-container"] div {
    color: white !important;
}

/* Section title */
h1, h2, h3 {
    color: #065f46;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #22c55e, #0ea5e9);
    margin: 1rem 0;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "ispu_dki_all1.csv")

    df = pd.read_csv(file_path, sep=";")
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    if "categori" in df.columns:
        df = df.rename(columns={"categori": "kategori"})

    for col in ["pm25", "pm10", "so2", "co", "o3", "no2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["tanggal"])

df = load_data()
if df.empty:
    st.error("❌ Data tidak ditemukan")
    st.stop()

# ======================================================
# HEADER
# ======================================================
st.title("🌿 Dashboard ISPU DKI Jakarta")
st.caption("Prediksi & Analisis Kualitas Udara Berbasis Machine Learning")
st.divider()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.subheader("⚙️ Pengaturan Dashboard")

    stasiun = st.selectbox(
        "🏢 Stasiun Pemantauan",
        sorted(df["stasiun"].dropna().unique())
    )

    polutan_map = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "so2": "SO₂",
        "co": "CO",
        "o3": "O₃",
        "no2": "NO₂"
    }

    polutan = st.selectbox(
        "🧪 Parameter Polutan",
        list(polutan_map.keys()),
        format_func=lambda x: polutan_map[x]
    )

    min_d = df["tanggal"].min().date()
    max_d = df["tanggal"].max().date()

    tanggal_awal = st.date_input("📅 Tanggal Awal", min_d, min_value=min_d, max_value=max_d)
    tanggal_akhir = st.date_input("📅 Tanggal Akhir", max_d, min_value=min_d, max_value=max_d)

# ======================================================
# VALIDASI
# ======================================================
if tanggal_awal > tanggal_akhir:
    st.warning("⚠️ Tanggal awal tidak boleh melebihi tanggal akhir")
    st.stop()

# ======================================================
# FILTER DATA
# ======================================================
filtered_df = df[
    (df["stasiun"] == stasiun) &
    (df["tanggal"] >= pd.to_datetime(tanggal_awal)) &
    (df["tanggal"] <= pd.to_datetime(tanggal_akhir))
].dropna(subset=[polutan])

if filtered_df.empty:
    st.warning("⚠️ Tidak ada data sesuai filter")
    st.stop()

# ======================================================
# METRIC
# ======================================================
st.subheader("📊 Ringkasan Statistik")

c1, c2, c3, c4 = st.columns(4)

avg_val = filtered_df[polutan].mean()
max_val = filtered_df[polutan].max()
min_val = filtered_df[polutan].min()
kategori = filtered_df["kategori"].mode()[0] if "kategori" in filtered_df.columns else "-"

c1.metric("Rata-rata", f"{avg_val:.2f}")
c2.metric("Maksimum", f"{max_val:.2f}")
c3.metric("Minimum", f"{min_val:.2f}")
c4.metric("Kategori Dominan", kategori)

st.divider()

# ======================================================
# VISUALISASI
# ======================================================
st.subheader("📈 Visualisasi Data")

v1, v2 = st.columns(2)

with v1:
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(filtered_df["tanggal"], filtered_df[polutan], linewidth=2)
    ax1.axhline(avg_val, linestyle="--")
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel("ISPU")
    ax1.grid(alpha=0.3)
    fig1.autofmt_xdate()
    st.pyplot(fig1, use_container_width=True)

with v2:
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.hist(filtered_df[polutan], bins=15, edgecolor="black")
    ax2.axvline(avg_val, linestyle="--")
    ax2.set_xlabel("ISPU")
    ax2.set_ylabel("Frekuensi")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2, use_container_width=True)

st.divider()

# ======================================================
# MACHINE LEARNING
# ======================================================
st.subheader("🔮 Prediksi ISPU")

if len(filtered_df) < 10:
    st.warning("⚠️ Data kurang untuk prediksi")
else:
    X = np.arange(len(filtered_df)).reshape(-1, 1)
    y = filtered_df[polutan].values

    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(Xtr, ytr)

    mae = mean_absolute_error(yts, model.predict(Xts))
    st.write(f"📌 MAE Model: **{mae:.2f}**")

    days = st.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

    future_X = np.arange(len(filtered_df), len(filtered_df) + days).reshape(-1, 1)
    future_y = np.maximum(model.predict(future_X), 0)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(filtered_df["tanggal"], y, label="Aktual")
    ax3.plot(
        pd.date_range(filtered_df["tanggal"].iloc[-1], periods=days + 1, freq="D")[1:],
        future_y,
        linestyle="--",
        label="Prediksi"
    )
    ax3.set_xlabel("Tanggal")
    ax3.set_ylabel("ISPU")
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig3, use_container_width=True)

