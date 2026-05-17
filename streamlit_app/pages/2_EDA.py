# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="🔍", layout="wide")

# ─────────────────────── Custom CSS ───────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Exploratory Data Analysis")
st.caption("Analisis pola data sebelum training model ContextFlow AI")

st.markdown("---")

# ─── Dataset Summary ───
st.subheader("📊 Dataset Summary")
summary_cols = st.columns(4)
dataset_info = [
    ("Dolly 15K", "~15,000", "🟦"),
    ("Alpaca", "~52,000", "🟧"),
    ("OpenAssistant", "~84,000", "🟩"),
    ("CoQA", "~108,000", "🟪"),
]
for (name, count, icon), col in zip(dataset_info, summary_cols):
    with col:
        st.metric(label=f"{icon} {name}", value=count)

st.markdown("---")

# ─── Simulasi data EDA ───
np.random.seed(42)
instruction_len = np.random.normal(60, 20, 1000).astype(int).clip(10, 200)
response_len = np.random.normal(120, 40, 1000).astype(int).clip(10, 500)
sources = np.random.choice(
    ["dolly", "alpaca", "openassistant", "coqa"],
    1000,
    p=[0.3, 0.3, 0.2, 0.2]
)

df = pd.DataFrame({
    "instruction_length": instruction_len,
    "response_length": response_len,
    "source": sources,
})

# ─── Chart styling helper ───
def style_ax(fig, ax):
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#21262d")

# ─── Distribution plots ───
col1, col2 = st.columns(2)

with col1:
    st.subheader("📏 Distribusi Panjang Instruksi")
    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(fig, ax)
    ax.hist(df["instruction_length"], bins=40, color="#58a6ff", edgecolor="#0e1117", alpha=0.85)
    ax.set_xlabel("Jumlah Karakter", color="#8b949e")
    ax.set_ylabel("Frekuensi", color="#8b949e")
    ax.grid(True, alpha=0.1, color="#8b949e")
    st.pyplot(fig)

with col2:
    st.subheader("📏 Distribusi Panjang Response")
    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(fig, ax)
    ax.hist(df["response_length"], bins=40, color="#f78166", edgecolor="#0e1117", alpha=0.85)
    ax.set_xlabel("Jumlah Karakter", color="#8b949e")
    ax.set_ylabel("Frekuensi", color="#8b949e")
    ax.grid(True, alpha=0.1, color="#8b949e")
    st.pyplot(fig)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🥧 Distribusi Sumber Dataset")
    counts = df["source"].value_counts()
    colors_pie = ["#58a6ff", "#f78166", "#3fb950", "#bc8cff"]
    fig, ax = plt.subplots(figsize=(6, 4))
    style_ax(fig, ax)
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        startangle=90, colors=colors_pie,
        textprops={'color': '#c9d1d9', 'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_color('#c9d1d9')
    st.pyplot(fig)

with col4:
    st.subheader("🔵 Scatter: Instruction vs Response")
    fig, ax = plt.subplots(figsize=(6, 4))
    style_ax(fig, ax)
    color_map = {
        "dolly": "#58a6ff",
        "alpaca": "#f78166",
        "openassistant": "#3fb950",
        "coqa": "#bc8cff"
    }
    for src, grp in df.groupby("source"):
        ax.scatter(
            grp["instruction_length"], grp["response_length"],
            alpha=0.4, s=12, label=src, color=color_map[src]
        )
    ax.set_xlabel("Instruction Length", color="#8b949e")
    ax.set_ylabel("Response Length", color="#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9", fontsize=8)
    ax.grid(True, alpha=0.1, color="#8b949e")
    st.pyplot(fig)

st.markdown("---")

# ─── Boxplot ───
st.subheader("📦 Boxplot: Panjang Response per Sumber")
fig, ax = plt.subplots(figsize=(10, 4))
style_ax(fig, ax)
bp = ax.boxplot(
    [df[df["source"] == s]["response_length"] for s in color_map.keys()],
    labels=color_map.keys(),
    patch_artist=True,
)
for patch, color in zip(bp['boxes'], colors_pie):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for element in ['whiskers', 'caps', 'medians']:
    for line in bp[element]:
        line.set_color('#8b949e')
ax.tick_params(colors="#8b949e")
ax.set_ylabel("Response Length", color="#8b949e")
ax.grid(True, alpha=0.1, color="#8b949e")
st.pyplot(fig)

st.markdown("---")

# ─── Descriptive Stats ───
st.subheader("📋 Statistik Deskriptif")
st.dataframe(df.describe().round(2), use_container_width=True)