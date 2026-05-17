# pyrefly: ignore [missing-import]
import streamlit as st
import time
import uuid
import sys
import os

# Tambahkan root project ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference.predictor import ContextFlowPredictor
from app.database.repository import save_prediction, save_feedback, get_conversation_history
from app.utils.logger import logger

st.set_page_config(
    page_title="ContextFlow AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── Custom CSS ───────────────────────
st.markdown("""
<style>
    /* Dark theme override */
    .stApp {
        background-color: #0e1117;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .sub-header {
        color: #8b949e;
        font-size: 1rem;
        margin-top: -10px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    .status-online {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }

    .status-base {
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.3);
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58a6ff;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Sidebar ───────────────────────
with st.sidebar:
    st.markdown("### 🧠 ContextFlow AI")
    st.markdown("---")

    st.markdown("#### ⚙️ Generation Settings")
    max_tokens = st.slider("Max Response Tokens", 64, 512, 256, key="max_tokens")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05, key="temperature")
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05, key="top_p")

    st.markdown("---")
    st.markdown("#### 📄 Document Context")
    context_input = st.text_area(
        "Paste konteks dokumen (opsional):",
        height=120,
        placeholder="Contoh: SOP perusahaan, kebijakan HR, panduan teknis...",
        key="context_input",
    )

    st.markdown("---")
    col_clear, col_history = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_history:
        if st.button("📜 History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get("show_history", False)

    # Model status
    st.markdown("---")
    if "predictor" in st.session_state:
        status = st.session_state.predictor.get_status()
        if "Fine-tuned" in status:
            st.markdown(f'<div class="status-badge status-online">{status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-badge status-base">{status}</div>', unsafe_allow_html=True)

# ─────────────────────── Session State ───────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Load model
if "predictor" not in st.session_state:
    with st.spinner("🔄 Loading AI model... (pertama kali mungkin perlu download)"):
        try:
            st.session_state.predictor = ContextFlowPredictor()
        except Exception as e:
            st.error(f"❌ Gagal load model: {e}")
            st.stop()

# ─────────────────────── Header ───────────────────────
st.markdown('<p class="main-header">🧠 ContextFlow AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise Knowledge Intelligence · Fine-Tuned LLM Assistant</p>', unsafe_allow_html=True)

# Model info
predictor = st.session_state.predictor
if predictor.is_finetuned:
    st.success("✅ Model fine-tuned aktif")
else:
    st.info("ℹ️ Menggunakan base model. Jalankan `python train.py` untuk fine-tuning.")

st.markdown("---")

# ─────────────────────── History Panel ───────────────────────
if st.session_state.show_history:
    with st.expander("📜 Conversation History (dari database)", expanded=True):
        history = get_conversation_history(st.session_state.session_id)
        if history:
            for h in history:
                st.markdown(f"**Q:** {h.get('instruction', '')}")
                st.markdown(f"**A:** {h.get('response', '')}")
                st.markdown("---")
        else:
            st.caption("Belum ada riwayat percakapan.")

# ─────────────────────── Chat History ───────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────── Chat Input ───────────────────────
if prompt := st.chat_input("Tanyakan seputar knowledge perusahaan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start = time.time()
            try:
                response = st.session_state.predictor.predict(
                    instruction=prompt,
                    context=context_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as e:
                response = f"⚠️ Error generating response: {e}"
                logger.error(f"Prediction error: {e}")
            latency = (time.time() - start) * 1000

        st.markdown(response)
        st.caption(f"⏱️ {latency:.0f}ms")

        # Simpan ke Supabase
        pred_id = save_prediction(
            session_id=st.session_state.session_id,
            instruction=prompt,
            response=response,
            context=context_input,
            latency_ms=latency,
        )

        # Feedback (compatible with Streamlit < 1.37)
        st.markdown("**Rate this response:**")
        feedback_cols = st.columns(5)
        for i, col in enumerate(feedback_cols):
            with col:
                if st.button(f"{'⭐' * (i + 1)}", key=f"fb_{pred_id}_{i+1}"):
                    save_feedback(pred_id, i + 1)
                    st.toast(f"Thanks! Rated {i + 1}/5 ⭐")

    st.session_state.messages.append({"role": "assistant", "content": response})