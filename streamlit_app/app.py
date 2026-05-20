# pyrefly: ignore [missing-import]
import streamlit as st
import time
import uuid
import sys
import os

# Tambahkan root project ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference.predictor import ContextFlowPredictor
from app.database.repository import save_prediction, save_feedback, save_message, get_conversation_history
from app.utils.logger import logger

st.set_page_config(
    page_title="ContextFlow AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── HelloMonday-Inspired CSS ───────────────────────
st.markdown("""
<style>
    /* ═══ Import premium font ═══ */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* ═══ Global Theme — Deep black, editorial feel ═══ */
    .stApp {
        background-color: #050505;
        font-family: 'Inter', sans-serif;
    }

    /* ═══ Remove all default Streamlit chrome ═══ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ═══ Hero Header ═══ */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.05;
        margin-bottom: 0;
        padding-top: 20px;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 300;
        color: #666666;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 8px;
        margin-bottom: 32px;
    }

    .hero-accent {
        background: linear-gradient(135deg, #ff6b35, #f7c948);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ═══ Status Pill ═══ */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 100px;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 24px;
    }

    .status-active {
        background: rgba(255, 107, 53, 0.1);
        color: #ff6b35;
        border: 1px solid rgba(255, 107, 53, 0.2);
    }

    .status-base {
        background: rgba(255, 255, 255, 0.05);
        color: #888888;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ═══ Divider ═══ */
    .hm-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
        margin: 32px 0;
        border: none;
    }

    /* ═══ Sidebar — Glass dark ═══ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #080808 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] .stMarkdown h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #666666;
    }

    section[data-testid="stSidebar"] label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #888888 !important;
        letter-spacing: 0.02em;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.04);
    }

    /* ═══ Slider customization ═══ */
    .stSlider > div > div > div > div {
        background: #ff6b35 !important;
    }

    /* ═══ Chat Messages — Minimal & clean ═══ */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 16px 0 !important;
        margin-bottom: 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    /* Chat input */
    .stChatInput > div {
        border-color: rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
    }

    .stChatInput > div:focus-within {
        border-color: #ff6b35 !important;
        box-shadow: 0 0 0 1px rgba(255, 107, 53, 0.2) !important;
    }

    .stChatInput textarea {
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }

    /* ═══ Buttons ═══ */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        color: #cccccc;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        border-color: #ff6b35;
        color: #ff6b35;
        background: rgba(255, 107, 53, 0.05);
        transform: translateY(-1px);
    }

    /* ═══ Text area ═══ */
    .stTextArea textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
        color: #cccccc !important;
        border-radius: 8px !important;
    }

    .stTextArea textarea:focus {
        border-color: #ff6b35 !important;
        box-shadow: 0 0 0 1px rgba(255, 107, 53, 0.15) !important;
    }

    /* ═══ Alert boxes ═══ */
    .stAlert {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }

    /* ═══ Expander ═══ */
    .streamlit-expanderHeader {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        color: #cccccc;
    }

    /* ═══ Toast ═══ */
    .stToast {
        font-family: 'Inter', sans-serif;
    }

    /* ═══ Feedback stars row ═══ */
    .feedback-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #555555;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* ═══ Model info tag ═══ */
    .model-info {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #555555;
        padding: 10px 16px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        margin-bottom: 24px;
    }

    .model-info strong {
        color: #999999;
    }

    /* ═══ Latency caption ═══ */
    .stChatMessage .stCaption {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #444444;
        letter-spacing: 0.04em;
    }

    /* ═══ Scrollbar ═══ */
    ::-webkit-scrollbar {
        width: 4px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.15);
    }

    /* ═══ Spinner ═══ */
    .stSpinner > div {
        border-top-color: #ff6b35 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Sidebar ───────────────────────
with st.sidebar:
    st.markdown("### ◆ ContextFlow")
    st.markdown("---")

    st.markdown("#### Settings")
    max_tokens = st.slider("Max tokens", 64, 512, 256, key="max_tokens")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05, key="temperature")
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05, key="top_p")

    st.markdown("---")
    st.markdown("#### Context")
    context_input = st.text_area(
        "Paste document context (optional):",
        height=120,
        placeholder="SOP, kebijakan HR, panduan teknis...",
        key="context_input",
    )

    st.markdown("---")
    col_clear, col_history = st.columns(2)
    with col_clear:
        if st.button("Clear ×", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_feedback = {}
            st.rerun()
    with col_history:
        if st.button("History ↗", use_container_width=True):
            st.session_state.show_history = not st.session_state.get("show_history", False)

    # Model status
    st.markdown("---")
    if "predictor" in st.session_state:
        status = st.session_state.predictor.get_status()
        if "Fine-tuned" in status:
            st.markdown(f'<div class="status-pill status-active">● Fine-tuned Active</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-pill status-base">○ Base Model</div>', unsafe_allow_html=True)

# ─────────────────────── Session State ───────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = {}  # {msg_index: pred_id}

# Load model
if "predictor" not in st.session_state:
    with st.spinner("Loading model..."):
        try:
            st.session_state.predictor = ContextFlowPredictor()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

# ─────────────────────── Header ───────────────────────
st.markdown('<p class="hero-title">Context<span class="hero-accent">Flow</span></p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Enterprise Knowledge Intelligence</p>', unsafe_allow_html=True)

# Model info
predictor = st.session_state.predictor
if predictor.is_finetuned:
    st.markdown('<div class="model-info">● <strong>Fine-tuned model active</strong> — Ready for enterprise queries</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="model-info">○ <strong>Base model</strong> — Run <code>python train.py</code> for fine-tuning</div>', unsafe_allow_html=True)

st.markdown('<div class="hm-divider"></div>', unsafe_allow_html=True)

# ─────────────────────── History Panel ───────────────────────
if st.session_state.show_history:
    with st.expander("Conversation History", expanded=True):
        history = get_conversation_history(st.session_state.session_id)
        if history:
            for h in history:
                st.markdown(f"**Q:** {h.get('instruction', '')}")
                st.markdown(f"**A:** {h.get('response', '')}")
                st.markdown('<div class="hm-divider"></div>', unsafe_allow_html=True)
        else:
            st.caption("No conversation history yet.")

# ─────────────────────── Chat History ───────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render feedback buttons untuk setiap assistant message
        if msg["role"] == "assistant" and idx in st.session_state.pending_feedback:
            pred_id = st.session_state.pending_feedback[idx]
            st.markdown('<div class="feedback-label">Rate this response</div>', unsafe_allow_html=True)
            feedback_cols = st.columns(5)
            for i, col in enumerate(feedback_cols):
                with col:
                    if st.button(f"{'★' * (i + 1)}", key=f"fb_{idx}_{i+1}"):
                        save_feedback(pred_id, i + 1)
                        st.toast(f"Rated {i + 1}/5 ★")
                        # Hapus dari pending setelah rated
                        del st.session_state.pending_feedback[idx]
                        st.rerun()

# ─────────────────────── Chat Input ───────────────────────
if prompt := st.chat_input("Ask about enterprise knowledge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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
                response = f"Error generating response: {e}"
                logger.error(f"Prediction error: {e}")
            latency = (time.time() - start) * 1000

        st.markdown(response)
        st.caption(f"{latency:.0f}ms")

        # Simpan prediction ke Supabase
        pred_id = save_prediction(
            session_id=st.session_state.session_id,
            instruction=prompt,
            response=response,
            context=context_input,
            latency_ms=latency,
        )

    # Simpan messages ke session_state
    assistant_idx = len(st.session_state.messages)  # index untuk assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Simpan pred_id untuk feedback buttons
    st.session_state.pending_feedback[assistant_idx] = pred_id

    # Rerun agar feedback buttons muncul dari chat history loop
    st.rerun()