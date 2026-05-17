
"""
Reusable Streamlit chat components untuk ContextFlow AI.
"""
# pyrefly: ignore [missing-import]
import streamlit as st


def render_chat_message(role: str, content: str, latency_ms: float = None):
    """Render satu pesan chat dengan styling."""
    with st.chat_message(role):
        st.markdown(content)
        if latency_ms and role == "assistant":
            st.caption(f"⏱️ {latency_ms:.0f}ms")


def render_chat_history(messages: list):
    """Render seluruh chat history."""
    for msg in messages:
        render_chat_message(msg["role"], msg["content"], msg.get("latency_ms"))


def render_model_status(predictor):
    """Render status model di sidebar."""
    status = predictor.get_status()
    if predictor.is_finetuned:
        st.success(status)
    else:
        st.warning(status)


def render_sidebar_settings():
    """Render settings di sidebar. Return dict of settings."""
    st.markdown("#### ⚙️ Generation Settings")
    max_tokens = st.slider("Max Response Tokens", 64, 512, 256)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.markdown("#### 📄 Document Context")
    context_input = st.text_area(
        "Paste konteks dokumen (opsional):",
        height=120,
        placeholder="Contoh: SOP perusahaan, kebijakan HR...",
    )

    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "context": context_input,
    }
