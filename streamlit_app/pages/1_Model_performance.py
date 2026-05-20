import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Model Performance — ContextFlow", page_icon="◆", layout="wide")

# ─────────────────────── HelloMonday-Inspired CSS ───────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #050505;
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0;
    }

    .page-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 300;
        color: #555555;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 8px;
        margin-bottom: 40px;
    }

    .hm-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
        margin: 40px 0;
        border: none;
    }

    .metric-grid {
        display: flex;
        gap: 16px;
        margin-bottom: 40px;
    }

    .metric-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-box:hover {
        border-color: rgba(255, 107, 53, 0.2);
        background: rgba(255, 107, 53, 0.03);
    }

    .metric-box .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    .metric-box .label {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #555555;
        margin-top: 8px;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 20px;
    }

    .training-info {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #555555;
        padding: 12px 16px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        margin-bottom: 24px;
        line-height: 1.6;
    }

    .training-info strong {
        color: #999999;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────── Load Real Training Data ───────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "contextflow-finetuned", "checkpoint-5400")
TRAINER_STATE_PATH = os.path.join(CHECKPOINT_DIR, "trainer_state.json")

# Real evaluation metrics dari log (hasil evaluasi model)
EVAL_METRICS = {
    "bleu": 0.3129,
    "rouge1": 0.0276,
    "rouge2": 0.0086,
    "rougeL": 0.0258,
}

# Load trainer_state.json untuk training curve
train_steps = []
train_losses = []
eval_steps = []
eval_losses = []

if os.path.exists(TRAINER_STATE_PATH):
    with open(TRAINER_STATE_PATH, "r") as f:
        trainer_state = json.load(f)

    for log in trainer_state.get("log_history", []):
        if "loss" in log and "eval_loss" not in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])
        elif "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_losses.append(log["eval_loss"])

    best_metric = trainer_state.get("best_metric")
    best_checkpoint = trainer_state.get("best_model_checkpoint", "")
    global_step = trainer_state.get("global_step", 0)
else:
    best_metric = None
    best_checkpoint = ""
    global_step = 0

final_train_loss = train_losses[-1] if train_losses else "N/A"
final_eval_loss = eval_losses[-1] if eval_losses else "N/A"

# ─────────────────────── Header ───────────────────────
st.markdown('<p class="page-title">Model Performance</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Actual fine-tuning results</p>', unsafe_allow_html=True)

# Training info
_best_metric_str = f"{best_metric:.4f}" if best_metric else "N/A"
st.markdown(f"""
<div class="training-info">
    <strong>Base Model:</strong> TinyLlama/TinyLlama-1.1B-Chat-v1.0 &nbsp;·&nbsp;
    <strong>Method:</strong> QLoRA (r=16, α=32) &nbsp;·&nbsp;
    <strong>Steps:</strong> {global_step:,} &nbsp;·&nbsp;
    <strong>Best Checkpoint:</strong> step 800 (eval_loss: {_best_metric_str})
</div>
""", unsafe_allow_html=True)


# ─── Metrics Summary ───
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-box">
        <div class="value">{EVAL_METRICS['bleu']}</div>
        <div class="label">BLEU Score</div>
    </div>
    <div class="metric-box">
        <div class="value">{EVAL_METRICS['rouge1']}</div>
        <div class="label">ROUGE-1</div>
    </div>
    <div class="metric-box">
        <div class="value">{EVAL_METRICS['rouge2']}</div>
        <div class="label">ROUGE-2</div>
    </div>
    <div class="metric-box">
        <div class="value">{EVAL_METRICS['rougeL']}</div>
        <div class="label">ROUGE-L</div>
    </div>
    <div class="metric-box">
        <div class="value">{final_train_loss if isinstance(final_train_loss, str) else f'{final_train_loss:.4f}'}</div>
        <div class="label">Final Train Loss</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hm-divider"></div>', unsafe_allow_html=True)

# ─── Training & Validation Loss (Real Data) ───
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">Training vs Validation Loss</p>', unsafe_allow_html=True)

    if train_steps and eval_steps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('#050505')
        ax.set_facecolor('#0a0a0a')

        # Subsample train loss for smoother plot (every 5th point)
        sub_steps = train_steps[::5]
        sub_losses = train_losses[::5]

        ax.plot(sub_steps, sub_losses, label="Train Loss", color="#ff6b35", linewidth=1.5, alpha=0.8)
        ax.plot(eval_steps, eval_losses, label="Eval Loss", color="#666666", linewidth=2, alpha=0.9, marker='o', markersize=3)

        # Mark best checkpoint
        if best_metric:
            best_step_idx = eval_losses.index(min(eval_losses))
            ax.axvline(x=eval_steps[best_step_idx], color="#4ecdc4", linestyle="--", linewidth=0.8, alpha=0.5, label=f"Best (step {eval_steps[best_step_idx]})")
            ax.plot(eval_steps[best_step_idx], min(eval_losses), 'o', color="#4ecdc4", markersize=8, zorder=5)

        ax.set_xlabel("Steps", color="#444444", fontsize=9)
        ax.set_ylabel("Loss", color="#444444", fontsize=9)
        ax.legend(facecolor="#0a0a0a", edgecolor="#1a1a1a", labelcolor="#888888", fontsize=7)
        ax.tick_params(colors="#333333", labelsize=8)
        ax.grid(True, alpha=0.04, color="#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#1a1a1a")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Training data not found. Run `python train.py` first.")

with col2:
    st.markdown('<p class="section-title">Evaluation Metrics Detail</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Metric": ["BLEU Score", "ROUGE-1", "ROUGE-2", "ROUGE-L",
                    "Train Loss (final)", "Eval Loss (final)", "Best Eval Loss"],
        "Value": [
            EVAL_METRICS["bleu"],
            EVAL_METRICS["rouge1"],
            EVAL_METRICS["rouge2"],
            EVAL_METRICS["rougeL"],
            round(final_train_loss, 4) if isinstance(final_train_loss, float) else final_train_loss,
            round(final_eval_loss, 4) if isinstance(final_eval_loss, float) else final_eval_loss,
            round(best_metric, 4) if best_metric else "N/A",
        ],
        "Note": [
            "50 sample eval",
            "50 sample eval",
            "50 sample eval",
            "50 sample eval",
            f"Step {global_step}",
            f"Step {global_step}",
            "Step 800 (best)",
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown('<div class="hm-divider"></div>', unsafe_allow_html=True)

# ─── Eval Loss Progression ───
st.markdown('<p class="section-title">Evaluation Loss Over Training</p>', unsafe_allow_html=True)

if eval_steps:
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    fig2.patch.set_facecolor('#050505')
    ax2.set_facecolor('#0a0a0a')

    ax2.plot(eval_steps, eval_losses, color="#ff6b35", linewidth=2, marker='o', markersize=4, alpha=0.9)
    ax2.fill_between(eval_steps, eval_losses, min(eval_losses), alpha=0.04, color="#ff6b35")

    # Annotate best and worst
    min_idx = eval_losses.index(min(eval_losses))
    ax2.annotate(f"Best: {min(eval_losses):.4f}",
                 xy=(eval_steps[min_idx], eval_losses[min_idx]),
                 xytext=(eval_steps[min_idx] + 300, eval_losses[min_idx] - 0.15),
                 color="#4ecdc4", fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#4ecdc4', lw=0.8))

    ax2.set_xlabel("Steps", color="#444444", fontsize=9)
    ax2.set_ylabel("Eval Loss", color="#444444", fontsize=9)
    ax2.tick_params(colors="#333333", labelsize=8)
    ax2.grid(True, alpha=0.04, color="#ffffff")
    for spine in ax2.spines.values():
        spine.set_color("#1a1a1a")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.warning("No evaluation data available.")

st.markdown('<div class="hm-divider"></div>', unsafe_allow_html=True)

# ─── Training Configuration ───
st.markdown('<p class="section-title">Training Configuration</p>', unsafe_allow_html=True)
config_df = pd.DataFrame({
    "Parameter": ["Base Model", "LoRA Rank (r)", "LoRA Alpha", "LoRA Dropout", "Target Modules",
                   "Batch Size", "Gradient Accumulation", "Learning Rate", "LR Scheduler",
                   "Epochs", "Max Seq Length", "Total Steps", "Best Checkpoint"],
    "Value": ["TinyLlama-1.1B-Chat-v1.0", "16", "32", "0.1", "q_proj, k_proj, v_proj, o_proj",
              "1", "4", "2e-4", "cosine",
              "3", "256", str(global_step), f"Step 800 (loss: {_best_metric_str})"],
})
st.dataframe(config_df, use_container_width=True, hide_index=True)