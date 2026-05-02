import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🛡️ LLM Spam/Phishing Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5;}
    .stTextArea textarea {height: 220px !important;}
    .result-box {padding: 20px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL LOADING ======================
@st.cache_resource(show_spinner="Loading LLM Model...")
def load_llm_model():
    model_path = Path("model")  # Path where notebook saved the model
    
    if model_path.exists() and (model_path / "config.json").exists():
        # Load fine-tuned DistilBERT from notebook
        classifier = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=128
        )
        st.success("✅ Loaded **Fine-tuned DistilBERT** model")
        return classifier, "fine-tuned"
    else:
        # Fallback: Use a strong zero-shot or general spam model
        st.info("Fine-tuned model not found. Using general-purpose spam detection model...")
        try:
            classifier = pipeline(
                "text-classification",
                model="mrm8488/distilroberta-finetuned-spam",
                device=0 if torch.cuda.is_available() else -1
            )
            st.success("✅ Loaded **DistilRoBERTa Spam Model**")
            return classifier, "general"
        except:
            # Ultimate fallback
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            st.warning("Using Zero-Shot classification (slower but works)")
            return classifier, "zero-shot"

# Load the model once
classifier, model_type = load_llm_model()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("🔧 Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
    
    st.markdown("---")
    st.markdown("### Model Info")
    if model_type == "fine-tuned":
        st.success("Using your trained DistilBERT model")
    elif model_type == "general":
        st.info("Using pre-trained spam model")
    else:
        st.info("Zero-shot classification")
    
    st.markdown("---")
    st.caption("Built with ❤️ using Hugging Face Transformers")

# ====================== MAIN UI ======================
st.title("🛡️ LLM-Powered Spam / Phishing Detector")
st.markdown("Detect spam and phishing attempts using **modern transformer models**")

tab1, tab2 = st.tabs(["📝 Single Message", "📊 Batch Analysis"])

with tab1:
    user_input = st.text_area(
        "Paste your email or SMS message here:",
        placeholder="Enter message to analyze...",
        height=180
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_button = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    if analyze_button and user_input.strip():
        with st
