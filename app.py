import streamlit as st
import torch
from transformers import pipeline
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="🛡️ LLM Spam & Phishing Detector",
    page_icon="📧",
    layout="centered"
)

st.title("🛡️ LLM-Powered Spam / Phishing Detector")
st.markdown("**Advanced detection using Transformer models**")

# ====================== MODEL LOADING ======================
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    model_dir = Path("model")
    
    # Try your fine-tuned model first
    if model_dir.exists() and (model_dir / "config.json").exists():
        try:
            pipe = pipeline(
                "text-classification",
                model=str(model_dir),
                tokenizer=str(model_dir),
                device=0 if torch.cuda.is_available() else -1,
                truncation=True,
                max_length=512
            )
            st.success("✅ **Your Fine-tuned DistilBERT** model loaded")
            return pipe, "fine-tuned"
        except:
            pass
    
    # Fallback 1: Good open spam model
    try:
        pipe = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-spam",
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("✅ Loaded **DistilRoBERTa Spam Classifier**")
        return pipe, "general"
    except:
        # Fallback 2: Zero-shot (always works)
        pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        st.info("Using Zero-Shot classification")
        return pipe, "zero-shot"

pipe, model_type = load_model()

# ====================== UI ======================
tab1, tab2 = st.tabs(["Single Message", "Batch Analysis"])

with tab1:
    message = st.text_area("Paste Email or SMS message:", height=200, 
                          placeholder="Enter message to analyze...")

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message")
        else:
            with st.spinner("Analyzing with LLM..."):
                try:
                    if model_type == "zero-shot":
                        result = pipe(message[:512], candidate_labels=["spam", "ham"])
                        is_spam = result["labels"][0] == "spam"
                        confidence = result["scores"][0]
                    else:
                        result = pipe(message[:512])[0]
                        is_spam = result["label"].lower() in ["spam", "1", "label_1"]
                        confidence = result["score"]

                    st.divider()
                    
                    if is_spam:
                        st.error("### 🚨 SPAM / PHISHING DETECTED", icon="⚠️")
                        st.markdown(f"**Confidence:** `{confidence:.1%}`")
                        st.warning("**Do not click any links or reply.**")
                    else:
                        st.success("### ✅ This message appears safe", icon="✅")
                        st.markdown(f"**Confidence:** `{confidence:.1%}`")
                    
                    with st.expander("Raw Model Output"):
                        st.json(result)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

with tab2:
    st.subheader("Batch Analysis")
    uploaded = st.file_uploader("Upload CSV or TXT (one message per line)", 
                               type=["csv", "txt"])
    
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            col = st.selectbox("Select message column", df.columns)
            messages = df[col].astype(str).tolist()
        else:
            messages = [line.decode().strip() for line in uploaded if line.strip()]
        
        if st.button("Analyze All Messages", type="primary"):
            with st.spinner(f"Processing {len(messages)} messages..."):
                results = []
                for msg in messages:
                    try:
                        if model_type == "zero-shot":
                            res = pipe(msg[:512], candidate_labels=["spam", "ham"])
                            spam = res["labels"][0] == "spam"
                            conf = res["scores"][0]
                        else:
                            res = pipe(msg[:512])[0]
                            spam = res["label"].lower() in ["spam", "1", "label_1"]
                            conf = res["score"]
                        
                        results.append({
                            "Message": msg[:80] + "..." if len(msg) > 80 else msg,
                            "Prediction": "SPAM" if spam else "HAM",
                            "Confidence": f"{conf:.1%}"
                        })
                    except:
                        results.append({"Message": msg[:80]+"...", "Prediction": "ERROR", "Confidence": "N/A"})
                
                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True)
                
                spam_count = sum(1 for r in results if r["Prediction"] == "SPAM")
                st.metric("Spam Detected", f"{spam_count} / {len(messages)}")

st.caption("Built with Hugging Face Transformers • Fine-tuned DistilBERT ready")
