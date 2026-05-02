import streamlit as st
import joblib
import os

# 1. Setup Page Config
st.set_page_config(page_title="Phishing Detection", page_icon="📧")

# 2. Cache the model loading
# This prevents reloading the 268MB file on every user interaction
@st.cache_resource
def load_phishing_model():
    model_path = "spam_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found! Please check your GitHub repository.")
        return None

# Load the model
model = load_phishing_model()

# 3. UI Design
st.title("📧 Phishing / Spam Detection")
st.write("Enter an Email or SMS message below to check for security threats.")

user_input = st.text_area("Message Content", placeholder="Paste message here...", height=200)

# 4. Prediction Logic
if st.button("Analyze Message"):
    if user_input.strip() != "":
        if model is not None:
            with st.spinner("Analyzing..."):
                # Make prediction
                # Note: This assumes your model handles text vectorization internally 
                # (e.g., a Pipeline with TfidfVectorizer + Classifier)
                prediction = model.predict([user_input])[0]
                
                # Handling probability if the model supports it
                try:
                    proba = model.predict_proba([user_input])
                    confidence = proba.max()
                except:
                    confidence = None

                # 5. Display Results
                st.divider()
                if prediction == 1: # Assuming 1 is Spam/Phishing
                    st.error("### ⚠️ Spam / Phishing Detected")
                    if confidence:
                        st.info(f"**Confidence Level:** {confidence:.2%}")
                else:
                    st.success("### ✅ Safe Message")
                    if confidence:
                        st.info(f"**Confidence Level:** {confidence:.2%}")
        else:
            st.error("Model failed to load.")
    else:
        st.warning("Please enter some text to analyze.")
