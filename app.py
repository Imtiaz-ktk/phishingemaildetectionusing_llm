import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model
tokenizer = DistilBertTokenizerFast.from_pretrained("./model")
model = DistilBertForSequenceClassification.from_pretrained("./model")

st.title("📧 Phishing / Spam Detection using LLM")

st.write("Enter Email, SMS, or Chat Message:")

user_input = st.text_area("Message")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(probs).item(), probs.detach().numpy()

if st.button("Check"):
    if user_input:
        label, prob = predict(user_input)

        if label == 1:
            st.error(f"⚠️ Spam / Phishing Detected\nConfidence: {prob[0][1]:.2f}")
        else:
            st.success(f"✅ Safe Message\nConfidence: {prob[0][0]:.2f}")
    else:
        st.warning("Please enter text")
