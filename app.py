import streamlit as st

st.set_page_config(
    page_title="AI Medical Report Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI Medical Report Assistant")

st.markdown("""
# Welcome 👋

This application can

- 📤 Upload Chest X-Rays
- 🧠 Predict Diseases using Deep Learning
- 🤖 Generate AI Medical Reports using Gemini
- 📜 Store Prediction History
- 📊 Visualize Model Performance

Use the sidebar to navigate through the application.
""")