import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ",
    layout="wide"
)

st.title("🏥 AI Medical Report Assistant")

st.caption(
    "Deep Learning + Artificial Intelligence + Medical Imaging"
)

st.divider()

st.header("📌 Project Overview")

st.write("""

AI Medical Report Assistant is an AI-powered healthcare application
that predicts chest diseases from Chest X-Ray images using Deep Learning.

After prediction, Google's Gemini AI automatically generates a
professional medical report.

The complete application is built using TensorFlow, Streamlit,
SQLite and Google's Generative AI.

""")

st.divider()

st.header("🧠 Deep Learning Model")

st.markdown("""

- Transfer Learning

- TensorFlow

- Image Size : **224 x 224**

- Binary Classification

- Classes

    • NORMAL

    • PNEUMONIA

""")

st.divider()

st.header("🩻 Dataset")

st.markdown("""

Dataset Used

Chest X-Ray Images (Pneumonia)

Training Images

Validation Images

Testing Images

Image Size

224 × 224

""")

st.divider()

st.header("⚙ Technology Stack")

col1,col2=st.columns(2)

with col1:

    st.markdown("""

### AI

- TensorFlow

- OpenCV

- NumPy

- Scikit-learn

- Gemini AI

""")

with col2:

    st.markdown("""

### Application

- Streamlit

- SQLite

- Python

- ReportLab

""")

st.divider()

st.header("✨ Features")

st.markdown("""

✅ Chest X-Ray Upload

✅ AI Disease Prediction

✅ Confidence Score

✅ AI Medical Report

✅ PDF Download

✅ SQLite History

✅ Dashboard

✅ Model Performance

""")

st.divider()

st.success(
    "Project Developed for AI/ML Technical Assessment"
)