import os
import streamlit as st

from predict import predict_image
from llm.report_generator import generate_report
from database.database import save_prediction
from components.pdf_generator import generate_pdf

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Upload & Predict",
    page_icon="📤",
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------
with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-title">
📤 Upload Chest X-Ray
</div>

<div class="sub-title">
Upload a Chest X-Ray image and let AI predict the disease.
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    left, right = st.columns([1, 1])

    with left:

        st.image(
            file_path,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

    with right:

        st.info("""
### Instructions

✔ Verify the uploaded X-Ray

✔ Click **Predict Disease**

✔ AI will classify the image

✔ Gemini AI will generate the medical report
""")

    st.divider()

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button(
        "🩺 Predict Disease",
        use_container_width=True
    ):

        with st.spinner("Analyzing X-Ray..."):

            prediction, confidence = predict_image(file_path)

            # Generate AI report
            report = generate_report(
                prediction,
                confidence
            )

            # Generate PDF
            pdf_path = generate_pdf(
                uploaded_file.name,
                prediction,
                confidence,
                report
            )

            # Save to database
            save_prediction(
                uploaded_file.name,
                prediction,
                confidence,
                report
            )

        st.toast(
            "Prediction Completed Successfully!",
            icon="✅"
        )

        st.success("Prediction Completed")

        # -----------------------------
        # Result Cards
        # -----------------------------

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(f"""
<div class="card">
<div class="card-title">
Prediction
</div>

<div class="card-value">
{prediction}
</div>
</div>
""", unsafe_allow_html=True)

        with c2:

            st.markdown(f"""
<div class="card">
<div class="card-title">
Confidence
</div>

<div class="card-value">
{confidence*100:.2f}%
</div>
</div>
""", unsafe_allow_html=True)

        st.write("### Confidence Score")

        st.progress(confidence)

        if prediction == "PNEUMONIA":

            st.error("🦠 Pneumonia Detected")

        else:

            st.success("🫁 Normal Chest X-Ray")

        st.divider()

        # -----------------------------
        # AI Report
        # -----------------------------

        st.subheader("🤖 AI Medical Report")

        st.markdown(
            f"""
<div class="report-card">
{report}
</div>
""",
            unsafe_allow_html=True
        )

        st.divider()

        # -----------------------------
        # Download
        # -----------------------------

        with open(pdf_path, "rb") as pdf_file:

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file,
                file_name="Medical_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )