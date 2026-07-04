import os
import json
import streamlit as st

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Dashboard")

st.caption(
    "Deep Learning Model Evaluation"
)

st.divider()

# ---------------------------------------
# Load Metrics
# ---------------------------------------

metrics = {
    "accuracy": 0,
    "precision": 0,
    "recall": 0,
    "f1_score": 0,
    "roc_auc": 0
}

if os.path.exists("model/metrics.json"):

    with open("model/metrics.json") as f:

        metrics = json.load(f)

# ---------------------------------------
# Metric Cards
# ---------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy']}%"
    )

with c2:
    st.metric(
        "Precision",
        f"{metrics['precision']}%"
    )

with c3:
    st.metric(
        "Recall",
        f"{metrics['recall']}%"
    )

with c4:
    st.metric(
        "F1 Score",
        f"{metrics['f1_score']}%"
    )

with c5:
    st.metric(
        "ROC AUC",
        f"{metrics['roc_auc']}%"
    )

st.divider()

# ---------------------------------------
# Accuracy & Loss
# ---------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("Training Accuracy")

    if os.path.exists("images/accuracy.png"):

        st.image(
            "images/accuracy.png",
            use_container_width=True
        )

with right:

    st.subheader("Training Loss")

    if os.path.exists("images/loss.png"):

        st.image(
            "images/loss.png",
            use_container_width=True
        )

st.divider()

# ---------------------------------------
# ROC Curve & Confusion Matrix
# ---------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("ROC Curve")

    if os.path.exists("images/roc_curve.png"):

        st.image(
            "images/roc_curve.png",
            use_container_width=True
        )

with right:

    st.subheader("Confusion Matrix")

    if os.path.exists("images/confusion_matrix.png"):

        st.image(
            "images/confusion_matrix.png",
            use_container_width=True
        )

st.divider()

# ---------------------------------------
# Dataset Visualization
# ---------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("Label Distribution")

    if os.path.exists("images/label_distribution.png"):

        st.image(
            "images/label_distribution.png",
            use_container_width=True
        )

with right:

    st.subheader("Sample Images")

    if os.path.exists("images/sample_images.png"):

        st.image(
            "images/sample_images.png",
            use_container_width=True
        )

st.divider()

# ---------------------------------------
# Classification Report
# ---------------------------------------

st.subheader("Classification Report")

if os.path.exists(
    "model/classification_report.txt"
):

    with open(
        "model/classification_report.txt"
    ) as f:

        st.code(
            f.read()
        )

else:

    st.warning(
        "Classification report not found."
    )

st.divider()

st.success(
    "✅ Model evaluation completed successfully."
)