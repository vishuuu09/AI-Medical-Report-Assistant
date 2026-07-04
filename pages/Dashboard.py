import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Dashboard",
    page_icon="🏥",
    layout="wide"
)

DATABASE = "database/history.db"


# --------------------------
# Load Data
# --------------------------

def load_data():
    conn = sqlite3.connect(DATABASE)

    try:
        df = pd.read_sql_query(
            "SELECT * FROM history ORDER BY created_at DESC",
            conn
        )
    except:
        df = pd.DataFrame(
            columns=[
                "id",
                "image_name",
                "prediction",
                "confidence",
                "report",
                "created_at"
            ]
        )

    conn.close()

    return df


df = load_data()


# --------------------------
# Header
# --------------------------

st.title("🏥 AI Medical Report Dashboard")

st.caption(
    "Deep Learning + Gemini AI + Streamlit"
)

st.divider()


# --------------------------
# Metrics
# --------------------------

total_predictions = len(df)

if total_predictions > 0:

    avg_confidence = round(
        df["confidence"].mean()*100,
        2
    )

    pneumonia = len(
        df[df["prediction"]=="PNEUMONIA"]
    )

    normal = len(
        df[df["prediction"]=="NORMAL"]
    )

else:

    avg_confidence = 0
    pneumonia = 0
    normal = 0


col1,col2,col3,col4 = st.columns(4)

with col1:

    st.metric(
        "🩻 Total Predictions",
        total_predictions
    )

with col2:

    st.metric(
        "🎯 Avg Confidence",
        f"{avg_confidence}%"
    )

with col3:

    st.metric(
        "🦠 Pneumonia",
        pneumonia
    )

with col4:

    st.metric(
        "🫁 Normal",
        normal
    )

st.divider()


# --------------------------
# Charts
# --------------------------

left,right = st.columns(2)

with left:

    st.subheader("Prediction Distribution")

    if total_predictions > 0:

        pie = pd.DataFrame({

            "Prediction":[
                "PNEUMONIA",
                "NORMAL"
            ],

            "Count":[
                pneumonia,
                normal
            ]

        })

        st.bar_chart(
            pie.set_index("Prediction")
        )

    else:

        st.info("No prediction data available.")


with right:

    st.subheader("Average Confidence")

    if total_predictions > 0:

        st.line_chart(
            df["confidence"]*100
        )

    else:

        st.info("No data available.")


st.divider()


# --------------------------
# Recent Predictions
# --------------------------

st.subheader("📜 Recent Predictions")

if total_predictions > 0:

    display = df[[
        "image_name",
        "prediction",
        "confidence",
        "created_at"
    ]].copy()

    display["confidence"] = (
        display["confidence"]*100
    ).round(2)

    display.rename(columns={
        "image_name":"Image",
        "prediction":"Prediction",
        "confidence":"Confidence (%)",
        "created_at":"Date"
    }, inplace=True)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No predictions yet."
    )


st.divider()


st.success(
    "✅ Model Loaded Successfully"
)