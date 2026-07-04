import streamlit as st
from database.database import history_dataframe, clear_history

st.set_page_config(
    page_title="Prediction History",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Prediction History")

st.caption(
    "View all previous AI predictions."
)

st.divider()

# -----------------------------
# Load Data
# -----------------------------
df = history_dataframe()

# -----------------------------
# Search
# -----------------------------
search = st.text_input(
    "🔍 Search by Image Name"
)

if search:

    df = df[
        df["Image"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total Predictions",
        len(df)
    )

with c2:
    st.metric(
        "Pneumonia Cases",
        len(
            df[
                df["Prediction"]=="PNEUMONIA"
            ]
        )
    )

with c3:
    st.metric(
        "Normal Cases",
        len(
            df[
                df["Prediction"]=="NORMAL"
            ]
        )
    )

st.divider()

# -----------------------------
# Table
# -----------------------------
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

st.divider()

# -----------------------------
# Download CSV
# -----------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download CSV",
    csv,
    "Prediction_History.csv",
    "text/csv",
    use_container_width=True
)

# -----------------------------
# Clear History
# -----------------------------
if st.button(
    "🗑 Clear History",
    use_container_width=True
):

    clear_history()

    st.success(
        "History Cleared Successfully."
    )

    st.rerun()