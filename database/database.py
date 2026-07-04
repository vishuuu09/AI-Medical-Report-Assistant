import sqlite3
import pandas as pd

DATABASE = "database/history.db"


def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS history(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT,

            prediction TEXT,

            confidence REAL,

            report TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

    """)

    conn.commit()

    conn.close()


def save_prediction(image_name, prediction, confidence, report):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

        INSERT INTO history(

            image_name,

            prediction,

            confidence,

            report

        )

        VALUES(?,?,?,?)

    """, (

        image_name,

        prediction,

        confidence,

        report

    ))

    conn.commit()

    conn.close()


def get_history():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM history

        ORDER BY created_at DESC

    """)

    history = cursor.fetchall()

    conn.close()

    return history


# -------------------------
# NEW
# -------------------------

def history_dataframe():

    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql_query("""

        SELECT

        image_name AS Image,

        prediction AS Prediction,

        ROUND(confidence*100,2) AS Confidence,

        created_at AS Date

        FROM history

        ORDER BY created_at DESC

    """, conn)

    conn.close()

    return df


# -------------------------
# NEW
# -------------------------

def clear_history():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()

    conn.close()