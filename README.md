# AI Medical Report Assistant

## Overview

An end-to-end AI application that predicts chest diseases from X-Ray images using TensorFlow Transfer Learning and generates an AI-assisted medical report using Google Gemini.

---

## Features

- Chest X-Ray Classification
- TensorFlow EfficientNetB0
- Gemini AI Report Generation
- Streamlit Web Application
- SQLite Prediction History
- ROC Curve
- Confusion Matrix
- Accuracy/Loss Graphs

---

## Dataset

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## Installation

```bash
pip install -r requirements.txt
```

Run

```bash
python train.py
```

Run Streamlit

```bash
streamlit run app.py
```

---

## Folder Structure

```
AI-Medical-Report-Assistant

app.py

train.py

predict.py

database.py

report_generator.py

utils.py

requirements.txt

README.md

model/

images/

dataset/
```

---

## Author

Sai Vishal