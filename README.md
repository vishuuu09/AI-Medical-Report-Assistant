<img width="1536" height="1024" alt="ChatGPT Image Jul 4, 2026, 06_47_55 PM" src="https://github.com/user-attachments/assets/9ea78c9c-1078-4208-a88a-a27a61895bab" /># AI Medical Report Assistant

## Overview

An end-to-end AI application that predicts chest diseases from X-Ray images using TensorFlow Transfer Learning and generates an AI-assisted medical report using Google Gemini.

---

#Architecture
<img width="1536" height="1024" alt="ChatGPT Image Jul 4, 2026, 06_47_55 PM" src="https://github.com/user-attachments/assets/8a641342-085b-49b2-87a6-71b5c1337152" />





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
<img width="1278" height="816" alt="{713C7890-5CB6-4B33-80B7-C69DCD0A0854}" src="https://github.com/user-attachments/assets/625140e9-59f1-4fcd-8d5a-eb0da264df01" />
<img width="1358" height="950" alt="{777B3977-6DDE-4323-BB93-019B6A25CF7D}" src="https://github.com/user-attachments/assets/741b5684-e5ef-4c18-a750-d2ca3248f540" />
<img width="1903" height="1028" alt="{B30AB968-6121-461C-BC57-CEDE68E303B4}" src="https://github.com/user-attachments/assets/cb7c81e5-cfe7-4022-8fa4-538ab85ea77f" />
<img width="1876" height="1030" alt="{337E2FA9-942A-4A38-B103-A8E6FCE2C665}" src="https://github.com/user-attachments/assets/a5461126-e2b9-43a8-ace5-b65317d31db6" />
<img width="1813" height="920" alt="{C7E47516-8D6A-4249-8B7C-1A2E16196B3E}" src="https://github.com/user-attachments/assets/1aafca8d-8240-419d-959c-f0c5d134eea0" />
<img width="1920" height="1038" alt="{33A17362-E4D7-4D19-8D6F-180555C615D2}" src="https://github.com/user-attachments/assets/cb335706-2b43-4016-92d0-428548bce9b2" />



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
