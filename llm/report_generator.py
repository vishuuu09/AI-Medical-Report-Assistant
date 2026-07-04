import os

from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_report(prediction, confidence):

    prompt = f"""

You are an expert radiologist.

Prediction:

{prediction}

Confidence:

{confidence*100:.2f}%

Generate a professional medical report.

Include

1 Findings

2 Impression

3 Recommendation

4 Disclaimer

"""

    try:

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"""

Prediction : {prediction}

Confidence : {confidence*100:.2f}%

Error

{e}

"""