from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile


def generate_pdf(image_name, prediction, confidence, report):

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    doc = SimpleDocTemplate(temp.name)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("<b>AI Medical Report Assistant</b>", styles["Title"])
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(f"<b>Image:</b> {image_name}", styles["BodyText"])
    )

    elements.append(
        Paragraph(f"<b>Prediction:</b> {prediction}", styles["BodyText"])
    )

    elements.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence*100:.2f}%",
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph("<b>AI Medical Report</b>", styles["Heading2"])
    )

    report = report.replace("\n", "<br/>")

    elements.append(
        Paragraph(report, styles["BodyText"])
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Generated using TensorFlow + Gemini AI",
            styles["Italic"]
        )
    )

    doc.build(elements)

    return temp.name