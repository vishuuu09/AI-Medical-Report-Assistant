import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------
# Paths
# -----------------------------------

DATASET = "dataset/chest_xray"

TEST_PATH = os.path.join(
    DATASET,
    "test"
)

MODEL_PATH = "model/best_model.keras"

IMAGE_FOLDER = "images"

MODEL_FOLDER = "model"

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# -----------------------------------
# Test Dataset
# -----------------------------------

generator = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = generator.flow_from_directory(

    TEST_PATH,

    target_size=(224, 224),

    batch_size=32,

    class_mode="binary",

    shuffle=False

)

# -----------------------------------
# Load Model
# -----------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel Loaded Successfully\n")

# -----------------------------------
# Prediction
# -----------------------------------

pred = model.predict(
    test_data,
    verbose=1
)

predictions = (pred > 0.5).astype(int)

true = test_data.classes

# -----------------------------------
# Metrics
# -----------------------------------

accuracy = np.mean(
    predictions.flatten() == true
)

precision = precision_score(
    true,
    predictions
)

recall = recall_score(
    true,
    predictions
)

f1 = f1_score(
    true,
    predictions
)

print("=" * 50)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

print("=" * 50)

# -----------------------------------
# Classification Report
# -----------------------------------

report = classification_report(
    true,
    predictions
)

print(report)

with open(
    "model/classification_report.txt",
    "w"
) as f:

    f.write(report)

# -----------------------------------
# Save Metrics
# -----------------------------------

metrics = {

    "accuracy": round(
        accuracy * 100,
        2
    ),

    "precision": round(
        precision * 100,
        2
    ),

    "recall": round(
        recall * 100,
        2
    ),

    "f1_score": round(
        f1 * 100,
        2
    )

}

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(
    true,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix")

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# ROC Curve
# -----------------------------------

fpr, tpr, _ = roc_curve(
    true,
    pred
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.grid(True)

plt.savefig(
    "images/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"ROC AUC : {roc_auc:.4f}")

# -----------------------------------
# Update Metrics
# -----------------------------------

metrics["roc_auc"] = round(
    roc_auc * 100,
    2
)

# -----------------------------------
# Save metrics.json
# -----------------------------------

with open(
    "model/metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# -----------------------------------
# Success
# -----------------------------------

print("\n" + "=" * 60)

print("✅ Evaluation Completed Successfully")

print("=" * 60)

print("Generated Files:")

print("✔ images/confusion_matrix.png")

print("✔ images/roc_curve.png")

print("✔ model/classification_report.txt")

print("✔ model/metrics.json")

print("=" * 60)