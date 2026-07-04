import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = 224

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]

model = tf.keras.models.load_model("model/best_model.keras")


def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image


def predict_image(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image)[0][0]

    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = prediction
    else:
        label = "NORMAL"
        confidence = 1 - prediction

    return label, float(confidence)