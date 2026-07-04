import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# -------------------------
# Dataset Path
# -------------------------

DATASET_PATH = "dataset/chest_xray"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

# -------------------------
# Data Generators
# -------------------------

train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_generator = ImageDataGenerator(
    rescale=1./255
)

train_data = train_generator.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

validation_data = validation_generator.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# -------------------------
# Show Sample Images
# -------------------------

images, labels = next(train_data)

plt.figure(figsize=(12,6))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(images[i])

    plt.title("PNEUMONIA" if labels[i] else "NORMAL")

    plt.axis("off")

plt.savefig("images/sample_images.png")

plt.close()

# -------------------------
# Label Distribution
# -------------------------

normal = len(os.listdir(os.path.join(TRAIN_PATH,"NORMAL")))

pneumonia = len(os.listdir(os.path.join(TRAIN_PATH,"PNEUMONIA")))

plt.figure(figsize=(5,5))

plt.bar(
    ["Normal","Pneumonia"],
    [normal,pneumonia]
)

plt.title("Label Distribution")

plt.savefig("images/label_distribution.png")

plt.close()

# -------------------------
# Build Model
# -------------------------

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128,activation="relu")(x)

x = Dropout(0.3)(x)

output = Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

checkpoint = ModelCheckpoint(

    "model/best_model.keras",

    monitor="val_accuracy",

    save_best_only=True

)

early = EarlyStopping(

    patience=3,

    restore_best_weights=True

)

# -------------------------
# Train
# -------------------------

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    callbacks=[checkpoint,early]

)

# -------------------------
# Accuracy Graph
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.savefig("images/accuracy.png")

plt.close()

# -------------------------
# Loss Graph
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.savefig("images/loss.png")

plt.close()

print("\nTraining Completed Successfully!")