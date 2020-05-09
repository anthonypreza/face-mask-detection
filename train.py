import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument(
    "-p",
    "--plot",
    type=str,
    default="plot.png",
    help="path to output loss/accuracy plot",
)
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="mask_detector.model",
    help="path to output face mask detector model",
)
args = vars(ap.parse_args())

LR = 1e-4
EPOCHS = 20
BS = 32

print("[INFO] Loading images...")
image_paths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for image_path in image_paths:
    print("[INFO] Loading image: ", image_path)
    label = image_path.split(os.path.sep)[-2]

    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = keras.applications.mobilenet_v2.preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print("Data shape: ", data.shape)
print("Labels shape: ", labels.shape)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = keras.utils.to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

aug = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

print("[INFO] Building model...")
base = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=keras.layers.Input(shape=(224, 224, 3)),
)

head = base.output
pool = keras.layers.AveragePooling2D(pool_size=(7, 7))(head)
flatten = keras.layers.Flatten()(pool)
dense1 = keras.layers.Dense(128, activation="relu")(flatten)
dropout = keras.layers.Dropout(0.5)(dense1)
dense2 = keras.layers.Dense(2, activation="softmax")(dropout)

model = keras.models.Model(inputs=base.input, outputs=dense2)

for layer in base.layers:
    layer.trainable = False

print("[INFO] Compiling model...")
adam = keras.optimizers.Adam(lr=LR, decay=LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

print("[INFO] Training model...")
history = model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BS,
    epochs=EPOCHS,
)

print("[INFO] Evaluating network...")
predictions = model.predict(X_test, batch_size=BS)
predictions = np.argmax(predictions, axis=1)

print(
    classification_report(y_test.argmax(axis=1), predictions, target_names=lb.classes_)
)

print(["INFO] saving model to disk..."])
model.save(args["model"], save_format="h5")

print("[INFO] Plotting loss curve...")
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
print(f'[INFO] Plot saved to {args["plot"]}...')
