from tensorflow import keras
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument(
    "-f",
    "--face",
    type=str,
    default="face_detector",
    help="path to face detector model directory",
)
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model",
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="minimum probability to filter weak detections",
)
args = vars(ap.parse_args())

print("[INFO] Loading face detector...")
prototxt_path = os.path.sep.join([args["face"], "deploy.prototxt"])
weights_path = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"]
)
net = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] Loading face mask detector...")
model = keras.models.load_model(args["model"])


image = cv2.imread(args["image"])
orig = image.copy()
h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        X_start, Y_start, X_end, Y_end = box.astype("int")

        X_start, Y_start = max(0, X_start), max(0, Y_start)
        X_end, Y_end = max(0, X_end), max(0, Y_end)

        face = image[Y_start:Y_end, X_start:X_end]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = keras.preprocessing.image.img_to_array(face)
        face = keras.applications.mobilenet_v2.preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        mask, without_mask = model.predict(face)[0]

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = f"{label}: {max(mask, without_mask)}"

        cv2.putText(
            image,
            label,
            (X_start, Y_start - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.rectangle(image, (X_start, Y_start), (X_end, Y_end), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
