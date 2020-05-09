from tensorflow import keras
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_predict(frame, face_net, mask_net):
    """
    Face detection and mask detection logic.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            X_start, Y_start, X_end, Y_end = box.astype("int")

            X_start, Y_start = max(0, X_start), max(0, Y_start)
            X_end, Y_end = min(w - 1, X_end), min(h - 1, Y_end)

            face = frame[Y_start:Y_end, X_start:X_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((X_start, Y_start, X_end, Y_end))

    if len(faces) > 0:
        preds = mask_net.predict(faces)

    return (locs, preds)


ap = argparse.ArgumentParser()
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

print("[INFO] Loading face detector model...")
proto_txt_path = os.path.sep.join([args["face"], "deploy.prototxt"])
weights_path = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"]
)
face_net = cv2.dnn.readNet(proto_txt_path, weights_path)

print("[INFO] Loading face mask detector model...")
mask_net = keras.models.load_model(args["model"])

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    locs, preds = detect_predict(frame, face_net, mask_net)

    for box, pred in zip(locs, preds):
        X_start, Y_start, X_end, Y_end = box
        mask, without_mask = pred

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = f"{label}: {max(mask, without_mask) * 100}"

        cv2.putText(
            frame,
            label,
            (X_start, Y_start - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.rectangle(frame, (X_start, Y_start), (X_end, Y_end), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
