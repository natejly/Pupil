# import CNN
import os
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
import tensorflow as tf

import numpy as np
import cv2
import time

def coarse_find(frame):
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(150, 150)
    )
    return eyes
CNN_path = "eye_tracking_model.keras"

model = load_model(CNN_path)

video_path = "videos/2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Cannot open video")
# importt eyedata frame
data = pd.read_csv("output/eye_data.csv")
frame_index = 0
while True:
    frame_index += 1
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[:frame.shape[0] // 2, :]
    # frame = frame[frame.shape[0] // 2:, :]

    eyes = coarse_find(frame)

    if len(eyes) > 0:
        prev_eyes = eyes.copy()
    elif prev_eyes is not None:
        eyes = prev_eyes
    else:
        continue

    x, y, w, h = eyes[0]
    size = max(w, h)
    # convert to grayscale

    image = frame[y:y+size, x:x+size]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize the image to 128x128
    image = cv2.resize(image, (128, 128))
    # convert to array
    image = image.reshape(1, 128, 128, 1)
    # normalize the image
    image = image / 255.0

    end = time.time()

    print(f"Load and scale time: {(end - start):.2f} s")

    pred = model.predict(image, verbose=0)[0]

    start = time.time()
    rescale = h/128

    pred_x = int(pred[0] * rescale + x)
    pred_y = int(pred[1] * rescale + y)
    pred_w = int(pred[2] * rescale)
    pred_h = int(pred[3] * rescale)
    pred_angle = int(pred[4])
    pred_angle = (pred_angle*90+180) % 180
    # transform elipse
    pred_color = (290, 100, 100)
    cv2.ellipse(frame, (pred_x, pred_y), (pred_w, pred_h), pred_angle, 0, 360, pred_color, 1)
    cv2.circle(frame, (pred_x, pred_y), 1, pred_color, -1)


    end = time.time()

    # show the eye crop

    cv2.imshow("Processed Frame", frame)
    end = time.time()
    # exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
