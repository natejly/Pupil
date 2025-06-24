import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2, time
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from trackingv2 import coarse_find

model = load_model("eye_tracking_model.keras")

# smoothing factor (0 < α ≤ 1). smaller α → slower updates
alpha = 1
ema = None  # will hold smoothed [x, y, w, h, angle_norm]

cap = cv2.VideoCapture("videos/2.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[: frame.shape[0]//2, :]
    eyes = coarse_find(frame)
    if len(eyes) > 0:
        prev_eyes = eyes.copy()
    elif prev_eyes is not None:
        eyes = prev_eyes
    else:
        continue

    x, y, w, h = eyes[0]
    size = max(w, h)
    crop = frame[y:y+size, x:x+size]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img128 = cv2.resize(gray, (128, 128)).astype("float32")/255.0
    inp = img128.reshape(1,128,128,1)

    pred = model.predict(inp, verbose=0)[0]
    # unpack and rescale
    px = pred[0]* (h/128) + x
    py = pred[1]* (h/128) + y
    pw = pred[2]* (h/128)
    ph = pred[3]* (h/128)
    # keep angle normalized in [-1,1]
    an = float(pred[4])

    current = np.array([px, py, pw, ph, an], dtype=np.float32)
    if ema is None:
        ema = current
    else:
        ema = alpha*current + (1-alpha)*ema

    ex, ey, ew, eh, ea = ema
    angle = (ea*90 + 180) % 180

    cv2.ellipse(frame,
                (int(ex), int(ey)),
                (int(ew), int(eh)),
                angle, 0, 360,
                (200,100,100), 1)
    cv2.circle(frame, (int(ex), int(ey)), 2, (200,100,100), -1)

    cv2.imshow("Smoothed Ellipse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
