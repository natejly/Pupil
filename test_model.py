import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from trackingv2 import coarse_find, process_eye_crop

# Load models
start = time.time()
model = load_model("eye_tracking_model.keras")
modelv = load_model("eye_tracking_modelv.keras")
print("models loaded in {:.2f} seconds".format(time.time() - start))

# Video capture
cap = cv2.VideoCapture("Lefts/igor1.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

# Configuration
top_half = True  # change to False to track bottom half
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 if top_half else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Variables
alpha = 1
ema = None
prev_eyes = None
frame_idx = 0
start_time = time.time()

# Frame processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop top or bottom half
    if top_half:
        frame = frame[:frame.shape[0] // 2, :]
    else:
        frame = frame[frame.shape[0] // 2:, :]

    eyes = coarse_find(frame)
    if len(eyes) > 0:
        prev_eyes = eyes.copy()
    elif prev_eyes is not None:
        eyes = prev_eyes
    else:
        continue

    eye_gray, x, y, size = process_eye_crop(frame, eyes)
    image = cv2.resize(eye_gray, (128, 128)) / 255.0
    image = np.reshape(image, (1, 128, 128, 1)).astype(np.float32)

    px = model.predict(image, verbose=0)[0] * size
    py = modelv.predict(image, verbose=0)[0] * size

    current = np.array([px, py], dtype=np.float32)
    if ema is None:
        ema = current
    else:
        ema = alpha * current + (1 - alpha) * ema

    ex, ey = ema

    # Draw tracking point and frame number
    cv2.circle(frame, (int(ex + x), int(ey + y)), 3, (200, 200, 100), -1)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show and save frame
    cv2.imshow("Smoothed Ellipse", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed {frame_idx} frames in {time.time() - start_time:.2f} seconds")
print(f"FPS: {frame_idx / (time.time() - start_time):.2f}")
