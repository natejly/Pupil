import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import numpy as np
import tensorflow as tf
from trackingv2 import coarse_find, process_eye_crop

# Load TFLite interpreters
interpreter_x = tf.lite.Interpreter(model_path="models/v1_small_lefts/eye_tracking_model.tflite")
interpreter_x.allocate_tensors()
input_details_x = interpreter_x.get_input_details()
output_details_x = interpreter_x.get_output_details()

interpreter_y = tf.lite.Interpreter(model_path="models/v1_small_lefts/eye_tracking_model.tflite")
interpreter_y.allocate_tensors()
input_details_y = interpreter_y.get_input_details()
output_details_y = interpreter_y.get_output_details()

print("TFLite models loaded.")

# Video capture
cap = cv2.VideoCapture("videos/2L.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

# Configuration
top_half = False  # change to True to track top half
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

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

    # Preprocess eye image
    eye_gray, x, y, size = process_eye_crop(frame, eyes)
    image = cv2.resize(eye_gray, (128, 128)) / 255.0
    image = np.reshape(image, (1, 128, 128, 1)).astype(np.float32)

    # Inference using TFLite (x coordinate)
    interpreter_x.set_tensor(input_details_x[0]['index'], image)
    interpreter_x.invoke()
    px = interpreter_x.get_tensor(output_details_x[0]['index'])[0][0] * size

    # Inference using TFLite (y coordinate)
    interpreter_y.set_tensor(input_details_y[0]['index'], image)
    interpreter_y.invoke()
    py = interpreter_y.get_tensor(output_details_y[0]['index'])[0][0] * size

    # TODO: work on smoothing idk why it makes it worse rn
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
