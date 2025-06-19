import cv2

cap = cv2.VideoCapture("videos/2.mp4")
if not cap.isOpened():
    raise IOError("Cannot open video")

# Try to get the frame count directly:
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Fallback: manually count if the property is unreliable:
if total <= 0:
    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

print(f"Total frames: {total}")
cap.release()