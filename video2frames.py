import os
import glob
import cv2

VIDEO_FOLDER  = "videos"
OUTPUT_FOLDER = "frames"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_paths = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4")))
if not video_paths:
    raise RuntimeError(f"No .mp4 files found in {VIDEO_FOLDER!r}")

global_idx = 0
h = 0
w = 0
h2 = 0
for vp in video_paths:
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print(f"⚠️ Skipping unreadable file: {vp}")
        continue

    print(f"Processing {vp} …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if global_idx == 0:
            h, w = frame.shape[:2]
            h2 = h // 2

        top   = frame[0:h2,    :]

        filename = f"{global_idx}.png"
        path     = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(path, top)
        global_idx += 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bottom = frame[h2:h,    :]

        filename = f"{global_idx}.png"
        path     = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(path, top)
        global_idx += 1

    cap.release()

print(f"✅ Saved {global_idx} frames into '{OUTPUT_FOLDER}/'")
