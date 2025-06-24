import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
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
def longest_nonzero_segment(arr):
    start = end = max_len = 0
    temp_start = None

    for i, val in enumerate(arr):
        if val != 0:
            if temp_start is None:
                temp_start = i
        else:
            if temp_start is not None:
                length = i - temp_start
                if length > max_len:
                    max_len = length
                    start = temp_start
                    end = i - 1
                temp_start = None

    if temp_start is not None:
        length = len(arr) - temp_start
        if length > max_len:
            start = temp_start
            end = len(arr) - 1

    return start, end
def longest_nonzero_segment(arr):
    incline_thresh = 2000
    n = len(arr)
    if n < 3:
        return (None, None)
    arr = np.asarray(arr, dtype=np.int64)

    incline = 0
    for i in range(1, n-1):
        if arr[i+1] - arr[i] > incline_thresh:
            incline = i
            break
    else:
        incline = 0

    start = 0
    end = len(arr) - 1
    max_len = 0
    temp_start = None

    for i in range(incline, n):
        if arr[i] != 0:
            if temp_start is None:
                temp_start = i
        else:
            if temp_start is not None:
                length = i - temp_start
                if length > max_len:
                    max_len = length
                    start, end = temp_start, i - 1
                temp_start = None

    if temp_start is not None:
        length = n - temp_start
        if length > max_len:
            start, end = temp_start, n - 1

    return (start, end)
    

if __name__ == "__main__":
    # get first arg

    frames_folder = "frames"
    center_alpha = .9
    size_alpha = .25
    rotation_alpha = 1
    prev_array = []
    prev_ellipse = None
    ema = None

    # thresholds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    # brute force this for now 
    thresholds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    TOP = False
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prev_eyes = None
    start_time = time.time()
# loop through images in frames_folder

    if not os.path.exists("frames"):
        print("No frames found in 'frames' folder. Please run 'processVideo.py' first.")
        sys.exit(1)
    # number of frames in frames_folder
    num_frames = len([f for f in os.listdir(frames_folder) if f.endswith('.png')])
    for i in range(num_frames):

        frame_path = os.path.join(frames_folder, f"{i}.png")
        frame = cv2.imread(frame_path)
        eyes = coarse_find(frame)

         if len(eyes) > 0:
            prev_eyes = eyes.copy()
        elif prev_eyes is not None:
            eyes = prev_eyes
        else:
            continue
        x, y, w, h = eyes[0]
        size = max(w, h)
        eye_crop = frame[y:y+size, x:x+size].copy()
        eye_crop = remove_bright_spots(eye_crop, threshold=220, replace=100)
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        dark_square, dark_val = find_dark_area(eye_gray)
        thresholded_images = threshold_images(eye_gray, dark_val, thresholds=thresholds)
        contours, contour_images = get_contours(thresholded_images)
        ellipse_images = []
        ellipses = []
        for cnt_list in contours:
            # blank single-channel canvas
            temp_img = eye_gray.copy()

            if len(cnt_list) == 0:
                ellipses.append(None)
                ellipse_images.append(temp_img)
                continue
            
            box = fit_ellipse(cnt_list)
            cv2.ellipse(temp_img, box, 255, 2)
            ellipses.append(box)

            ellipse_images.append(temp_img)

        N = len(thresholded_images) 
        percents = []
        for i in range(N):
            eye_thresh = thresholded_images[i]
            ellipse     = ellipses[i]
            if ellipse is None:
                percents.append(0)
                continue
            # get height to width ratio
            ellipse_ratio = ellipse[1][1] / ellipse[1][0]
            # if ratio is too high, skip
            if ellipse_ratio > 1.75 or ellipse_ratio < 0.8:
                percents.append(0)
                continue

            mask = np.zeros_like(eye_thresh)
            (cx, cy), (w, h), ang = ellipse
            cv2.ellipse(mask,
                        (int(cx), int(cy)),
                        (int(w/2), int(h/2)),
                        ang, 0, 360,
                        255, -1)

            inside_total  = cv2.countNonZero(mask)
            inside_white  = cv2.countNonZero(cv2.bitwise_and(eye_thresh, mask))
            inside_ratio  = inside_white / inside_total if inside_total>0 else 0

            outside_mask  = cv2.bitwise_not(mask)
            outside_total = cv2.countNonZero(outside_mask)
            # invert thresholded so white→0, black→255, then AND
            outside_black = cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_not(eye_thresh), outside_mask))
            outside_ratio = outside_black / outside_total if outside_total>0 else 0

            percent = ((inside_ratio + outside_ratio*.25) / 1.5)
            roundness = 1.0 - abs(w - h) / max(w, h)
            percents.append(percent + roundness*0)

        best_idx     = int(np.argmax(percents))
        best_ellipse = ellipses[best_idx]

        # fallback to previous if current is still None
        if best_ellipse is None:
            if prev_ellipse is not None:
                print("Using previous ellipse")
                best_ellipse, prev_x, prev_y = prev_ellipse
                x, y = prev_x, prev_y
            else:
                print("No valid ellipse yet")
                continue
        if prev_ellipse is not None:
            (pcx, pcy), (pw, ph), pang = prev_ellipse[0]
            (cx, cy), (w, h), ang = best_ellipse
            # check tthat ellipse is not telleporting
            if abs(cy - pcy) > 100 or abs(cx - pcx) > 100:
                print(f"Teleporting detected, using previous ellipse{frame_idx}")
                best_ellipse = prev_ellipse[0]
                x, y = prev_ellipse[1], prev_ellipse[2]
            # check that current area is withing 75% of previous size
            elif (w * h) < 0.3 * (pw * ph):
                print(f"Current ellipse too small, using previous ellipse{frame_idx}")
                best_ellipse = prev_ellipse[0]
                x, y = prev_ellipse[1], prev_ellipse[2]
        # if current
        prev_ellipse = (best_ellipse, x, y)
        # shift ellipse back into full‐frame coords
        best_ellipse = check_flip(best_ellipse)
        (cx, cy), (w, h), ang = best_ellipse

        # full_ellipse = project_ellipse(full_ellipse)
        # past 3 elipses array
    
        alphas = np.array([
            center_alpha,   # cx
            center_alpha,   # cy
            size_alpha,     # w
            size_alpha,     # h
            rotation_alpha  # ang
        ], dtype=np.float32)

        # …inside your loop, once you have (cx,cy),(w,h),ang and the crop offset (x,y):
        current = np.array([cx + x, cy + y, w, h, ang], dtype=np.float32)

        if ema is None:
            ema = current.copy()
        else:
            # elementwise EMA with three different alphas
            ema = alphas * current + (1.0 - alphas) * ema

        # unpack smoothed values
        sm_cx, sm_cy, sm_w, sm_h, sm_ang = ema

        full_ellipse = (
            (float(sm_cx), float(sm_cy)),
            (float(sm_w),  float(sm_h)),
            float(sm_ang)
        )

        






        cv2.ellipse(eye_crop, best_ellipse, (200, 190, 140), 2)
        cv2.circle(eye_crop, (int(best_ellipse[0][0]), int(best_ellipse[0][1])), 2, (255, 200, 100), -1)
        prev_elipse = best_ellipse
        

        frame[y:y+h, x:x+w] = eye_crop

        cv2.imshow("Processed Frame", frame)
        # save the frame
        if good_frame:
            cv2.imwrite(f"output/{frame_idx}.png", to_save)
            # save frame index, eye coordinates, and ellipse parameters
            # createt eye_data.txt if it doesn't exist
            cv2.imwrite(f"output/eye_{frame_idx}.png", eye_crop)

            with open("output/eye_data.csv", "a") as f:
                f.write(f"{frame_idx},{x},{y},{w},{h},"
                        f"{best_ellipse[0][0]},{best_ellipse[0][1]},"
                        f"{best_ellipse[1][0]},{best_ellipse[1][1]},{best_ellipse[2]}\n")
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # save video
    end_time = time.time()
    print(f"FPS: {frame_idx / (end_time - start_time):.2f}")
    cv2.destroyAllWindows()