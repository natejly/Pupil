import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
import matplotlib.pyplot as plt
mask = np.zeros((128, 128), dtype=np.uint8)


#
def coarse_find(frame):
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(100, 100)
    )
    return eyes

def longest_nonzero_segment(arr):
    incline_thresh = 2000
    n = len(arr)
    if n < 3:
        return (None, None)
    arr = np.asarray(arr, dtype=np.int64)

    # 1) Locate first large incline
    incline = 0
    for i in range(1, n-1):
        if arr[i+1] - arr[i] > incline_thresh:
            incline = i
            break
    else:
        incline = 0

    # 2) Search for longest non-zero segment in arr[valley:]
    start = end = 0
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

    # Handle case where array ends in a non-zero segment
    if temp_start is not None:
        length = n - temp_start
        if length > max_len:
            start, end = temp_start, n - 1

    return (start, end, incline)

    

if __name__ == "__main__":
    # get first arg
    if len(sys.argv) != 3:
        print("Usage: python pupil.py <video_path> <-r/-c>")
    video_path = sys.argv[1]
    if sys.argv[2] == "r":
        clean = False
    elif sys.argv[2] == "c":
        clean = True
    # video_path = f"videos/{video_path}.mp4"
    if not os.path.exists(video_path):
        print(f"Video {video_path} does not exist")
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/eye_data.csv", "w") as f:
        f.write("frame_idx,x,y,w,h,ellipse_x,ellipse_y,ellipse_w,ellipse_h,ellipse_angle\n")

    frame_idx = 0
    prev_eyes = None
    start = time.time()
    prev_elipse = None
    start_time = time.time()
    while True:
        good_frame = True
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
        eye_crop = frame[y:y+size, x:x+size]
        plt.imshow(eye_crop)
        images = []
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        to_save = eye_gray.copy()
        thresholds = [45, 55, 60, 70]  # Low, Medium, High
        ellipses = []
        mask_images = []
        for i in range(4):
            image = eye_gray.copy()
            _, eye_thresh = cv2.threshold(image, thresholds[i], 100, cv2.THRESH_BINARY_INV)

            h, w = eye_thresh.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            flood = eye_thresh.copy()
            cv2.floodFill(flood, mask, (0, 0), 255)
            flood_inv = cv2.bitwise_not(flood)
            eye_thresh_filled = cv2.bitwise_or(eye_thresh, flood_inv)
            eye_thresh_filled = cv2.morphologyEx(eye_thresh_filled, cv2.MORPH_CLOSE, np.zeros((3, 3), np.uint8))
            eye_thresh_filled = cv2.morphologyEx(eye_thresh_filled, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            x_hist = np.sum(eye_thresh_filled, axis=0)

            # Find x-coordinate of the peak
            middle = x_hist[np.argmax(x_hist)]/3
            # find places where the histogram is above the middle value
            above = np.where(x_hist > middle, x_hist, 0)

            # Find the longest segment above the middle value
            start, end, incline = longest_nonzero_segment(above)

            middle = x_hist[np.argmax(x_hist[incline:])] / 2
            above = np.where(x_hist > middle, x_hist, 0)
            descend_step = 100
            # Extend start and end based on slope
            while start > 0 and x_hist[start-1] < x_hist[start] - descend_step:
                start -= 2
            while end < len(x_hist) - 1 and x_hist[end+1] < x_hist[end] - descend_step:
                end += 2
            
            center = (start + end) // 2
            delta = end - center
            if delta == 0:
                delta = 1
            # check not out of bounds
            start -= int((center-start)*100/delta)
            end += int((end-center)*100/delta)
            start = max(0, start)
            end = min(len(x_hist) - 1, end)

            eye_thresh_filled[:, :start] = 0
            eye_thresh_filled[:, end:] = 0


            mask_images.append(eye_thresh_filled)
            contours, _ = cv2.findContours(eye_thresh_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            margin = 3
            h_crop, w_crop = eye_crop.shape[:2]

            cleaned_contours = []
            for cnt in contours:
                # Filter out points near the edge
                kept_points = [pt for pt in cnt if
                            margin < pt[0][0] < (w_crop - margin) and
                            margin < pt[0][1] < (h_crop - margin)]
                
                if len(kept_points) >= 3:  # Need at least 3 points for a valid contour
                    cleaned_contours.append(np.array(kept_points).reshape(-1, 1, 2))

            contours = cleaned_contours
            # take only the largest contour
            if len(contours) > 0:
                contours = [max(contours, key=cv2.contourArea)]
            #convert contours to concave hulls
            contours = [cv2.convexHull(cnt) for cnt in contours]
            if len(contours) == 0:
                ellipses.append(None)
                print("no eyes", frame_idx)
                continue
            # get largest contour
            if len(contours) > 0:
                best = max(contours, key=cv2.contourArea)
            if len(best) < 5:
                ellipses.append(None)
                print("no eyes", frame_idx)
                continue
            ellipse = cv2.fitEllipse(best)
            # check that elipse center is in frame
            if not (0 < ellipse[0][0] < w and 0 < ellipse[0][1] < h):
                ellipses.append(None)
                print("no eyes", frame_idx)
                continue
            ellipses.append(ellipse)
        
        percents = []
        for i in range(4):
            eye_thresh = mask_images[i]
            ellipse_mask = np.zeros_like(eye_thresh)
            
            if ellipses[i]:
                ellipse = ellipses[i]
                if ellipse is None:
                    percents.append(0)
                    continue
                
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
                angle = int(ellipse[2])
                
                cv2.ellipse(ellipse_mask, center, axes, angle, 0, 360, 255, -1)

                white_pixels = cv2.countNonZero(cv2.bitwise_and(eye_thresh, ellipse_mask))
                total_pixels = cv2.countNonZero(ellipse_mask)
                percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                roundness = axes[0] / axes[1] if axes[1] != 0 else 0
                ellipse_ratio = ellipse[1][0] / ellipse[1][1] if ellipse[1][1] != 0 else 0

                percents.append(percentage + roundness*100)
            else:
                percents.append(0)

        best_index = np.argmax(percents)
        best_ellipse = ellipses[best_index]
        prev_elipse_area = prev_elipse[1][0] * prev_elipse[1][1] if prev_elipse else 0
        best_ellipse_area = best_ellipse[1][0] * best_ellipse[1][1] if best_ellipse else 0

        if best_ellipse is None:
            best_ellipse = prev_elipse
        else:
            prev_elipse = best_ellipse

        # calculate distance from center of prev ellipse to center of best ellipse
        if prev_elipse is not None:
            prev_center = (int(prev_elipse[0][0]), int(prev_elipse[0][1]))
            best_center = (int(best_ellipse[0][0]), int(best_ellipse[0][1]))
            distance = np.linalg.norm(np.array(prev_center) - np.array(best_center))
            if distance > best_ellipse[1][0]:
                best_ellipse = prev_elipse
                good_frame = False
                if clean:
                    frame_idx += 1
                    continue    

        if best_ellipse_area < prev_elipse_area*.95 or best_ellipse_area > prev_elipse_area*1.1:
            best_ellipse = prev_elipse
            good_frame = False
            if clean:
                frame_idx += 1
                continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)  # draw ellipse using value 1
        mean_val = image[mask == 1].mean()
        if mean_val > 100:
            print("blink detected", frame_idx)
            good_frame = False
            if clean:
                frame_idx += 1
                continue

        # average best and prev ellipse
        multiplier = .25
        if prev_elipse is not None:
            best_ellipse = (
                ((best_ellipse[0][0] + prev_elipse[0][0]*multiplier) / (1+multiplier), (best_ellipse[0][1] + prev_elipse[0][1]*multiplier) / (1+multiplier)),
                ((best_ellipse[1][0] + prev_elipse[1][0]*multiplier) / (1+multiplier), (best_ellipse[1][1] + prev_elipse[1][1]*multiplier) / (1+multiplier)),
                (best_ellipse[2] + prev_elipse[2]*multiplier) / (1+multiplier)
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
    cap.release()
    cv2.destroyAllWindows()