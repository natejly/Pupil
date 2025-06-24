import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
import matplotlib.pyplot as plt
mask = np.zeros((128, 128), dtype=np.uint8)

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

def remove_bright_spots(image, threshold=200, replace=0):
    """
    Remove bright spots from the image by setting pixels above a certain threshold to black.
    """
    # Create a mask where pixels are below the threshold
    mask = image < threshold
    # Set bright pixels to black
    image[~mask] = replace
    return image
def find_dark_area(image):
    num_grids = 9
    h, w = image.shape[:2]
    grid_h = h // num_grids
    grid_w = w // num_grids 
    darkest_val = 255
    darkest_square = None
    for i in range(num_grids):
        for j in range(num_grids):
            grid = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            mean_val = np.mean(grid)
            if mean_val < darkest_val:
                darkest_val = mean_val
                darkest_square = (i*grid_h, j*grid_w, grid_h, grid_w)
    return darkest_square, darkest_val
    
def threshold_images(image, dark_point, thresholds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50]):
    images = []
    h, w = image.shape
    denoised = cv2.GaussianBlur(image, (5, 5), 0)   
    kernel = np.ones((3, 3), np.uint8)

    for t in thresholds:
        _, binary = cv2.threshold(
            denoised,
            dark_point + t,
            255,
            cv2.THRESH_BINARY_INV
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        mask = np.zeros((h + 2, w + 2), np.uint8)

        flood = opened.copy()
        cv2.floodFill(flood, mask, (0, 0), 255)

        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(opened, flood_inv)

        images.append(filled)

    return images

def get_contours(images, min_area=1500, margin=3):

    filtered_contours = []
    contour_images    = []

    for img in images:
        h, w = img.shape[:2]
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept = []
        for cnt in cnts:
            all_pts = cnt.reshape(-1, 2)
            if cv2.contourArea(cnt) < min_area:
                continue
            pts = cnt.reshape(-1, 2)
            # skip if any point is too close to the border
            if (pts[:,0] < margin).any() or (pts[:,0] > w - margin).any() \
            or (pts[:,1] < margin).any() or (pts[:,1] > h - margin).any():
                continue
            kept.append(cnt)
        # keep two largest contours
        kept = sorted(kept, key=cv2.contourArea, reverse=True)[:1]
        # combine all contours into one
        if len(kept) > 0:
            all_pts = np.vstack([c.reshape(-1,2) for c in kept])
            all_pts = all_pts.reshape(-1,1,2).astype(np.int32)
        else:
            all_pts = np.array([], dtype=np.int32).reshape(-1,1,2)
        hull = cv2.convexHull(all_pts)

        # draw kept contours onto a blank image
        ci = np.zeros_like(img)
        cv2.drawContours(ci, hull, -1, 255, 2)
        filtered_contours.append(kept)
        contour_images.append(ci)

        

    return filtered_contours, contour_images

def fit_ellipse(contour, bias_factor=-1):
    pts = np.vstack([c.reshape(-1,2) for c in contour])
    
    # 2) Compute the halfway‐down (mean Y) and pick bottom half
    mean_y     = np.mean(pts[:,1])
    bottom_pts = pts[pts[:,1] > mean_y]
    
    if bottom_pts.size and bias_factor > 0:
        weighted_pts = np.concatenate(
            [pts] + [bottom_pts]*bias_factor,
            axis=0
        )
    else:
        weighted_pts = pts
    
    weighted_pts = weighted_pts.reshape(-1,1,2).astype(np.int32)
    # remove points above the mean Y
    # weighted_pts = weighted_pts[weighted_pts[:,0,1] > mean_y]
    if len(weighted_pts) < 5:
        return None
    return cv2.fitEllipse(weighted_pts)


def check_flip(ellipse):
    (cx, cy), (w, h), ang = ellipse

    if w < h:
        w, h = h, w
        ang += 90
    if ang >= 90:
        ang -= 180
    elif ang < -90:
        ang += 180

    return (cx, cy), (w, h), ang

if __name__ == "__main__":
    # for training set alpha to 0
    video_path = "videos/igor1.mp4"
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if TOP:
            frame = frame[:frame.shape[0] // 2, :]
        else:
            frame = frame[frame.shape[0] // 2:, :]
        eyes = coarse_find(frame)
        # if can't find eyes use previous location 

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
        H, W = thresholded_images[0].shape
        # 3 rows this time
        grid = np.zeros((3 * H, N * W), dtype=np.uint8)
        for i in range(N):
            grid[0   :  H, i*W:(i+1)*W] = thresholded_images[i]
            grid[H   :2*H, i*W:(i+1)*W] = contour_images[i]
            grid[2*H :3*H, i*W:(i+1)*W] = ellipse_images[i]
                    # resize for display if you like
        grid_disp = cv2.resize(grid, (1024, 512))  # just keep aspect
        cv2.imshow("Threshold | Contour | Ellipse", grid_disp)
        cv2.ellipse(frame, full_ellipse, (0, 255, 0), 2)
        # plot center point
        cv2.circle(frame, (int(cx+x), int(cy+y)), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 0, 0), 2)
        cv2.imshow("Eye Tracking", frame)

        frame_idx += 1
        prev_array.append(best_ellipse)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = time.time()
    print(f"Processed {frame_idx} frames in {end_time - start_time:.2f} seconds.")
    print(f"Average FPS: {frame_idx / (end_time - start_time):.2f}")
    cap.release()
    cv2.destroyAllWindows()