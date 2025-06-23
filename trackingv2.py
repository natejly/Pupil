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
        minSize=(100, 100)
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
    
def threshold_images(image, dark_point, thresholds=[15, 30, 45, 0]):
    images = []
    h, w = image.shape

    # 1) Pre-smooth the grayscale to reduce sensor noise
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    # small kernel for morphology
    kernel = np.ones((3, 3), np.uint8)

    for t in thresholds:
        # 2) Inverse binary threshold (dark ⇒ white)
        _, binary = cv2.threshold(
            denoised,
            dark_point + t,
            255,
            cv2.THRESH_BINARY_INV
        )

        # 3) Morphological opening to remove speckles
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4) Prepare mask for floodFill (2 px border)
        mask = np.zeros((h + 2, w + 2), np.uint8)

        flood = opened.copy()
        cv2.floodFill(flood, mask, (0, 0), 255)

        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(opened, flood_inv)

        images.append(filled)

    return images

def get_contours(images, min_area=100, margin=3):

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

if __name__ == "__main__":

    video_path = "videos/2.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prev_eyes = None
    start = time.time()
    start_time = time.time()
    prev_elipse = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:frame.shape[0] // 2, :]
        # frame = frame[frame.shape[0] // 2:, :]
        eyes = coarse_find(frame)
        # if can't find eyes use previous location 
        # TODO: can just use prev elipse so don't have to recalculate
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
        
        # plot dark square
        
# Threshold the image using multiple thresholds
        thresholded_images = threshold_images(eye_gray, dark_val)
        contours, contour_images = get_contours(thresholded_images)
        ellipse_images = []
        elipses = []
        for cnt_list in contours:
            # blank single-channel canvas
            temp_img = eye_gray.copy()

            if len(cnt_list) == 0:
                elipses.append(None)
                ellipse_images.append(temp_img)
                continue
            
                # fit ellipse to the largest contour
            box = fit_ellipse(cnt_list)
            cv2.ellipse(temp_img, box, 255, 2)
            elipses.append(box)

            ellipse_images.append(temp_img)

        N = len(thresholded_images) 
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
        percents = []
        for i in range(4):
            eye_thresh   = thresholded_images[i]
            ellipse      = elipses[i]
            if ellipse is None:
                percents.append(0)
                continue

            # build filled mask of the ellipse
            mask = np.zeros_like(eye_thresh)
            (cx, cy), (w, h), ang = ellipse
            cv2.ellipse(mask,
                        (int(cx), int(cy)),
                        (int(w/2), int(h/2)),
                        ang, 0, 360,
                        255, -1)

            white_pixels = cv2.countNonZero(cv2.bitwise_and(eye_thresh, mask))
            total_pixels = cv2.countNonZero(mask)
            coverage     = (white_pixels/total_pixels)*100 if total_pixels>0 else 0

            # simple shape metric: how close axes are
            roundness    = min(w, h)/max(w, h) if max(w,h)>0 else 0

            percents.append(coverage + roundness*500)

        # pick best
        if not any(elipses):
            # use previous if no valid ellipse found
            if prev_elipse is not None:
                best_ellipse, prev_x, prev_y = prev_elipse
                x, y = prev_x, prev_y


        best_idx     = int(np.argmax(percents))
        best_ellipse = elipses[best_idx]

        # fallback to previous if current is still None
        if best_ellipse is None:
            if prev_ellipse is not None:
                best_ellipse, prev_x, prev_y = prev_ellipse
                x, y = prev_x, prev_y
            else:
                print("No valid ellipse yet")
                continue
        else:
            # remember for next frame
            prev_ellipse = (best_ellipse, x, y)

        # shift ellipse back into full‐frame coords
        (cx, cy), (w, h), ang = best_ellipse
        full_ellipse = (
            (cx + x, cy + y),
            (w, h),
            ang
        )

    

        cv2.ellipse(frame, full_ellipse, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 0, 0), 2)
        cv2.imshow("Eye Tracking", frame)

        frame_idx += 1
        prev_elipse = (full_ellipse, x, y)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = time.time()
    print(f"Processed {frame_idx} frames in {end_time - start_time:.2f} seconds.")
    print(f"Average FPS: {frame_idx / (end_time - start_time):.2f}")
    cap.release()
    cv2.destroyAllWindows()