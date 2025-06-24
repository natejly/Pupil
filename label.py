import cv2
import numpy as np
import time
import os
import sys
import math
from math import pi
from trackingv2 import (coarse_find, remove_bright_spots, find_dark_area, threshold_images, 
                        get_contours, fit_ellipse, check_flip, prepare_frame, process_eye_crop, 
                        generate_ellipse_candidates, calculate_ellipse_scores, select_best_ellipse, 
                        apply_smoothing, display_results)

if __name__ == "__main__":
    # get first arg

    frames_folder = "frames"
    center_alpha = 1
    size_alpha = 1
    rotation_alpha = 1
    prev_array = []
    prev_ellipse = None
    ema = None

    # thresholds=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    # brute force this for now 
    thresholds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]
    TOP = False
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

        eye_gray, x, y, size = process_eye_crop(frame, eyes)
        dark_square, dark_val = find_dark_area(eye_gray)
        
        thresholded_images, contour_images, ellipse_images, ellipses = generate_ellipse_candidates(
            eye_gray, dark_val, thresholds)
        
        percents = calculate_ellipse_scores(thresholded_images, ellipses)
        
        best_ellipse, x, y = select_best_ellipse(ellipses, percents, prev_ellipse, x, y, frame_idx)
        
        if best_ellipse is None:
            continue
            
        prev_ellipse = (best_ellipse, x, y)
        
        full_ellipse, ema = apply_smoothing(best_ellipse, x, y, ema, 
                                          center_alpha, size_alpha, rotation_alpha)
        
        (cx, cy), (w, h), ang = best_ellipse
        
        display_results(frame, thresholded_images, contour_images, ellipse_images,
                       full_ellipse, cx, cy, x, y, frame_idx)

        frame_idx += 1
        prev_array.append(best_ellipse)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # save video
    end_time = time.time()
    print(f"FPS: {frame_idx / (end_time - start_time):.2f}")
    cv2.destroyAllWindows()