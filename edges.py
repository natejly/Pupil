import cv2
import numpy as np


def coarse_find(frame):
    """Haar cascade to find cropping bounds for eyes."""
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50)
    )
    return eyes  # just return whatever we got (possibly empty)

def edge_detection(eye_crop):
    """Edge detection using Canny."""
    edges = cv2.Canny(eye_crop, 50, 60, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    filtered_contours = [contour for contour in contours if len(contour) >= 15 and len(contour) <= 500]
    return filtered_contours

def get_curves(contours, aspect_threshold=10, min_length=30):
    """Uses rotating calipersto filter out linear contours."""
    kept = []
    for cnt in contours:
        # Need at least 2 points to form a line
        if len(cnt) < 2:
            continue

        # Compute the min-area rectangle (center, (w,h), angle)
        rect = cv2.minAreaRect(cnt) 
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / short_side

        # Check linearity: if very “long and skinny,” drop it
        if aspect < aspect_threshold and long_side >= min_length:
            kept.append(cnt)

    return kept

def convex_hull(contours):
    """Compute convex hulls for the contours."""
    hulls = []
    for cnt in contours:
        if len(cnt) >= 3:  # Need at least 3 points to form a hull
            hull = cv2.convexHull(cnt)
            hulls.append(hull)
    return hulls

if __name__ == "__main__":
    path = "videos/2.mp4"
    cap = cv2.VideoCapture(path)
    prev_eyes = None 
    num = 0

    while True:
        ret, frame = cap.read()
        cv2.imwrite(f"output/raw_{num:04d}.png", frame)

        if not ret:
            break

        h_full, _ = frame.shape[:2]
        frame = frame[: h_full // 2, :]
        # binarize the frame to black and white
        

        eyes = coarse_find(frame)

        if len(eyes) > 0:
            prev_eyes = eyes.copy()  
        elif prev_eyes is not None:
            eyes = prev_eyes         
        else:
            continue

        x, y, w, h = eyes[0] 
        eye_crop = frame[y : y + h, x : x + w]
        eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        contours = edge_detection(eye_crop)
        # remove contours that are close to the edge of the crop
        contours = [cnt for cnt in contours if not np.any(cnt < 20) and not np.any(cnt > np.array([w - 20, h - 20]))]

        curves = get_curves(contours)

        #draw cobntours and cirves in frames next to each other
        if len(curves) == 0:
            continue
        #draw contours on full pictures
        masked = eye_crop.copy()
        orig_vis = frame.copy()
        filt_vis = frame.copy()

        _, masked = cv2.threshold(masked, 50, 100, cv2
        .THRESH_TOZERO)
        # 2) Draw all raw contours on orig_vis
        for cnt in contours:
            rand_vals = np.random.randint(0, 256, size=3)
            color = tuple(int(c) for c in rand_vals)
            cv2.drawContours(orig_vis, [cnt + (x, y)], -1, color, 1)

        # 3) Draw only the “curves” on filt_vis
        for cnt in curves:
            rand_vals = np.random.randint(0, 256, size=3)
            color = tuple(int(c) for c in rand_vals)
            cv2.drawContours(filt_vis, [cnt + (x, y)], -1, color, 1)

        # 4) Concatenate side by side
        #    (both must be same height)
        comparison = cv2.hconcat([ orig_vis, filt_vis])
        # 5) Show the combined image
        cv2.imshow("Raw vs Filtered Contours", comparison)
        # save the comparison image
        cv2.imwrite(f"output/frame_{num:04d}.png", comparison)
        num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()