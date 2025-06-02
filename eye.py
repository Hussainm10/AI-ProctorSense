import cv2
import dlib
import numpy as np
from collections import deque

# Load dlib’s face detector and 68 landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\CheatingDetectionSystem\models\shape_predictor_68_face_landmarks.dat")

GAZE_BUFFER_SIZE = 5
gaze_history = deque(maxlen=GAZE_BUFFER_SIZE)

def detect_pupil(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    gray_eye = cv2.equalizeHist(gray_eye)

    # Blur to reduce noise
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)

    # Adaptive threshold for better lighting robustness
    threshold_eye = cv2.adaptiveThreshold(
        blurred_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 5
    )

    # Find contours
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter out very small contours
        pupil_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(pupil_contour) < 20:
            return None, None
        px, py, pw, ph = cv2.boundingRect(pupil_contour)
        return (px + pw // 2, py + ph // 2), (px, py, pw, ph)
    
    return None, None

def get_eye_region(frame, eye_points):
    x, y, w, h = cv2.boundingRect(eye_points)
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, w_frame), min(y + h, h_frame)
    return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def classify_gaze(left_pupil, right_pupil, left_rect, right_rect):
    if left_pupil is None or right_pupil is None:
        return "No eyes detected"

    lx, ly = left_pupil
    rx, ry = right_pupil

    lw = left_rect[2]
    lh = left_rect[3]
    rw = right_rect[2]
    rh = right_rect[3]

    norm_ly = ly / lh
    norm_ry = ry / rh

    if lx < lw / 3 and rx < rw / 3:
        return "Looking Left"
    elif lx > 2 * lw / 3 and rx > 2 * rw / 3:
        return "Looking Right"
    elif norm_ly < 0.3 and norm_ry < 0.3:
        return "Looking Up"
    elif norm_ly > 0.6 and norm_ry > 0.6:
        return "Looking Down"
    else:
        return "Looking Center"

def smooth_gaze(gaze):
    gaze_history.append(gaze)
    return max(set(gaze_history), key=gaze_history.count)

def process_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_direction = "No face detected"

    if len(faces) == 0:
        gaze_history.clear()
        return frame, gaze_direction

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

        left_eye, left_rect = get_eye_region(frame, left_eye_points)
        right_eye, right_rect = get_eye_region(frame, right_eye_points)

        left_pupil, left_bbox = detect_pupil(left_eye)
        right_pupil, right_bbox = detect_pupil(right_eye)

        # Draw eye bounding boxes
        x, y, w, h = left_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x, y, w, h = right_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw red pupil indicators
        if left_pupil:
            lx, ly = left_rect[0] + left_pupil[0], left_rect[1] + left_pupil[1]
            cv2.circle(frame, (lx, ly), 6, (0, 0, 255), -1)
        if right_pupil:
            rx, ry = right_rect[0] + right_pupil[0], right_rect[1] + right_pupil[1]
            cv2.circle(frame, (rx, ry), 6, (0, 0, 255), -1)

        gaze_direction = classify_gaze(left_pupil, right_pupil, left_rect, right_rect)
        gaze_direction = smooth_gaze(gaze_direction)
        break

    # Changed position to bottom left, smaller font, keep white color
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, gaze_direction
