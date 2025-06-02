import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Parameters
MAX_STRIKES = 3
STRIKE_TIMEOUT = 10  # seconds
head_strikes = 0
head_last_bad_time = None
head_alert_displayed = False
FONT_SCALE = 0.5  # smaller font as requested
FONT_THICKNESS = 1

def detect_head_direction(frame):
    global head_strikes, head_last_bad_time, head_alert_displayed

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        nose = lm[1]
        x_center = int(nose.x * w)

        center_margin = w * 0.1  # ~10% margin around center

        if x_center < w / 2 - center_margin or x_center > w / 2 + center_margin:
            if head_last_bad_time is None:
                head_last_bad_time = time.time()
            elif time.time() - head_last_bad_time > STRIKE_TIMEOUT:
                if head_strikes < MAX_STRIKES:
                    head_strikes += 1
                    head_last_bad_time = None
                    if not head_alert_displayed:
                        print("ALERT! Head turned too long.")
                        head_alert_displayed = True
        else:
            head_last_bad_time = None
            head_alert_displayed = False
    else:
        head_last_bad_time = None
        head_alert_displayed = False

    # Display warnings bottom-left corner, smaller font, red color
    pos_y = frame.shape[0] - 90  # start near bottom left
    line_height = 20

    if 0 < head_strikes < MAX_STRIKES:
        cv2.putText(frame, f"Head misaligned! {MAX_STRIKES - head_strikes} strikes left",
                    (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
    elif head_strikes >= MAX_STRIKES:
        cv2.putText(frame, "ALERT! Head strikes breached. Exam cancelled.",
                    (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

    return frame
