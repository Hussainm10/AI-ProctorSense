import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

MAX_STRIKES = 3
STRIKE_TIMEOUT = 10  # seconds

hands_strikes = 0
hands_last_bad_time = None
hands_alert_displayed = False
FONT_SCALE = 0.5  # smaller font
FONT_THICKNESS = 1

# Finger connections based on mediapipe hand landmarks to draw grids
FINGER_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def detect_hands_visibility(frame):
    global hands_strikes, hands_last_bad_time, hands_alert_displayed

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hands_last_bad_time = None
        hands_alert_displayed = False

        # Draw linear white grids connecting finger joints
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for connection in FINGER_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_pos, end_pos, (255, 255, 255), 1)  # White thin lines

    else:
        if hands_last_bad_time is None:
            hands_last_bad_time = time.time()
        elif time.time() - hands_last_bad_time > STRIKE_TIMEOUT:
            if hands_strikes < MAX_STRIKES:
                hands_strikes += 1
                hands_last_bad_time = None
                hands_alert_displayed = False

    # Display warnings bottom-left corner, smaller font, red color
    pos_y = frame.shape[0] - 60  # a bit above head warning
    line_height = 20

    if 0 < hands_strikes < MAX_STRIKES:
        cv2.putText(frame, f"Hands not visible! {MAX_STRIKES - hands_strikes} strikes left",
                    (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
    elif hands_strikes >= MAX_STRIKES:
        cv2.putText(frame, "ALERT! Hands strikes breached. Exam cancelled.",
                    (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
        if not hands_alert_displayed:
            print("ALERT! Hands not visible for too long. Exam cancelled.")
            hands_alert_displayed = True

    return frame
