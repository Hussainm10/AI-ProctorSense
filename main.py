import cv2
import streamlit as st
import base64
from Head import detect_head_direction
from eye import process_eye_movement
from hands import detect_hands_visibility

# Load and encode background image
def get_base64_bg(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

bg_image = get_base64_bg("D:/CheatingDetectionSystem/background.png")

# Streamlit page config and styling
st.set_page_config(page_title="Proctor Sense", layout="centered", page_icon="🧠")

# Background and button styling CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        margin: 0;
        padding: 0;
    }}
    .css-18e3th9 {{
        background-color: transparent;
        padding: 0;
        margin: 0;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, #4a90e2, #63b8ff) !important;
        color: white !important;
        font-size: 20px !important;
        border-radius: 15px !important;
        padding: 1em 2.5em !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.5) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, #357abd, #4dabf7) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 7px 20px rgba(74, 144, 226, 0.7) !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Add vertical spacing to approximate vertical centering
st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)

# Use columns to center the button horizontally
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("📷 Start Proctoring")

def letterbox(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))
    top = (target_h - nh) // 2
    bottom = target_h - nh - top
    left = (target_w - nw) // 2
    right = target_w - nw - left
    color = [0, 0, 0]
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded

def run_cheating_detection():
    cap = cv2.VideoCapture(0)
    window_width, window_height = 1280, 720

    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame = detect_head_direction(frame)
        frame, _ = process_eye_movement(frame)
        frame = detect_hands_visibility(frame)

        text = "Press 'q' to quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        display_frame = letterbox(frame, window_width, window_height)
        cv2.imshow("Cheating Detection System", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Trigger the detection system on button click
if start_button:
    run_cheating_detection()
