# demo.py
import os
import sys
import time
import cv2
import torch
import numpy as np
from collections import deque
from src.config import (
    DEVICE, SAVE_PATH, IMAGENET_MEAN, IMAGENET_STD,
    TARGET_FRAMES, IMG_SIZE, NUM_CLASSES, LABEL_MAP_PATH,
    SEED
)
from src.model import TimeSformerClassifier
from src.utils import seed_everything
import pickle

# ------------------- Cấu hình cho phần "đứng yên để dự đoán" -------------------
MOTION_THRESHOLD   = 4.0   # ngưỡng chênh lệch trung bình giữa 2 frame liên tiếp (0-255)
STILLNESS_DURATION = 0.8   # số giây đứng yên liên tục thì mới trigger dự đoán
MIN_CAPTURE_TIME   = 0.5   # thời gian tối thiểu (giây) trước khi cho phép trigger,
                            # tránh trigger ngay lập tức nếu người dùng chưa kịp cử động

# ------------------- Kiểm tra model và train nếu cần -------------------
if not os.path.exists(SAVE_PATH):
    print(f"Không tìm thấy {SAVE_PATH}, tiến hành train lại...")
    os.environ['WANDB_MODE'] = 'disabled'
    from scripts.train import main as train_main
    train_main()
    print("Train hoàn tất, tiếp tục demo.")

# ------------------- Load model -------------------
seed_everything(SEED)
model = TimeSformerClassifier(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from {SAVE_PATH}")

# ------------------- Load label mapping -------------------
with open(LABEL_MAP_PATH, 'rb') as f:
    label_mapping = pickle.load(f)
idx_to_label = {v: k for k, v in label_mapping.items()}

# ------------------- Tiền xử lý khung hình -------------------
def preprocess_frames(frames_list):
    processed = []
    for frame in frames_list:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        processed.append(frame_resized)
    frames = np.stack(processed, axis=0)
    frames = torch.from_numpy(frames).float()
    frames = frames.permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames.unsqueeze(0).to(DEVICE)

# ------------------- Dự đoán -------------------
@torch.no_grad()
def predict(model, frames_tensor):
    outputs = model(frames_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_idx = outputs.argmax(1).item()
    confidence = probs[0, pred_idx].item()
    label = idx_to_label[pred_idx]
    return label, confidence

# ------------------- Vẽ overlay text -------------------
def draw_banner(display, lines, color=(0, 255, 0)):
    h, w, _ = display.shape
    banner_h = 40 + 35 * len(lines)
    cv2.rectangle(display, (10, 10), (w - 10, 10 + banner_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (20, 45 + 35 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ------------------- Demo webcam -------------------
STATE_IDLE = 'idle'
STATE_CAPTURING = 'capturing'
STATE_RESULT = 'result'

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    state = STATE_IDLE
    buffer = deque(maxlen=TARGET_FRAMES)
    prev_gray = None
    start_time = 0.0
    last_motion_time = 0.0
    pred_label = ""
    confidence = 0.0

    print("Nhấn 'q' để bắt đầu dự đoán, 'r' để làm lại, 'e' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame.")
            break

        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        # ---- Thoát chương trình ----
        if key == ord('e'):
            break

        if state == STATE_IDLE:
            draw_banner(display, ["Nhan 'q' de bat dau dự đoan"], color=(255, 255, 0))
            if key == ord('q'):
                state = STATE_CAPTURING
                buffer.clear()
                prev_gray = None
                start_time = time.time()
                last_motion_time = time.time()

        elif state == STATE_CAPTURING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = float(np.mean(diff))
                if motion_score > MOTION_THRESHOLD:
                    last_motion_time = time.time()
            prev_gray = gray

            buffer.append(frame.copy())

            now = time.time()
            elapsed_since_start = now - start_time
            still_duration = now - last_motion_time

            draw_banner(display, [
                "Dang quan sat... hay thuc hien ky hieu",
                f"Dung yen: {still_duration:.1f}s / {STILLNESS_DURATION:.1f}s"
            ], color=(0, 255, 255))

            # Điều kiện trigger dự đoán: đủ frame + qua thời gian tối thiểu + đứng yên đủ lâu
            if (len(buffer) == TARGET_FRAMES
                    and elapsed_since_start >= MIN_CAPTURE_TIME
                    and still_duration >= STILLNESS_DURATION):
                frames_tensor = preprocess_frames(list(buffer))
                pred_label, confidence = predict(model, frames_tensor)
                state = STATE_RESULT

        elif state == STATE_RESULT:
            draw_banner(display, [
                f"Ket qua: {pred_label} ({confidence*100:.1f}%)",
                "Nhan 'r' de du doan lai, 'e' de thoat"
            ], color=(0, 255, 0))

            if key == ord('r'):
                state = STATE_CAPTURING
                buffer.clear()
                prev_gray = None
                start_time = time.time()
                last_motion_time = time.time()

        cv2.imshow("Sign Language Demo", display)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()