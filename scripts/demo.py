# demo.py
import os
import time
import cv2 # type: ignore
import torch
import numpy as np
from src.config import (
    DEVICE, SAVE_PATH, IMAGENET_MEAN, IMAGENET_STD,
    TARGET_FRAMES, IMG_SIZE, NUM_CLASSES, LABEL_MAP_PATH,
    SEED
)
from src.model import TimeSformerClassifier
from src.utils import seed_everything
import pickle

# ------------------- wehcam mode -------------------
MOTION_THRESHOLD    = 4.0
STILLNESS_DURATION  = 0.8
MIN_RECORD_SECONDS  = 0.5
MAX_RECORD_SECONDS  = 6.0

CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480
DISPLAY_WIDTH  = 480
WINDOW_NAME    = "Sign Language Demo"

if not os.path.exists(SAVE_PATH):
    print(f"{SAVE_PATH} not found, training activated...")
    os.environ['WANDB_MODE'] = 'disabled'
    from scripts.train import main as train_main
    train_main()
    print("Training completed, continuing demo.")

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


# =============================================================================
# SHARE MODE
# =============================================================================

def sample_frames_uniform(frames_list, target_frames=TARGET_FRAMES):
    """
    frames_list: list các frame BGR (np.ndarray) độ dài T bất kỳ.
    Lấy đều target_frames điểm từ đầu đến cuối clip
    -- giống hệt VideoDataset._sample_frames() lúc train.
    """
    T = len(frames_list)
    if T == target_frames:
        return frames_list
    if T < target_frames:
        pad = target_frames - T
        return frames_list + [frames_list[-1]] * pad
    idx = np.linspace(0, T - 1, target_frames).astype(int)
    return [frames_list[i] for i in idx]


def preprocess_frames(frames_list):
    """
    frames_list: list of np.ndarray (H, W, 3) uint8 BGR, length = TARGET_FRAMES
    (đã qua sample_frames_uniform)
    Trả về tensor (1, T, C, H, W) đã normalize, sẵn sàng cho model.
    """
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


@torch.no_grad()
def predict(model, frames_tensor):
    outputs = model(frames_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_idx = outputs.argmax(1).item()
    confidence = probs[0, pred_idx].item()
    label = idx_to_label[pred_idx]
    return label, confidence


def draw_banner(display, lines, color=(0, 255, 0)):
    h, w, _ = display.shape
    banner_h = 34 + 26 * len(lines)
    cv2.rectangle(display, (8, 8), (w - 8, 8 + banner_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (16, 32 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def resize_keep_ratio(frame, target_w, target_h):
    """Resize giữ nguyên tỉ lệ khung hình, thêm viền đen nếu cần (letterbox)."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

# =============================================================================
# VIDEO MODE
# =============================================================================

def pick_video_file():
    """Mở hộp thoại chọn file bằng tkinter. Trả về đường dẫn, hoặc None nếu huỷ."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Selecting video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None


def run_video_mode():
    video_path = pick_video_file()
    if not video_path:
        print("Video selection cancelled, returning to menu.")
        return

    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return

    # ---- Đọc toàn bộ video, sample đều, và dự đoán MỘT LẦN ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open file: {video_path}")
        return

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0 or fps > 120:
        fps = 30
    delay_ms = max(1, int(1000 / fps))

    if not all_frames:
        print(f"[Error] Cannot read video's content: {video_path}")
        return

    sampled = sample_frames_uniform(all_frames, TARGET_FRAMES)
    frames_tensor = preprocess_frames(sampled)
    label_name, confidence = predict(model, frames_tensor)
    conf_pct = confidence * 100

    print('\n' + '=' * 50)
    print(f'Video     : {os.path.basename(video_path)}')
    print(f'Label     : {label_name}')
    print(f'Confidence: {conf_pct:.2f}%')
    print(f'So frame goc: {len(all_frames)} | Sample con lai: {len(sampled)}')
    print('=' * 50)

    # ---- Phát lại video, có replay ('r') và quay lại menu ('m') ----
    print(">> Video playing. Press 'r' to replay, 'm' to return to menu, 'e' to exit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9 / 16))

    frame_idx = 0
    total_frames = len(all_frames)

    while True:
        frame = all_frames[frame_idx].copy()
        display = resize_keep_ratio(frame, DISPLAY_WIDTH, int(DISPLAY_WIDTH * 9 / 16))

        text = f"{label_name} ({conf_pct:.1f}%)"
        cv2.putText(display, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 1, cv2.LINE_AA)

        frame_idx += 1
        if frame_idx >= total_frames:
            draw_banner(display, [
                "Video ended - 'r': replay, 'm': return to menu, 'e': exit"
            ], color=(0, 255, 255))
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(0) & 0xFF
        else:
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord('e'):
            cv2.destroyAllWindows()
            raise SystemExit
        elif key == ord('m'):
            cv2.destroyWindow(WINDOW_NAME)
            return
        elif key == ord('r'):
            frame_idx = 0


# =============================================================================
# WEBCAM MODE
# =============================================================================

STATE_IDLE = 'idle'
STATE_CAPTURING = 'capturing'
STATE_RESULT = 'result'

def run_webcam_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot turn on webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    display_height_init = int(CAPTURE_HEIGHT * (DISPLAY_WIDTH / CAPTURE_WIDTH))
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, display_height_init)

    state = STATE_IDLE
    recorded_frames = []
    prev_gray = None
    start_time = 0.0
    last_motion_time = 0.0
    pred_label = ""
    confidence = 0.0

    print("Press 'q' to start recording, 'r' to retry, 'm' to return to menu, and 'e' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cant read video.")
            break

        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit
        if key == ord('m'):
            break  # thoát hàm, quay lại menu chọn chế độ

        if state == STATE_IDLE:
            draw_banner(display, ["Press 'q' to start recording"], color=(255, 255, 0))
            if key == ord('q'):
                state = STATE_CAPTURING
                recorded_frames = []
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

            recorded_frames.append(frame.copy())

            now = time.time()
            elapsed = now - start_time
            still_duration = now - last_motion_time

            draw_banner(display, [
                "Recording... ",
                f"Num of frames recorded: {len(recorded_frames)}",
                f"Standstill: {still_duration:.1f}s / {STILLNESS_DURATION:.1f}s"
            ], color=(0, 255, 255))

            reached_min_time = elapsed >= MIN_RECORD_SECONDS
            is_still_enough = still_duration >= STILLNESS_DURATION
            reached_max_time = elapsed >= MAX_RECORD_SECONDS

            if (reached_min_time and is_still_enough) or reached_max_time:
                sampled = sample_frames_uniform(recorded_frames, TARGET_FRAMES)
                frames_tensor = preprocess_frames(sampled)
                pred_label, confidence = predict(model, frames_tensor)
                state = STATE_RESULT

        elif state == STATE_RESULT:
            draw_banner(display, [
                f"Result: {pred_label} ({confidence*100:.1f}%)",
                f"(From {len(recorded_frames)} frames)",
                "'r': re-predicting | 'm': menu | 'e': exit"
            ], color=(0, 255, 0))

            if key == ord('r'):
                state = STATE_CAPTURING
                recorded_frames = []
                prev_gray = None
                start_time = time.time()
                last_motion_time = time.time()

        h, w = display.shape[:2]
        scale = DISPLAY_WIDTH / w
        display_resized = cv2.resize(display, (DISPLAY_WIDTH, int(h * scale)))
        cv2.imshow(WINDOW_NAME, display_resized)

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)
    
# =============================================================================
# MENU
# =============================================================================

def run_mode_selector():
    canvas = np.zeros((280, 560, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 560, 280)

    lines = [
        ("Sign Language Demo", (255, 255, 255), 1.0),
        ("", (0, 0, 0), 0.1),
        ("Press 'v': VIDEO MODE - Prediction from pre-recorded video", (0, 255, 255), 0.7),
        ("Press 'w' : WEBCAM MODE - Prediction using webcam", (0, 255, 255), 0.7),
        ("Press 'e' : EXIT", (0, 255, 255), 0.7),
    ]

    while True:
        frame = canvas.copy()
        y = 50
        for text, color, scale in lines:
            if text:
                cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2 if scale >= 1.0 else 1, cv2.LINE_AA)
            y += 45
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('e'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('v'):
            return 'video'
        elif key == ord('w'):
            return 'webcam'


def main():
    while True:
        mode = run_mode_selector()
        if mode is None:
            break
        try:
            if mode == 'video':
                run_video_mode()
            elif mode == 'webcam':
                run_webcam_mode()
        except SystemExit:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()