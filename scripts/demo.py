"""Streamlit interface for Vietnamese sign language video classification."""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Nhận diện ngôn ngữ ký hiệu",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: "Segoe UI", Arial, sans-serif; }
      .stApp { background: #f3f6fb; color: #24364b; }
      .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }
      .stMarkdown, [data-testid="stMarkdownContainer"] { color: #34465d; }
      [data-testid="stMarkdownContainer"] h1,
      [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMarkdownContainer"] h3,
      [data-testid="stMarkdownContainer"] h4 { color: #173b5e !important; }
      .hero { padding: 2rem 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(120deg, #123c69 0%, #1b6ca8 58%, #35a7a0 100%);
        box-shadow: 0 16px 38px rgba(20, 65, 105, .18); margin-bottom: 1.5rem; }
      .hero h1 { color: white !important; font-size: 2.1rem; margin: .3rem 0 .5rem; }
      .hero p { color: #e3f2ff !important; font-size: 1rem; margin: 0; }
      .eyebrow { color: #b9f2e8; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
      .panel { padding: 1.35rem 1.5rem; background: white; border: 1px solid #e8edf4;
        border-radius: 18px; box-shadow: 0 5px 18px rgba(25, 50, 80, .045); margin-bottom: 1rem; }
      .panel h3 { color: #173b5e !important; margin: 0 0 .25rem; }
      .muted { color: #68788c !important; }
      .feature-card { height: 100%; padding: 1rem 1.15rem; border-radius: 16px;
        background: white; border: 1px solid #e5ebf3; box-shadow: 0 5px 18px rgba(25, 50, 80, .04); }
      .feature-card strong { display: block; color: #173b5e; font-size: 1rem; margin-bottom: .25rem; }
      .feature-card span { color: #718096; font-size: .88rem; }
      [data-testid="stFileUploader"] { background: white; border: 1px solid #e2e8f0;
        border-radius: 16px; padding: .8rem 1rem; }
      [data-testid="stFileUploader"] section { background: #f7faff !important;
        border: 1px dashed #a9c1dd !important; border-radius: 12px !important; }
      [data-testid="stFileUploader"] * { color: #34465d !important; }
      [data-testid="stFileUploader"] button { background: white !important;
        border: 1px solid #cbd8e7 !important; color: #173b5e !important; }
      div.stButton > button { border-radius: 12px; font-weight: 700; min-height: 3rem;
        border: 0; background: linear-gradient(105deg, #175b9a, #258d9d); color: white; }
      div.stButton > button:hover { background: linear-gradient(105deg, #124a80, #1b7482); color: white; }
    </style>
<section class="hero">
      <div class="eyebrow">AI · SIGN LANGUAGE</div>
      <h1>Nhận diện ngôn ngữ ký hiệu</h1>
      <p>Tải video lên để mô hình phân loại ký hiệu tiếng Việt.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


def load_inference_assets():
    """Load heavy ML dependencies and model only when inference is requested."""
    import pickle

    import torch
    from src.config import DEVICE, LABEL_MAP_PATH, NUM_CLASSES, SAVE_PATH
    from src.model import TimeSformerClassifier

    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            "Chưa tìm thấy best_model_timesformer.pth. Hãy huấn luyện mô hình "
            "bằng `python scripts/train.py` trước khi dùng demo."
        )
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError("Không tìm thấy tệp label_mapping.pkl trong thư mục project.")

    model = TimeSformerClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    with open(LABEL_MAP_PATH, "rb") as label_file:
        label_mapping = pickle.load(label_file)
    return model, {index: label for label, index in label_mapping.items()}, DEVICE


@st.cache_resource(show_spinner=False)
def get_inference_assets():
    return load_inference_assets()


def read_sampled_frames(video_bytes, filename, target_frames):
    """Read only uniformly spaced frames to avoid holding the whole video in memory."""
    import cv2
    import numpy as np

    suffix = Path(filename).suffix.lower() or ".mp4"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("Không thể mở video. Hãy thử tệp MP4, AVI, MOV hoặc MKV khác.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            raise ValueError("Không đọc được khung hình từ video này.")

        indices = np.linspace(0, frame_count - 1, target_frames).astype(int)
        frames = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        capture.release()

        if not frames:
            raise ValueError("Video không có khung hình đọc được.")
        while len(frames) < target_frames:
            frames.append(frames[-1])
        return frames
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@st.cache_data(show_spinner=False)
def make_preview_sheet(video_bytes, filename):
    """Build a browser-safe contact sheet from frames decoded by OpenCV."""
    from io import BytesIO

    from PIL import Image

    frames = read_sampled_frames(video_bytes, filename, target_frames=8)
    tile_width, tile_height = 320, 180
    gap = 12
    sheet = Image.new(
        "RGB",
        (tile_width * 2 + gap * 3, tile_height * 4 + gap * 5),
        "#e8eef5",
    )
    for index, frame in enumerate(frames):
        rgb_frame = Image.fromarray(frame[:, :, ::-1].copy())
        rgb_frame.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "#142235")
        x_offset = (tile_width - rgb_frame.width) // 2
        y_offset = (tile_height - rgb_frame.height) // 2
        tile.paste(rgb_frame, (x_offset, y_offset))
        x = gap + (index % 2) * (tile_width + gap)
        y = gap + (index // 2) * (tile_height + gap)
        sheet.paste(tile, (x, y))

    output = BytesIO()
    sheet.save(output, format="PNG", optimize=True)
    return output.getvalue()


def classify_video(video_bytes, filename):
    import cv2
    import numpy as np
    import torch
    from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, TARGET_FRAMES

    model, index_to_label, device = get_inference_assets()
    sampled_frames = read_sampled_frames(video_bytes, filename, TARGET_FRAMES)
    processed_frames = []
    for frame in sampled_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        processed_frames.append(frame)

    tensor = torch.from_numpy(np.stack(processed_frames)).float()
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
    return index_to_label[predicted_index], float(probabilities[predicted_index].item())


feature_a, feature_b, feature_c = st.columns(3, gap="medium")
with feature_a:
    st.markdown('<div class="feature-card"><strong>100 lớp ký hiệu</strong><span>Phân loại từ vựng tiếng Việt</span></div>', unsafe_allow_html=True)
with feature_b:
    st.markdown('<div class="feature-card"><strong>8 khung hình</strong><span>Lấy mẫu xuyên suốt video</span></div>', unsafe_allow_html=True)
with feature_c:
    st.markdown('<div class="feature-card"><strong>Thao tác đơn giản</strong><span>Tải video lên và nhận kết quả</span></div>', unsafe_allow_html=True)

st.markdown('<div class="panel"><h3>①  Tải video lên</h3><p class="muted">Chọn video rõ nét, có đầy đủ động tác ký hiệu. Hỗ trợ MP4, AVI, MOV và MKV.</p></div>', unsafe_allow_html=True)
uploaded_video = st.file_uploader(
    "Chọn tệp video",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed",
)

if uploaded_video is not None:
    video_bytes = uploaded_video.getvalue()
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="panel"><h3>②  Xem trước video</h3><p class="muted">Một số khung hình được lấy từ đầu đến cuối clip.</p></div>', unsafe_allow_html=True)
        try:
            st.image(make_preview_sheet(video_bytes, uploaded_video.name), use_container_width=True)
        except Exception as error:
            st.warning(f"Chưa thể tạo ảnh xem trước: {error}")
    with right:
        st.markdown('<div class="panel"><h3>③  Nhận diện ký hiệu</h3><p class="muted">Mô hình phân tích các khung hình và trả về nhãn dự đoán kèm độ tin cậy.</p></div>', unsafe_allow_html=True)
        st.caption(f"Tệp đã chọn: {uploaded_video.name}")
        run_prediction = st.button("🤟  Nhận diện ký hiệu", type="primary", use_container_width=True)

    if run_prediction:
        st.session_state.pop("prediction_result", None)
        try:
            with st.spinner("Đang tải mô hình và phân tích video… lần đầu có thể mất thêm chút thời gian."):
                label, confidence = classify_video(video_bytes, uploaded_video.name)
            st.session_state["prediction_result"] = {
                "filename": uploaded_video.name,
                "label": label,
                "confidence": confidence,
            }
        except Exception as error:
            st.error(str(error))

    result = st.session_state.get("prediction_result")
    if result and result["filename"] == uploaded_video.name:
        st.markdown('<div class="panel"><div class="muted">Kết quả dự đoán</div></div>', unsafe_allow_html=True)
        st.subheader(result["label"])
        st.progress(result["confidence"], text=f'Độ tin cậy: {result["confidence"]:.1%}')
else:
    st.info("Chọn một video ở trên để xem trước và bắt đầu nhận diện.")
