from .config import *
from .dataset import VideoDataset, read_video

@torch.no_grad()
def evaluate(
    model, folder_path, label_to_idx_path,
    output_csv='predictions.csv', device=DEVICE,
    model_path=None, target_frames=TARGET_FRAMES,
):
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded weights từ {model_path}')

    model.to(device).eval()

    with open(label_to_idx_path, 'rb') as f:
        label_mapping = pickle.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # Tạo dataset helper để dùng _sample_frames và _normalize
    ds = VideoDataset.__new__(VideoDataset)
    ds.target_frames = target_frames
    ds.mean = IMAGENET_MEAN
    ds.std  = IMAGENET_STD

    video_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])
    print(f'Tìm thấy {len(video_files)} video trong "{folder_path}"')

    predictions = []
    for video_file in tqdm(video_files, desc='Inference'):
        try:
            frames = read_video(os.path.join(folder_path, video_file))
            frames = ds._sample_frames(frames)
            frames = ds._normalize(frames).unsqueeze(0).to(device)  # (1,T,C,H,W)
            with autocast():
                out = model(frames)
            label_name = idx_to_label[out.argmax(1).item()]
            predictions.append((video_file, label_name))
        except Exception as e:
            print(f'Lỗi với {video_file}: {e}')

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'label'])
        writer.writerows(predictions)

    print(f'✓ Saved {len(predictions)} predictions → "{output_csv}"')