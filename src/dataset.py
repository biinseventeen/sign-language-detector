from .config import *

#=======================#
#===DATA AUGMENTATION===#
#=======================#
class VideoAugmentation:
    """
    Augmentation cho video ký hiệu.
    Thứ tự áp dụng: speed → crop → color
    Input/output frames: torch.uint8, shape (T, H, W, C)
    """
    def __init__(
        self,
        crop_scale=(0.85, 1.0),
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        speed_range=(0.8, 1.2),
        flip_prob=0.0,   # Tắt flip vì ký hiệu tay có thể đổi nghĩa
    ):
        self.crop_scale   = crop_scale
        self.brightness   = brightness
        self.contrast     = contrast
        self.saturation   = saturation
        self.speed_range  = speed_range
        self.flip_prob    = flip_prob

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self._speed_augment(frames)
        frames = self._random_resized_crop(frames)
        frames = self._color_jitter(frames)
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            frames = torch.flip(frames, dims=[2])  # flip W axis
        return frames

    def _speed_augment(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]
        speed   = random.uniform(*self.speed_range)
        new_T   = max(4, int(T / speed))
        if new_T == T:
            return frames
        indices = torch.linspace(0, T - 1, new_T).long().clamp(0, T - 1)
        return frames[indices]

    def _random_resized_crop(self, frames: torch.Tensor) -> torch.Tensor:
        T, H, W, C = frames.shape
        scale  = random.uniform(*self.crop_scale)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top    = random.randint(0, H - crop_h)
        left   = random.randint(0, W - crop_w)
        frames = frames[:, top:top+crop_h, left:left+crop_w, :]
        frames = frames.permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=(IMG_SIZE, IMG_SIZE),
                               mode='bilinear', align_corners=False)
        return frames.permute(0, 2, 3, 1).to(torch.uint8)

    def _color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float()
        frames = frames * (1 + random.uniform(-self.brightness, self.brightness))
        mean   = frames.mean(dim=(1, 2), keepdim=True)
        frames = (frames - mean) * (1 + random.uniform(-self.contrast, self.contrast)) + mean
        gray   = frames.mean(dim=-1, keepdim=True)
        frames = gray + (frames - gray) * (1 + random.uniform(-self.saturation, self.saturation))
        return frames.clamp(0, 255).to(torch.uint8)

#===========================#
#==========DATASET==========#
#===========================#
def read_video(video_path: str) -> torch.Tensor:
    """Đọc video → tensor uint8 (T, H, W, 3)."""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f'Không đọc được frame từ {video_path}')
    return torch.from_numpy(np.stack(frames, axis=0))


class VideoDataset(Dataset):
    """
    Load video theo cấu trúc: root_dir/<class_name>/<video_file>
    Trả về tensor frames shape (T, C, H, W) đã normalize, sẵn sàng cho TimeSformer.
    """
    def __init__(
        self,
        root_dir: str,
        label_to_idx_path: str,
        target_frames: int = TARGET_FRAMES,
        augment: VideoAugmentation = None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.root_dir      = root_dir
        self.target_frames = target_frames
        self.augment       = augment
        self.mean          = mean
        self.std           = std

        # Load & normalize label mapping
        with open(label_to_idx_path, 'rb') as f:
            raw = pickle.load(f)
        self.label_mapping = {
            unicodedata.normalize('NFC', k).strip(): v
            for k, v in raw.items()
        }

        # Collect (path, label_name, label_idx)
        self.instances = []
        self.label_idx = []
        self.labels    = []

        for label_folder in sorted(os.listdir(root_dir))[:NUM_CLASSES]:
            path = os.path.join(root_dir, label_folder)
            if not os.path.isdir(path):
                continue
            clean = unicodedata.normalize('NFC', label_folder).strip()
            if clean not in self.label_mapping:
                raise KeyError(f"Nhãn '{clean}' không có trong mapping")
            idx = self.label_mapping[clean]
            for video_file in os.listdir(path):
                self.instances.append(os.path.join(path, video_file))
                self.labels.append(label_folder)
                self.label_idx.append(idx)

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Uniform sampling về đúng TARGET_FRAMES."""
        T = frames.shape[0]
        if T == self.target_frames:
            return frames
        if T < self.target_frames:
            # Pad bằng frame cuối
            pad = self.target_frames - T
            return torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)
        idx = torch.linspace(0, T - 1, self.target_frames).long()
        return frames[idx]

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """(T,H,W,C) uint8 → (T,C,H,W) float normalized."""
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        mean   = torch.tensor(self.mean).view(1, 3, 1, 1)
        std    = torch.tensor(self.std).view(1, 3, 1, 1)
        return (frames - mean) / std

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        frames    = read_video(self.instances[idx])   # (T, H, W, C) uint8
        if self.augment is not None:
            frames = self.augment(frames)             # augment trước khi sample
        frames    = self._sample_frames(frames)       # → (target_frames, H, W, C)
        frames    = self._normalize(frames)           # → (target_frames, C, H, W) float
        return {
            'frames':    frames,
            'label_idx': self.label_idx[idx],
            'label':     self.labels[idx],
        }


def collate_fn(batch):
    return {
        'frames':    torch.stack([b['frames'] for b in batch]),
        'label_idx': torch.tensor([b['label_idx'] for b in batch]),
        'label':     [b['label'] for b in batch],
    }


def create_balanced_sampler(dataset, max_oversample_ratio=10):
    all_labels  = (
        [dataset.dataset.label_idx[i] for i in dataset.indices]
        if hasattr(dataset, 'dataset')
        else dataset.label_idx
    )
    label_count = np.bincount(all_labels)
    label_count = np.where(label_count == 0, 1, label_count)
    max_count   = label_count.max()
    min_count   = label_count[label_count > 1].min()
    capped      = np.clip(label_count, min_count / max_oversample_ratio, None)
    weights     = 1.0 / capped
    sample_w    = torch.FloatTensor([weights[l] for l in all_labels])
    target_n    = min(int(max_count * len(label_count)), len(all_labels) * 2)
    print(f'Balanced sampler: {len(all_labels)} → {target_n} samples/epoch')
    return WeightedRandomSampler(sample_w, num_samples=target_n, replacement=True)