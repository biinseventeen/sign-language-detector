import wandb
from src.config import (
    SEED, DEVICE, NUM_CLASSES,
    TRAIN_ROOT, LABEL_MAP_PATH, SAVE_PATH,
    BATCH_SIZE, ACCUM_STEPS, LR_BACKBONE, LR_HEAD, EPOCHS,
    IMAGENET_MEAN, IMAGENET_STD,
    WANDB_PROJECT, WANDB_ENTITY,
)
from src.utils import seed_everything
from src.model import TimeSformerClassifier
from src.dataset import (
    VideoAugmentation, VideoDataset,
    collate_fn, create_balanced_sampler,
)
from src.train import train_phase

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def build_dataloaders():
    augment = VideoAugmentation(
        crop_scale=(0.85, 1.0),
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        speed_range=(0.8, 1.2),
    )

    full_train = VideoDataset(TRAIN_ROOT, LABEL_MAP_PATH, augment=augment)
    full_val   = VideoDataset(TRAIN_ROOT, LABEL_MAP_PATH, augment=None)

    indices = list(range(len(full_train)))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))

    train_dataset = Subset(full_train, indices[:split])
    val_dataset   = Subset(full_val,   indices[split:])

    sampler = create_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, collate_fn=collate_fn,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True,
    )

    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')
    return train_loader, val_loader


def main():
    seed_everything(SEED)

    wandb.login()
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name='TimeSformer_FullFinetune',
        config={
            'target_frames': 8,
            'batch_size':    BATCH_SIZE,
            'accum_steps':   ACCUM_STEPS,
            'lr_backbone':   LR_BACKBONE,
            'lr_head':       LR_HEAD,
            'epochs':        EPOCHS,
        }
    )

    train_loader, val_loader = build_dataloaders()

    model = TimeSformerClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    best_f1 = train_phase(
        model, train_loader, val_loader,
        num_epochs=EPOCHS,
        lr_backbone=LR_BACKBONE,
        lr_head=LR_HEAD,
        phase_name='FullFinetune',
    )

    print(f'Best F1: {best_f1:.2f}%')
    wandb.finish()


if __name__ == '__main__':
    main()