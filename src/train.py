from .config import *

def train_epoch(
    model, dataloader, criterion, optimizer, scaler,
    device=DEVICE, accum_steps=ACCUM_STEPS
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc='Train')
    for i, batch in enumerate(progress):
        frames = batch['frames'].to(device)      # (B, T, C, H, W)
        labels = batch['label_idx'].to(device)

        with autocast():                         # AMP: fp16 tự động
            outputs = model(frames)
            loss    = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        progress.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device=DEVICE):
    model.eval()
    total_loss, preds, all_labels = 0.0, [], []

    for batch in tqdm(dataloader, desc='Val'):
        frames = batch['frames'].to(device)
        labels = batch['label_idx'].to(device)
        with autocast():
            outputs = model(frames)
            loss    = criterion(outputs, labels)
        total_loss += loss.item()
        preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='macro', zero_division=0
    )
    return total_loss / len(dataloader), {
        'precision': p * 100,
        'recall':    r * 100,
        'f1':        f1 * 100,
    }


def train_phase(
    model, train_loader, val_loader,
    num_epochs, lr_backbone, lr_head, phase_name, device=DEVICE
):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.head.parameters(),     'lr': lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr_backbone * 0.1
    )
    scaler    = GradScaler()
    best_f1   = 0.0

    for epoch in range(num_epochs):
        print(f'\n=== [{phase_name}] Epoch {epoch+1}/{num_epochs} ===')
        train_loss             = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics  = validate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            'phase':        phase_name,
            'epoch':        epoch + 1,
            'train_loss':   train_loss,
            'val_loss':     val_loss,
            'val_f1':       val_metrics['f1'],
            'val_precision':val_metrics['precision'],
            'val_recall':   val_metrics['recall'],
            'lr':           optimizer.param_groups[0]['lr'],
        })

        print(f"Val F1: {val_metrics['f1']:.2f}% | "
              f"Precision: {val_metrics['precision']:.2f}% | "
              f"Recall: {val_metrics['recall']:.2f}%")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'✓ Saved best model — F1: {best_f1:.2f}%')

    return best_f1