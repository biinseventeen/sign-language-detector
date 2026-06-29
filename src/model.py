from .config import *

class TimeSformerClassifier(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.3):
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(
            'facebook/timesformer-base-finetuned-k400',
            ignore_mismatched_sizes=True,
        )
        hidden_size = self.backbone.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    def _print_trainable(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Trainable: {trainable:,} / {total:,} params '
              f'({100*trainable/total:.1f}%)')

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out    = self.backbone(pixel_values=x)  # last_hidden_state: (B, 1+T*patches, 768)
        cls    = out.last_hidden_state[:, 0]    # CLS token: (B, 768)
        return self.head(cls)                   # (B, 100)