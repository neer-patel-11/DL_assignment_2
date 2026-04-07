import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model — classification fully loaded, others return empty tensors."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        dropout_p: float = 0.4,
    ):
        super().__init__()

        # ── Download checkpoints ─────────────────────────────────────────
        import gdown
        gdown.download(id="1rKqCIeyfgEAKiRdCbUmjJAKbWAXgVt5O", output=classifier_path, quiet=False)
        gdown.download(id="1BpOe9YyojShsXoSTBvdBflrubHEF2wQK", output=localizer_path, quiet=False)
        gdown.download(id="1H59EmgH6IACggQ_jaSY0OAGPwz2Ai7WU", output=unet_path, quiet=False)

        # ── Shared backbone ──────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ──────────────────────────────────────────
        # Mirrors VGG11Classifier.classifier exactly:
        #   Flatten(0) → Linear(1) → ReLU(2) → Dropout(3)
        #   → Linear(4) → ReLU(5) → Dropout(6) → Linear(7)
        # Bottleneck from encoder.pool5 is already [B, 512, 7, 7]
        self.classification_head = nn.Sequential(
            nn.Flatten(),                        # 0
            nn.Linear(512 * 7 * 7, 4096),        # 1  ← classifier.1
            nn.ReLU(inplace=True),               # 2
            CustomDropout(dropout_p),            # 3
            nn.Linear(4096, 4096),               # 4  ← classifier.4
            nn.ReLU(inplace=True),               # 5
            CustomDropout(dropout_p),            # 6
            nn.Linear(4096, num_breeds),         # 7  ← classifier.7
        )

        # ── Stub heads (unused, kept for interface compatibility) ────────
        # loc_head: outputs [B, 4]
        self.loc_head = nn.Identity()
        # seg decoder stubs — just enough to satisfy any attribute access
        self.seg_final = nn.Identity()

        # ── Load weights ─────────────────────────────────────────────────
        self._load_classifier(classifier_path)

    # ─────────────────────────────────────────────────────────────────────

    def _load_classifier(self, path: str):
        """
        Load encoder + classification head weights from VGG11Classifier checkpoint.

        VGG11Classifier saves:
          features.block1…block5.*  →  self.encoder.*
          classifier.1/4/7.*        →  self.classification_head.1/4/7.*
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        # ── 1. Encoder weights ───────────────────────────────────────────
        # VGG11Classifier wraps the encoder as `self.features` which IS a
        # VGG11Encoder instance, so keys are features.block1.0.weight etc.
        enc_sd = {}
        for k, v in sd.items():
            if k.startswith("features."):
                # strip "features." prefix → block1.0.weight etc.
                new_key = k[len("features."):]
                enc_sd[new_key] = v

        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        print(f"[classifier] encoder  | matched: {len(enc_sd)} "
              f"| missing: {len(missing)} | unexpected: {len(unexpected)}")

        # ── 2. Classification head weights ───────────────────────────────
        # classifier.N.weight/bias → classification_head.N.weight/bias
        # Only the three Linear layers (indices 1, 4, 7) have parameters.
        cls_sd = {}
        for k, v in sd.items():
            if k.startswith("classifier."):
                # "classifier.1.weight" → "classification_head.1.weight"
                new_key = "classification_head." + k[len("classifier."):]
                cls_sd[new_key] = v

        missing, unexpected = self.load_state_dict(cls_sd, strict=False)
        print(f"[classifier] cls_head | matched: {len(cls_sd)} "
              f"| missing: {len(missing)} | unexpected: {len(unexpected)}")

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: [B, in_channels, H, W]

        Returns:
            dict with keys:
              'classification' : [B, num_breeds]  — real logits
              'localization'   : [B, 4]            — zeros (stub)
              'segmentation'   : [B, seg_classes, H, W] — zeros (stub)
        """
        B = x.size(0)

        # Encoder — pool5 included, bottleneck is [B, 512, 7, 7]
        bottleneck = self.encoder(x, return_features=False)

        # Classification
        cls_out = self.classification_head(bottleneck)   # [B, num_breeds]

        # Stubs with correct output shapes
        loc_out = torch.zeros(B, 4, device=x.device, dtype=x.dtype)
        seg_out = torch.zeros(B, 3, x.size(2), x.size(3),
                              device=x.device, dtype=x.dtype)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }