
import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth",dropout_p=0.3):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        import gdown
        gdown.download(id="1rKqCIeyfgEAKiRdCbUmjJAKbWAXgVt5O", output=classifier_path, quiet=False)
        gdown.download(id="1BpOe9YyojShsXoSTBvdBflrubHEF2wQK", output=localizer_path, quiet=False)
        gdown.download(id="1H59EmgH6IACggQ_jaSY0OAGPwz2Ai7WU", output=unet_path, quiet=False)

        super().__init__()

        # ── Shared backbone ──────────────────────────────────────────────
        # self.encoder = VGG11Encoder(in_channels=in_channels)

        super().__init__()

        # ── Shared backbone ──────────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Task 1: Classification head (updated) ────────────────────────────
        # Mirrors VGG11Classifier exactly: Flatten → 4096 → 4096 → num_classes
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # ── Classification head ──────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),   # 0
            nn.Flatten(),                   # 1
            nn.Linear(512 * 7 * 7, 4096),  # 2
            nn.ReLU(inplace=True),          # 3
            CustomDropout(dropout_p),       # 4
            nn.Linear(4096, 4096),          # 5
            nn.ReLU(inplace=True),          # 6
            CustomDropout(dropout_p),       # 7
            nn.Linear(4096, num_breeds),    # 8
        )

        # ── Localisation head ────────────────────────────────────────────
        self.loc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),   # 0
            nn.Flatten(),                   # 1
            nn.Linear(512 * 7 * 7, 4096),  # 2
            nn.ReLU(inplace=True),          # 3
            CustomDropout(dropout_p),       # 4
            nn.Linear(4096, 1024),          # 5
            nn.ReLU(inplace=True),          # 6
            CustomDropout(dropout_p),       # 7
            nn.Linear(1024, 4),             # 8
        )

        # ── Segmentation decoder ─────────────────────────────────────────
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self._dec_block(1024, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._dec_block(768, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._dec_block(384, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._dec_block(192, 64)

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._dec_block(128, 64)

        self.seg_final = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ── Load pretrained weights ──────────────────────────────────────
        # Order matters: classifier encoder is loaded first (best backbone),
        # then unet decoder, then localizer head.
        if classifier_path:
            self._load_classifier(classifier_path)
        if unet_path:
            self._load_unet(unet_path)
        if localizer_path:
            self._load_localizer(localizer_path)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _load_classifier(self, path: str):
        """
        Load encoder weights from VGG11Classifier checkpoint.

        VGG11Classifier.features is a flat nn.Sequential.
        Index map (features.N → VGG11Encoder block):

        Block 1  : Conv(0)  BN(1)  ReLU(2)  Pool(3)
        Block 2  : Conv(4)  BN(5)  ReLU(6)  Pool(7)
        Block 3  : Conv(8)  BN(9)  ReLU(10) Conv(11) BN(12) ReLU(13) Pool(14)
        Block 4  : Conv(15) BN(16) ReLU(17) Conv(18) BN(19) ReLU(20) Pool(21)
        Block 5  : Conv(22) BN(23) ReLU(24) Conv(25) BN(26) ReLU(27) Pool(28)
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        # features.N.param → N.param  (strip prefix)
        feat = {k.replace("features.", ""): v
                for k, v in sd.items() if k.startswith("features.")}

        idx_to_enc = {
            # ── Block 1 ──────────────────────────────
            "0.weight": "block1.0.weight",
            "0.bias":   "block1.0.bias",
            "1.weight": "block1.1.weight",
            "1.bias":   "block1.1.bias",
            "1.running_mean":        "block1.1.running_mean",
            "1.running_var":         "block1.1.running_var",
            "1.num_batches_tracked": "block1.1.num_batches_tracked",
            # ── Block 2 ──────────────────────────────
            "4.weight": "block2.0.weight",
            "4.bias":   "block2.0.bias",
            "5.weight": "block2.1.weight",
            "5.bias":   "block2.1.bias",
            "5.running_mean":        "block2.1.running_mean",
            "5.running_var":         "block2.1.running_var",
            "5.num_batches_tracked": "block2.1.num_batches_tracked",
            # ── Block 3 conv1 ────────────────────────
            "8.weight":  "block3.0.weight",
            "8.bias":    "block3.0.bias",
            "9.weight":  "block3.1.weight",
            "9.bias":    "block3.1.bias",
            "9.running_mean":        "block3.1.running_mean",
            "9.running_var":         "block3.1.running_var",
            "9.num_batches_tracked": "block3.1.num_batches_tracked",
            # ── Block 3 conv2 ────────────────────────
            "11.weight": "block3.3.weight",
            "11.bias":   "block3.3.bias",
            "12.weight": "block3.4.weight",
            "12.bias":   "block3.4.bias",
            "12.running_mean":        "block3.4.running_mean",
            "12.running_var":         "block3.4.running_var",
            "12.num_batches_tracked": "block3.4.num_batches_tracked",
            # ── Block 4 conv1 ────────────────────────
            "15.weight": "block4.0.weight",
            "15.bias":   "block4.0.bias",
            "16.weight": "block4.1.weight",
            "16.bias":   "block4.1.bias",
            "16.running_mean":        "block4.1.running_mean",
            "16.running_var":         "block4.1.running_var",
            "16.num_batches_tracked": "block4.1.num_batches_tracked",
            # ── Block 4 conv2 ────────────────────────
            "18.weight": "block4.3.weight",
            "18.bias":   "block4.3.bias",
            "19.weight": "block4.4.weight",
            "19.bias":   "block4.4.bias",
            "19.running_mean":        "block4.4.running_mean",
            "19.running_var":         "block4.4.running_var",
            "19.num_batches_tracked": "block4.4.num_batches_tracked",
            # ── Block 5 conv1 ────────────────────────
            "22.weight": "block5.0.weight",
            "22.bias":   "block5.0.bias",
            "23.weight": "block5.1.weight",
            "23.bias":   "block5.1.bias",
            "23.running_mean":        "block5.1.running_mean",
            "23.running_var":         "block5.1.running_var",
            "23.num_batches_tracked": "block5.1.num_batches_tracked",
            # ── Block 5 conv2 ────────────────────────
            "25.weight": "block5.3.weight",
            "25.bias":   "block5.3.bias",
            "26.weight": "block5.4.weight",
            "26.bias":   "block5.4.bias",
            "26.running_mean":        "block5.4.running_mean",
            "26.running_var":         "block5.4.running_var",
            "26.num_batches_tracked": "block5.4.num_batches_tracked",
        }

        new_sd = {enc_k: feat[flat_k]
                  for flat_k, enc_k in idx_to_enc.items()
                  if flat_k in feat}

        missing, unexpected = self.encoder.load_state_dict(new_sd, strict=False)
        print(f"[classifier] encoder loaded | matched: {len(new_sd)} | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")

        # ── Also load classification head ────────────────────────────────
        # VGG11Classifier.classifier layout:
        #   Flatten(0) Linear(1) ReLU(2) Dropout(3)
        #   Linear(4)  ReLU(5)   Dropout(6) Linear(7)
        # self.cls_head layout:
        #   AvgPool(0) Flatten(1) Linear(2) ReLU(3) Dropout(4)
        #   Linear(5)  ReLU(6)   Dropout(7) Linear(8)
        cls_map = {
            "classifier.1.weight": "classification_head.1.weight",
            "classifier.1.bias":   "classification_head.1.bias",
            "classifier.4.weight": "classification_head.4.weight",
            "classifier.4.bias":   "classification_head.4.bias",
            "classifier.7.weight": "classification_head.7.weight",
            "classifier.7.bias":   "classification_head.7.bias",
        }

        cls_sd = {head_k: sd[cls_k]
                for cls_k, head_k in cls_map.items()
                if cls_k in sd}

        # missing, unexpected = self.load_state_dict(cls_sd, strict=False)
        # print(f"[classifier] classification_head loaded | matched: {len(cls_sd)} | "
        #     f"missing: {len(missing)} | unexpected: {len(unexpected)}")

        missing, unexpected = self.load_state_dict(cls_sd, strict=False)
        print(f"[classifier] cls_head loaded | matched: {len(cls_sd)} | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")

    def _load_localizer(self, path: str):
        """
        Load localizer regression head from VGG11Localizer checkpoint.

        VGG11Localizer.regressor layout:
          Flatten(0) Linear(1) ReLU(2) Dropout(3)
          Linear(4)  ReLU(5)   Dropout(6) Linear(7)
        self.loc_head layout:
          AvgPool(0) Flatten(1) Linear(2) ReLU(3) Dropout(4)
          Linear(5)  ReLU(6)   Dropout(7) Linear(8)
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        loc_map = {
            "regressor.1.weight": "loc_head.2.weight",
            "regressor.1.bias":   "loc_head.2.bias",
            "regressor.4.weight": "loc_head.5.weight",
            "regressor.4.bias":   "loc_head.5.bias",
            "regressor.7.weight": "loc_head.8.weight",
            "regressor.7.bias":   "loc_head.8.bias",
        }

        new_sd = {head_k: sd[loc_k]
                  for loc_k, head_k in loc_map.items()
                  if loc_k in sd}

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        print(f"[localizer] loc_head loaded | matched: {len(new_sd)} | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")

    def _load_unet(self, path: str):
        """
        Load encoder + decoder weights from VGG11UNet checkpoint.

        VGG11UNet uses VGG11Encoder internally so encoder keys match directly.
        Decoder keys (up5/dec5/..) also match self directly.
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        # encoder
        enc_sd = {k[len("encoder."):]: v
                  for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        print(f"[unet] encoder loaded | matched: {len(enc_sd)} | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")

        # decoder + seg_final
        dec_keys = ("up5","dec5","up4","dec4","up3","dec3",
                    "up2","dec2","up1","dec1","seg_final")
        dec_sd = {k: v for k, v in sd.items()
                  if any(k.startswith(dk) for dk in dec_keys)}
        missing, unexpected = self.load_state_dict(dec_sd, strict=False)
        print(f"[unet] decoder loaded | matched: {len(dec_sd)} | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # shared encoder
        bottleneck, skips = self.encoder(x, return_features=True)

        # classification
        # cls_out = self.cls_head(bottleneck)

        # localization
        loc_out = self.loc_head(bottleneck)

        # segmentation decoder
        s = self.up5(bottleneck)
        s = torch.cat([s, skips["block5"]], dim=1)
        s = self.dec5(s)

        s = self.up4(s)
        s = torch.cat([s, skips["block4"]], dim=1)
        s = self.dec4(s)

        s = self.up3(s)
        s = torch.cat([s, skips["block3"]], dim=1)
        s = self.dec3(s)

        s = self.up2(s)
        s = torch.cat([s, skips["block2"]], dim=1)
        s = self.dec2(s)

        s = self.up1(s)
        s = torch.cat([s, skips["block1"]], dim=1)
        s = self.dec1(s)

        seg_out = self.seg_final(s)

        cls_out = self.classification_head(bottleneck)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }


