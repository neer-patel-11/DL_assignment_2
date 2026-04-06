import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier


class MultiTaskPerceptionModel(nn.Module):

    def __init__(
        self,
        num_breeds:  int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path:  str = "checkpoints/localizer.pth",
        unet_path:       str = "checkpoints/unet.pth",
        dropout_p: float = 0.3
    ):
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
        gdown.download(id="1wsj_OO2fE26Hn_czJCZkMSDZl1jFu29r", output=classifier_path, quiet=False)
        gdown.download(id="1BpOe9YyojShsXoSTBvdBflrubHEF2wQK", output=localizer_path,  quiet=False)
        gdown.download(id="1H59EmgH6IACggQ_jaSY0OAGPwz2Ai7WU", output=unet_path,       quiet=False)

        super().__init__()

        # ── 1. Classification: full VGG11Classifier (own backbone + head) ─
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)

        # ── 2. Localization: own backbone + regression head ───────────────
        self.loc_encoder = VGG11Encoder(in_channels=in_channels)
        self.loc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),       # 0
            nn.Flatten(),                       # 1
            nn.Linear(512 * 7 * 7, 4096),      # 2
            nn.ReLU(inplace=True),              # 3
            CustomDropout(dropout_p),           # 4
            nn.Linear(4096, 1024),              # 5
            nn.ReLU(inplace=True),              # 6
            CustomDropout(dropout_p),           # 7
            nn.Linear(1024, 4),                 # 8
        )

        # ── 3. Segmentation: own backbone + UNet decoder ──────────────────
        self.seg_encoder = VGG11Encoder(in_channels=in_channels)
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self._dec_block(1024, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._dec_block(768,  256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._dec_block(384,  128)
        self.up2  = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.dec2 = self._dec_block(192,  64)
        self.up1  = nn.ConvTranspose2d(64,  64,  kernel_size=2, stride=2)
        self.dec1 = self._dec_block(128,  64)
        self.seg_final = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ── Load weights ──────────────────────────────────────────────────
        if classifier_path: self._load_classifier(classifier_path)
        if localizer_path:  self._load_localizer(localizer_path)
        if unet_path:       self._load_unet(unet_path)

    # ── helpers ───────────────────────────────────────────────────────────

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
        VGG11Classifier has its own features + classifier keys.
        Load directly into self.classifier — zero remapping needed.
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)
        result = self.classifier.load_state_dict(sd, strict=True)
        print(f"✓ [classifier] Loaded {len(sd)} weights (strict=True)")

    def _load_localizer(self, path: str):
        """
        VGG11Localizer checkpoint has:
          encoder.*    → self.loc_encoder
          regressor.*  → self.loc_head (with index offset due to AdaptiveAvgPool)
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        # Encoder
        enc_sd = {k[len("encoder."):]: v
                  for k, v in sd.items() if k.startswith("encoder.")}
        result = self.loc_encoder.load_state_dict(enc_sd, strict=False)
        print(f"✓ [localizer] loc_encoder: {len(enc_sd)} weights | missing: {len(result.missing_keys)}")

        # Regression head
        # VGG11Localizer.regressor: Flatten(0) Linear(1) ReLU(2) Dropout(3) Linear(4) ReLU(5) Dropout(6) Linear(7)
        # self.loc_head:            AvgPool(0) Flatten(1) Linear(2) ReLU(3) Dropout(4) Linear(5) ReLU(6) Dropout(7) Linear(8)
        loc_map = {
            "regressor.1.weight": "loc_head.2.weight",
            "regressor.1.bias":   "loc_head.2.bias",
            "regressor.4.weight": "loc_head.5.weight",
            "regressor.4.bias":   "loc_head.5.bias",
            "regressor.7.weight": "loc_head.8.weight",
            "regressor.7.bias":   "loc_head.8.bias",
        }
        head_sd = {head_k: sd[loc_k] for loc_k, head_k in loc_map.items() if loc_k in sd}
        result  = self.load_state_dict(head_sd, strict=False)
        print(f"✓ [localizer] loc_head:    {len(head_sd)} weights | missing: {len(result.missing_keys)}")

    def _load_unet(self, path: str):
        """
        VGG11UNet checkpoint has:
          encoder.*         → self.seg_encoder
          up5/dec5/...      → self decoder layers
        """
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)

        # Encoder
        enc_sd = {k[len("encoder."):]: v
                  for k, v in sd.items() if k.startswith("encoder.")}
        result = self.seg_encoder.load_state_dict(enc_sd, strict=False)
        print(f"✓ [unet] seg_encoder: {len(enc_sd)} weights | missing: {len(result.missing_keys)}")

        # Decoder
        dec_keys = ("up5","dec5","up4","dec4","up3","dec3","up2","dec2","up1","dec1","seg_final")
        dec_sd   = {k: v for k, v in sd.items() if any(k.startswith(dk) for dk in dec_keys)}
        result   = self.load_state_dict(dec_sd, strict=False)
        print(f"✓ [unet] seg_decoder: {len(dec_sd)} weights | missing: {len(result.missing_keys)}")

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Load localizer regression head from VGG11Localizer checkpoint.

        VGG11Localizer.regressor layout:
        Flatten(0) Linear(1) ReLU(2) Dropout(3)
        Linear(4)  ReLU(5)   Dropout(6) Linear(7)
        self.loc_head layout:
        AvgPool(0) Flatten(1) Linear(2) ReLU(3) Dropout(4)
        Linear(5)  ReLU(6)   Dropout(7) Linear(8)
        """
        # Classification — VGG11Classifier runs its own features + classifier
        cls_out = self.classifier(x)

        # Localization — dedicated encoder
        loc_feat, _ = self.loc_encoder(x, return_features=True)
        loc_out = self.loc_head(loc_feat)

        # Segmentation — dedicated encoder + UNet decoder
        bottleneck, skips = self.seg_encoder(x, return_features=True)
        s = self.up5(bottleneck)
        s = torch.cat([s, skips["block5"]], dim=1);  s = self.dec5(s)
        s = self.up4(s)
        s = torch.cat([s, skips["block4"]], dim=1);  s = self.dec4(s)
        s = self.up3(s)
        s = torch.cat([s, skips["block3"]], dim=1);  s = self.dec3(s)
        s = self.up2(s)
        s = torch.cat([s, skips["block2"]], dim=1);  s = self.dec2(s)
        s = self.up1(s)
        s = torch.cat([s, skips["block1"]], dim=1);  s = self.dec1(s)
        seg_out = self.seg_final(s)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }