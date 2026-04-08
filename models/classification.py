import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = Encoder + Improved Classification Head."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.4):
        super().__init__()

        # ----- ENCODER -----
        self.features = VGG11Encoder(in_channels=in_channels)

        # ----- SAFE POOLING (important improvement) -----
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ----- CLASSIFIER -----
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_classes)
        )

        self._init_weights()

    # ----- COMBINED INITIALIZATION -----
    def _init_weights(self):
        for m in self.modules():

            # 🔹 Keep your strong conv init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # 🔹 Use friend's stable classifier init
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ----- FORWARD -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, 3, 224, 224]
        Output: [B, num_classes]
        """

        x = self.features(x, return_features=False)   # [B, 512, H, W]
        x = self.avgpool(x)                           # [B, 512, 7, 7]
        x = torch.flatten(x, 1)                       # [B, 512*7*7]
        x = self.classifier(x)

        return x