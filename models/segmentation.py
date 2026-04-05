"""Segmentation model
"""

import torch
import torch.nn as nn
from vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        
        # Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Decoder with transposed convolutions
        # Note: VGG11 has blocks with channels: 64, 128, 256, 256, 512, 512
        # But in the provided classifier, it's: 64, 128, 256, 512, 512
        
        # Upsample from bottleneck (512, 7x7) -> (512, 14x14)
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        # Concat with block5 skip (512) = 1024 channels
        self.dec5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Upsample (512, 14x14) -> (256, 28x28)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Concat with block4 skip (512) = 768 channels
        self.dec4 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),  # 768 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Upsample (256, 28x28) -> (128, 56x56)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Concat with block3 skip (256) = 384 channels
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),  # 384 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsample (128, 56x56) -> (64, 112x112)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Concat with block2 skip (128) = 192 channels
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),  # 192 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling (64, 112x112) -> (64, 224x224)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # Concat with block1 skip (64) = 128 channels
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        
        # Encoder with skip connections
        bottleneck, skip_features = self.encoder(x, return_features=True)
        
        # Print shapes for debugging (remove after fixing)
        # print(f"Bottleneck: {bottleneck.shape}")
        # for k, v in skip_features.items():
        #     print(f"{k}: {v.shape}")
        
        # Decoder with skip connections
        # Level 5
        x = self.up5(bottleneck)
        x = torch.cat([x, skip_features['block5']], dim=1)
        x = self.dec5(x)
        
        # Level 4
        x = self.up4(x)
        x = torch.cat([x, skip_features['block4']], dim=1)
        x = self.dec4(x)
        
        # Level 3
        x = self.up3(x)
        x = torch.cat([x, skip_features['block3']], dim=1)
        x = self.dec3(x)
        
        # Level 2
        x = self.up2(x)
        x = torch.cat([x, skip_features['block2']], dim=1)
        x = self.dec2(x)
        
        # Level 1
        x = self.up1(x)
        x = torch.cat([x, skip_features['block1']], dim=1)
        x = self.dec1(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x