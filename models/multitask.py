import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
import os


def _load_state(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("state_dict", ckpt)


class MultiTaskPerceptionModel(nn.Module):
    """
    Evaluation-ready model:
    - Real: classification
    - Dummy: localization + segmentation (zeros)
    """

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

        # -------- DOWNLOAD CHECKPOINTS (REQUIRED) --------         import gdown
        import gdown
        os.makedirs(
            os.path.dirname(classifier_path) if os.path.dirname(classifier_path) else "checkpoints",
            exist_ok=True
        )

        gdown.download(id="19Grc7A4q9J6Dq9dJM9w9VOscjQu_V-o9", output=classifier_path, quiet=False)
        gdown.download(id="1BpOe9YyojShsXoSTBvdBflrubHEF2wQK", output=localizer_path, quiet=False)
        gdown.download(id="1H59EmgH6IACggQ_jaSY0OAGPwz2Ai7WU", output=unet_path, quiet=False)

        # -------- BACKBONE --------
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # -------- MATCH CLASSIFIER --------
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classification_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # -------- LOAD PRETRAINED --------
        self._load_classifier(classifier_path)

        self.seg_classes = seg_classes

    def _load_classifier(self, path: str):
        """
        Load pretrained classifier weights cleanly
        """
        clf = VGG11Classifier()
        clf.load_state_dict(_load_state(path))

        # copy encoder
        self.encoder.load_state_dict(clf.features.state_dict())

        # copy classifier
        self.classification_head.load_state_dict(clf.classifier.state_dict())

        print(" Loaded pretrained classifier weights")

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # -------- ENCODER --------
        x = self.encoder(x, return_features=False)

        # -------- CLASSIFICATION --------
        pooled = self.avgpool(x)
        flat = torch.flatten(pooled, 1)
        cls_out = self.classification_head(flat)

        # -------- DUMMY OUTPUTS --------
        loc_out = torch.zeros(B, 4, device=x.device, dtype=x.dtype)

        seg_out = torch.zeros(
            B, self.seg_classes, H, W,
            device=x.device,
            dtype=x.dtype
        )

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }