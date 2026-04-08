import torch
import torch.nn as nn
import os

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer


def _load_state(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("state_dict", ckpt)


class MultiTaskPerceptionModel(nn.Module):
    """
    Evaluation-ready model:
    - Classification → real pretrained model
    - Localization → real pretrained model
    - Segmentation → dummy (zeros)
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super().__init__()

        import gdown

        os.makedirs(
            os.path.dirname(classifier_path) if os.path.dirname(classifier_path) else "checkpoints",
            exist_ok=True
        )

        gdown.download(id="19Grc7A4q9J6Dq9dJM9w9VOscjQu_V-o9", output=classifier_path, quiet=False)
        gdown.download(id="1dLju5vUvjlij2Q_Xq1dwtIWKx7tjm0In", output=localizer_path, quiet=False)
        gdown.download(id="1H59EmgH6IACggQ_jaSY0OAGPwz2Ai7WU", output=unet_path, quiet=False)

        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer = VGG11Localizer()

        # self.seg_classes = seg_classes

        # -------- LOAD WEIGHTS --------
        self._load_classifier(classifier_path)
        self._load_localizer(localizer_path)

    # ---------------- LOADERS ----------------

    def _load_classifier(self, path):
        if os.path.exists(path):
            self.classifier.load_state_dict(_load_state(path))
            print(" Classifier loaded")
        else:
            print(" Classifier checkpoint missing")

    def _load_localizer(self, path):
        if os.path.exists(path):
            self.localizer.load_state_dict(_load_state(path))
            print(" Localizer loaded")
        else:
            print(" Localizer checkpoint missing")

    # ---------------- FORWARD ----------------

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # Classification
        cls_out = self.classifier(x)

        # Localization
        loc_out = self.localizer(x)

        # Dummy segmentation
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