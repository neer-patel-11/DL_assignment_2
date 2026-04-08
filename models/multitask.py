import torch
import torch.nn as nn
import os

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

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
        gdown.download(id="1NLNyJhUkDmE_prWq0ixzxa4UmaDj8neY", output=localizer_path, quiet=False)
        gdown.download(id="1sw0a8if_hrFzuzRI6v29_LNz22UMDUAg", output=unet_path, quiet=False)

        self.classifier = VGG11Classifier(num_classes=num_breeds)
        self.localizer = VGG11Localizer()
        self.segmentor = VGG11UNet(num_classes=seg_classes)
        self.seg_classes = seg_classes

        self._load_classifier(classifier_path)
        self._load_localizer(localizer_path)
        self._load_segmentor(unet_path)

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
    def _load_segmentor(self, path):
        if os.path.exists(path):
            self.segmentor.load_state_dict(_load_state(path))
            print(" Segmentor loaded")
        else:
            print(" Segmentor checkpoint missing")


    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # Classification
        cls_out = self.classifier(x)

        # Localization
        loc_out = self.localizer(x)

        # segmentation
        seg_out = self.segmentor(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }