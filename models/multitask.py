import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier


def _load_state(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("state_dict", ckpt)


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
        dropout_p: float = 0.0,
    ):
        super().__init__()

        # Download checkpoints  
        import gdown
        gdown.download(id="1zvfoMy1ds9v1Df9ZeSanBHK8XVu045WD", output=classifier_path, quiet=False)
        gdown.download(id="1BpOe9YyojShsXoSTBvdBflrubHEF2wQK", output=localizer_path, quiet=False)
        gdown.download(id="1H59EmgH6IACggQ_jaSY0OAGPwz2Ai7WU", output=unet_path, quiet=False)

        # Shared backbone 
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head 
        self.classification_head = nn.Sequential(
            nn.Flatten(),                        
            nn.Linear(512 * 7 * 7, 4096),        
            nn.ReLU(inplace=True),               
            CustomDropout(dropout_p),            
            nn.Linear(4096, 4096),               
            nn.ReLU(inplace=True),               
            CustomDropout(dropout_p),            
            nn.Linear(4096, num_breeds),         
        )

        # Stub heads 
        self.loc_head = nn.Identity()
        self.seg_final = nn.Identity()

        # Load weights 
        self._load_classifier(classifier_path)


    def _load_classifier(self, path: str):
        """
        Clean loading:
        1. Load full classifier
        2. Copy encoder + classifier weights
        """

        # Load pretrained classifier
        clf = VGG11Classifier()
        clf.load_state_dict(_load_state(path))

        # Copy encoder weights
        self.encoder.load_state_dict(clf.features.state_dict())

        # Copy classification head weights
        self.classification_head.load_state_dict(clf.classifier.state_dict())

        print("[classifier] Loaded encoder + classification head successfully")


    def forward(self, x: torch.Tensor):
        B = x.size(0)

        # Encoder → [B, 512, 7, 7]
        bottleneck = self.encoder(x, return_features=False)

        # Classification
        cls_out = self.classification_head(bottleneck)

        # Stub outputs
        loc_out = torch.zeros(B, 4, device=x.device, dtype=x.dtype)
        seg_out = torch.zeros(
            B, 3, x.size(2), x.size(3),
            device=x.device, dtype=x.dtype
        )

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }