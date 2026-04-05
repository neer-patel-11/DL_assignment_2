"""Inference script for MultiTaskPerceptionModel"""

import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.multitask import MultiTaskPerceptionModel

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BREED_NAMES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay",
    "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired",
    "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
    "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue",
    "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu",
    "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]


def preprocess(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0)


def decode_bbox(raw: torch.Tensor) -> dict:
    t = raw[0]
    return {
        "x_center": round(torch.sigmoid(t[0]).item() * IMAGE_SIZE, 2),
        "y_center": round(torch.sigmoid(t[1]).item() * IMAGE_SIZE, 2),
        "width":    round(torch.sigmoid(t[2]).item() * IMAGE_SIZE, 2),
        "height":   round(torch.sigmoid(t[3]).item() * IMAGE_SIZE, 2),
    }


def predict(
    image_path:      str,
    classifier_path: str,
    localizer_path:  str,
    unet_path:       str,
    device:          str = "cpu",
):
    # 1. Build model and load all three checkpoints
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=3,
        # classifier_path=classifier_path,
        # localizer_path=localizer_path,
        # unet_path=unet_path,
    )
    model.to(device).eval()

    # 2. Preprocess
    x = preprocess(image_path).to(device)

    # 3. Forward pass
    with torch.no_grad():
        outputs = model(x)

    # 4. Classification
    cls_logits = outputs["classification"]
    cls_idx    = cls_logits.argmax(dim=1).item()
    cls_conf   = cls_logits.softmax(dim=1)[0, cls_idx].item()
    breed      = BREED_NAMES[cls_idx]

    # 5. Localization
    bbox = decode_bbox(outputs["localization"])

    # 6. Segmentation
    seg_map = torch.argmax(outputs["segmentation"][0], dim=0).cpu().numpy()
    seg_classes, seg_counts = np.unique(seg_map, return_counts=True)
    seg_summary = {int(c): int(n) for c, n in zip(seg_classes, seg_counts)}

    # 7. Print
    print("=" * 45)
    print(f"  Image      : {image_path}")
    print("-" * 45)
    print(f"  Breed      : {breed}")
    print(f"  Confidence : {cls_conf:.2%}")
    print("-" * 45)
    print(f"  BBox (224px space)")
    print(f"    x_center : {bbox['x_center']}")
    print(f"    y_center : {bbox['y_center']}")
    print(f"    width    : {bbox['width']}")
    print(f"    height   : {bbox['height']}")
    print("-" * 45)
    print(f"  Segmentation pixel counts")
    for cls_id, count in seg_summary.items():
        label = ["Foreground", "Background", "Border"][cls_id]
        print(f"    {cls_id} ({label:10s}) : {count}")
    print("=" * 45)

    return {
        "breed":       breed,
        "confidence":  cls_conf,
        "bbox":        bbox,
        "seg_map":     seg_map,
    }


if __name__ == "__main__":
    # Usage:
    # python inference.py image.jpg classifier.pth localizer.pth unet.pth [cpu|cuda]
    predict(
        image_path      = sys.argv[1],
        # classifier_path = sys.argv[2],
        # localizer_path  = sys.argv[3],
        # unet_path       = sys.argv[4],
        device          = sys.argv[5] if len(sys.argv) > 5 else "cpu",
    )