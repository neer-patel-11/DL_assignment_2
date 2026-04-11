
import os
import torch
import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.multitask import MultiTaskPerceptionModel

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 224
NUM_CLASSES = 3   # segmentation classes
WILD_IMAGES = [
    "/test_images/1.jpeg",
    "/test_images/2.jpg",
    "/test_images/3.jpeg",
]

# Oxford-IIIT breed names (37 classes, index 0-36)
BREED_NAMES = [
    "Abyssinian","Bengal","Birman","Bombay","British Shorthair",
    "Egyptian Mau","Maine Coon","Persian","Ragdoll","Russian Blue",
    "Siamese","Sphynx","American Bulldog","American Pit Bull Terrier",
    "Basset Hound","Beagle","Boxer","Chihuahua","English Cocker Spaniel",
    "English Setter","German Shorthaired","Great Pyrenees","Havanese",
    "Japanese Chin","Keeshond","Leonberger","Miniature Pinscher",
    "Newfoundland","Pomeranian","Pug","Saint Bernard","Samoyed",
    "Scottish Terrier","Shiba Inu","Staffordshire Bull Terrier",
    "Wheaten Terrier","Yorkshire Terrier",
]

# Trimap colour map: 0=background, 1=foreground/pet, 2=boundary
TRIMAP_COLOURS = {
    0: (0,   0,   128),
    1: (255, 140,   0),
    2: (200, 200, 200),
}


# ===============================
# 🔹 PREPROCESS
# ===============================
def preprocess(img_path):
    """Load any image and return a normalised [1,3,224,224] tensor."""
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.open(img_path).convert("RGB")
    return transform(img_pil).unsqueeze(0), img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))


# ===============================
# 🔹 DENORMALISE TENSOR → PIL
# ===============================
def denorm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return T.ToPILImage()((tensor.cpu() * std + mean).clamp(0,1))


# ===============================
# 🔹 DRAW BOUNDING BOX
# ===============================
def draw_bbox(img_pil, box_cxcywh, colour="red", width=3):
    """box_cxcywh: [cx, cy, w, h] in pixels (0-224)."""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    cx, cy, bw, bh = box_cxcywh
    x1 = max(0, cx - bw / 2)
    y1 = max(0, cy - bh / 2)
    x2 = min(IMAGE_SIZE, cx + bw / 2)
    y2 = min(IMAGE_SIZE, cy + bh / 2)
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
    return img


# ===============================
# 🔹 SEGMENTATION MASK → PIL
# ===============================
def mask_to_pil(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in TRIMAP_COLOURS.items():
        rgb[mask_np == cls_id] = colour
    return Image.fromarray(rgb)


# ===============================
# 🔹 SEGMENTATION OVERLAY
# ===============================
def overlay_mask(img_pil, mask_np, alpha=0.5):
    """Blend colour-coded mask over original image."""
    mask_pil = mask_to_pil(mask_np).resize(img_pil.size)
    return Image.blend(img_pil.convert("RGB"), mask_pil, alpha=alpha)


# ===============================
# 🔹 CONFIDENCE SCORE (from classifier)
# ===============================
def top3_preds(logits):
    probs = torch.softmax(logits.squeeze(0), dim=0)
    top3  = probs.topk(3)
    return [(BREED_NAMES[idx.item()], round(prob.item()*100, 2))
            for prob, idx in zip(top3.values, top3.indices)]


# ===============================
# 🔹 DRAW CLASSIFICATION RESULT
# ===============================
def draw_classification(img_pil, top3):
    """Add top-3 breed labels directly on the image."""
    img = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Semi-transparent dark banner at the bottom
    banner_h = 60
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle(
        [0, IMAGE_SIZE - banner_h, IMAGE_SIZE, IMAGE_SIZE],
        fill=(0, 0, 0, 160)
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    y = IMAGE_SIZE - banner_h + 6
    for rank, (breed, pct) in enumerate(top3, 1):
        label = f"#{rank} {breed}: {pct:.1f}%"
        draw.text((6, y), label, fill=(255, 255, 255))
        y += 17

    return img


# ===============================
# 🔹 GENERALIZATION COMMENT
# ===============================
def generalization_comment(img_idx, conf, bbox, mask_np):
    """
    Auto-generate a brief generalization note based on metrics.
    """
    cx, cy, bw, bh = bbox
    box_area_pct = round(100 * (bw * bh) / (IMAGE_SIZE ** 2), 1)
    fg_pct = round(100 * (mask_np == 1).sum() / mask_np.size, 1)
    bg_pct = round(100 * (mask_np == 0).sum() / mask_np.size, 1)

    notes = []

    # Classification confidence
    if conf >= 70:
        notes.append(f"✅ Classifier confident ({conf:.1f}% top-1) — breed generalised well.")
    elif conf >= 40:
        notes.append(f"⚠️ Moderate classifier confidence ({conf:.1f}%) — likely unseen breed/style.")
    else:
        notes.append(f"❌ Low classifier confidence ({conf:.1f}%) — model struggled with this image.")

    # Bounding box coverage
    if box_area_pct < 10:
        notes.append(f"❌ BBox very small ({box_area_pct}% of image) — localizer may have failed.")
    elif box_area_pct > 80:
        notes.append(f"⚠️ BBox covers {box_area_pct}% of image — likely too loose (complex background?).")
    else:
        notes.append(f"✅ BBox covers {box_area_pct}% — reasonable crop for the classifier.")

    # Segmentation foreground coverage
    if fg_pct < 15:
        notes.append(f"❌ Segmentor predicted little foreground ({fg_pct}%) — U-Net may have collapsed.")
    elif fg_pct > 70:
        notes.append(f"⚠️ Foreground ({fg_pct}%) seems over-predicted — unusual background?")
    else:
        notes.append(f"✅ Foreground ({fg_pct}%) looks plausible.")

    return "\n".join(notes)


# ===============================
# 🔹 MAIN
# ===============================
def run_task_2_7():

    # -------- W&B init --------
    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",
        group="TASK_2_7",
        job_type="pipeline_showcase",
        name="TASK_2_7_wild_pipeline",
        config={
            "task":       "2.7",
            "num_images": len(WILD_IMAGES),
            "image_size": IMAGE_SIZE,
        }
    )

    # -------- Model --------
    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    # -------- W&B Table --------
    table = wandb.Table(columns=[
        "image_id",
        "original",
        "classification_result",
        "localization_result",
        "segmentation_mask",
        "segmentation_overlay",
        "top1_breed",
        "top1_conf_%",
        "bbox_area_%",
        "fg_pct_%",
        "generalization_notes",
    ])

    print("Running pipeline on in-the-wild images...\n")

    for img_idx, img_path in enumerate(WILD_IMAGES):

        if not os.path.exists(img_path):
            print(f"  ⚠️  {img_path} not found — skipping.")
            continue

        print(f"[{img_idx+1}/{len(WILD_IMAGES)}] {img_path}")

        # -------- Preprocess --------
        image_t, img_224 = preprocess(img_path)
        image_t = image_t.to(DEVICE)

        # -------- Forward (single call to MultiTaskPerceptionModel) --------
        with torch.no_grad():
            outputs = model(image_t)

        cls_logits = outputs["classification"]   # [1, 37]
        loc_box    = outputs["localization"]     # [1, 4]  cx,cy,w,h pixels
        seg_logits = outputs["segmentation"]     # [1, 3, 224, 224]

        # -------- Post-process --------
        top3     = top3_preds(cls_logits)
        top1_conf = top3[0][1]

        bbox     = loc_box.squeeze(0).cpu().numpy()          # [4]
        pred_mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [224,224]

        # -------- Build output images --------
        orig_pil  = img_224.copy()

        cls_pil   = draw_classification(img_224.copy(), top3)

        loc_pil   = draw_bbox(img_224.copy(), bbox, colour="red", width=3)

        seg_pil   = mask_to_pil(pred_mask)

        ov_pil    = overlay_mask(img_224.copy(), pred_mask, alpha=0.45)

        # -------- Generalization note --------
        note = generalization_comment(img_idx, top1_conf, bbox, pred_mask)
        print(f"  {note.replace(chr(10), '  |  ')}\n")

        box_area_pct = round(100 * (bbox[2] * bbox[3]) / (IMAGE_SIZE**2), 1)
        fg_pct       = round(100 * (pred_mask == 1).sum() / pred_mask.size, 1)

        # -------- Add to table --------
        table.add_data(
            img_idx + 1,
            wandb.Image(orig_pil,
                        caption=f"Original — {os.path.basename(img_path)}"),
            wandb.Image(cls_pil,
                        caption=f"Top-1: {top3[0][0]} ({top1_conf:.1f}%)"),
            wandb.Image(loc_pil,
                        caption=f"BBox area: {box_area_pct}%"),
            wandb.Image(seg_pil,
                        caption="Trimap mask (orange=pet, blue=bg, grey=boundary)"),
            wandb.Image(ov_pil,
                        caption="Segmentation overlay (α=0.45)"),
            top3[0][0],
            top1_conf,
            box_area_pct,
            fg_pct,
            note,
        )

    wandb.log({"pipeline_showcase": table})

    # -------- Written evaluation (logged as W&B text artifact) --------
    evaluation_text = """
# Task 2.7 — Generalization Evaluation

## Pipeline Overview
Each image passes through the full MultiTaskPerceptionModel in a single forward call,
producing three simultaneous outputs: breed classification, bounding box, and trimap segmentation.

## What to look for per task

### Classification
The classifier uses softmax confidence as a proxy for certainty.
In-the-wild images often have:
  - Unusual lighting or camera angle → confidence drop
  - Mixed-breed or uncommon coats → top-1 wrong but correct genus in top-3
  - High confidence wrong answers (overconfidence) → common in fine-grained models

### Localization
The regressor outputs a single box in [cx, cy, w, h] pixel format.
Failure modes on wild images:
  - Very small pets in large scenes → box drifts to image centre
  - Multiple animals → box averages across subjects
  - Cluttered / patterned backgrounds → false activation regions

### Segmentation (U-Net)
The U-Net was trained on clean Oxford-IIIT images with relatively uniform backgrounds.
In-the-wild failures:
  - Non-standard lighting (harsh shadows, over-exposure) → boundary class collapses
  - Complex backgrounds (grass, patterned furniture) → background misclassified as foreground
  - Unusual poses (crouched, turned away) → foreground under-predicted

## Colour legend for trimap
  Orange  = foreground (pet body)
  Dark blue = background
  Light grey = uncertain boundary region
"""
    artifact = wandb.Artifact("task_2_7_evaluation", type="report")
    with artifact.new_file("evaluation.md", mode="w") as f:
        f.write(evaluation_text)
    run.log_artifact(artifact)

    print(evaluation_text)
    run.finish()
    print("\n Task 2.7 complete — check W&B for the pipeline showcase table.")


# ===============================
# 🔹 ENTRY POINT
# ===============================
if __name__ == "__main__":
    run_task_2_7()