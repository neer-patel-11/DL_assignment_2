
import torch
import torch.nn as nn
import numpy as np
import wandb
from PIL import Image
import torchvision.transforms as T

from models.multitask import MultiTaskPerceptionModel
from data.dataset import OxfordIIITPetDataset   

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VISUAL = 5          # images to visualise
NUM_EVAL   = 50         # images to compute aggregate metrics over
NUM_CLASSES = 3

TRIMAP_PALETTE = {
    0: (0,   0,   128),   # background  → dark blue
    1: (255, 140,   0),   # foreground  → orange
    2: (200, 200, 200),   # boundary    → light grey
}


def mask_to_pil(mask_np):
    """mask_np: H×W numpy array with values 0/1/2 → RGB PIL Image."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in TRIMAP_PALETTE.items():
        rgb[mask_np == cls_id] = colour
    return Image.fromarray(rgb)


def tensor_to_pil(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return T.ToPILImage()(img)


def pixel_accuracy(pred, target):
    """pred, target: [H, W] or [B, H, W] LongTensor."""
    correct = (pred == target).sum().item()
    total   = target.numel()
    return correct / total


def dice_coefficient(pred, target, num_classes=NUM_CLASSES, eps=1e-7):
    dice_scores = []
    for c in range(num_classes):
        pred_c   = (pred   == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union        = pred_c.sum() + target_c.sum()
        dice_scores.append(((2.0 * intersection + eps) / (union + eps)).item())
    return sum(dice_scores) / len(dice_scores)


def class_distribution(mask_np, num_classes=NUM_CLASSES):
    total = mask_np.size
    return {f"cls{c}_pct": round(100 * (mask_np == c).sum() / total, 2)
            for c in range(num_classes)}


def run_task_2_6(root_dir=None):

    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",
        group="TASK_2_6",
        job_type="segmentation_eval",
        name="TASK_2_6_segmentation",
        config={
            "task":        "2.6",
            "num_visual":  NUM_VISUAL,
            "num_eval":    NUM_EVAL,
            "num_classes": NUM_CLASSES,
        }
    )

    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    test_dataset = OxfordIIITPetDataset(root_dir=root_dir, isTrain=False)

    
    visual_table = wandb.Table(columns=[
        "index", "original", "gt_trimap", "pred_trimap",
        "pixel_accuracy", "dice_score",
        "bg_pct", "fg_pct", "boundary_pct",
    ])

    print(f"Logging {NUM_VISUAL} visual samples...")

    for i in range(NUM_VISUAL):
        sample   = test_dataset[i]
        image_t  = sample["image"].unsqueeze(0).to(DEVICE)  # [1,3,224,224]
        gt_mask  = sample["mask"]                            # [224,224] long

        with torch.no_grad():
            logits   = model.segmentor(image_t)              # [1,3,224,224]
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu()   # [224,224]

        pa   = pixel_accuracy(pred_mask, gt_mask)
        dice = dice_coefficient(pred_mask, gt_mask)

        gt_np   = gt_mask.numpy()
        pred_np = pred_mask.numpy()
        dist    = class_distribution(gt_np)

        original_pil = tensor_to_pil(sample["image"])
        gt_pil       = mask_to_pil(gt_np)
        pred_pil     = mask_to_pil(pred_np)

        visual_table.add_data(
            i,
            wandb.Image(original_pil,
                        caption=f"[{i}] Original"),
            wandb.Image(gt_pil,
                        caption=f"[{i}] GT  | bg={dist['cls0_pct']}% fg={dist['cls1_pct']}% bnd={dist['cls2_pct']}%"),
            wandb.Image(pred_pil,
                        caption=f"[{i}] Pred | PA={pa:.3f}  Dice={dice:.3f}"),
            round(pa,   4),
            round(dice, 4),
            dist["cls0_pct"],
            dist["cls1_pct"],
            dist["cls2_pct"],
        )

        print(f"  [{i}] PA={pa:.4f}  Dice={dice:.4f}  "
              f"(bg={dist['cls0_pct']}%  fg={dist['cls1_pct']}%  bnd={dist['cls2_pct']}%)")

    wandb.log({"segmentation_visuals": visual_table})


    print(f"\nComputing aggregate metrics over {NUM_EVAL} samples...")

    all_pa, all_dice = [], []
    class_dice = {c: [] for c in range(NUM_CLASSES)}

    for i in range(min(NUM_EVAL, len(test_dataset))):
        sample  = test_dataset[i]
        image_t = sample["image"].unsqueeze(0).to(DEVICE)
        gt_mask = sample["mask"]

        with torch.no_grad():
            logits    = model.segmentor(image_t)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu()

        all_pa.append(pixel_accuracy(pred_mask, gt_mask))
        all_dice.append(dice_coefficient(pred_mask, gt_mask))

        # Per-class dice
        for c in range(NUM_CLASSES):
            pred_c   = (pred_mask == c).float()
            target_c = (gt_mask   == c).float()
            inter    = (pred_c * target_c).sum()
            union    = pred_c.sum() + target_c.sum()
            class_dice[c].append(((2 * inter + 1e-7) / (union + 1e-7)).item())

    mean_pa        = np.mean(all_pa)
    mean_dice      = np.mean(all_dice)
    per_class_dice = {c: np.mean(v) for c, v in class_dice.items()}

    wandb.log({
        "mean_pixel_accuracy":     round(mean_pa,   4),
        "mean_dice_score":         round(mean_dice, 4),
        "dice_background":         round(per_class_dice[0], 4),
        "dice_foreground":         round(per_class_dice[1], 4),
        "dice_boundary":           round(per_class_dice[2], 4),
        "pa_vs_dice_gap":          round(mean_pa - mean_dice, 4),
    })

    print(f"\n{'='*45}")
    print(f"  Mean Pixel Accuracy : {mean_pa:.4f}")
    print(f"  Mean Dice Score     : {mean_dice:.4f}")
    print(f"  Gap (PA - Dice)     : {mean_pa - mean_dice:.4f}  ← imbalance artefact")
    print(f"  Dice  background    : {per_class_dice[0]:.4f}")
    print(f"  Dice  foreground    : {per_class_dice[1]:.4f}")
    print(f"  Dice  boundary      : {per_class_dice[2]:.4f}")
    print(f"{'='*45}")

    
    explanation = """
## Why Pixel Accuracy is artificially high on trimaps

### Pixel distribution in Oxford-IIIT Pet trimaps
The trimap has 3 classes: background (0), foreground/pet (1), boundary (2).
Background typically dominates — often ~60-70% of pixels per image.

### The mathematical problem with Pixel Accuracy

    PA = (correctly classified pixels) / (total pixels)

If a naive model predicts *everything* as background (the majority class),
it trivially achieves PA ≈ 0.65 while learning nothing about the pet region.

### Why Dice is immune to this

For class c:

    Dice_c = (2 × |Pred_c ∩ GT_c|) / (|Pred_c| + |GT_c|)

If the model predicts zero foreground pixels:
    Pred_foreground = 0  →  Dice_foreground = 0

Dice collapses to 0 for any ignored class, regardless of how well the
majority class is handled. The mean Dice across classes therefore penalises
the model harshly for missing minority classes (boundary, foreground).

### Summary
| Metric          | Model predicts all-background | Balanced model |
|-----------------|-------------------------------|----------------|
| Pixel Accuracy  | ~0.65 (looks OK!)             | ~0.80          |
| Mean Dice       | ~0.22 (exposed!)              | ~0.72          |

Dice is the correct metric whenever class imbalance exists, because it
evaluates every class on equal footing independently of its pixel count.
"""

    print(explanation)
    wandb.log({"explanation_dice_vs_pa": wandb.Html(
        explanation.replace("\n", "<br>").replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
    )})

    run.finish()
    print("\n✅ Task 2.6 complete — check W&B for visuals and metrics.")


# if __name__ == "__main__":
#     root_dir = None  
#     run_task_2_6(root_dir=root_dir)