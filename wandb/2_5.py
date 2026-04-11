
import torch
import torch.nn as nn
import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
NUM_IMAGES = 10


def get_confidence(model, image_tensor):
    """
    Derive a confidence score from the classifier's softmax output.
    Uses max softmax probability as a proxy for detection confidence.
    """
    with torch.no_grad():
        cls_out = model.classifier(image_tensor)          # [1, 37]
        probs = torch.softmax(cls_out, dim=1)
        conf = probs.max().item()
    return conf


def compute_iou(pred_box, gt_box):
    """
    Compute IoU between two boxes in [cx, cy, w, h] pixel format.
    Converts to [x1, y1, x2, y2] internally.
    """
    def to_xyxy(box):
        cx, cy, w, h = box
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    px1, py1, px2, py2 = to_xyxy(pred_box)
    gx1, gy1, gx2, gy2 = to_xyxy(gt_box)

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area   = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area


def draw_boxes(image_tensor, pred_box, gt_box):
    """
    Draw GT (green) and Pred (red) boxes on the image.
    Boxes are in [cx, cy, w, h] pixel format (0-224).
    Returns a PIL Image.
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img_pil = T.ToPILImage()(img)
    img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))

    draw = ImageDraw.Draw(img_pil)

    def cx_cy_to_xyxy(box):
        cx, cy, w, h = box
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    # Ground Truth — Green
    gt_xyxy = cx_cy_to_xyxy(gt_box)
    draw.rectangle(gt_xyxy, outline="green", width=3)

    # Prediction — Red
    pred_xyxy = cx_cy_to_xyxy(pred_box)
    draw.rectangle(pred_xyxy, outline="red", width=3)

    return img_pil


def analyze_failure(iou, confidence, threshold_conf=0.7, threshold_iou=0.3):
    """
    Classify a prediction as a failure case.
    Failure = high confidence but low IoU.
    """
    if confidence >= threshold_conf and iou < threshold_iou:
        return " HIGH CONF / LOW IoU (failure)"
    elif iou < 0.3:
        return " LOW IoU"
    elif confidence < 0.4:
        return " LOW CONFIDENCE"
    else:
        return " OK"


def run_task_2_5(root_dir=None):

    # -------- WandB init --------
    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",
        group="TASK_2_5",
        job_type="detection_eval",
        name="TASK_2_5_detection",
        config={
            "task": "2.5",
            "num_images": NUM_IMAGES,
            "image_size": IMAGE_SIZE,
        }
    )

    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    test_dataset = OxfordIIITPetDataset(
        root_dir=root_dir,
        isTrain=False,
        need_box=True        # only samples that have XML bounding boxes
    )
    table = wandb.Table(columns=[
        "index",
        "image",
        "confidence",
        "iou",
        "status",
        "pred_box_cxcywh",
        "gt_box_cxcywh",
    ])

    failure_cases = []

    print(f"Running on {NUM_IMAGES} test images...")

    for i in range(NUM_IMAGES):
        sample = test_dataset[i]

        image_t = sample["image"].unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]
        gt_box  = sample["bbox"].numpy()                    # [cx, cy, w, h] pixels

        with torch.no_grad():
            pred_box_t = model.localizer(image_t)           # [1, 4] pixels

        pred_box = pred_box_t.squeeze(0).cpu().numpy()      # [cx, cy, w, h]

        conf = get_confidence(model, image_t)

        iou = compute_iou(pred_box.tolist(), gt_box.tolist())

        status = analyze_failure(iou, conf)

        annotated_img = draw_boxes(sample["image"], pred_box, gt_box)

        table.add_data(
            i,
            wandb.Image(
                annotated_img,
                caption=f"[{i}] Conf={conf:.2f} | IoU={iou:.2f} | {status}"
            ),
            round(conf, 4),
            round(iou, 4),
            status,
            str(np.round(pred_box, 1).tolist()),
            str(np.round(gt_box, 1).tolist()),
        )

        if "failure" in status.lower() or "LOW IoU" in status:
            failure_cases.append({
                "index": i,
                "conf": conf,
                "iou": iou,
                "status": status,
            })

        print(f"  [{i}] conf={conf:.3f}  iou={iou:.3f}  {status}")

    wandb.log({"detection_results": table})

    all_ious   = [float(table.data[r][3]) for r in range(len(table.data))]
    all_confs  = [float(table.data[r][2]) for r in range(len(table.data))]
    wandb.log({
        "mean_iou":        round(np.mean(all_ious), 4),
        "mean_confidence": round(np.mean(all_confs), 4),
        "num_failures":    len(failure_cases),
    })

    print("\n=== FAILURE CASE ANALYSIS ===")
    if failure_cases:
        for fc in failure_cases:
            print(f"  Image {fc['index']}: conf={fc['conf']:.3f}, iou={fc['iou']:.3f} → {fc['status']}")
        print("\nPossible causes (inspect these images):")
        print("  • Occlusion — part of the pet hidden behind furniture/other objects")
        print("  • Scale mismatch — very small or very large pet relative to image")
        print("  • Complex background — cluttered scenes confuse the regressor")
        print("  • Unusual pose — pet not in canonical upright position")
    else:
        print("  No hard failures found in this sample.")

    print(f"\nMean IoU:        {np.mean(all_ious):.4f}")
    print(f"Mean Confidence: {np.mean(all_confs):.4f}")

    run.finish()
    print("\n Task 2.5 complete — check W&B for the detection table.")


# if __name__ == "__main__":
#     run_task_2_5(root_dir=root_dir)