"""Training entrypoint
"""
"""Training entrypoint
"""
from data.pets_dataset import get_data_loader
from models.classification import VGG11Classifier

import torch
import torch.optim as optim 
import torch.nn as nn
import os

def train_vgg11(batch_size=32, lr=0.002, epochs=10, device="cuda" if torch.cuda.is_available() else "cpu"):

    train_loader, test_loader = get_data_loader(batch_size=batch_size)

    model = VGG11Classifier().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

        val_loss /= len(test_loader)


        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_metric": best_val_loss,
            }

            torch.save(checkpoint, "checkpoints/classifier.pth")
            print(" Saved best model")


from losses.iou_loss import IoULoss
def compute_iou(pred_boxes, target_boxes, eps=1e-6):
    def to_corners(box):
        x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2
        return xmin, ymin, xmax, ymax

    pxmin, pymin, pxmax, pymax = to_corners(pred_boxes)
    txmin, tymin, txmax, tymax = to_corners(target_boxes)

    ixmin = torch.max(pxmin, txmin)
    iymin = torch.max(pymin, tymin)
    ixmax = torch.min(pxmax, txmax)
    iymax = torch.min(pymax, tymax)

    inter_w = (ixmax - ixmin).clamp(min=0)
    inter_h = (iymax - iymin).clamp(min=0)
    intersection = inter_w * inter_h

    pred_area = (pxmax - pxmin).clamp(min=0) * (pymax - pymin).clamp(min=0)
    target_area = (txmax - txmin).clamp(min=0) * (tymax - tymin).clamp(min=0)

    union = pred_area + target_area - intersection + eps
    iou = intersection / union

    return iou


def train_localizer(
    model,
    train_loader,
    val_loader,
    epochs=25,
    lr=1e-4,
    device=None,
    save_dir="./checkpoints"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = IoULoss()  #  YOUR custom IoU loss

    best_iou = -1.0

    for epoch in range(epochs):
        # ================= TRAIN =================
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            H, W = 224, 224

            preds[:, 0] = torch.sigmoid(preds[:, 0]) * W   # x_center
            preds[:, 1] = torch.sigmoid(preds[:, 1]) * H   # y_center
            preds[:, 2] = torch.sigmoid(preds[:, 2]) * W   # width
            preds[:, 3] = torch.sigmoid(preds[:, 3]) * H   # height
            # optional safety: ensure width/height positive
            preds[:, 2:] = torch.relu(preds[:, 2:])

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)

                H, W = 224, 224

                preds[:, 0] = torch.sigmoid(preds[:, 0]) * W   # x_center
                preds[:, 1] = torch.sigmoid(preds[:, 1]) * H   # y_center
                preds[:, 2] = torch.sigmoid(preds[:, 2]) * W   # width
                preds[:, 3] = torch.sigmoid(preds[:, 3]) * H   # height
                preds[:, 2:] = torch.relu(preds[:, 2:])

                loss = criterion(preds, targets)
                val_loss += loss.item()

                iou = compute_iou(preds, targets)
                val_iou += iou.mean().item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val IoU: {val_iou:.4f}")

        # ================= SAVE BEST =================
        if val_iou > best_iou:
            best_iou = val_iou

            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_iou,
            }

            save_path = os.path.join(save_dir, "localizer.pth")
            torch.save(checkpoint, save_path)

            print(" Saved best model (localizer.pth)")

    print(f"\n🏁 Training complete. Best IoU: {best_iou:.4f}")

# if __name__ == "__main__":
#     train_vgg11(batch_size=64,epochs=1)