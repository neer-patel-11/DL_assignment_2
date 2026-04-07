
from data.pets_dataset import get_data_loader
from models.classification import VGG11Classifier


import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import os


def train_vgg11(batch_size=32, lr=1e-4, epochs=40, device="cuda" if torch.cuda.is_available() else "cpu"):

    train_loader, test_loader = get_data_loader(batch_size=batch_size,test_size=0.1)

    model = VGG11Classifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0

    os.makedirs("/content/checkpoints", exist_ok=True)

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0

        all_train_preds = []
        all_train_labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # outputs = model(x)
            # loss = criterion(outputs, y)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()   # <-- BEFORE forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(y.cpu().numpy())

        train_loss /= len(train_loader)

        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")
        train_acc = accuracy_score(all_train_labels, all_train_preds)


        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0

        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())

        val_loss /= len(test_loader)

        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
        val_acc = accuracy_score(all_val_labels, all_val_preds)


        # ================= LOG =================
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}"
        )


        # ================= SAVE BEST =================
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_f1": best_val_f1,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }

            torch.save(checkpoint, "/checkpoints/classifier.pth")
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



from models.segmentation import VGG11UNet
from data.segmentation_dataset import get_segmentation_data_loader


def dice_coefficient(pred, target, num_classes=3, eps=1e-7):
    """
    Calculate Dice coefficient for evaluation.
    
    Args:
        pred: Predicted segmentation [B, H, W]
        target: Ground truth segmentation [B, H, W]
        num_classes: Number of classes
        eps: Small epsilon for numerical stability
    
    Returns:
        Mean Dice coefficient across all classes
    """
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())
    
    return sum(dice_scores) / len(dice_scores)


def train_vgg11_unet(
    root_dir,
    batch_size=16,
    lr=0.001,
    epochs=50,
    image_size=224,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = get_segmentation_data_loader(
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model = VGG11UNet(num_classes=3, in_channels=3).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function: Cross-Entropy Loss
    # We use ignore_index for border class if needed, or train on all 3 classes
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float("inf")
    best_dice = 0.0
    
    print(f"Training on {device}...")
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total test samples: {len(test_loader.dataset)}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            
            # Calculate Dice coefficient
            preds = torch.argmax(outputs, dim=1)
            dice = dice_coefficient(preds, masks, num_classes=3)
            train_dice += dice
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} Dice: {dice:.4f}")
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Metrics
                val_loss += loss.item()
                
                # Calculate Dice coefficient
                preds = torch.argmax(outputs, dim=1)
                dice = dice_coefficient(preds, masks, num_classes=3)
                val_dice += dice
        
        val_loss /= len(test_loader)
        val_dice /= len(test_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print("=" * 60)
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print("=" * 60)
        
        # Save best model based on Dice coefficient
        if val_dice > best_dice:
            best_dice = val_dice
            best_val_loss = val_loss
            
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_dice": best_dice,
                "best_val_loss": best_val_loss,
            }
            
            torch.save(checkpoint, "checkpoints/unet_segmentation_best.pth")
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
        
        print()
    
    print("Training complete!")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


# if __name__ == "__main__":
#     # Example usage
#     train_vgg11_unet(
#         root_dir="path/to/oxford-iiit-pet",  # Update this path
#         batch_size=16,
#         lr=0.001,
#         epochs=50,
#         image_size=224
#     )