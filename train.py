
from data.pets_dataset import get_data_loader
from models.classification import VGG11Classifier
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import os
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss
from models.segmentation import VGG11UNet
from data.pets_dataset import get_data_loader

def train_vgg11(batch_size=64, lr=1e-4, epochs=40,
                device="cuda" if torch.cuda.is_available() else "cpu"):

    train_loader, test_loader = get_data_loader(batch_size=batch_size)

    model = VGG11Classifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = "/content/drive/MyDrive/DL-assignment_2/checkpoints/classifier.pth"

    start_epoch = 0
    best_val_f1 = 0.0

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if os.path.exists(checkpoint_path):
        print("Loading existing checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer state loaded")
        else:
            print(" No optimizer state found → starting fresh optimizer")

        start_epoch = checkpoint.get("epoch", 0)
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)

        print(f" Resumed from epoch {start_epoch} | Best F1: {best_val_f1:.4f}")

    for epoch in range(start_epoch, epochs):

        model.train()
        train_loss = 0.0

        all_train_preds = []
        all_train_labels = []

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()

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

        model.eval()
        val_loss = 0.0

        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())

        val_loss /= len(test_loader)

        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
        val_acc = accuracy_score(all_val_labels, all_val_preds)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),   
                "epoch": epoch + 1,
                "best_val_f1": best_val_f1,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }

            torch.save(checkpoint, checkpoint_path)
            print(" Saved best model")


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

def is_valid_bbox(bbox):
    """
    bbox: [cx, cy, w, h] (pixel space)
    filter full image boxes
    """
    cx, cy, w, h = bbox

    # full image (approx)
    if w >= 0.95 * 224 and h >= 0.95 * 224:
        return False

    return True


def train_localizer(batch_size=64, lr=1e-4, epochs=40,
                    device="cuda" if torch.cuda.is_available() else "cpu"):

    train_loader, test_loader = get_data_loader(batch_size=batch_size)

    model = VGG11Localizer().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.IoULoss()  

    checkpoint_path = "/content/drive/MyDrive/DL-assignment_2/checkpoints/localizer.pth"

    start_epoch = 0
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if os.path.exists(checkpoint_path):
        print(" Loading checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(" Optimizer loaded")
        else:
            print(" No optimizer found")

        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f" Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):

        model.train()
        train_loss = 0.0

        total_samples = 0
        valid_samples = 0
        skipped_samples = 0

        for batch in train_loader:

            x = batch["image"].to(device)
            bbox = batch["bbox"].to(device)

            valid_idx = []
            for i in range(len(bbox)):
                if is_valid_bbox(bbox[i]):
                    valid_idx.append(i)

            if len(valid_idx) == 0:
                skipped_samples += len(bbox)
                continue

            x = x[valid_idx]
            bbox = bbox[valid_idx]

            valid_samples += len(valid_idx)
            total_samples += len(valid_idx)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, bbox)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        print(f"\n Train Stats:")
        print(f"Total used: {valid_samples}")
        print(f"Skipped: {skipped_samples}")

        model.eval()
        val_loss = 0.0

        val_valid = 0
        val_skipped = 0

        with torch.no_grad():
            for batch in test_loader:

                x = batch["image"].to(device)
                bbox = batch["bbox"].to(device)

                valid_idx = []
                for i in range(len(bbox)):
                    if is_valid_bbox(bbox[i]):
                        valid_idx.append(i)

                if len(valid_idx) == 0:
                    val_skipped += len(bbox)
                    continue

                x = x[valid_idx]
                bbox = bbox[valid_idx]

                val_valid += len(valid_idx)

                outputs = model(x)

                loss = criterion(outputs, bbox)

                val_loss += loss.item()

        val_loss /= len(test_loader)

        print(f" Val Stats:")
        print(f"Used: {val_valid} | Skipped: {val_skipped}")

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} || "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
            }

            torch.save(checkpoint, checkpoint_path)
            print(" Saved best localizer")



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
    train_loader, test_loader = get_data_loader(
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

