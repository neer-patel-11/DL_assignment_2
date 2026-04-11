

import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb

from models.segmentation import VGG11UNet
from data.pets_dataset import get_data_loader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# 🔹 DICE METRIC
# ===============================
def dice_coefficient(pred, target, num_classes=3, eps=1e-7):
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)


# ===============================
# 🔹 FREEZING STRATEGIES
# ===============================
def apply_transfer_strategy(model, strategy):

    encoder = model.encoder  # VGG11 backbone inside UNet

    if strategy == "feature_extractor":
        print("🔒 Freezing entire encoder")
        for param in encoder.parameters():
            param.requires_grad = False

    elif strategy == "partial_finetune":
        print("🔒 Freezing early layers, unfreezing last blocks")

        for name, param in encoder.named_parameters():
            if "block5" in name or "block4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif strategy == "full_finetune":
        print("🔓 Training full model")
        for param in encoder.parameters():
            param.requires_grad = True

    else:
        raise ValueError("Invalid strategy")


def train_segmentation(
    strategy,
    epochs=20,
    batch_size=16,
    lr=1e-3
):

    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",

        group="TASK_2_3",

        job_type=strategy,
        name=f"TASK_2_3_{strategy}",

        config={
            "task": "2.3",
            "strategy": strategy,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size
        }
    )

    train_loader, val_loader = get_data_loader(
        batch_size=batch_size
    )

    model = VGG11UNet(num_classes=3).to(DEVICE)

    apply_transfer_strategy(model, strategy)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    best_dice = 0

    for epoch in range(epochs):

        start_time = time.time()

        model.train()
        train_loss = 0
        train_dice = 0

        for batch in train_loader:

            images = batch["image"].to(DEVICE)

            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_dice += dice_coefficient(preds, masks)

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for batch in val_loader:
                images =  batch["image"].to(DEVICE)
                masks =  batch["mask"].to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_dice += dice_coefficient(preds, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        epoch_time = time.time() - start_time

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
            "epoch_time_sec": epoch_time
        })

        print(f"[{strategy}] Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print("-" * 50)


    run.finish()


# if __name__ == "__main__":


# 1 Strict Feature Extractor
# train_segmentation("feature_extractor")

# 2 Partial Fine-Tuning
# train_segmentation("partial_finetune")

# 3 Full Fine-Tuning
# train_segmentation("full_finetune")