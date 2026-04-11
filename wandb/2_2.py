
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from data.pets_dataset import get_data_loader
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout   # your custom dropout
from models.classification import VGG11Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in loader:

        images = torch.stack([item["image"] for item in batch]).to(DEVICE)
        labels = torch.stack([item["label"] for item in batch]).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:

            images = torch.stack([item["image"] for item in batch]).to(DEVICE)
            labels = torch.stack([item["label"] for item in batch]).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


def run_experiment(dropout_p, epochs=10, lr=0.01):

    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",

        group="TASK_2_2",

        # Helps separation inside group
        job_type=f"dropout_{dropout_p}",

        name=f"TASK_2_2_dropout_{dropout_p}",

        config={
            "task": "2.2",
            "dropout_p": dropout_p,
            "learning_rate": lr,
            "epochs": epochs,
            "dataset": "OxfordPets"
        }
    )

    train_loader, val_loader = get_data_loader(batch_size=64)

    model = VGG11Classifier(dropout_p=dropout_p).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f"[Dropout {dropout_p}] Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    run.finish()


# if __name__ == "__main__":

#     # 1 No Dropout
#     run_experiment(dropout_p=0.0)

#     # 2 Moderate Dropout
#     run_experiment(dropout_p=0.2)

#     # 3 High Dropout
#     run_experiment(dropout_p=0.5)

