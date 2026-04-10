
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
from data.pets_dataset import get_data_loader
from models.classification import VGG11Classifier



class VGG11EncoderNoBN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.block1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = nn.Sequential(
            conv_block(256, 512),
            conv_block(512, 512)
        )
        self.pool4 = nn.MaxPool2d(2)

        self.block5 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))
        return x




class VGG11ClassifierNoBN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.features = VGG11EncoderNoBN()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),

            nn.Linear(4096, 4096),
            nn.ReLU(True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_activation_hook(storage_dict, name):
    def hook(model, input, output):
        storage_dict[name] = output.detach().cpu()
    return hook

# train_task_2_1.py




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, activations=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    captured = False

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def log_activation_distribution(activations, tag):
    act = activations.flatten().numpy()

    plt.figure()
    plt.hist(act, bins=50)
    plt.title(f"Activation Distribution - {tag}")

    wandb.log({f"{tag}_activation_hist": wandb.Image(plt)})
    plt.close()


def run_experiment(use_bn=True, lr=1e-4,epochs=20):

    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",
        group="TASK_2_1",
        name=f"TASK_2_1_{'BN' if use_bn else 'NO_BN'}_lr_{lr}",
        config={
            "task": "2.1",
            "use_batchnorm": use_bn,
            "learning_rate": lr,
            "epochs": epochs,
            "dataset": "OxfordPets"
        }
    )

    train_loader, val_loader = get_data_loader(batch_size=32)

    # Model selection
    if use_bn:
        model = VGG11Classifier().to(DEVICE)
        hook_layer = model.features.block3
    else:
        model = VGG11ClassifierNoBN().to(DEVICE)
        hook_layer = model.features.block3

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Hook storage
    activations = {}
    hook_layer.register_forward_hook(get_activation_hook(activations, "block3"))


    fixed_batch = next(iter(val_loader))["image"].to(DEVICE)
    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        prefix = "BN" if use_bn else "NO_BN"

        wandb.log({
            "epoch": epoch,
            f"{prefix}/train_loss": train_loss,
            f"{prefix}/val_loss": val_loss,
            f"{prefix}/val_acc": val_acc
        })

        print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        with torch.no_grad():
                _ = model(fixed_batch)
        
        log_activation_distribution(
            activations["block3"],
            tag=f"{'BN' if use_bn else 'NO_BN'}_epoch_{epoch}"
        )


    run.finish()

# if __name__ == "__main__":

#     # Normal learning rate
#     run_experiment(use_bn=True, lr=0.01)
#     run_experiment(use_bn=False, lr=0.01)

#     # High LR (to test stability)
#     run_experiment(use_bn=True, lr=0.1)
#     run_experiment(use_bn=False, lr=0.1)

