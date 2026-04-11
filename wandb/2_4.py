import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import wandb
import numpy as np

from models.multitask import MultiTaskPerceptionModel  # adjust filename if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiTaskPerceptionModel().to(DEVICE)
model.eval()

classifier = model.classifier  # use only classifier

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

first_conv = None
last_conv = None

for name, layer in classifier.features.named_modules():
    if isinstance(layer, nn.Conv2d):
        if first_conv is None:
            first_conv = layer   
        last_conv = layer        

first_conv.register_forward_hook(get_activation("first"))
last_conv.register_forward_hook(get_activation("last"))

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def feature_maps_to_wandb(feature_map, layer_name, num_maps=10):
    """
    Convert feature map tensor to a list of wandb.Image objects.
    Sends exactly num_maps images (or fewer if channels < num_maps).
    """
    feature_map = feature_map.squeeze(0)          # (C, H, W)
    num_maps = min(num_maps, feature_map.shape[0])

    wandb_images = []
    for i in range(num_maps):
        fm = feature_map[i].cpu().numpy()

        # Normalize to [0, 255] for clean display
        fm_min, fm_max = fm.min(), fm.max()
        if fm_max - fm_min > 1e-8:
            fm = (fm - fm_min) / (fm_max - fm_min)
        fm = (fm * 255).astype(np.uint8)

        wandb_images.append(
            wandb.Image(fm, caption=f"{layer_name} | channel {i}")
        )

    return wandb_images


def plot_feature_maps(feature_map, title, num_maps=10):
    feature_map = feature_map.squeeze(0)
    num_maps = min(num_maps, feature_map.shape[0])

    fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
    if num_maps == 1:
        axes = [axes]

    for i in range(num_maps):
        axes[i].imshow(feature_map[i].cpu(), cmap="viridis")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def run_feature_map_logging(img_path: str):

    # -------- WandB init --------
    run = wandb.init(
        entity="da25m021-iitm-indi",
        project="dl_assignment_2",
        group="TASK_2_4",
        job_type="feature_maps",
        name="TASK_2_4_feature_maps",
        config={
            "task": "2.4",
            "image": img_path,
            "num_feature_maps_logged": 10,
        }
    )

    image = load_image(img_path).to(DEVICE)

    with torch.no_grad():
        _ = classifier(image)

    wandb.log({
        "input_image": wandb.Image(img_path, caption="Input image")
    })

    first_images = feature_maps_to_wandb(activations["first"], "first_conv", num_maps=10)
    last_images  = feature_maps_to_wandb(activations["last"],  "last_conv",  num_maps=10)

    wandb.log({
        "first_conv_feature_maps": first_images,
        "last_conv_feature_maps":  last_images,
        "first_conv_channels_total": activations["first"].shape[1],
        "last_conv_channels_total":  activations["last"].shape[1],
    })

    print(f" Logged 10 feature maps from first conv "
          f"(total channels: {activations['first'].shape[1]})")
    print(f" Logged 10 feature maps from last conv  "
          f"(total channels: {activations['last'].shape[1]})")

    plot_feature_maps(activations["first"], "First Conv Layer Features (top 10)")
    plot_feature_maps(activations["last"],  "Last Conv Layer Features (top 10)")

    run.finish()

# img_path = "/content/dataset/data/dataset/images/american_bulldog_68.jpg"   
# run_feature_map_logging(img_path)