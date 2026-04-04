import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torch.utils.data import DataLoader

class OxfordPetLocalizationDataset(Dataset):
    def __init__(self, root_dir, image_dir="images", xml_dir="annotations/xmls",
                 transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.xml_dir = os.path.join(root_dir, xml_dir)
        self.transform = transform

        all_images = [
            f for f in os.listdir(self.image_dir)
            if f.endswith(".jpg")
        ]

        # keep only valid (image, xml) pairs
        self.samples = []
        for img_name in all_images:
            xml_name = img_name.replace(".jpg", ".xml")

            img_path = os.path.join(self.image_dir, img_name)
            xml_path = os.path.join(self.xml_dir, xml_name)

            if os.path.exists(img_path) and os.path.exists(xml_path):
                self.samples.append((img_path, xml_path))
            else:
                # optional debug log
                print(f"Skipping: {img_name} (missing xml or image)")

    def __len__(self):
        return len(self.samples)

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        return xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        xmin, ymin, xmax, ymax = self._parse_xml(xml_path)

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        bbox = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

        if self.transform:
            image, bbox = self.transform(image, bbox)

        return image, bbox    

class ResizeWithBBox:
    def __init__(self, size):
        self.size = size
        self.resize = T.Resize(size)

    def __call__(self, image, bbox):
        w_old, h_old = image.size
        image = self.resize(image)
        w_new, h_new = self.size

        scale_x = w_new / w_old
        scale_y = h_new / h_old

        x, y, w, h = bbox

        x *= scale_x
        y *= scale_y
        w *= scale_x
        h *= scale_y

        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        return image, bbox
    
class TransformWrapper:
    def __init__(self, size=(224, 224)):
        self.resize = ResizeWithBBox(size)
        self.to_tensor = T.ToTensor()

    def __call__(self, image, bbox):
        image, bbox = self.resize(image, bbox)
        image = self.to_tensor(image)
        return image, bbox
    


def get_data_loader(root_dir, batch_size=32, split=0.8):
    dataset = OxfordPetLocalizationDataset(
        root_dir=root_dir,
        transform=TransformWrapper((224, 224))
    )

    # split
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    return train_loader, val_loader