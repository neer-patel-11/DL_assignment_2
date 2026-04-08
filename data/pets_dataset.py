import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 224


def get_transforms(split):
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(0.05, 0.1, 15, p=0.5),
                A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.5),
                A.CoarseDropout(num_holes_range=(1, 4),
                                hole_height_range=(16, 32),
                                hole_width_range=(16, 32), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="albumentations",
                                     label_fields=["bbox_labels"],
                                     clip=True,
                                     min_visibility=0.3),
        )
    else:
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="albumentations",
                                     label_fields=["bbox_labels"],
                                     clip=True,
                                     min_visibility=0.3),
        )


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir=None, test_size=0.2, random_state=42, isTrain=True,need_box=False):

        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(__file__), "dataset")

        root_dir = os.path.abspath(root_dir)

        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.xmls_dir = os.path.join(root_dir, "annotations", "xmls")
        annotations_file = os.path.join(root_dir, "annotations", "list.txt")

        images_paths = []
        labels = []
        bboxes = []

        # ---------- BUILD DATA ----------
        with open(annotations_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue

                parts = line.strip().split()
                name = parts[0]
                class_id = int(parts[1]) - 1


                img_path = os.path.join(self.images_dir, name + ".jpg")
                xml_path = os.path.join(self.xmls_dir, name + ".xml")
                
                if need_box and not os.path.exists(xml_path):
                    continue

                bbox = self._parse_bbox(xml_path)

                images_paths.append(img_path)
                labels.append(class_id)
                bboxes.append(bbox)

        # ---------- SPLIT ----------
        train_img, test_img, y_train, y_test, b_train, b_test = train_test_split(
            images_paths, labels, bboxes,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        if isTrain:
            self.images_paths = train_img
            self.labels = y_train
            self.bboxes = b_train
            self.split = "train"
        else:
            self.images_paths = test_img
            self.labels = y_test
            self.bboxes = b_test
            self.split = "val"

        self.transform = get_transforms(self.split)

    # ---------- PARSE XML ----------
    def _parse_bbox(self, xml_path):
        if not os.path.exists(xml_path):
            return [0.0, 0.0, 1.0, 1.0]

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find("size")
            w = float(size.find("width").text)
            h = float(size.find("height").text)

            bb = root.find("object").find("bndbox")

            xmin = float(bb.find("xmin").text) / w
            ymin = float(bb.find("ymin").text) / h
            xmax = float(bb.find("xmax").text) / w
            ymax = float(bb.find("ymax").text) / h

            return [xmin, ymin, xmax, ymax]

        except:
            return [0.0, 0.0, 1.0, 1.0]

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        label = self.labels[idx]
        bbox = self.bboxes[idx]

        # ---------- LOAD ----------
        image = np.array(Image.open(img_path).convert("RGB"))

        name = os.path.basename(img_path).replace(".jpg", "")
        mask_path = os.path.join(self.masks_dir, name + ".png")
        mask = np.array(Image.open(mask_path).convert("L"))

        # ---------- TRANSFORM ----------
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=[bbox],
            bbox_labels=[0]
        )

        image_t = transformed["image"].float()

        mask_t = transformed["mask"]
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.tensor(mask_t)
        mask_t = torch.clamp(mask_t.long() - 1, 0, 2)

        # ---------- BBOX ----------
        if len(transformed["bboxes"]) > 0:
            x1, y1, x2, y2 = transformed["bboxes"][0]
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

        cx = ((x1 + x2) / 2) * IMAGE_SIZE
        cy = ((y1 + y2) / 2) * IMAGE_SIZE
        bw = (x2 - x1) * IMAGE_SIZE
        bh = (y2 - y1) * IMAGE_SIZE

        bbox_t = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        return {
            "image": image_t,
            "label": torch.tensor(label, dtype=torch.long),
            "bbox": bbox_t,
            "mask": mask_t
        }


def get_data_loader(root_dir=None, batch_size=64):

    train_dataset = OxfordIIITPetDataset(root_dir=root_dir, isTrain=True)
    test_dataset = OxfordIIITPetDataset(root_dir=root_dir, isTrain=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader