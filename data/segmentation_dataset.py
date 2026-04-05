"""Dataset and DataLoader for Oxford-IIIT Pet Segmentation
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class OxfordPetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet Dataset for Segmentation with trimaps."""
    
    def __init__(self, root_dir, split='trainval', image_size=224):
        """
        Args:
            root_dir: Root directory containing images/ and trimaps/ folders
            split: 'trainval' or 'test'
            image_size: Size to resize images and masks to
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.trimap_dir = os.path.join(root_dir, 'annotations/trimaps')
        self.image_size = image_size
        
        # Read split file
        split_file = os.path.join(os.path.join(root_dir,'annotations'), f'{split}.txt')
        with open(split_file, 'r') as f:
            all_names = [line.strip().split()[0] for line in f.readlines()]
        
        # Filter valid samples (both image and trimap exist)
        self.image_names = []
        skipped_count = 0
        
        for name in all_names:
            # Try different image extensions
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                temp_path = os.path.join(self.image_dir, f'{name}{ext}')
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            # Try different trimap filename patterns
            trimap_path = None
            # Pattern 1: regular filename
            temp_trimap = os.path.join(self.trimap_dir, f'{name}.png')
            if os.path.exists(temp_trimap) and self._is_valid_image(temp_trimap):
                trimap_path = temp_trimap
            else:
                # Pattern 2: with ._ prefix (skip if it's a macOS metadata file)
                temp_trimap = os.path.join(self.trimap_dir, f'._{name}.png')
                if os.path.exists(temp_trimap) and self._is_valid_image(temp_trimap):
                    trimap_path = temp_trimap
            
            # Check if both exist and are valid
            if img_path and trimap_path:
                self.image_names.append((name, img_path, trimap_path))
            else:
                skipped_count += 1
                if not img_path:
                    print(f"Skipped (missing image): {name}")
                if not trimap_path:
                    print(f"Skipped (missing/invalid trimap): {name}")
        
        print(f"\n{split.upper()} Dataset Summary:")
        print(f"  Total samples in split file: {len(all_names)}")
        print(f"  Valid samples loaded: {len(self.image_names)}")
        print(f"  Skipped samples: {skipped_count}")
        print("-" * 60)
        
        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Mask transformation (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), 
                            interpolation=transforms.InterpolationMode.NEAREST)
        ])
    
    def _is_valid_image(self, path):
        """Check if file is a valid image (not a macOS metadata file)."""
        try:
            # Try to open the image to verify it's valid
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name, img_path, trimap_path = self.image_names[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)
        
        # Load trimap
        trimap = Image.open(trimap_path)
        trimap = self.mask_transform(trimap)
        
        # Convert trimap to tensor and adjust labels
        # Original: 1=Foreground, 2=Background, 3=Not classified
        # Convert to: 0=Background, 1=Foreground, 2=Border/Not classified
        trimap = torch.from_numpy(np.array(trimap)).long()
        trimap = trimap - 1  # Now: 0=Foreground, 1=Background, 2=Not classified
        
        # Remap to: 0=Background, 1=Foreground, 2=Border
        mask = torch.zeros_like(trimap)
        mask[trimap == 1] = 0  # Background
        mask[trimap == 0] = 1  # Foreground
        mask[trimap == 2] = 2  # Border/Not classified
        
        return image, mask


def get_segmentation_data_loader(root_dir, batch_size=16, image_size=224, num_workers=4):
    """
    Create train and test data loaders for segmentation.
    
    Args:
        root_dir: Root directory of Oxford-IIIT Pet dataset
        batch_size: Batch size
        image_size: Image size (will be resized to image_size x image_size)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader
    """
    print("\n" + "=" * 60)
    print("LOADING SEGMENTATION DATASETS")
    print("=" * 60)
    
    train_dataset = OxfordPetSegmentationDataset(
        root_dir=root_dir, 
        split='trainval', 
        image_size=image_size
    )
    
    test_dataset = OxfordPetSegmentationDataset(
        root_dir=root_dir, 
        split='test', 
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\nDataLoaders created successfully!")
    print("=" * 60 + "\n")
    
    return train_loader, test_loader