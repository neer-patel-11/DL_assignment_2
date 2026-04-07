"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, root_dir=None , test_size = 0.2 , random_state=42 , isTrain = True):

        if root_dir is None:
            root_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        
        root_dir = os.path.abspath(root_dir)  # Convert to absolute path

        annotations_file = os.path.join(root_dir, "annotations", "list.txt")

        images_dir = os.path.join(root_dir ,'images')
        
        # valid_ext = ('.jpg', '.jpeg', '.png')

        # images_file_name = [
        #     f for f in os.listdir(images_dir)
        #     if f.lower().endswith(valid_ext)
        # ]

        # self.labels = [''.join(char for char in cur_image if not char.isdigit()) for cur_image in images_file_name]

        # self.labels = [s[:-5] for s in self.labels]

        # self.images_paths = [os.path.join(images_dir,cur_image_path) for cur_image_path in images_file_name]

        self.images_paths = []
        self.labels = []

        with open(annotations_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                
                parts = line.strip().split()
                image_name = parts[0] + ".jpg"
                class_id = int(parts[1]) - 1   

                self.images_paths.append(os.path.join(images_dir, image_name))
                self.labels.append(class_id)


        train_images_path, test_images_path, y_train, y_test = train_test_split(self.images_paths,self.labels , random_state=random_state,test_size=test_size, shuffle=True , stratify=self.labels)
        
        unique_classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        if isTrain:
            self.images_paths = train_images_path
            self.labels = y_train
        else:
            self.images_paths = test_images_path
            self.labels = y_test


    def __len__(self):
        return len(self.images_paths) 

    def __getitem__(self, idx):
        import torch

        image = Image.open(self.images_paths[idx]).convert("RGB")        
        image = image.resize((224, 224))
        
        image = np.array(image)
        
        # Convert to tensor (HWC → CHW)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        label = self.labels[idx]

        return image, torch.tensor(label, dtype=torch.long)
    
def get_data_loader(root_dir=None , test_size = 0.2 , random_state=42 , batch_size = 64):

    dataset_train = OxfordIIITPetDataset(root_dir=root_dir , test_size=test_size , random_state=random_state ,isTrain = True )

    dataset_test = OxfordIIITPetDataset(root_dir=root_dir , test_size=test_size , random_state=random_state , isTrain=False)


    train_dataloader = DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0 # For parallel data loading (optional)
    )

    test_dataloader = DataLoader(
    dataset=dataset_test,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0 # For parallel data loading (optional)
    )

    return train_dataloader , test_dataloader




# if __name__ == "__main__":
#     # dataset = OxfordIIITPetDataset()

#     train_loader , _ = get_data_loader()

#     for i, (x,y) in enumerate(train_loader):
#         print(i)
#         print(y[0])
#         print(x[0])
#         print(type(x[0]))
#         break

