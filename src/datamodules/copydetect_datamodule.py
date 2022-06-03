import os
import re
import csv
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import LightningDataModule

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_SDEV = [0.229, 0.224, 0.225]


# Return all the image paths in a folder
get_image_file = lambda image_dir: [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
get_image = lambda folder, index: Image.open(folder[index])
    
class CopyDetectPretrainDataset(Dataset):
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.image_files = np.array([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        image_id = self.image_files[index]
        image = Image.open(os.path.join(self.image_dir, image_id))
        image_id = re.findall(r'\d+', image_id)[0] # Get file name
        return (image, int(image_id))

class CopyDetectCollateFn(nn.Module):
    def __init__(self,
                 transform,
                 augment: object,
                 n_crops: Optional[int] = 1):
        super().__init__()
        self.transform  = transform
        self.augment = augment
        self.n_crops = n_crops

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        indices = np.arange(batch_size)
        
        imgs = [i[0] for i in batch]
        ids  = [i[1] for i in batch]
        # Transform image in batch and give a dimension for batching
        ref_imgs = list(map(lambda x: self.transform(x).unsqueeze_(dim = 0), imgs))
        
        ref_imgs_list, aug_imgs_list, ref_ids_list, aug_ids_list = [], [], [], []
        
        for _ in range(self.n_crops):
            rand_bool = np.random.uniform(size = batch_size) < 0.5
            rand_indices = np.random.randint(0, batch_size, size = batch_size)
            # If p > 0.5, augmented image is another image from the same batch
            aug_indices = np.where(rand_bool, indices, rand_indices)
            # From the augmented indices, select the images and the image indices
            aug_imgs = list(map(lambda i: imgs[i], aug_indices.tolist()))
            aug_ids = list(map(lambda i: ids[i], aug_indices.tolist()))
            # Transform image in batch and give a dimension for batching
            aug_imgs = list(map(lambda x: self.transform(self.augment(x)).unsqueeze_(dim = 0), aug_imgs))
            
            ref_imgs_list.extend(ref_imgs), aug_imgs_list.extend(aug_imgs), ref_ids_list.extend(ids), aug_ids_list.extend(aug_ids)
            
        return torch.vstack(ref_imgs_list), torch.vstack(aug_imgs_list), torch.tensor(ref_ids_list), torch.tensor(aug_ids_list)
        
class CopyDetectValDataset(Dataset):
    def __init__(self,
                 references_dir: str,
                 queries_dir: str,
                 dev_validation_set: str,
                 transform):
        
        self.refernces_dir = references_dir
        self.queries_dir = queries_dir
        self.transform = transform
        
        val_data = []
        with open(dev_validation_set, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ',')
            for row in csvreader:
                val_data.append((row[0], row[1], row[2]))
                                
        self.val_data = np.array(val_data)

    def __len__(self) -> int:
        return len(self.val_data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        ref_id, query_id, label = self.val_data[index]
        ref_image = Image.open(os.path.join(self.refernces_dir, ref_id))
        query_image = Image.open(os.path.join(self.queries_dir, query_id))
        
        return (self.transform(ref_image), self.transform(query_image), float(label))
    
@dataclass
class CopyDetectDataModule(LightningDataModule):
    train_dir: str                  # Train directory
    references_dir: str             # Reference directory
    dev_queries_dir: str            # Dev queries directory
    final_queries_dir: str          # Final queries directory
    augment: object                 # Augmentation object from augment.py
    dev_validation_set: str = None  # Validation set created from dev ground truth, with randomly selected negative reference image pair  
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    n_crops: int = 1
    image_size: int = 224
    
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(logger = False)
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((self.image_size, self.image_size)),
                                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_SDEV)])
        
        self.collate_fn = CopyDetectCollateFn(transform = transform,
                                              augment = self.augment,
                                              n_crops = self.n_crops)
                                                               
        self.train_dataset = CopyDetectPretrainDataset(image_dir = self.train_dir)
        
        self.val_dataset = CopyDetectValDataset(dev_validation_set = self.dev_validation_set,
                                                references_dir = self.references_dir,
                                                queries_dir = self.dev_queries_dir,
                                                transform = transform)
    """
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.train_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = self.pin_memory,
                          collate_fn = self.collate_fn,
                          shuffle = True,
                          drop_last = True)
    """
    
    def train_dataloader(self):
        return DataLoader(dataset = self.val_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = self.pin_memory,
                          shuffle = False)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.val_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = self.pin_memory,
                          shuffle = False)
    
        