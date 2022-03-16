import os
import random
import csv
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import LightningDataModule

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_SDEV = [0.229, 0.224, 0.225]

# Return all the image paths in a folder
get_image_file = lambda image_dir: [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

class CopyDetectDataset(Dataset):
    def __init__(self,
                 image_files: list,
                 image_size: int,
                 augment: object = None,
                 val: bool = False,
                 pretrain: bool = False,
                 n_crops: int = 1):
        
        self.image_files = image_files
        self.pretrain = pretrain
        self.val = val
        self.augment = augment
        self.n_crops = n_crops

        self.trfm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((image_size, image_size)),
                                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_SDEV)])
        
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                               List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                               torch.Tensor]:
        if self.val:
            ref_image_path, query_image_path, label = self.image_file[index]
            ref_image, query_image = Image.open(ref_image_path), Image.open(query_image_path)
            return (self.trfm(ref_image), self.trfm(query_image), torch.tensor(label))
        else:
            image_path = self.image_files[index]
            image = Image.open(image_path)

            if self.pretrain:
                imgs = []
                for _ in range(self.n_crops):
                    aug_index = index
                    if random.random() > 0.5:
                        aug_index = random.randint(0, len(self.image_files)) # Get random image file
                    label = torch.tensor(aug_index == index, dtype = torch.long) # label 1: modified copy: aug_index == index 
                    aug_image = Image.open(self.image_files[aug_index])
                    imgs.append((self.trfm(image), self.trfm(self.augment(aug_image)), label))
                return imgs
            else:
                return self.trfm(image)
        

@dataclass
class CopyDetectDataModule(LightningDataModule):
    train_dir: str # Train directory
    references_dir: str # Reference directory
    dev_queries_dir: str  # Dev queries directory
    final_queries_dir: str # Final queries directory
    augment: object # Augmentation object from augment.py
    dev_validation_set: str # Validation set created from dev ground truth, with randomly selected negative reference image pair  
    dev_ground_truth: str = None # Dev ground truth containing queries and corresponding reference pair
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    n_crops: int = 1
    image_size: int = 224
    
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(logger = False)
        
    def prepare_data(self) -> None:
        # Download data
        urls_list = {'train_dir': [],
                     'references_dir': [],
                     'dev_queries_dir': [],
                     'final_queries_dir': []}
        
        for d, urls in urls_list.item():
            folder_dir = getattr(self, d)
            for url in urls:
                download_and_extract_archive(url, folder_dir)
                
                # move files
                # delete zip file
            # delete folder
        
        # Create validation set by using the ground truth csv, which contain query images and its corresponding reference images
        # We create 'false' label by pairing query imge with a random reference image
        # True label is query image with its corresponding reference image as given in the ground truth csv
        
        val_data = []
        with open(self.dev_ground_truth) as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ',')
            next(csvreader) # Skip header row
            for row in csvreader: # Row return query_id, reference_id
                if row[1] == '':
                    continue # Filter row with reference id, ignoring all distractors images
                else:
                    query_image_path = os.path.join(self.hparams.dev_queries_dir, row[0])
                    ref_image_path = os.path.join(self.hparams.references_dir, row[1])
                    rand_image = row[1]
                    while(rand_image == row[1]):
                        rand_image = np.random.choice(self.hparams.references_dir) # pick one reference image randomly
                    
                    rand_image_path = os.path.join(self.hparams.references_dir, rand_image)
                    val_data.append([ref_image_path, query_image_path, 1])
                    val_data.append([rand_image_path, query_image_path, 0])
        
        folder_dir, _ = os.path.split(self.dev_ground_truth)
        val_path = os.path.join(folder_dir, 'dev_validation_set.csv')
        
        with open(val_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(val_data)
    
    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        val_data = []
        with open(self.dev_validation_set, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ',')
            for row in csvreader:
                val_data.append(row)
        
        self.train_dataset = CopyDetectDataset(image_files = get_image_file(self.hparams.train_dir),
                                               image_size = self.hparams.image_size,
                                               pretrain = True,
                                               augment = self.augment,
                                               n_crops = self.hparams.n_crops)
        
        self.val_dataset = CopyDetectDataset(image_files = val_data,
                                             image_size = self.hparams.image_size,
                                             pretrain = False,
                                             val = True)
        
        self.reference_dataset = CopyDetectDataset(image_files = get_image_file(self.hparams.references_dir),
                                                   image_size = self.hparams.image_size,
                                                   pretrain = False)
    
        self.query_final_dataset = CopyDetectDataset(image_files = get_image_file(self.hparams.final_queries_dir),
                                                     image_size = self.hparams.image_size,
                                                     pretrain = False)
        

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.train_dataset,
                          batch_size = self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          shuffle = True,
                          drop_last = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.val_dataset,
                          batch_size = self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          shuffle = False)
            
    def ref_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.reference_dataset,
                          batch_size = self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          shuffle = False)
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.query_final_dataset,
                          batch_size = self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          shuffle = False)
        