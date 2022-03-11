"""
{Description}

http://www.apache.org/licenses/LICENSE-2.0
"""

__author__ = 'Lee Jiahe'
__credits__ = ['']
__version__ = '0.0.1'

import os
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.io import read_image
from torchvision.transforms import transforms

from ..utils.kfold import BaseKFoldDataModule
from ..utils import download_and_unzip

class CopyDetectDataset(Dataset):
    def __init__(self, image_dir: str, pretrain: bool = False) -> None:
        assert(os.path.exists(image_dir))
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.pretrain = pretrain
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        self.aug = self.transforms
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        image = Image.open(image_path)

        if self.pretrain:
            return self.transforms(image), self.aug(image)
        else:
            return self.transforms(image)
        

@dataclass
class CDKFoldDataModule(BaseKFoldDataModule):
    training_dir: str
    reference_dir: str
    query_dev_dir: str 
    query_final_dir: str
    num_folds: int
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False

    def prepare_data(self) -> None:
        for dir in [self.hparams.training_dir, self.hparams.reference_dir, self.hparams.query_dev_dir, self.hparams.query_final_dir]:
            if os.path.exists(dir) == False:
                os.mkdir(dir)
        # download the data.
        from multiprocessing import Pool, RLock
        from tqdm import tqdm
        
        urls = []
        urls.append(("https://dl.fbaipublicfiles.com/image_similarity_challenge/public/dev_queries.zip", self.hparams.query_dev_dir))
        urls.append(("https://dl.fbaipublicfiles.com/image_similarity_challenge/public/final_queries.zip", self.hparams.query_final_dir))
        urls.extend([(f"https://dl.fbaipublicfiles.com/image_similarity_challenge/public/train_{i}.zip", self.hparams.training_dir) for i in range(20)])
        urls.extend([(f"https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_{i}.zip", self.hparams.reference_dir) for i in range(20)])

        with Pool(processes = 8, initargs = (RLock(),), initializer = tqdm.set_lock) as p:
            jobs = [p.apply_async(download_and_unzip, args = (url, dir, i)) for i, (url, dir) in enumerate(urls)]
            results = [job.get() for job in jobs]
        
        print('Finish downloading all data!')
            
        
    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        self.train_dataset = CopyDetectDataset(self.hparams.training_dir, pretrain = True)
        #self.reference_dataset = CopyDetectDataset(self.hparams.reference_dir, pretrain = False)
        #self.query_dev_dataset = CopyDetectDataset(self.hparams.query_dev_dir, pretrain = False)
        #self.query_final_dataset = CopyDetectDataset(self.hparams.query_final_dir, pretrain = False)

    def setup_folds(self) -> None:
        self.splits = [split for split in KFold(self.hparams.num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.train_fold,
                          batch_size = self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.val_fold,
                          batch_size = self.hparams.batch_size,
                          num_workers = self.hparams.num_workers,
                          pin_memory = self.hparams.pin_memory,
                          shuffle = False)

    def dev_dataloader(self) -> DataLoader:
        return DataLoader(dataset = self.query_dev_dataset,
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
        
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(logger = False)