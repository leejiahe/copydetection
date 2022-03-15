import os
from matplotlib import image
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule
import augly.image as imaugs

from components.augmentation import *

class CopyDetectDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 image_size: int,
                 pretrain: bool = False,
                 n_crops: int = 1,
                 n_upper: int = 2,
                 n_lower: int = 1,
                 overlay_image_dir: str = None):
        
        assert(os.path.exists(image_dir))
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.pretrain = pretrain
        self.n_crops = n_crops
        
        if pretrain:
            transforms_list = [OverlayRandomStripes(),
                               OverlayRandomEmoji(),
                               EncodingRandomQuality(),
                               MemeRandomFormat(),
                               OverlayRandomText(),
                               RandomSaturation(),
                               ApplyRandomPILFilter(),
                               OverlayOntoRandomBackgroundImage(overlay_image_dir),
                               OverlayOntoRandomForegroundImage(overlay_image_dir),
                               RandomShufflePixels(),
                               OverlayOntoRandomScreenshot(),
                               RandomPadSquare(),
                               ConvertRandomColor(),
                               RandomCropping(),
                               imaugs.RandomAspectRatio(),
                               imaugs.RandomPixelization(0, 0.7),
                               imaugs.RandomBlur(2, 10),
                               imaugs.RandomBrightness(0.1, 1),
                               imaugs.RandomRotation(-90, 90),
                               imaugs.Grayscale(),
                               imaugs.PerspectiveTransform(),
                               imaugs.VFlip(),
                               imaugs.HFlip()]
            
            self.aug = N_Compositions(transforms_list, n_upper = n_upper, n_lower = n_lower)

        self.trfm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((image_size, image_size)),
                                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_SDEV)])
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
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
                imgs.append((self.trfm(image), self.trfm(self.aug(aug_image)), label))
            return imgs
        else:
            return self.trfm(image)
        

@dataclass
class CopyDetectDataModule(LightningDataModule):
    training_dir: str
    reference_dir: str
    query_dev_dir: str 
    query_final_dir: str
    num_folds: int
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    
    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(logger = False)
    
    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        self.train_dataset = CopyDetectDataset(image_dir = self.hparams.training_dir,
                                               image_size = self.hparams.image_size,
                                               pretrain = True,
                                               n_crops = self.hparams.n_crops,
                                               n_upper = self.hparams.n_upper,
                                               n_lower = self.hparams.n_lower,
                                               overlay_image_dir = self.hparams.training_dir)
        
        self.reference_dataset = CopyDetectDataset(image_dir = self.hparams.training_dir,
                                                   image_size = self.hparams.image_size,
                                                   pretrain = False)
        
        self.query_dev_dataset = CopyDetectDataset(image_dir = self.hparams.training_dir,
                                                   image_size = self.hparams.image_size,
                                                   pretrain = False)
        
        self.query_final_dataset = CopyDetectDataset(image_dir = self.hparams.training_dir,
                                                     image_size = self.hparams.image_size,
                                                     pretrain = False)
        
        self.setup_folds()
        

    def setup_folds(self) -> None:
        self.splits = [split for split in KFold(self.hparams.num_folds).split(range(len(self.train_dataset)))]
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
        
