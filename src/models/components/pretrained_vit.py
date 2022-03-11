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
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.io import read_image
from torchvision.transforms import transforms

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
        if self.pretrain:
            rand_ref_index = np.random.randint(low = 0, high = len(self.image_files))
            if rand_ref_index == index:
                rand_ref_index = rand_ref_index + 1 if index == 0 else rand_ref_index - 1
                
            ref_image = self.transforms(self.get_image(index))
            rand_ref_image = self.transforms(self.get_image(rand_ref_index))
            aug_image = self.aug(self.get_image(index))
            return (ref_image, aug_image, torch.tensor(1)), (rand_ref_image, aug_image, torch.tensor(0))
        else:
            return self.transforms(self.get_image(index))
        
    def get_image(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        image = Image.open(image_path)
        return image

class CopyDetectViT(nn.module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass

def main():
    pass
    
    #train_dataset = CopyDetectDataset('/home/leejiahe/copydetection/data/trial/')
    #train_dataloader = DataLoader(train_dataset, batch_size = 1)
    #b = next(iter(train_dataloader))
    #print (b)
    
if __name__ == "__main__":
    main()
    
