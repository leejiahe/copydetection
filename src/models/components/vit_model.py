from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification

#from ...datamodules.copydetect_datamodule import CopyDetectDataset

from embeddings import CopyDetectEmbedding

import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

class CopyDetectDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int, pretrain: bool = False):
        assert(os.path.exists(image_dir))
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.transforms = transforms.Compose([transforms.RandomResizedCrop((image_size, image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        self.aug = self.transforms
        self.pretrain = pretrain
                
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        image = Image.open(image_path)

        if self.pretrain:
            return self.transforms(image), self.aug(image)
        else:
            return self.transforms(image)

class SimImageHead(nn.Module):
    hidden_size: int = 768
    def __init__(self, config):
        super().__init__()
        self.project = nn.Linear(config.hidden_size, 1)

    def forward(self, img_emb_rq):
        sim_img_score = self.project(img_emb_rq)
        return sim_img_score
    
class Projection(nn.Module):
    # Inspired by:
    # https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    hidden_size: int = 768
    projected_hidden_size:int = 2048
    projected_size: int = 512
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(self.hidden_size, self.projected_hidden_size),
                                   nn.BatchNorm1d(self.projected_hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.projected_hidden_size, self.projected_size, bias = False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim = 1)

class CopyDetectViT(nn.Module):
    def __init__(self, pretrained_arch, config):
        super().__init__()
        pretrained_model = ViTForImageClassification.from_pretrained(pretrained_arch)
        # We only use the encoder and layernorm from ViT, we will use a different embedding and prediction head
        self.encoder = nn.Sequential(pretrained_model.vit.encoder,
                                     pretrained_model.vit.layernorm)
        # We will copy the ViT cls token and positional embedding
        self.embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                             vit_cls = pretrained_model.vit.embeddings.cls_token,
                                             pos_emb = pretrained_model.vit.embeddings.position_embeddings)
        
        self.contrastive_head = Projection(input_size = pretrained_model.config.hidden_size, 
                                           hidden_size = config.projected_hidden_size,
                                           projected_size = config.projected_size)
        
        self.sim_img_head = SimImageHead(hidden_size = pretrained_model.config.hidden_size)
        
    def forward(self, img_r = None, img_q = None, pretrain = False):
        if pretrain:
            assert img_r is not None and img_q is not None
            emb_r, emb_q, emb_rq, labels_rq = self.embedding(img_r, img_q)
            encoded_r = self.encoder(emb_r)
            encoded_q = self.encoder(emb_q)
            encoded_rq = self.encoder(emb_rq)
            
            projected_r = self.contrastive_head(encoded_r[:, 0]) # Take the vit cls token of image r to the projection head
            projected_q = self.contrastive_head(encoded_q[:, 0]) # Take the vit cls token of image q to the projection head
            contrastive_loss = contrastive_loss(projected_r, projected_q)
            
            sim_img_logits = self.sim_img_head(encoded_rq[:, 0]) # Take the cls token of the sequence of image r and q to the similar image head 
            sim_img_loss = sim_img_loss(sim_img_logits, labels_rq)
            
            loss = contrastive_loss + sim_img_loss
            
            return loss
        else:
            raise NotImplementedError

class Config:
    image_size = 224
    projected_hidden_size = 2048
    projected_size = 512
    
    

def main():
    #model = CopyDetectViT('google/vit-base-patch16-224')
    pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    train_dataset = CopyDetectDataset(image_dir = '/home/leejiahe/copydetection/data/query_dev/',
                                      image_size = pretrained_model.config.image_size, 
                                      pretrain = True)
    train_dataloader = DataLoader(train_dataset, batch_size = 2)
    b = next(iter(train_dataloader))
    img_r, img_q = b[0], b[1]
    embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                    vit_cls = pretrained_model.vit.embeddings.cls_token,
                                    pos_emb = pretrained_model.vit.embeddings.position_embeddings)
    print (embedding(img_r, img_q))
    
if __name__ == "__main__":
    main()
#%%
import torch

pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
train_dataset = CopyDetectDataset(image_dir = '/home/leejiahe/copydetection/data/query_dev/',
                                  image_size = pretrained_model.config.image_size, 
                                  pretrain = True)
train_dataloader = DataLoader(train_dataset, batch_size = 2)
b = next(iter(train_dataloader))
img_r, img_q = b[0], b[1]
embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                vit_cls = pretrained_model.vit.embeddings.cls_token,
                                pos_emb = pretrained_model.vit.embeddings.position_embeddings)

# %%
