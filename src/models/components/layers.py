# Inspired bny
# https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/vit/modeling_vit.py

import collections.abc
from typing import Optional, List
from PIL import Image 

import torch
import torch.nn as nn
import torch.nn.functional as F

import collections.abc
from torch import nn

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
            
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
        
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight.data, nonlinearity = 'relu')
        if module.bias is not None:
            module.bias.zero_()

class PatchEmbeddings(nn.Module):
    # Based on timm implementation, which can be found here:
    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

    def __init__(self, image_size = 224, patch_size = 16, num_channels = 3, embed_dim = 768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

class SimImagePred(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768):
        super().__init__()
        self.projec1 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.ReLU(),
                                     nn.Linear(embedding_dim, 1))
        self.projec2 = nn.Sequential(nn.Linear(embedding_dim, 1),
                                     nn.Sigmoid())
        self.project = nn.Linear(embedding_dim, 1)
        self.apply(init_weights)

    def forward(self, emb_rq):
        cls = emb_rq[:, 0, :] # Only get the cls token
        sim_img_score = self.project(cls)
        return sim_img_score
    
class ContrastiveProj(nn.Module):
    # Inspired by:
    # https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim:int = 2048,
                 projected_dim: int = 512):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, projected_dim, bias = False))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        cls = x[:, 0, :] # Only get the vit cls token
        cls = self.model(cls)
        return F.normalize(cls, dim = 1)

class NormalizedFeatures(nn.Module):
    def __init__(self,
                 hidden_dim: int = 768,
                 layer_norm_eps = 1e-5):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_dim, eps = layer_norm_eps)
        
    def forward(self, outputs):
        return self.layernorm(outputs.last_hidden_state)

class CopyDetectEmbedding(nn.Module):
    def __init__(self,
                 config: object,
                 vit_cls: torch.Tensor,
                 pos_emb: torch.Tensor):
        super().__init__()
        self.vit_cls = vit_cls #pretrained cls token from ViT model
        self.position_embeddings = pos_emb #pretrained positional embedding from ViT model
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) #cls token for the similar image prediction
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) #sep token to separate sequence of the two images
        self.img_segment = nn.Embedding(2, config.hidden_size) #image segment to differentiate image r and image q

        self.patch_embeddings = PatchEmbeddings(image_size = config.image_size,
                                                patch_size = config.patch_size,
                                                num_channels = config.num_channels,
                                                embed_dim = config.hidden_size)
        
        self.patch_size = config.patch_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self,
                img_r: List[Image.Image],
                img_q: Optional[List[Image.Image]] = None):
        batch_size, _, _, _ = img_r.size() # shape: batch_size, num channel, h, w
        vit_cls = self.vit_cls.expand(batch_size, -1, -1)
        
        emb_r = self.patch_embeddings(img_r)
        emb_r = torch.cat((vit_cls, emb_r), dim = 1)
        emb_r = self.dropout(emb_r)
        if img_q is None:
            return emb_r            
        else:
            emb_q = self.patch_embeddings(img_q)
            emb_q = torch.cat((vit_cls, emb_q), dim = 1)
            emb_q = self.dropout(emb_q)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            
            emb_rq = torch.cat((cls_token, emb_r, emb_q), dim = 1)
            return emb_rq
            
            """
            # First image segment (similar to sentence A in NSP) 
            segment_r = self.img_segment(torch.zeros((batch_size, emb_r.size(1)), device = self.img_segment.weight.device, dtype = torch.int))
            # Second image segment (similar to sentence B in NSP)
            segment_q = self.img_segment(torch.ones((batch_size, emb_q.size(1)), device = self.img_segment.weight.device, dtype = torch.int))
            # Add segment embedding to reference and query embeddings
            emb_seg_r = emb_r + segment_r
            emb_seg_q = emb_q + segment_q
            # Added first segment (similar to NSP) to cls and sep tokens, there is not positional encoding for this two tokens
            cls_token = self.cls_token.expand(batch_size, -1, -1) + segment_r 
            sep_token = self.sep_token.expand(batch_size, -1, -1) + segment_r
            # Concat cls, ref emb, sep, query emb
            emb_rq = torch.cat((cls_token, emb_seg_r, sep_token, emb_seg_q), dim = 1)
            return emb_rq
            """