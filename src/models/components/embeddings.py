# Inspired bny
# https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/vit/modeling_vit.py

from Typing import Optional, List, Union
from PIL import Image 

import torch
import torch.nn as nn

from patch_embeddings import PatchEmbeddings

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
                img_q: Optional[List[Image.Image]] = None,
                )-> Union[torch.Tensor, List[torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        vit_cls = self.vit_cls.expand(batch_size, -1, -1)
        
        emb_r = self.patch_embeddings(img_r)
        emb_r = torch.cat((vit_cls, emb_r), dim = 1)
        if img_q is None:
            return self.dropout(emb_r)            
        else:
            emb_q = self.patch_embeddings(img_q)
            emb_q = torch.cat((vit_cls, emb_q), dim = 1)
        
            batch_size, seq_len_r, _ = emb_r.shape # shape: batch_size, seq_len, dim
            # First image segment (similar to sentence A in NSP) 
            segment_r = self.img_segment(torch.zeros((batch_size, seq_len_r), device = self.img_segment.weight.device, dtype = torch.int))
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
            return emb_r, emb_q, emb_rq
