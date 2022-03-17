#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel



class CopyDetectViT(nn.Module):
    def __init__(self, pretrained_arch, config):
        super().__init__()
        pretrained_model = ViTModel.from_pretrained(pretrained_arch)
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
