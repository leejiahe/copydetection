import torch
import torch.nn as nn
from transformers import ViTForImageClassification

#from ...datamodules.copydetect_datamodule import CopyDetectDataset

from embeddings import CopyDetectEmbedding

class SimImageHead(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
    
class ContrastiveHead(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class CopyDetectViT(nn.Module):
    def __init__(self, pretrained_arch):
        super().__init__()
        pretrained_model = ViTForImageClassification.from_pretrained(pretrained_arch)
        self.model = pretrained_model.vit.encoder # We only use the encoder from ViT, we will use a different embedding and prediction head
        self.embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                             vit_cls = pretrained_model.vit.embeddings.cls_token)
        self.contrastive_head = ContrastiveHead()
        self.sim_img_head = SimImageHead()
        
    def forward(self, img_r = None, img_q = None, pretrain = False):
        if pretrain:
            assert img_r is not None and img_q is not None
            emb_r, emb_q, emb_seg_rq, labels_rq = self.embedding(img_r, img_q)
            encoded_r = self.model(emb_r)
            encoded_q = self.model(emb_q)
            encoded_rq = self.model(emb_seg_rq)
            
            projected_r = self.contrastive_head(encoded_r) #goto contrastive head
            projected_q = self.contrastive_head(encoded_q)
            contrastive_loss = contrastive_loss(projected_r, projected_q)
            
            sim_img_logits = self.sim_img_head(encoded_rq)
            sim_img_loss = sim_img_loss(sim_img_logits, labels_rq)
            
            loss = contrastive_loss + sim_img_loss
            
            return loss
        else:
            raise NotImplementedError

def main():
    model = CopyDetectViT('google/vit-base-patch16-224')
    
    #train_dataset = CopyDetectDataset('/home/leejiahe/copydetection/data/trial/')
    #train_dataloader = DataLoader(train_dataset, batch_size = 1)
    #b = next(iter(train_dataloader))
    #print (b)
    
if __name__ == "__main__":
    main()

#%%
from transformers import ViTForImageClassification
pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# %%
