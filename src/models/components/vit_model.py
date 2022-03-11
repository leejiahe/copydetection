import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification

#from ...datamodules.copydetect_datamodule import CopyDetectDataset

from embeddings import CopyDetectEmbedding

class SimImageHead(nn.Module):
    hidden_size: int = 768
    def __init__(self, config):
        super().__init__()
        self.project = nn.Linear(config.hidden_size, 1)

    def forward(self, img_emb_rq):
        sim_img_score = self.project(img_emb_rq)
        return sim_img_score
    
class Projection(nn.Module):
    input_dim: int = 768
    hidden_dim:int = 2048
    output_dim: int = 512
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                   nn.BatchNorm1d(self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.output_dim, bias = False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim = 1)

class CopyDetectViT(nn.Module):
    def __init__(self, pretrained_arch):
        super().__init__()
        pretrained_model = ViTForImageClassification.from_pretrained(pretrained_arch)
        self.encoder = pretrained_model.vit.encoder # We only use the encoder from ViT, we will use a different embedding and prediction head
        self.embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                             vit_cls = pretrained_model.vit.embeddings.cls_token,
                                             pos_emb = pretrained_model.vit.embeddings.position_embeddings)
        self.contrastive_head = Projection()
        self.sim_img_head = SimImageHead()
        
    def forward(self, img_r = None, img_q = None, pretrain = False):
        if pretrain:
            assert img_r is not None and img_q is not None
            emb_r, emb_q, emb_rq, labels_rq = self.embedding(img_r, img_q)
            encoded_r = self.encoder(emb_r)
            encoded_q = self.encoder(emb_q)
            encoded_rq = self.encoder(emb_rq)
            
            projected_r = self.contrastive_head(encoded_r)
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
