from typing import Any, List, Optional

import einops
import deepspeed

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.precision_recall import Precision
from timm.scheduler.cosine_lr import CosineLRScheduler

from transformers import ViTModel

from src.models.components.layers import CopyDetectEmbedding, NormalizedFeatures, SimImagePred, ContrastiveProj

class CopyDetectModule(LightningModule):
    def __init__(self,
                 pretrained_arch: str,          # Pretrained ViT architecture
                 ntxentloss: object,            # Contrastive loss
                 lr: float,                     # Learning rate
                 gamma1: float = 1,             # SimImagePredictor loss multiplier
                 gamma2: float = 1,             # Contrastive loss multiplier
                 beta1: float = 0.9,            # beta1 term for Adam optimizer
                 beta2: float = 0.999,          # beta2 term for Adam optimizer
                 weight_decay: float  = 0,      # Weight decay for optimizer
                 t_initial: int = 10,           # Initial number of epoch
                 k_decay: float = 1,            # K Decay of learning rate
                 warmup_t: int = 0,             # Number of warmup epoch
                 warmup_lr_init: float = 0,     # Warmup learning rate
                 hidden_dim: int = 2048,        # Contrastive projection size of hidden layer
                 projected_dim: int = 512,      # Contrastive projection size of projection head 
                 ):               
        super().__init__()
        self.save_hyperparameters(logger = False)
         
        # Instantiate ViT encoder from pretrained model
        pretrained_model = ViTModel.from_pretrained(pretrained_arch)
        self.encoder = pretrained_model.encoder
        
        #for parent in self.encoder.named_children():
        #    for name, params in parent[1].named_children():
        #        if int(name) < 6:
        #            params.requires_grad_(False)
                    
        self.patch_size = pretrained_model.config.patch_size
                
        # Instantiate embedding, we use the pretrained ViT cls and position embedding
        self.embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                             vit_cls = pretrained_model.embeddings.cls_token,
                                             pos_emb = pretrained_model.embeddings.position_embeddings)
        
        # Normalized features
        self.normfeats = NormalizedFeatures(hidden_dim = pretrained_model.config.hidden_size,
                                            layer_norm_eps = pretrained_model.config.layer_norm_eps)
                
        # Instantiate SimImagePredictor
        self.simimagepred = SimImagePred(embedding_dim = pretrained_model.config.hidden_size)

        # Instantiate ContrastiveProjection
        self.contrastiveproj = ContrastiveProj(embedding_dim = pretrained_model.config.hidden_size,
                                               hidden_dim = hidden_dim,
                                               projected_dim = projected_dim)
        
        # Contrastive loss 
        self.contrastive_loss = ntxentloss
        
        # Binary cross entropy loss for similar image pair
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
        # Model precision in detecting modified copy
        self.train_precision, self.val_precision = Precision(average = 'micro'), Precision(average = 'micro')   
          
        # For logging best validation precision
        self.val_precision_best = MaxMetric()

    def feature_extract(self, batch: Any) -> torch.Tensor:
        # To extract feature vector
        img_r, img_id = batch
        encoding = self.norm_feats(self.encoder(self.embedding(img_r)))
        batch_size, num_ch, H, W, = img_r.size()
        h, w = int(H/self.patch_size), int(W/self.patch_size)
        cls, feats = encoding[:,0,:], encoding[:,1:,:] # Get the cls token and all the images features
            
        feats = einops.rearrange(feats, 'b (h w) d -> b d h w', h = h, w = w).clamp(min = 1e-6)
        # GeM Pooling
        feats = F.avg_pool2d(feats.pow(4), (h,w)).pow(1./4)
        feats = einops.rearrange(feats, 'b d () () -> b d')
        # Concatenate cls tokens with image patches to give local and global views of image
        feature_vector = torch.cat((cls, feats), dim = 1)

        return feature_vector, img_id
        
    def predict_copy(self, batch):
        # For copy detection 
        img_r, img_q = batch
        logits = self.simimagepred(self.encoder_checkpoint(self.embedding(img_r, img_q)))
        preds = torch.argmax(logits, dim = 1)
        return preds

    
    def training_step(self, batch: Any, batch_idx: int):
        img_r, img_q, id_r, id_q = batch
        label = torch.tensor(id_r == id_q, dtype = torch.float, device = img_r.device)
        
        # SimImagePredictor
        logits = self.simimagepred(self.normfeats(self.encoder(self.embedding(img_r, img_q))))

        # Calculate binary cross entropy loss of similar image pair
        simimage_loss = self.bce_loss(logits, label.unsqueeze(dim = 1))
        preds = torch.argmax(logits, dim = 1)
        
        # Contrastive loss
        proj_r = self.contrastiveproj(self.normfeats(self.encoder(self.embedding(img_r))))
        proj_q = self.contrastiveproj(self.normfeats(self.encoder(self.embedding(img_q))))
        # Calculate contrastive loss between un-augmented img_r and augmented positive pair of img_q
        contrastive_loss = self.contrastive_loss(proj_r, proj_q, id_r, id_q)
        
        # Weighted sum of bce and contrastive loss
        total_loss = self.hparams.gamma1 * simimage_loss + self.hparams.gamma2 * contrastive_loss
        
        # Log train metrics
        precision = self.train_precision(preds, label.int())
        self.log("train/simimage_loss", simimage_loss, on_step = False, on_epoch = True, prog_bar = False)
        self.log("train/contrastive_loss", contrastive_loss, on_step = False, on_epoch = True, prog_bar = False)
        self.log("train/total_loss", total_loss, on_step = False, on_epoch = True, prog_bar = False)
        self.log("train/precision", precision, on_step = False, on_epoch = True, prog_bar = True)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        img_r, img_q, label = batch
        
        # SimImagePredictor
        logits = self.simimagepred(self.normfeats(self.encoder(self.embedding(img_r, img_q))))
        # Calculate binary cross entropy loss of similar image pair
        simimage_loss = self.bce_loss(logits, label.unsqueeze(dim = 1))
        preds = torch.argmax(logits, dim = 1)

        # Log val metrics
        precision = self.val_precision(preds, label.int())
        self.log("val/simimage_loss", simimage_loss, on_step = False, on_epoch = True, prog_bar = False)
        self.log("val/precision", precision, on_step = False, on_epoch = True, prog_bar = True)
    
    def validation_epoch_end(self, outputs: Any) -> None:
        precision = self.val_precision.compute()
        self.val_precision_best.update(precision)
        self.log('val/precision_best', self.val_precision_best.compute(), on_epoch = True, prog_bar = True)
    
    def on_epoch_end(self):
        # Reset metrics at the end of every epoch
        self.train_precision.reset()
        self.val_precision.reset()

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        scheduler.step(epoch = self.current_epoch)

    def configure_optimizers(self):
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model_params = self.parameters(),
                                                        lr = self.hparams.lr,
                                                        betas = (self.hparams.beta1, self.hparams.beta2),
                                                        weight_decay = self.hparams.weight_decay)
        
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial = self.hparams.t_initial, 
                                      lr_min = self.hparams.lr,
                                      k_decay = self.hparams.k_decay,
                                      warmup_t = self.hparams.warmup_t,
                                      warmup_lr_init = self.hparams.warmup_lr_init,)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        