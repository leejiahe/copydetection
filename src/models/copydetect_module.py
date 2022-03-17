from typing import Any, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import ViTModel

from src.models.components.layers import CopyDetectEmbedding, NormalizedFeatures, SimImagePred, ContrastiveProj

class CopyDetectModule(LightningModule):
    def __init__(self,
                 pretrained_arch: str,          # Pretrained ViT architecture
                 ntxentloss: object,            # Contrastive loss
                 hidden_dim: int = 2048,        # Contrastive projection size of hidden layer
                 projected_dim: int = 512,      # Contrastive projection size of projection head 
                 beta1: int = 1,                # Similar image BCE loss multiplier
                 beta2: int = 1,                # Contrastive loss multiplier
                 lr: float = 0.001,
                 weight_decay: float = 0.0005):               
        super().__init__()
        self.save_hyperparameters(logger = False)
         
        # Instantiate ViT encoder from pretrained model
        pretrained_model = ViTModel.from_pretrained(pretrained_arch)
        encoder = pretrained_model.encoder
                
        # Instantiate embedding, we use the pretrained ViT cls and position embedding
        embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                        vit_cls = pretrained_model.embeddings.cls_token,
                                        pos_emb = pretrained_model.embeddings.position_embeddings)
        
        # Normalized features
        normfeats = NormalizedFeatures(hidden_dim = pretrained_model.config.hidden_size,
                                       layer_norm_eps = pretrained_model.config.layer_norm_eps)
        # Feature Vector Extractor
        self.feature_extractor = nn.Sequential(embedding, encoder, normfeats)
        
        # Instantiate SimImagePredictor
        simimagepred = SimImagePred(embedding_dim = pretrained_model.config.hidden_size)
        self.embedding = embedding
        self.simimagepred = nn.Sequential(encoder, normfeats, simimagepred)

        # Instantiate ContrastiveProjection
        contrastiveproj = ContrastiveProj(embedding_dim = pretrained_model.config.hidden_size,
                                          hidden_dim = hidden_dim,
                                          projected_dim = projected_dim)
        self.contrastiveproj = nn.Sequential(embedding, encoder, normfeats, contrastiveproj)
        
        # Contrastive loss 
        self.contrastive_loss = ntxentloss
        
        # Binary cross entropy loss for similar image pair
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
        # Model accuracy in detecting modified copy
        self.train_acc, self.val_acc = Accuracy(), Accuracy()   
          
        # For logging best validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self,
                img_r: torch.Tensor,
                img_q: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        if img_q is not None:
            return self.feature_extractor(img_r)
        else:
            embedding_rq = self.embedding(img_r, img_q)
            logits = self.simimagepred(embedding_rq)
            preds = torch.argmax(logits, dim = 1)
            return preds

    def step(self,
             img_r: List[torch.Tensor],
             img_q: List[torch.Tensor],
             label: List[torch.Tensor]):
        
        # img_r, img_q to SimImagePredictor
        embedding_rq = self.embedding(img_r, img_q) ## nn sequential don't take multiple input
        logits = self.simimagepred(embedding_rq)
        # Calculate binary cross entropy loss of similar image pair
        simimage_loss = self.bce_loss(logits, label.unsqueeze(dim = 1))
        # Predictions
        preds = torch.argmax(logits, dim = 1)
        
        # Get positive indices
        pos_indices = label.bool()
        # Forward positive indices of img_r and img_q to ContrastiveProjection
        proj_r = self.contrastiveproj(img_r[pos_indices])
        proj_q = self.contrastiveproj(img_q[pos_indices])

        # Calculate contrastive loss between un-augmented img_r and augmented positive pair of img_q
        contrastive_loss = self.contrastive_loss(proj_r, proj_q)
        
        # Weighted sum of bce and contrastive loss
        total_loss = self.hparams.beta1 * simimage_loss + self.hparams.beta2 * contrastive_loss
        
        return {'simimage': simimage_loss, 'contrastive': contrastive_loss, 'total': total_loss}, preds

    def training_step(self, batch: Any, batch_idx: int):
        img_r, img_q, label = batch
        img_r, img_q, label = torch.vstack(img_r), torch.vstack(img_q), torch.hstack(label)

        losses, preds = self.step(img_r, img_q, label)
        
        # Log train metrics
        acc = self.train_acc(preds, label.int())
        self.log("train/total_loss", losses['total'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("train/simimage_loss", losses['simimage'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("train/contrastive_loss", losses['contrastive'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("train/acc", acc, on_step = True, on_epoch = True, prog_bar = True)

        return losses['total']

    def validation_step(self, batch: Any, batch_idx: int):
        img_r, img_q, label = batch
        losses, preds = self.step(img_r, img_q, label)

        # Log val metrics
        acc = self.val_acc(preds, label.int())
        self.log("val/total_loss", losses['total'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("val/simimage_loss", losses['simimage'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("val/contrastive_loss", losses['contrastive'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("val/acc", acc, on_step = True, on_epoch = True, prog_bar = True)

        return losses['total']

    def validation_epoch_end(self, outputs: Any):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch = True, prog_bar = True)

    def on_epoch_end(self):
        # Reset metrics at the end of every epoch
        self.train_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.parameters(),
                                lr = self.hparams.lr,
                                weight_decay = self.hparams.weight_decay)