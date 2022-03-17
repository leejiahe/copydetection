#%%
import os

import torch

get_path = lambda x: os.path.join(os.getcwd(),'data', x)

from src.datamodules.components.augmentation import Augment

augment = Augment(overlay_image_dir = get_path('train/'),
                  n_upper = 2,
                  n_lower = 1)

from src.datamodules.copydetect_datamodule import CopyDetectDataModule

cdm = CopyDetectDataModule(train_dir = get_path('train/'),
                           references_dir = get_path('references/'),
                           dev_queries_dir = get_path('dev_queries/'),
                           final_queries_dir = get_path('final_queries/'),
                           augment = augment,
                           dev_validation_set = get_path('dev_validation_set.csv'),
                           batch_size = 16,
                           pin_memory = True,
                           num_workers = 10,
                           n_crops = 2
                           )

cdm.setup()
#%%
img_rt, img_qt, labelt = next(iter(cdm.train_dataloader()))
img_rt, img_qt, labelt = torch.vstack(img_rt), torch.vstack(img_qt), torch.hstack(labelt)


#%%
pos_indices = labelt.bool()
img_rt[pos_indices]

#%%


#%%
from typing import Any, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import ViTModel

from pytorch_metric_learning.losses import CrossBatchMemory, NTXentLoss
from pytorch_metric_learning.utils import distributed as pml_dist

from src.models.components.embeddings import CopyDetectEmbedding, NormalizedFeatures

def create_labels(num_pos_pairs, previous_max_label):
    labels = torch.arange(0, num_pos_pairs)
    labels = torch.cat((labels, labels))
    # Offset so that the labels do not overlap with any labels in the memory queue
    labels += previous_max_label + 1
    # To enqueue the output of img_q, which is the 2nd half of the batch
    enqueue_idx = torch.arange(num_pos_pairs, num_pos_pairs * 2)
    return labels, enqueue_idx

class CopyDetectModule(LightningModule):
    def __init__(self,
                 pretrained_arch: str,          # Pretrained ViT architecture
                 simimagepred: nn.Module,       # Similar image predictor
                 contrastiveproj: nn.Module,    # Contrastive projection
                 beta1: int = 1,                # Similar image BCE loss multiplier
                 beta2: int = 1):               # Contrastive loss multiplier
        super().__init__()
        self.save_hyperparameters(logger = False)
        self.beta1 = beta1
        self.beta2 = beta2
         
        # Instantiate ViT encoder from pretrained model
        pretrained_model = ViTModel.from_pretrained(pretrained_arch)
        encoder = nn.Sequential(pretrained_model.vit.encoder,
                                pretrained_model.vit.layernorm)
                
        # Instantiate embedding, we use the pretrained ViT cls and position embedding
        embedding = CopyDetectEmbedding(config = pretrained_model.config,
                                        vit_cls = pretrained_model.embeddings.cls_token,
                                        pos_emb = pretrained_model.embeddings.positional_embeddings)
        
        # Normalized features
        normfeats = NormalizedFeatures(hidden_dim = pretrained_model.config.hidden_size,
                                      eps = pretrained_model.config.layer_norm_eps)
        # Feature Vector Extractor
        self.feature_extractor = nn.Sequential(embedding, encoder, normfeats)
        
        # Instantiate SimImagePredictor
        self.simimagepred = nn.Sequential(embedding, normfeats, simimagepred)
        
        # Instantiate ContrastiveProjection
        self.contrastiveproj = nn.Sequential(embedding, normfeats, contrastiveproj)
        
        # Distributed cross batch memory contrastive loss 
        self.xbm_contrastive_loss = pml_dist(CrossBatchMemory(NTXentLoss(self.temperature),
                                                              embedding_size = pretrained_model.config.hidden_size,
                                                              memory_size = self.memory_size))
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
            logits = self.simimagepred(img_r, img_q)
            preds = torch.argmax(logits, dim = 1)
            return preds

    def step(self, img_r: List[torch.Tensor], img_q: List[torch.Tensor], label: List[torch.Tensor], val: Optional[bool] = False):
        # img_r, img_q to SimImagePredictor
        logits = self.simimagepred(img_r, img_q)
        # Calculate binary cross entropy loss of similar image pair
        simimage_loss = self.bce_loss(logits, label)
        # Predictions
        preds = torch.argmax(logits, dim = 1)
        
        # Get positive indices
        pos_indices = label.bool()
        # Forward positive indices of img_r and img_q to ContrastiveProjection
        proj_r = self.contrastiveproj(img_r[pos_indices])
        proj_q = self.contrastiveproj(img_q[pos_indices])
        proj_rq = torch.cat([proj_r, proj_q], dim = 0)
        
        # Calculate contrastive loss between un-augmented img_r and augmented positive pair of img_q
        previous_max_label = torch.max(self.xbm_contrastive_loss.label_memory)
        labels, enqueue_idx = create_labels(proj_rq.size(0), previous_max_label)
        enqueue_idx = None if val else enqueue_idx # Don't enqueue validation embedding into memory during validation step
        contrastive_loss = self.xbm_contrastive_loss(proj_rq, labels, enqueue_idx = enqueue_idx)
        
        # Weighted sum of bce and contrastive loss
        total_loss = self.beta1 * simimage_loss + self.beta2 * contrastive_loss
        
        return {'simimage': simimage_loss, 'contrastive': contrastive_loss, 'total': total_loss}, preds, label

    def training_step(self, batch: Any, batch_idx: int):
        img_r, img_q, label = batch
        img_r, img_q, label = torch.vstack(img_r), torch.vstack(img_q), torch.hstack(label)
        losses, preds, label = self.step(batch)
        
        # Log train metrics
        acc = self.train_acc(preds, label)
        self.log("train/total_loss", losses['total'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("train/simimage_loss", losses['simimage'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("train/contrastive_loss", losses['contrastive'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("train/acc", acc, on_step = True, on_epoch = True, prog_bar = True)

        return losses['total']

    def validation_step(self, batch: Any, batch_idx: int):
        img_r, img_q, label = batch
        losses, preds, label = self.step(img_r, img_q, label, val = True)

        # Log val metrics
        acc = self.val_acc(preds, label)
        self.log("val/total_loss", losses['total'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("val/simimage_loss", losses['simimage'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("val/contrastive_loss", losses['contrastive'], on_step = True, on_epoch = True, prog_bar = False)
        self.log("val/acc", acc, on_step = True, on_epoch = True, prog_bar = True)

        return losses['total']

    def validation_epoch_end(self, outputs: Any):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        # Reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            params = self.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.weight_decay
        )


# %%
