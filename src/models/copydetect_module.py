from typing import Any, List, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import ViTModel

from src.models.components.layers import CopyDetectEmbedding, NormalizedFeatures, SimImagePred, ContrastiveProj

from src.utils import create_labels
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from pytorch_metric_learning.utils.distributed import DistributedLossWrapper

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
        self.encoder = pretrained_model.encoder
        #for name, child in encoder.layer.named_children():
        #    if (int(name) < 11):
        #        for params in child.parameters():
        #            params.requires_grad = False
                    
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
        
        # XBM
        self.xbm_loss = DistributedLossWrapper(loss = CrossBatchMemory(loss = NTXentLoss(temperature = 0.9),
                                                                       embedding_size = pretrained_model.config.hidden_size,
                                                                       memory_size = 1024),
                                               efficient = False)
        
        # Binary cross entropy loss for similar image pair
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
        # Model accuracy in detecting modified copy
        self.train_acc, self.val_acc = Accuracy(), Accuracy()   
          
        # For logging best validation accuracy
        self.val_acc_best = MaxMetric()

    def feature_extract(self, batch: Any) -> torch.Tensor:
        # To extract feature vector
        img_r, img_id = batch
        encoding = self.normfeats(self.encoder(self.embedding(img_r)))
        batch_size, num_ch, H, W, = img_r.size()
        #dim = encoding.size(2) # batch_size, seq_len, dim 
        h, w = int(H/self.patch_size), int(W/self.patch_size)
        cls, feats = encoding[:,0,:], encoding[:,1:,:] # Get the cls token and all the images features
            
        #feats = feats.reshape(batch_size, h, w, dim).clamp(min = 1e-6).permute(0,3,1,2)
        feats = einops.rearrange(feats, 'b (h w) d -> b d h w', h = h, w = w).clamp(min = 1e-6)
        # GeM Pooling
        feats = F.avg_pool2d(feats.pow(4), (h,w)).pow(1./4)
        feats = einops.rearrange(feats, 'b d () () -> b d')
        # Concatenate cls tokens with image patches to give local and global views of image
        feature_vector = torch.cat((cls, feats), dim = 1)

        return feature_vector, img_id
        
    def predict_copy(self, batch):
        # For copy detection 
        logits = self.simimagepred(self.normfeats(self.encoder(self.embedding(batch))))
        preds = torch.argmax(logits, dim = 1)
        return preds

    def step(self,
             img_r: List[torch.Tensor],
             img_q: List[torch.Tensor],
             label: List[torch.Tensor]):
        
        # img_r, img_q to SimImagePredictor
        logits = self.simimagepred(self.normfeats(self.encoder(self.embedding(img_r, img_q)))) ## nn sequential don't take multiple input
        
        # Calculate binary cross entropy loss of similar image pair
        simimage_loss = self.bce_loss(logits, label.unsqueeze(dim = 1))
        # Predictions
        preds = torch.argmax(logits, dim = 1)
        
        # Get positive indices
        pos_indices = label.bool()
        # Forward positive indices of img_r and img_q to ContrastiveProjection
        proj_r = self.contrastiveproj(self.normfeats(self.encoder(self.embedding(img_r[pos_indices]))))
        proj_q = self.contrastiveproj(self.normfeats(self.encoder(self.embedding(img_q[pos_indices]))))

        # Calculate contrastive loss between un-augmented img_r and augmented positive pair of img_q
        contrastive_loss = self.contrastive_loss(proj_r, proj_q)
        
        #XBM
        proj_rq = torch.cat((proj_r, proj_q), dim = 0)
        previous_max_label = torch.max(self.xbm_loss.label_memory)
        indices, enqueue_idx = create_labels(proj_r.size(0), previous_max_label, proj_rq.device)
        xbm_loss = self.xbm_loss(proj_rq, indices, enqueue_idx = enqueue_idx)
        
        # Weighted sum of bce and contrastive loss
        total_loss = self.hparams.beta1 * simimage_loss + self.hparams.beta2 * contrastive_loss
        
        return {'simimage': simimage_loss, 'contrastive': contrastive_loss, 'total': total_loss}, preds

    def training_step(self, batch: Any, batch_idx: int):
        img_r, img_q, label = batch

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
        
    def test_step(self, batch: Any, batch_idx: int):
        feats = self.feature_extract(batch) # Get feat
        
        return feats
    
    def test_epoch_end(self, test_step_outputs: Any):
        all_feats, all_ids = [], []
        for step_output in test_step_outputs:
            all_feats.append(step_output[0])
            all_ids.extend(step_output[1])
            
        all_feats = torch.vstack(all_feats)
        #self.test_results = (all_feats, all_ids)
        
        return (all_feats, all_ids)
        
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        score = self.predict_copy(batch)
        
        return score
    
    def on_epoch_end(self):
        # Reset metrics at the end of every epoch
        self.train_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.parameters(),
                                lr = self.hparams.lr,
                                weight_decay = self.hparams.weight_decay)