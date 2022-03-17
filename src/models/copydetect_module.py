from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from pytorch_metric_learning.losses import CrossBatchMemory, NTXentLoss
from pytorch_metric_learning.utils import distributed as pml_dist

class CopyDetectModule(LightningModule):
    def __init__(self):
        super().__init__()
        # To access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # Cross batch memory contrastive loss and binary cross entropy loss for similar image pair
        #! embedding size
        self.xbm_contrastive_loss = CrossBatchMemory(NTXentLoss(self.hparams.temperature),
                                                embedding_size = self.hparams.embedding_size,
                                                memory_size = self.hparams.memory_size)
        self.sim_img_bce_loss = torch.nn.BCEWithLogitsLoss()

        # Separate metric instance for train, val and test step to ensure a proper reduction over the epoch
        self.train_acc, self.val_acc, self.test_acc = Accuracy(), Accuracy(), Accuracy()
        
        # For logging best validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
