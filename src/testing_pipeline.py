import os
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from PIL import Image

import hydra
import torch
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import DataLoader
from omegaconf import DictConfig
from torchvision import transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_SDEV = [0.229, 0.224, 0.225]


# Return all the image paths in a folder
get_image_file = lambda image_dir: [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
get_image = lambda folder, index: Image.open(folder[index])
get_file_ids = lambda x: [f for f in os.listdir(x)]

class CopyDetectDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 transform):
        self.image_files = np.array(get_image_file(image_dir))
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_files[index]
        image = Image.open(image_path)
        image_id = os.path.split(image_path)[-1]
        return self.transform(image), image_id

class PredictedDataset(Dataset):
    def __init__(self,
                 predictions: list,
                 references_dir: str,
                 final_queries_dir: str,
                 transform):
        self.predictions = predictions
        self.references_images = np.array(get_image_file(references_dir))
        self.final_queries_images = np.array(get_image_file(final_queries_dir))
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.predictions)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        reference_image = get_image(self.references_images, index)
        final_queries_image = get_image(self.final_queries_images, index)
        
        return self.transform(reference_image). self.transform(final_queries_image) 


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline.
    Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model and load from checkpoint
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    # To eval mode
    model.eval()
    model.freeze()
    
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    
    
    
    #! Specify image size at config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((config.image_size, config.image_size)),
                                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_SDEV)])
    
    reference_dataset = CopyDetectDataset(config.references_dir, transform = transform)
    final_queries_dataset = CopyDetectDataset(config.final_queries_dir, transform = transform)
    
    reference_dataloader = DataLoader(dataset           = reference_dataset,
                                      batch_size        = config.batch_size,
                                      num_workers       = config.num_workers,
                                      pin_memory        = config.pin_memory,
                                      shuffle           = False)
    final_queries_dataloader = DataLoader(dataset       = final_queries_dataset,
                                          batch_size    = config.batch_size,
                                          num_workers   = config.num_workers,
                                          pin_memory    = config.pin_memory,
                                          shuffle       = False)
    
    #! Check if test append output from different batches
    # Reference image embedding
    # Using the cls token and GeM pooled local patches, it gives a global and important local representations of the image

    with torch.no_grad():
        log.info("Extracting features for reference images")
        references_feats, references_id = [], []
        for ref_image, ref_id in reference_dataloader:
            ref_feat = model.feature_extract(ref_image)
            references_feats.append(ref_feat)
            references_id.append(ref_id)
            
        references_feats = torch.vstack(references_feats).detach().cpu().numpy()
        
        final_queries_feats, final_queries_id = [], []
        for query_image, query_id in final_queries_dataloader:
            query_feat = model.feature_extract(query_image)
            final_queries_feats.append(query_feat)
            final_queries_id.append(query_id)
            
        final_queries_feat = torch.vstack(final_queries_feat).detach().cpu().numpy()
    
    # Search for closest references image from queries image
    nq = len(final_queries_feats)
    lims, dis, ids = utils.search_with_capped_res(final_queries_feats, references_feats, num_results = nq * 10)

    predictions_list = []
    for i in range(nq):
        for j in range(lims[i], lims[i+1]):
            predictions_list.append([final_queries_id[i], references_id[ids[j]]])
            
    predicted_dataset = PredictedDataset(predictions       = predictions_list,
                                         references_dir    = config.references_dir,
                                         final_queries_dir = config.final_queries_dir,
                                         transform         = transform)
    
    predicted_dataloader = DataLoader(dataset      = predicted_dataset,
                                      batch_size   = config.batch_size,
                                      num_workers  = config.num_workers,
                                      pin_memory   = config.pin_memory)
    # Get copy detection from model
    log.info("Testing model on final queries and references image to get similar image score")
    scores = []
    with torch.no_grad():
        for ref_image, query_image in predicted_dataloader:
            score = model.simimage(ref_image, query_image)
            scores.append(score)
    scores = torch.vstack(scores).detach().cpu().numpy()
    #! Check this
    # Get scores into dataframe for evaluation
    matching_df = pd.DataFrame({"query_id": predictions_list[i][0],
                                "reference_id": predictions_list[i][1],
                                "score": scores[i]} 
                               for i in range(len(scores)))
    
    final_ground_truth_df = pd.read_csv(config.final_ground_truth)
    
    ap, rp90 = utils.evaluate_metrics(matching_df, final_ground_truth_df)
    log.info(f"The ap is {ap} and rp90 is {rp90}")
