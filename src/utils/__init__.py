import logging
import warnings
from typing import List, Sequence, Tuple, Optional, Any
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug",
                  "info",
                  "warning",
                  "error",
                  "exception",
                  "fatal",
                  "critical",):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(config: DictConfig,
                 print_order: Sequence[str] = ("datamodule",
                                               "model",
                                               "callbacks",
                                               "logger",
                                               "trainer"),
                 resolve: bool = True,
                 ) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(config: DictConfig,
                        model: pl.LightningModule,
                        datamodule: pl.LightningDataModule,
                        trainer: pl.Trainer,
                        callbacks: List[pl.Callback],
                        logger: List[pl.loggers.LightningLoggerBase],
                        ) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(config: DictConfig,
           model: pl.LightningModule,
           datamodule: pl.LightningDataModule,
           trainer: pl.Trainer,
           callbacks: List[pl.Callback],
           logger: List[pl.loggers.LightningLoggerBase],
           ) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
            
def argsort(seq: Sequence[Any]):
    """Like np.argsort but for 1D sequences. Based on https://stackoverflow.com/a/3382369/3853462"""
    return sorted(range(len(seq)), key=seq.__getitem__)


def precision_recall(y_true: np.ndarray,
                     probas_pred: np.ndarray,
                     num_positives: int,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precisions, recalls, and thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.

    Returns
    -------
    precisions, recalls, thresholds
        ordered by increasing recall, as for a precision-recall curve
    """
    probas_pred = probas_pred.flatten()
    y_true = y_true.flatten()
    # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
    # eg,the final order will be (0.5, False), (0.5, False), (0.5, True), (0.4, False), ...
    # This allows to have the worst possible AP.
    # It prevents participants from putting the same score for all predictions to get a good AP.
    order = argsort(list(zip(probas_pred, ~y_true)))
    order = order[::-1]  # sort by decreasing score
    probas_pred = probas_pred[order]
    y_true = y_true[order]

    ntp = np.cumsum(y_true)  # number of true positives <= threshold
    nres = np.arange(len(y_true)) + 1  # number of results

    precisions = ntp / nres
    recalls = ntp / num_positives
    return precisions, recalls, probas_pred


def average_precision(recalls: np.ndarray, precisions: np.ndarray):
    """
    Compute the micro-average precision score (μAP).

    Parameters
    ----------
    recalls : np.ndarray
        Recalls. Must be sorted by increasing recall, as in a PR curve.
    precisions : np.ndarray
        Precisions for each recall value.

    Returns
    -------
    μAP: float
    """

    # Check that it's ordered by increasing recall
    if not np.all(recalls[:-1] <= recalls[1:]):
        raise ValueError("recalls array must be sorted before passing in")
    return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()


def find_operating_point(x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray, required_x: float,
                         ) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find the highest y (and corresponding z) with x at least `required_x`.

    Returns
    -------
    x, y, z
        The best operating point (highest y) with x at least `required_x`.
        If we can't find a point with the required x value, return
        x=required_x, y=None, z=None
    """
    valid_points = x >= required_x
    if not np.any(valid_points):
        return required_x, None, None

    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_z = z[valid_points]
    best_idx = np.argmax(valid_y)
    return valid_x[best_idx], valid_y[best_idx], valid_z[best_idx]


def evaluate_metrics(submission_df: pd.DataFrame, gt_df: pd.DataFrame):
    """Given a matching track submission dataframe and a ground truth dataframe,
    compute the competition metrics."""

    # Subset submission_df to query_ids that we have labels for in gt_df
    submission_df = submission_df[submission_df["query_id"].isin(gt_df["query_id"])]

    gt_pairs = {tuple(row) for row in gt_df[["query_id", "reference_id"]].itertuples(index=False) if not pd.isna(row.reference_id)}

    # Binary indicator for whether prediction is a true positive or false positive
    y_true = np.array([tuple(row) in gt_pairs for row in submission_df[["query_id", "reference_id"]].itertuples(index=False)])
    
    # Confidence score, as if probability. Only property required is greater score == more confident.
    probas_pred = submission_df["score"].values

    p, r, t = precision_recall(y_true, probas_pred, len(gt_pairs))

    # Micro-average precision
    ap = average_precision(r, p)

    # Metrics @ Precision>=90%
    pp90, rp90, tp90 = find_operating_point(p, r, t, required_x=0.9)

    if rp90 is None:
        # Precision was never above 90%
        rp90 = 0.0

    return ap, rp90

def query_iterator(xq: np.ndarray):
    """Produces batches of progressively increasing sizes."""
    nq = len(xq)
    bs = 32
    i = 0
    while i < nq:
        xqi = xq[i : i + bs]  # noqa: E203
        yield xqi
        if bs < 20_000:
            bs *= 2
        i += len(xqi)


def search_with_capped_res(xq: np.ndarray, xb: np.ndarray, num_results: int):
    """Searches xq (queries) into xb (reference), with a maximum total number of results."""
    import faiss
    from faiss.contrib import exhaustive_search

    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)

    radius, lims, dis, ids = exhaustive_search.range_search_max_results(index,
                                                                        query_iterator(xq),
                                                                        1e10,  # initial radius is arbitrary
                                                                        max_results = 2 * num_results,
                                                                        min_results = num_results,
                                                                        ngpu=-1)  # use GPU if available
    n = len(dis)
    nq = len(xq)
    if n > num_results:
        # crop to num_results exactly
        o = dis.argpartition(num_results)[:num_results]
        mask = np.zeros(n, bool)
        mask[o] = True
        new_dis = dis[mask]
        new_ids = ids[mask]
        nres = [0] + [mask[lims[i] : lims[i + 1]].sum() for i in range(nq)]  # noqa: E203
        new_lims = np.cumsum(nres)
        lims, dis, ids = new_lims, new_dis, new_ids

    return lims, dis, ids

def create_labels(num_pos_pairs, previous_max_label, mod_device):
    # create labels that indicate what the positive pairs are
    labels = torch.arange(0, num_pos_pairs)
    labels = torch.cat((labels, labels), device = mod_device)
    # add an offset so that the labels do not overlap with any labels in the memory queue
    labels += previous_max_label + 1
    # we want to enqueue the output of encK, which is the 2nd half of the batch
    enqueue_idx = torch.arange(num_pos_pairs, num_pos_pairs * 2)
    return labels, enqueue_idx