import argparse
import math
import os
from collections import defaultdict
from re import S
from typing import Dict, Tuple

import hydra
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pytorch_lightning as pl
import timm
import torch
import wandb
from kornia.losses.focal import FocalLoss
from matplotlib import cm
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import build_from_cfg
from mmpose.apis import multi_gpu_test, single_gpu_test, train_model
from mmpose.models import HRNet
from mmpose.models.builder import BACKBONES, HEADS, LOSSES, NECKS, POSENETS
from numpy import heaviside
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d
from torchmetrics import MeanAbsoluteError

from data_loader import KeypointsDataModule
from losses import JointsMSELoss, Loss_weighted
from utils import Keypoints, draw_keypoints


def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)


def build_posenet(cfg):
    """Build posenet."""
    return build(cfg, POSENETS)


criterion = Loss_weighted()

mean_absolute_error = MeanAbsoluteError()


class Keypointdetector(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
        inferencing: bool = False,
        num_keypoints: int = 12,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.valdation_count = 0
        self.cfg = Config.fromfile(
            "/mnt/vol_c/code/sketchpad/experiments/hrnet_light_config.py"
        )
        model = build_posenet(self.cfg.model)
        load_checkpoint(
            model,
            "/mnt/vol_c/code/sketchpad/pretrained_models/naive_litehrnet_18_coco_256x192.pth",
            map_location="cpu",
        )
        self.backbone = [o for o in model.children()][0]
        self.head = [o for o in model.children()][1]
        self.upsample = nn.Upsample(
            size=tuple(config.model.input_size), mode="bilinear", align_corners=False
        )
        self.model = nn.Sequential(self.backbone, self.head, self.upsample)

    def forward(self, x):

        output_heatmap = self.model(x)
        return output_heatmap

    def training_step(self, batch, batch_idx):
        x, (targets, M, keypoints, visible, labels, captions) = batch
        y_hat = self(x)
        loss = criterion(y_hat, targets, M)
        # loss = self.head.get_loss(y_hat, targets, M)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x, (targets, M, keypoints, visible, labels, captions) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)
        y_hat = self(x)
        loss = criterion(y_hat, targets, M)
        self.log("val_loss", loss)
        if len(labels[0]) != (len(keypoints[0]) // 2):
            raise ValueError("Data is broken. missing labels")
        self.valdation_count += 1
        if self.valdation_count % 10 == 0:
            return (x, y_hat, targets, M, visible, labels, captions, loss)

    def test_step(self, batch, batch_idx):
        x, (targets, M, keypoints, visible, labels, captions) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)
        y_hat = self(x)
        loss = criterion(y_hat, targets, M)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        print(self.hparams.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_epoch_end(self, outputs: Dict) -> None:
        wandb.log(
            {f"train_loss_hist": wandb.Histogram([[o["loss"].cpu() for o in outputs]])}
        )

    def validation_epoch_end(self, all_batches_of_outputs) -> None:
        """Compute metrics on the full validation set.
        Args:
            outputs (Dict[str, Any]): Dict of values collected over each batch put through model.eval()(..)
        """
        # FIXME: don't call cuda here do .to(device)
        MEAN = 255 * torch.tensor([0.485, 0.456, 0.406]).cuda()
        STD = 255 * torch.tensor([0.229, 0.224, 0.225]).cuda()
        pred_images, truth_images, captions = [], [], []
        heatmaps = []
        truth_maps = []
        mae = defaultdict(list)
        pred_kps = defaultdict(list)  # [key_point_index]->[mae_of_sin]
        target_kps = defaultdict(list)
        wandb.log(
            {
                f"val_loss_hist": wandb.Histogram(
                    [o[-1].cpu() for o in all_batches_of_outputs]
                )
            }
        )

        for batch_of_outputs in all_batches_of_outputs[::10]:
            for image, y_hat, target, M, visible, labels, caption in zip(
                batch_of_outputs[0],
                batch_of_outputs[1],
                batch_of_outputs[2],
                batch_of_outputs[3],
                batch_of_outputs[4],
                batch_of_outputs[5],
                batch_of_outputs[6],
            ):

                image = image * STD[:, None, None] + MEAN[:, None, None]

                pred_keypoints = []
                for i in range(y_hat.shape[0]):
                    pred_keypoints.append(
                        (y_hat[i] == torch.max(y_hat[i])).nonzero()[0].tolist()[::-1]
                    )
                label_names = [Keypoints._fields[o - 1] for o in labels.cpu()]
                for i, p_kps in enumerate(pred_keypoints):
                    pred_kps[i].append(p_kps)
                heat_image = Image.new(
                    "RGB", (y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2])
                )

                x_offset = 0
                for i in range(y_hat.shape[0]):
                    heat_image.paste(
                        Image.fromarray(
                            np.uint8(cm.viridis(y_hat[i].cpu().numpy()) * 255)
                        ),
                        (x_offset, 0),
                    )
                    x_offset += y_hat.shape[1]
                heatmaps.append(heat_image)
                truth_maps.append(
                    Image.fromarray(
                        np.uint8(cm.viridis(target.max(axis=0)[0].cpu().numpy()) * 255)
                    )
                )

                res_val = draw_keypoints(
                    image,
                    pred_keypoints,
                    label_names,
                    show_labels=True,
                    short_names=True,
                    visible=visible,
                    show_all=True,
                )
                pred_images.append(res_val)

                target_keypoints = []
                for i in range(target.shape[0]):
                    target_keypoints.append(
                        (target[i] == torch.max(target[i])).nonzero()[0].tolist()[::-1]
                    )
                label_names = [Keypoints._fields[o - 1] for o in labels.cpu()]
                for i, t_kps in enumerate(target_keypoints):
                    target_kps[i].append(t_kps)
                res = draw_keypoints(
                    image,
                    target_keypoints,
                    label_names,
                    show_labels=True,
                    short_names=True,
                    visible=visible,
                    show_all=True,
                )
                truth_images.append(res)
                captions.append(caption)

        for i in range(12):
            wandb.log(
                {
                    f"mean_absolute_error_kps{i}": mean_absolute_error(
                        torch.tensor(pred_kps[i]), torch.tensor(target_kps[i])
                    ).item()
                }
            )
        wandb.log(
            {
                f"Pred Keypoints": [
                    wandb.Image(o, caption=c) for o, c in zip(pred_images, captions)
                ],
                f"Truth Keypoints": [
                    wandb.Image(o, caption=c) for o, c in zip(truth_images, captions)
                ],
                f"Heatmaps": [
                    wandb.Image(o, caption=c) for o, c in zip(heatmaps, captions)
                ],
                f"TruthMaps": [
                    wandb.Image(o, caption=c) for o, c in zip(truth_maps, captions)
                ],
            }
        )


@hydra.main(config_path="./experiments", config_name="config_1.yaml")
def cli_main(cfg: DictConfig):
    wb = cfg.wb
    wandb.init(name=wb.name, project=wb.project)
    wandb_logger = WandbLogger(
        name=wb.name, project=wb.project, save_dir=wb.save_dir, log_model=True
    )
    cfg.model.input_size

    model = Keypointdetector(config=cfg, learning_rate=cfg.model.learning_rate)
    wandb.watch(model)
    data_dirs = [os.path.join(cfg.data.base_dir, o) for o in cfg.data.sets]
    print(data_dirs)
    early_stopping = EarlyStopping("val_loss")

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        auto_lr_find=cfg.trainer.auto_lr_find,
        track_grad_norm=2,
        # precision=16,
        stochastic_weight_avg=True,
        gradient_clip_val=0.5,
        # accumulate_grad_batches=3,
        log_every_n_steps=10,  # For large batch_size and small samples
        callbacks=[early_stopping],
        resume_from_checkpoint=cfg.trainer.resume_from_checkpoint,
    )
    trainer.tune(
        model,
        KeypointsDataModule(
            data_dirs=data_dirs,
            input_size=cfg.model.input_size,
            batch_size=cfg.trainer.batch_size,
        ),
    )
    trainer.fit(
        model,
        KeypointsDataModule(
            data_dirs=data_dirs,
            input_size=cfg.model.input_size,
            batch_size=cfg.trainer.batch_size,
        ),
    )


if __name__ == "__main__":
    cli_main()
