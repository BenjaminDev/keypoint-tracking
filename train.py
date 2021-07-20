
import argparse
from typing import Dict, Tuple

import hydra
from numpy import heaviside
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pytorch_lightning as pl
import timm
import torch
# criterion = FocalLoss({"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'})
import wandb
from kornia.losses.focal import FocalLoss
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d
from PIL import Image
from matplotlib import cm
import numpy as np
from data_loader import KeypointsDataModule
from utils import Keypoints, draw_keypoints

criterion = nn.MSELoss(reduction='sum')


class Keypointdetector(pl.LightningModule):
    def __init__(
        self,
        num_keypoints: int = 12,
        learning_rate: float = 0.0001,
        output_image_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        # self.model = timm.create_model("hrnet_w18", pretrained=True, num_classes=2*num_keypoints)
        self.features = timm.create_model('hrnet_w18', pretrained=True,features_only=True, num_classes=0, global_pool='')
        # self.model = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool='')
        # self.features = timm.create_model('hrnet_w18', pretrained=True, num_classes=0, global_pool='')
        # final_inp_channels = 2048
        # BN_MOMENTUM = 0.01
        # FINAL_CONV_KERNEL=1
        # num_points=24
        final_inp_channels = 960
        BN_MOMENTUM = 0.01
        FINAL_CONV_KERNEL=1
        num_points=12
        self.head = nn.Sequential(
                    nn.Conv2d(
                        in_channels=final_inp_channels,
                        out_channels=final_inp_channels,
                        kernel_size=1,
                        # stride=1,
                        dilation=2,
                        padding=1 ),
                    nn.BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=final_inp_channels,
                        out_channels=num_points,
                        kernel_size=FINAL_CONV_KERNEL,
                        dilation=2,
                        stride=1,
                        padding=1),
                        nn.Upsample(size=(output_image_size[0],output_image_size[1] ), mode="bilinear", align_corners=False),
                        # nn.Conv2d(in_channels=num_points, out_channels=1, kernel_size=1)

                )


        # self.head[0].register_hook(
        #     lambda grad: self.logger.experiment.log({"head_grads": grad.cpu()})
        # )
        # self.feature_extractor.freeze()
        # self.head = nn.Linear(2048, out_features = 2*num_keypoints)
        # self.transform_input = True
        # mm.create_model('resnet18', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
        # self.model.classifier = nn.Linear(1280, 2*num_keypoints)
        # x = self.model.features(x)
        # x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        # l0 = self.l0(x

    def forward(self, x):
        # x = x.view(x.size(0), -1)
                # imagenet normalisation

        # self.feature_extractor.eval()
        # with torch.no_grad():
        #     representations = self.feature_extractor(x).flatten(1)
        # x = self.head(representations)
        # return self.model(x)
        # return self.head(self.features(x)).squeeze(-1).squeeze(-1)
        x = self.features(x)

        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)

        return self.head(x).squeeze(-1).squeeze(-1)


    def training_step(self, batch, batch_idx):
        x, (targets, keypoints, visible, labels, captions) = batch
        y_hat = self(x)
        # Non visible keypoints should be dealt with. Set to zero loss?
        # keypoints = keypoints*torch.repeat_interleave(visible, 2, dim=1)
        # y_hat = y_hat*torch.repeat_interleave(visible, 2, dim=1)
        #
        # keypoints = keypoints.view(keypoints.size(0), -1)
        #

        loss = criterion(y_hat, targets)

        return loss

    def validation_step(self, batch, batch_idx):

        x, (targets,keypoints, visible, labels, captions) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)
        y_hat = self(x)
        #

        loss = criterion(y_hat, targets)
        self.log("val_loss", loss)
        if len(labels[0]) != (len(keypoints[0])//2):
            raise ValueError("Data is broken. missing labels")
        return (x, y_hat, targets, visible, labels, captions)

    def test_step(self, batch, batch_idx):
        x, (targets, keypoints, visible, labels, captions) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)

        y_hat = self(x)

        loss = criterion(y_hat, targets)
        # loss = loss*visible
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
        # wandb.log(
        #     {f"val_loss_hist": wandb.Histogram([[h[-1] for h in batch_of_outputs]])}
        # )
        pred_images, truth_images, captions = [], [], []
        heatmaps = []
        for batch_of_outputs in all_batches_of_outputs[::4]:
            #
            # if len(outputs) != 5:
            #
            for image, y_hat, target, visible, labels, caption in zip(batch_of_outputs[0],batch_of_outputs[1],batch_of_outputs[2],batch_of_outputs[3], batch_of_outputs[4], batch_of_outputs[5]) :

                keypoints=[]
                for i in range(y_hat.shape[0]):
                    keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
                label_names = [Keypoints._fields[o-1] for o in labels.cpu()]

                heatmaps.append(Image.fromarray(np.uint8(cm.viridis(y_hat[i].cpu().numpy())*255)))


                res_val = draw_keypoints(
                    image, keypoints, label_names, show_labels = True, short_names = True, visible=visible, show_all=True
                )
                pred_images.append(res_val)

                keypoints=[]
                for i in range(target.shape[0]):
                    keypoints.append((target[i]==torch.max(target[i])).nonzero()[0].tolist()[::-1])
                label_names = [Keypoints._fields[o-1] for o in labels.cpu()]
                res = draw_keypoints(
                    image, keypoints, label_names, show_labels = True, short_names = True, visible=visible, show_all=True
                )
                truth_images.append(res)
                captions.append(caption)

        wandb.log({
            f"Pred Keypoints":  [wandb.Image(o, caption=c) for o, c in zip(pred_images, captions)],
            f"Truth Keypoints" : [wandb.Image(o, caption=c) for o, c in zip(truth_images, captions)],
            f"Heatmaps" : [wandb.Image(o) for o in heatmaps]
        })



import os

@hydra.main(config_path="./experiments", config_name="config_1.yaml")
def cli_main(cfg: DictConfig):
    wb=cfg.wb
    wandb.init(name=wb.name, project=wb.project)
    wandb_logger = WandbLogger(
        name=wb.name, project=wb.project, save_dir=wb.save_dir, log_model=True
    )
    cfg.model.input_size

    model = Keypointdetector(output_image_size=cfg.model.input_size)
    # wandb.watch(model)
    # checkpoint_callback = ModelCheckpoint(dirpath='/mnt/vol_c/models/', save_top_k=3)
    # `mc = ModelCheckpoint(monitor='your_monitor')` and use it as `Trainer(callbacks=[mc])`
    data_dirs=[os.path.join(cfg.data.base_dir, o) for o in cfg.data.sets]
    print (data_dirs)

    trainer = pl.Trainer(gpus=1, max_epochs=200, logger=wandb_logger, auto_lr_find=True, track_grad_norm=2)
    trainer.tune(model, KeypointsDataModule(data_dirs=data_dirs, input_size=cfg.model.input_size))
    trainer.fit(model,  KeypointsDataModule(data_dirs=data_dirs, input_size=cfg.model.input_size))


if __name__ == "__main__":
    cli_main()
