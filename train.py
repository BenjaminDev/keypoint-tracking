import torch
from torch.nn import functional as F

import pytorch_lightning as pl

from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn, optim
import timm
from utils import draw_keypoints
from data_loader import KeypointsDataModule
class Keypointdetector(pl.LightningModule):

    def __init__(
        self,
        hidden_dim: int = 128,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.classifier = nn.Linear(1280, 30)


    def forward(self, x):
        # x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, (keypoints, visible, labels) = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, keypoints)


        return loss

    def validation_step(self, batch, batch_idx):
        x, (keypoints, visible, labels) = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, keypoints)
        self.log('valid_loss', loss)
        return (x[0], y_hat[0], visible[0], labels[0])

    def test_step(self, batch, batch_idx):
        x, (keypoints, visible, labels) = batch

        y_hat = self(x)
        # Non visible keypoints should be dealt with. Set to zero loss?
        # keypoints = keypoints*torch.repeat_interleave(visible, 2, dim=1)
        # y_hat = y_hat*torch.repeat_interleave(visible, 2, dim=1)
        loss = F.mse_loss(y_hat, keypoints)
        # loss = loss*visible
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def validation_epoch_end(self, outputs) -> None:
        """Compute metrics on the full validation set.
        Args:
            outputs (Dict[str, Any]): Dict of values collected over each batch put through model.eval()(..)
        """
        breakpoint()
        c=0
        for image, keypoints, labels, visible in outputs:

            res = draw_keypoints(image, keypoints.reshape(-1, 2), labels, visible, show_all=True)
            res.save(f"tmp_{c}.png")
            c += 1






from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(
    name="keypoints", project="wf", save_dir="/mnt/vol_b/models/keypoints"
)
def cli_main():

    model = Keypointdetector()

    trainer = pl.Trainer(gpus=1,  logger=wandb_logger)
    trainer.fit(model, KeypointsDataModule("/home/ubuntu/clean_data/v004"))

if __name__ == '__main__':
    # cli_lightning_logo()
    cli_main()