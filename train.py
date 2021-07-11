
import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn, optim
from torch.nn import functional as F

from data_loader import KeypointsDataModule
from utils import draw_keypoints, Keypoints


class Keypointdetector(pl.LightningModule):
    def __init__(
        self,
        num_keypoints: int = 12,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.feature_extractor = timm.create_model("resnet50", pretrained=True, num_classes=0)
        # self.feature_extractor.freeze()
        self.head = nn.Linear(2048, out_features = 2*num_keypoints)
        # mm.create_model('resnet18', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
        # self.model.classifier = nn.Linear(1280, 2*num_keypoints)
        # x = self.model.features(x)
        # x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        # l0 = self.l0(x

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.head(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, (keypoints, visible, labels) = batch
        y_hat = self(x)
        # Non visible keypoints should be dealt with. Set to zero loss?
        # keypoints = keypoints*torch.repeat_interleave(visible, 2, dim=1)
        # y_hat = y_hat*torch.repeat_interleave(visible, 2, dim=1)
        # breakpoint()
        keypoints = keypoints.view(keypoints.size(0), -1)
        loss = F.mse_loss(y_hat, keypoints)

        return loss

    def validation_step(self, batch, batch_idx):
        x, (keypoints, visible, labels) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)
        y_hat = self(x)
        # breakpoint()

        loss = F.mse_loss(y_hat, keypoints)
        self.log("valid_loss", loss)
        if len(labels[0]) != (len(keypoints[0])//2):
            raise ValueError("Data is broken. missing labels")
        return (x[0], y_hat[0], visible[0], labels[0])

    def test_step(self, batch, batch_idx):
        x, (keypoints, visible, labels) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)

        y_hat = self(x)
        loss = F.mse_loss(y_hat, keypoints)
        # loss = loss*visible
        self.log("test_loss", loss)

    def configure_optimizers(self):
        print(self.hparams.learning_rate)
        return torch.optim.Adam(self.head.parameters(), lr=self.learning_rate)

    def validation_epoch_end(self, outputs) -> None:
        """Compute metrics on the full validation set.
        Args:
            outputs (Dict[str, Any]): Dict of values collected over each batch put through model.eval()(..)
        """
        # breakpoint()
        c = 0
        for image, keypoints, visible, labels in outputs:
            label_names = [Keypoints._fields[o-1] for o in labels.cpu()]
            res = draw_keypoints(
                image, keypoints.reshape(-1, 2), label_names, visible, show_all=True
            )
            res.save(f"tmp_{c}.png")
            c += 1


from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    name="keypoints", project="wf", save_dir="/mnt/vol_b/models/keypoints"
)


def cli_main():

    model = Keypointdetector()

    trainer = pl.Trainer(gpus=1, logger=wandb_logger, auto_lr_find=True)
    trainer.tune(model, KeypointsDataModule("/mnt/vol_b/clean_data/tmp2"))
    trainer.fit(model, KeypointsDataModule("/mnt/vol_b/clean_data/tmp2"))


if __name__ == "__main__":
    # cli_lightning_logo()
    cli_main()
