
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import timm
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.conv import Conv2d

from data_loader import KeypointsDataModule
from utils import draw_keypoints, Keypoints
from kornia.losses.focal import FocalLoss

# criterion = FocalLoss({"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'})
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
                        nn.Upsample(size=output_image_size, mode="bilinear", align_corners=False),
                        # nn.Conv2d(in_channels=num_points, out_channels=1, kernel_size=1)

                )



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
        x, (targets, keypoints, visible, labels) = batch
        y_hat = self(x)
        # Non visible keypoints should be dealt with. Set to zero loss?
        # keypoints = keypoints*torch.repeat_interleave(visible, 2, dim=1)
        # y_hat = y_hat*torch.repeat_interleave(visible, 2, dim=1)
        # breakpoint()
        # keypoints = keypoints.view(keypoints.size(0), -1)
        # breakpoint()

        loss = criterion(y_hat, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        x, (targets,keypoints, visible, labels) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)
        y_hat = self(x)
        # breakpoint()

        loss = criterion(y_hat, targets)
        self.log("valid_loss", loss)
        if len(labels[0]) != (len(keypoints[0])//2):
            raise ValueError("Data is broken. missing labels")
        return (x[0], y_hat[0], targets[0], visible[0], labels[0])

    def test_step(self, batch, batch_idx):
        x, (targets, keypoints, visible, labels) = batch
        keypoints = keypoints.view(keypoints.size(0), -1)

        y_hat = self(x)

        loss = criterion(y_hat, targets)
        # loss = loss*visible
        self.log("test_loss", loss)

    def configure_optimizers(self):
        print(self.hparams.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def validation_epoch_end(self, outputs) -> None:
        """Compute metrics on the full validation set.
        Args:
            outputs (Dict[str, Any]): Dict of values collected over each batch put through model.eval()(..)
        """
        # wandb.log({"loss": 0.314, "epoch": 5,
        #    "inputs": wandb.Image(inputs),
        #    "logits": wandb.Histogram(ouputs),
        #    "captions": wandb.HTML(captions)
        #    })
        c = 0
        for image, y_hat, target, visible, labels in outputs:
            # breakpoint()
            keypoints=[]
            for i in range(y_hat.shape[0]):
                keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
            label_names = [Keypoints._fields[o-1] for o in labels.cpu()]
            res = draw_keypoints(
                image, keypoints, label_names, visible, show_all=True
            )
            res.save(f"tmp_{c}.png")

            keypoints=[]
            for i in range(target.shape[0]):
                keypoints.append((target[i]==torch.max(target[i])).nonzero()[0].tolist()[::-1])
            label_names = [Keypoints._fields[o-1] for o in labels.cpu()]
            res = draw_keypoints(
                image, keypoints, label_names, visible, show_all=True
            )
            res.save(f"truth_{c}.png")

            c += 1


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

wandb_logger = WandbLogger(
    name="keypoints", project="wf", save_dir="/mnt/vol_b/models/keypoints", log_model=True
)


def cli_main():

    model = Keypointdetector()
    checkpoint_callback = ModelCheckpoint(dirpath='/mnt/vol_b/models/')

    trainer = pl.Trainer(gpus=1, max_epochs=3, logger=wandb_logger, auto_lr_find=True, track_grad_norm=2)
    trainer.tune(model, KeypointsDataModule("/mnt/vol_b/clean_data/tmp2"))
    trainer.fit(model, KeypointsDataModule("/mnt/vol_b/clean_data/tmp2"))
    trainer.save_checkpoint("example.ckpt")

if __name__ == "__main__":
    # cli_lightning_logo()
    cli_main()
