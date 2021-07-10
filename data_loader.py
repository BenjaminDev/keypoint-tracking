import os
import warnings
from albumentations.augmentations.geometric.functional import keypoint_affine

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from torchvision.datasets import D
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import CIFAR10, MNIST
import PIL
from albumentations.pytorch import ToTensorV2
import cv2
import os
from torchvision import utils
from utils import Keypoints, MetaData, read_meta, COLORS, draw_keypoints
from pathlib import Path
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 16 if AVAIL_GPUS else 2
import albumentations as A

tfs = A.Compose(
                [
                    A.Resize(480, 540),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.ColorJitter(),
                    A.RandomBrightnessContrast(),
                ],
                keypoint_params=A.KeypointParams(format='xy')
            )


class KeypointsDataset(Dataset):
    def __init__(self, data_path:Path, train=True, transform=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.image_files = sorted(list(self.data_path.glob("*.png")))
        self.label_files = sorted(list(self.data_path.glob("*.json")))
        if (not any([l.stem == i.stem for l,i in zip(self.label_files, self.image_files)])):
            raise ValueError("Image files and label files mismatch")
        self.category_names = Keypoints._fields
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        if self.train:
            image = cv2.imread(os.fsdecode(self.image_files[index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            metadata = read_meta(self.label_files[index])
            keypoints = metadata.keypoints
            visible = torch.tensor(metadata.visible, dtype=torch.float)
            labels = metadata.keypoint_labels

        if self.transform:

            trasformed = self.transform(image=image,keypoints=keypoints)
            image = torch.tensor(trasformed["image"], dtype=torch.float32).permute(2, 0, 1)
            keypoints= torch.tensor(trasformed["keypoints"]).type(torch.float)

        return image, (keypoints.flatten(), visible, labels)
    def plot_sample(self, index, show_all=True):
        image, (keypoints, visible, labels) = self[index]
        if not image.shape[0] in {1, 3}:
            warnings.WarningMessage("Assuming image needs to be permuted into (c x h x w)")
            image = image.permute(2, 0, 1)
        image = image.type(torch.uint8)
        keypoints = keypoints.reshape(-1, 2)
        return draw_keypoints(image, keypoints, labels, visible, show_all)


class KeypointsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.Resize(480),transforms.ToTensor()])
        self.transform = tfs

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.num_classes = 10

    # def prepare_data(self):
    #     # download


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.keypoints_train = KeypointsDataset(self.data_dir, train=True, transform=self.transform)
            sample_image, _ = self.keypoints_train[0]
            self.dims = sample_image.shape
        # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:


    def train_dataloader(self):
        return DataLoader(self.keypoints_train, batch_size=BATCH_SIZE, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.keypoints_train, batch_size=BATCH_SIZE, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.keypoints_train, batch_size=BATCH_SIZE)
if __name__ == "__main__":

    ds = KeypointsDataset(data_path=Path("/home/ubuntu/clean_data/v004"), transform=tfs)
    sample = ds.plot_sample(0)
    sample.save("tmp.png")
