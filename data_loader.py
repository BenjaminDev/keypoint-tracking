import os
import warnings
from pathlib import Path

import cv2
import PIL
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from albumentations.augmentations.geometric.functional import keypoint_affine
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
# from torchvision.datasets import D
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
from utils import COLORS, Keypoints, MetaData, draw_keypoints, read_meta

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 4 if AVAIL_GPUS else 2
import albumentations as A

tfs = A.Compose(
    [
        A.Resize(224, 224),

        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1), # need to fix c
        # A.ColorJitter(),
        # A.RandomBrightnessContrast(),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2()
    ],
    keypoint_params=A.KeypointParams(format="xy",label_fields=['labels',"visible"]),
)

def generate_target(img, pt, sigma=3.5,  label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    # REF: https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/f776dbe8eb6fec831774a47209dae5547ae2cda5/lib/utils/transforms.py#L216
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        raise Exception("sss")
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        # breakpoint()
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

class KeypointsDataset(Dataset):
    def __init__(self, data_path: Path, train=True, transform=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.image_files = sorted(list(self.data_path.glob("*.png")))
        self.label_files = sorted(list(self.data_path.glob("*.json")))
        if not any(
            [l.stem == i.stem for l, i in zip(self.label_files, self.image_files)]
        ):
            raise ValueError("Image files and label files mismatch")
        self.category_names = Keypoints._fields
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        # index = 0
        # index = index % len(self.label_files)
        if self.train:
            image = cv2.imread(os.fsdecode(self.image_files[index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            metadata = read_meta(self.label_files[index])
            keypoints = metadata.keypoints
            visible = torch.tensor(metadata.visible, dtype=torch.float)
            labels = metadata.keypoint_labels
            numeric_labels = [Keypoints._fields_defaults[o] for o in metadata.keypoint_labels]
            # if tpts[i, 1] > 0:
                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                #                                scale, self.output_size, rot=r)
        if self.transform:
            trasformed = self.transform(image=image,keypoints=keypoints, labels=numeric_labels, visible=visible)
            image = torch.tensor(trasformed["image"], dtype=torch.float32).permute(
                2, 0, 1
            )
            # breakpoint()
            # image = trasformed["image"]
            _, h, w = image.shape  # (C x H x W)
            keypoints = [(x / w, y / h) for x, y in trasformed["keypoints"]]
            keypoints = torch.tensor(keypoints).type(torch.float)
            numeric_labels = torch.tensor(trasformed["labels"]).type(torch.int64)
            visible = torch.tensor(trasformed['visible']).type(torch.float)

            nparts = len(metadata.keypoint_labels)
            target = np.zeros((nparts, w, h))
            for i in range(nparts):
                pt_x, pt_y = keypoints[i][0]*w,  keypoints[i][1]*h
                target[i] = generate_target(target[i], (pt_x, pt_y))
            target = torch.tensor(target, dtype=torch.float)*100

        if len(numeric_labels) != len(keypoints):
            raise ValueError("Data is broken. missing labels")
        return image, (target,keypoints.view(keypoints.size(0), -1), visible, numeric_labels)

    def plot_sample(self, index, show_all=True):
        image, (target, keypoints, visible, numeric_labels) = self[index]
        label_names = [Keypoints._fields[o-1] for o in numeric_labels]
        if not image.shape[0] in {1, 3}:
            warnings.WarningMessage(
                "Assuming image needs to be permuted into (c x h x w)"
            )
            image = image.permute(2, 0, 1)
        image = image.type(torch.uint8)

        keypoints = []
        for i in range(target.shape[0]):
                keypoints.append((target[i]==torch.max(target[i])).nonzero()[0].tolist()[::-1])
        return draw_keypoints(image, keypoints, label_names, visible, show_all), target


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
        if stage == "fit" or stage is None:
            self.keypoints_train = KeypointsDataset(
                self.data_dir, train=True, transform=self.transform
            )
            self.keypoints_val = KeypointsDataset(
                self.data_dir+'/val', train=True, transform=self.transform
            )
            sample_image, _ = self.keypoints_train[0]
            self.dims = sample_image.shape
        # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:

    def train_dataloader(self):
        return DataLoader(self.keypoints_train, batch_size=BATCH_SIZE, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.keypoints_val, batch_size=BATCH_SIZE, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.keypoints_train, batch_size=BATCH_SIZE)


if __name__ == "__main__":

    ds = KeypointsDataset(data_path=Path("/mnt/vol_b/clean_data/tmp2"), transform=tfs)
    sample, target = ds.plot_sample(0)
    sample.save("tmp2.png")

    target = target*255
    for i in range(target.shape[0]):
        kp = (target[i]==torch.max(target[i])).nonzero()[0]

    # target = target.permute(2, 0, 1)
    for i in range(target.shape[0]):
        PIL.Image.fromarray(target[i].numpy().astype('uint8')).convert("RGB").save(f"{i}_target.png")
    # t.save("mask.png")
    v, _ = torch.max(target,0)

    new_img = PIL.Image.blend(sample, PIL.Image.fromarray(v.numpy().astype('uint8')).convert("RGB"), 0.5).save("target.png")

