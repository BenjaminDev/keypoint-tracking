import os
import warnings
from pathlib import Path
from random import randint, random
from typing import List, Tuple

import albumentations as A
import coremltools as ct
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import PIL
import pytorch_lightning as pl
import scipy.misc
import torch
import torch.nn.functional as F
from albumentations.augmentations.geometric.functional import keypoint_affine
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.geometric.transforms import Perspective
from albumentations.augmentations.transforms import ChannelDropout, RGBShift
from albumentations.core.composition import OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.metrics.functional import accuracy
from scipy import ndimage
from scipy.ndimage.morphology import grey_dilation
from torch import nn
from torch._C import dtype

# from torchvision.datasets import D
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm

from utils import COLORS, Keypoints, MetaData, draw_keypoints, read_meta


class HeatmapGenerator:
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res / 64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros(
            (self.num_joints, self.output_res, self.output_res), dtype=np.float32
        )
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue

                    ul = (
                        int(np.round(x - 3 * sigma - 1)),
                        int(np.round(y - 3 * sigma - 1)),
                    )
                    br = (
                        int(np.round(x + 3 * sigma + 2)),
                        int(np.round(y + 3 * sigma + 2)),
                    )

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d]
                    )
        M = np.zeros_like(hms)
        # mask = np.clip(hms.max().astype(np.int),0,63)
        for i in range(len(M)):
            M[i] = grey_dilation(hms[i], size=(3, 3))
        # breakpoint()
        M = np.where(M >= 0.5, 1, 0)
        return hms, M


import numpy as np
import torch
import torch.nn.functional as F


class ModelLabeller:
    # TODO: Get ref model running on linux.
    def __init__(self, output_res: int, num_joints: int, model_path: Path) -> None:
        self.output_res = output_res
        self.num_joints = num_joints
        self.ort_session = ort.InferenceSession(
            "/mnt/vol_c/code/sketchpad/coreml_models/reference/model.onnx"
        )

    def __call__(self, image_path):
        image = PIL.Image.open(image_path).resize((160, 160))
        # image.save("/tmp/input.png")
        # y_hat = self.model.predict({"input0:0" : image})["output0"]
        img = np.array(image).astype(np.float32)  # .transpose(0,3,1,2)
        # np.expand_dims(np.array(image).astype(np.float32).transpose(1,2,0),-1)
        breakpoint()
        MEAN = 255 * torch.tensor([0.485, 0.456, 0.406])
        STD = 255 * torch.tensor([0.229, 0.224, 0.225])
        img = np.array(
            torch.tensor(img.transpose(2, 1, 0)) * STD[:, None, None]
            + MEAN[:, None, None]
        )
        img = np.expand_dims(img, 0).transpose(0, 2, 3, 1)
        y_hat = self.ort_session.run(None, {"input0:0": img})[
            1
        ]  # np.random.randn(10,  160, 160,3).astype(np.float32)})
        y_hat = y_hat.transpose(3, 1, 2, 0).squeeze(-1)
        # y_hat = np.array(torch.sigmoid(torch.tensor(y_hat)/y_hat.max()))*255
        y_hat = np.concatenate(
            [y_hat[:6, ...], y_hat[7:13, ...]]
        )  # using only 6 points per foot
        hms = np.zeros((12, 480, 480), dtype=float)
        for i, hm in enumerate(y_hat):
            img = PIL.Image.fromarray(hm).convert("L").resize((480, 480))
            # img.save(f"/tmp/check_{i}.png")
            hms[i] = np.array(img)
        M = np.zeros_like(hms)
        for i in range(len(M)):
            M[i] = grey_dilation(hms[i], size=(3, 3))

        M = np.where(M >= 0.5, 1, 0)
        return hms, M


class KeypointsDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        image_size: Tuple[int, int],
        target_scale: float = 800.0,
        sigma: float = 3,
        train=True,
        transform=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.image_files = sorted(list(self.data_path.glob("*.png")))
        self.label_files = sorted(list(self.data_path.glob("*.json")))
        self.target_scale = target_scale  # TODO: look at a 'robust' loss so we don't need this to force the issue.
        self.sigma = sigma
        self.custom_transforms = True
        self.rotater = A.Compose(
            [A.SafeRotate(limit=(-180, 180), p=0.8, border_mode=cv2.BORDER_CONSTANT),],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["labels", "visible"], remove_invisible=False
            ),
        )
        assert image_size[0] == image_size[1], "Only square images are supported!"
        self.image_size = image_size
        self.heatmapper = HeatmapGenerator(image_size[0], 12)
        # self.heatmapper=ModelLabeller(480,14,model_path=Path("/mnt/vol_c/code/sketchpad/coreml_models/reference/model3.mlmodel"))
        if not any(
            [l.stem == i.stem for l, i in zip(self.label_files, self.image_files)]
        ):
            for l, i in zip(self.label_files, self.image_files):
                if l.stem != i.stem:
                    print(f"{l.stem} {i.stem}")
            breakpoint()
            raise ValueError("Image files and label files mismatch")
        self.category_names = Keypoints._fields
        self.transform = transform
        self.train = train

    def crop_keep_points(self, image, keypoints):
        if random() < 0.2:
            return image, keypoints
        h, w, _ = image.shape
        total_border = randint(10, max(w, h))
        x_ratio = random()
        y_ratio = random()
        x_max = min(
            max(keypoints, key=lambda kp: kp[0])[0] + int(total_border * x_ratio), w
        )
        x_min = max(
            min(keypoints, key=lambda kp: kp[0])[0] - int(total_border * (1 - x_ratio)),
            0,
        )
        y_max = min(
            max(keypoints, key=lambda kp: kp[1])[1] + int(total_border * y_ratio), h
        )
        y_min = max(
            min(keypoints, key=lambda kp: kp[1])[1] - int(total_border * (1 - y_ratio)),
            0,
        )
        keypoints = [
            (max(0, o[0] - x_min - 1), max(0, o[1] - y_min - 1)) for o in keypoints
        ]
        return image[y_min:y_max, x_min:x_max, :], keypoints

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
            numeric_labels = [
                Keypoints._fields_defaults[o] for o in metadata.keypoint_labels
            ]

        if self.custom_transforms:
            # rotated_image_and_points = self.rotater(image=image, keypoints=keypoints, labels=numeric_labels, visible=visible)
            # image = rotated_image_and_points["image"]
            # keypoints = [(int(x),int(y)) for x,y in rotated_image_and_points["keypoints"]]
            image, keypoints = self.crop_keep_points(image, keypoints)
        if self.transform:
            trasformed = self.transform(
                image=image, keypoints=keypoints, labels=numeric_labels, visible=visible
            )
            image = torch.tensor(trasformed["image"], dtype=torch.float32).permute(
                2, 0, 1
            )
            _, h, w = image.shape  # (C x H x W)
            self.sigma = (
                float(h) / 500.0
            )  # TODO: See how to change this and ideally make it smaller as the model trains. self.trainer.current_epoch should be it!
            keypoints = [(x / w, y / h) for x, y in trasformed["keypoints"]]

            keypoints = torch.tensor(keypoints).type(torch.float)
            numeric_labels = torch.tensor(trasformed["labels"]).type(torch.int64)
            visible = torch.tensor(trasformed["visible"]).type(torch.float)

            nparts = len(metadata.keypoint_labels)
            target = np.zeros((nparts, w, h))
            xs_ys = []
            for i in range(nparts):
                pt_x, pt_y = keypoints[i][0] * w, keypoints[i][1] * h
                xs_ys.append([pt_x, pt_y, 1])
            if isinstance(self.heatmapper, ModelLabeller):
                target, M = self.heatmapper(self.image_files[index])
            else:
                target, M = self.heatmapper([xs_ys])
            target = torch.tensor(target, dtype=torch.float)
            M = torch.tensor(M, dtype=torch.float)

        if len(numeric_labels) != len(keypoints):
            raise ValueError("Data is broken. missing labels")
        return (
            image,
            (
                target,
                M,
                keypoints.view(keypoints.size(0), -1),
                visible,
                numeric_labels,
                self.image_files[index].stem,
            ),
        )

    def plot_sample(self, index, show_all=True, denormalize=True):
        image, (target, M, keypoints, visible, numeric_labels, source_filename) = self[
            index
        ]
        label_names = [Keypoints._fields[o - 1] for o in numeric_labels]
        if not image.shape[0] in {1, 3}:
            warnings.WarningMessage(
                "Assuming image needs to be permuted into (c x h x w)"
            )
            image = image.permute(2, 0, 1)
        if denormalize:
            MEAN = 255 * torch.tensor([0.485, 0.456, 0.406])
            STD = 255 * torch.tensor([0.229, 0.224, 0.225])

            image = image * STD[:, None, None] + MEAN[:, None, None]
        image = image.type(torch.uint8)
        keypoints = []
        for i in range(target.shape[0]):
            keypoints.append(
                (target[i] == torch.max(target[i])).nonzero()[0].tolist()[::-1]
            )
        return (
            draw_keypoints(
                image,
                keypoints,
                label_names,
                show_labels=True,
                short_names=True,
                visible=visible,
                show_all=show_all,
            ),
            target,
            M,
        )


class KeypointsDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dirs: List[str], input_size: Tuple[int, int], batch_size: int
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_transforms = A.Compose(
            [
                # A.RandomCrop(*input_size),
                A.SafeRotate(limit=(-180, 180), p=0.8, border_mode=cv2.BORDER_CONSTANT),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
                A.Resize(*input_size),
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.ChannelShuffle(p=0.2),
                        A.RGBShift(p=0.2),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.4,
                ),
                A.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]),
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["labels", "visible"], remove_invisible=False
            ),
        )
        self.test_transforms = A.Compose(
            [
                A.Resize(*input_size),
                A.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]),
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["labels", "visible"], remove_invisible=False
            ),
        )

    # def prepare_data(self):
    #     # download

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.keypoints_train = ConcatDataset(
                [
                    KeypointsDataset(
                        data_dir,
                        image_size=self.input_size,
                        train=True,
                        transform=self.train_transforms,
                    )
                    for data_dir in self.data_dirs
                ]
            )
            self.keypoints_val = ConcatDataset(
                [
                    KeypointsDataset(
                        data_dir + "/val",
                        image_size=self.input_size,
                        train=True,
                        transform=self.train_transforms,
                    )
                    for data_dir in self.data_dirs
                ]
            )
            # sample_image, _ = self.keypoints_train[0]
            # self.dims = sample_image.shape
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.keypoints_test = ConcatDataset(
                [
                    KeypointsDataset(
                        data_dir + "/test",
                        image_size=self.input_size,
                        train=True,
                        transform=self.test_transforms,
                    )
                    for data_dir in self.data_dirs
                ]
            )

    def train_dataloader(self):
        return DataLoader(
            self.keypoints_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.keypoints_val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.keypoints_test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False,
        )


if __name__ == "__main__":
    data_dirs = ["/mnt/vol_b/training_data/clean/0014-IMG_1037"]
    input_size = (160, 160)
    dm = KeypointsDataModule(data_dirs, input_size, batch_size=1)
    # ds = KeypointsDataset(data_path=Path("/mnt/vol_b/clean_data/tmp2"))
    dm.setup("fit")
    dl = dm.train_dataloader()
    for o, _ in dl:
        pass

    breakpoint()
    sample, target, M = dl.dataset.datasets[0].plot_sample(10)
    sample.save("/tmp/tmp2.png")
    breakpoint()
    target = target * 255
    M = M * 255
    for i in range(target.shape[0]):
        kp = (target[i] == torch.max(target[i])).nonzero()[0]

    # target = target.permute(2, 0, 1)
    for i in range(target.shape[0]):
        PIL.Image.fromarray(target[i].numpy().astype("uint8")).convert("RGB").save(
            f"/tmp/{i}_target.png"
        )
    for i in range(M.shape[0]):
        PIL.Image.fromarray(M[i].numpy().astype("uint8")).convert("L").save(
            f"/tmp/{i}_M.png"
        )
    # t.save("mask.png")
    v, _ = torch.max(target, 0)

    new_img = PIL.Image.blend(
        sample, PIL.Image.fromarray(v.numpy().astype("uint8")).convert("RGB"), 0.5
    ).save("/tmp/target.png")
