
import onnxruntime as ort
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
from kornia.geometry import subpix
# from torchvision.datasets import D
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm

from utils import COLORS, Keypoints, MetaData, draw_keypoints, read_meta
from torchvision import transforms as T
from torchvision import utils
from torchvision.transforms import ToTensor

image_to_tensor = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
        # T.ConvertImageDtype(torch.uint8),
    ]
)

ort_session = ort.InferenceSession(
            "/mnt/vol_b/data/real/reference.onnx"
        )
image_path = "/mnt/vol_b/training_data/clean/0020-IMG_1045/frame_111.png"
image = PIL.Image.open(image_path).resize((160, 160))
# image_in = np.array(image_to_tensor(image).unsqueeze(0).permute(0,3,2,1))
y_hat = ort_session.run(None, {"input0": np.expand_dims(np.array(image, dtype=np.float32),0)})[0]
y_hat = y_hat.transpose(0,3,2,1)
y_hat = torch.tensor(np.concatenate([y_hat[:,:6,...], y_hat[:,7:13,...]])).reshape(1,12,160,160)
PIL.Image.fromarray(np.array(y_hat[0,1,:,:],dtype=np.float32)).convert("L").save("map.png")
keypoints=subpix.spatial_soft_argmax.spatial_soft_argmax2d(y_hat, normalized_coordinates=False).squeeze(0)
breakpoint()
y_hat = y_hat.transpose(0,3,1,2).reshape(1,160,160,16)
PIL.Image.fromarray(np.array(y_hat[0,:,:,3],dtype=np.float32)).convert("L").save("map.png")
keypoints=subpix.spatial_soft_argmax.spatial_soft_argmax2d(torch.tensor(y_hat[0,:,:,1]), normalized_coordinates=False).squeeze(0)
label_names = [Keypoints._fields[o-1] for o in range(12)]
h, w = image.size
manual_decoded_keypoints=[]
for kp in keypoints[:12]:
    manual_decoded_keypoints.append(((kp[1].item()/160.0)*h, (kp[0].item()/160.0)*w))
annotated_image = draw_keypoints(image, manual_decoded_keypoints, labels=label_names, show_labels=False, show_all=True)
annotated_image.save(f"tmp.png")

# y_hat = model.predict({"input0:0" : image})["output0"]
#     # breakpoint()
#     # y_hat=torch.tensor(y_hat).squeeze(0).permute(2,1,0)
#     y_hat = y_hat.transpose(0,3,2,1)
#     # y_hat = torch.tensor(np.concatenate([y_hat[:,:6,...], y_hat[:,7:13,...]])).reshape(1,12,160,160)
#     # breakpoint()
#     keypoints=subpix.spatial_soft_argmax.spatial_soft_argmax2d(y_hat, normalized_coordinates=False).squeeze(0)
#     input_image = PIL.Image.open(fn)
#     # for j in range(12): #range(y_hat.shape[0]):
#     #     manual_decoded_keypoints.append((y_hat[j]==torch.max(y_hat[j])).nonzero()[0].tolist())
#     if args.annotate:





# hm=PIL.Image.fromarray(hms[0::,:,1])