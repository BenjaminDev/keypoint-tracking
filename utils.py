import json
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union

# import Imath
import numpy as np
import OpenEXR
import PIL
import torch
from matplotlib import colors
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage
from skimage.measure import regionprops
from torch.autograd.grad_mode import F
from torchvision import transforms as T
from torchvision import utils
from torchvision.transforms import ToTensor

image_to_tensor = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.uint8)])
# FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
FLOAT =float

COLORS = {
    "left_shoe_front": "#F40001",
    "left_shoe_back": "#FF5700",
    "left_shoe_inner_side": "#FFB300",
    "left_shoe_outer_side": "#35D900",
    "left_shoe_top": "#00DBB4",
    "left_shoe_ankle": "#00C7FF",
    "right_shoe_front": "#F40001",
    "right_shoe_back": "#FF5700",
    "right_shoe_inner_side": "#FFB300",
    "right_shoe_outer_side": "#35D900",
    "right_shoe_top": "#00DBB4",
    "right_shoe_ankle": "#00C7FF",
    "left_shoe": "#F7B500",
    "right_shoe": "#FA6400",
    "body": "#FFFFFF",
}
sn = {
    "RIGHT_SHOE_FRONT" : "RF",
        "RIGHT_SHOE_TOP" : "RT",
        "RIGHT_SHOE_OUTER_SIDE" : "RO",
        "RIGHT_SHOE_INNER_SIDE" : "RI",
        "RIGHT_SHOE_BACK": "RB",
        "RIGHT_SHOE_ANKLE" : "RA",
        "LEFT_SHOE_BACK" : "LB",
        "LEFT_SHOE_ANKLE" : "LA",
        "LEFT_SHOE_OUTER_SIDE" : "LO",
        "LEFT_SHOE_TOP": "LT",
        "LEFT_SHOE_INNER_SIDE" : "LI",
        "LEFT_SHOE_FRONT" : "LF",
}
KeypointInfo = namedtuple(
    "KeypointInfo",
    [
        "RIGHT_SHOE_FRONT",
        "RIGHT_SHOE_TOP",
        "RIGHT_SHOE_OUTER_SIDE",
        "RIGHT_SHOE_INNER_SIDE",
        "RIGHT_SHOE_BACK",
        "RIGHT_SHOE_ANKLE",
        "LEFT_SHOE_BACK",
        "LEFT_SHOE_ANKLE",
        "LEFT_SHOE_OUTER_SIDE",
        "LEFT_SHOE_TOP",
        "LEFT_SHOE_INNER_SIDE",
        "LEFT_SHOE_FRONT",
        # "BODY",
        # "RIGHT_SHOE",
        # "LEFT_SHOE",
    ],
    defaults=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # , 13, 14, 15],
)

RegionInfo = namedtuple(
    "RegionInfo",
    [
        "BODY",
        "RIGHT_SHOE",
        "LEFT_SHOE",
    ],
    defaults=[13, 14, 15],
)
# Common Types
Keypoints = KeypointInfo()
Regions = RegionInfo()

PImage = PIL.Image.Image
TImage = torch.Tensor
Points = List[Tuple[int, int]]
TPoints = torch.Tensor

class MetaData(BaseModel):
    source: Dict[str, str]
    keypoints: List[Tuple[int, int]]  # x,y
    keypoint_labels: List[str] = Keypoints._fields
    visible: List[bool] = [False] * len(Keypoints._fields)
    bounding_boxes: Dict[str, Tuple[int, int, int, int]]  #  (xmin, ymin, xmax, ymax)


def exr_channel_to_np(
    exr_file_discriptor: OpenEXR.InputFile,
    size: Tuple[int, int, int],
    channel_name: str,
) -> List[np.array]:
    channel_str = exr_file_discriptor.channel(channel_name, FLOAT)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)
    return channel


def EncodeToSRGB(v):
    return np.where(
        v <= 0.0031308, (v * 12.92) * 255, (1.055 * (v ** (1.0 / 2.4)) - 0.055) * 255
    )


def exr2rgb(exr_file_discriptor, size):
    channels = []
    channel_names = ["R", "G", "B"]
    for channel_name in channel_names:
        channel = exr_channel_to_np(exr_file_discriptor, size, channel_name)
        channels.append(EncodeToSRGB(channel))
    return np.dstack(channels), exr_file_discriptor.header()


def mask_to_bounding_boxes(mask):
    bounding_boxes = defaultdict(tuple)
    props = [o for o in regionprops(mask.astype("int")) if o.label in Regions]
    region_min_value = min(Regions._fields_defaults.values())
    for prop in props:
        ymin, xmin, ymax, xmax = prop.bbox
        bounding_boxes[Regions._fields[prop.label - region_min_value]] = (
            xmin,
            ymin,
            xmax,
            ymax,
        )
    return bounding_boxes


def mask_to_keypoints(mask) -> Tuple[List[Tuple[int, int]], List[bool]]:
    keypoints = []
    visible = []
    present_keypoints_in_mask = np.unique(mask).astype("int")
    for kp_number in Keypoints:
        if not kp_number in present_keypoints_in_mask:
            keypoints.append((0, 0))
            visible.append(False)
            continue
        y, x = ndimage.measurements.center_of_mass((mask == kp_number))
        kp = (x, y)
        visible.append(not any([np.isnan(o) for o in kp]))
        if any([np.isnan(o) for o in kp]):
            kp = (0, 0)
        keypoints.append(kp)

    return [(int(o[0]), int(o[1])) for o in keypoints], visible


def read_meta(file_path: Path):
    with open(file_path, mode="r") as fp:
        return MetaData(**json.load(fp))


def draw_bounding_box(image, bounding_boxes: Dict):
    if not isinstance(image, torch.Tensor):
        image = image_to_tensor(image)
    bboxes = torch.tensor([bbox for bbox in bounding_boxes.values()], dtype=torch.float)
    labels = list(bounding_boxes.keys())
    colors = [COLORS[o.lower()] for o in bounding_boxes.keys()]
    image = utils.draw_bounding_boxes(
        image.type(torch.uint8), bboxes, labels=labels, colors=colors
    )
    return PIL.Image.fromarray(image.permute(1, 2, 0).numpy())




def draw_keypoints(
    image: Union[PImage, TImage],
    keypoints: Union[Points, TPoints],
    labels: List[str],
    show_labels: bool = True,
    short_names:bool = True,
    visible: Union[List, torch.Tensor]=None,
    show_all: bool = False,
):
    if not isinstance(image, torch.Tensor):
        image = image_to_tensor(image)
    image = image.cpu().type(torch.uint8)
    if isinstance(keypoints, TPoints):
        keypoints = keypoints.cpu()
        keypoints = keypoints.reshape(-1, 2)
    if keypoints[0][0] < 1:
        # We have normalised coordinates. Convert back to image
        _, h, w = image.shape  # (C x H x W)
        keypoints = [(int(x * w), int(y * h)) for x, y in keypoints]
    if not visible is None and isinstance(visible, torch.Tensor):
        visible = visible.cpu()
    d = image.size()[1] // 300
    bboxes = [(xy[0] - d, xy[1] - d, xy[0] + d, xy[1] + d) for xy in keypoints]
    colors = None
    if not labels is  None:
        colors = [COLORS[o.lower()] for o in labels]

    if not show_all:
        bboxes = [o for i, o in enumerate(bboxes) if visible[i]]
        colors = [o for i, o in enumerate(colors) if visible[i]]
    bboxes = torch.tensor(bboxes, dtype=torch.float)
    if not show_labels:
        labels = None
    if short_names and show_labels:
        labels = [sn[o] for o in labels]
    image = utils.draw_bounding_boxes(
        image, bboxes, labels=labels, colors=colors, fill=True
    )
    return PIL.Image.fromarray(image.permute(1, 2, 0).numpy())

import cv2
import os
def load_image(image_path:Path, size:Tuple[int, int]):
    image = PIL.Image.open(image_path).resize(size)
    # image = cv2.imread(os.fsdecode(image_path))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_to_tensor(image).unsqueeze(0).type(torch.float)