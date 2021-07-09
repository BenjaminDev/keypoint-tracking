import json
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import Imath
import numpy as np
import OpenEXR
import PIL
import torch
from matplotlib import colors
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage
from skimage.measure import regionprops
from torchvision import transforms as T
from torchvision import utils
from torchvision.transforms import ToTensor

image_to_tensor = T.Compose([T.ToTensor(), T.ConvertImageDtype(torch.uint8)])
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

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
        "BODY",
        "RIGHT_SHOE",
        "LEFT_SHOE",
    ],
    defaults=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
)
Keypoints = KeypointInfo()


class MetaData(BaseModel):
    source: Dict[str, str]
    keypoints: List[Tuple[int, int]]
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
    props = [o for o in regionprops(mask.astype("int")) if o.label in Keypoints]
    for prop in props:
        ymin, xmin, ymax, xmax = prop.bbox
        bounding_boxes[Keypoints._fields[prop.label - 1]] = (xmin, ymin, xmax, ymax)
    return bounding_boxes


def mask_to_keypoints(mask) -> Tuple[List[Tuple[int, int]], List[bool]]:
    keypoints = []
    visible = []
    present_keypoints_in_mask = np.unique(mask).astype("int")
    for kp_number in Keypoints:
        if not kp_number in present_keypoints_in_mask:
            continue
        kp = ndimage.measurements.center_of_mass((mask == kp_number))
        visible.append(not any([np.isnan(o) for o in kp]))
        if any([np.isnan(o) for o in kp]):
            kp = (0, 0)
        keypoints.append(kp)

    return [(int(o[0]), int(o[1])) for o in keypoints], visible


def read_meta(file_path: Path):
    with open(file_path, mode="r") as fp:
        return MetaData(**json.load(fp))


def draw_bounding_box(image, bounding_boxes: Dict):
    image_t = image_to_tensor(image)
    bboxes = torch.tensor([bbox for bbox in bounding_boxes.values()], dtype=torch.float)
    labels = list(bounding_boxes.keys())
    colors = [COLORS[o.lower()] for o in bounding_boxes.keys()]
    image_t = utils.draw_bounding_boxes(
        image_t.type(torch.uint8), bboxes, labels=labels, colors=colors
    )
    return PIL.Image.fromarray(image_t.permute(1, 2, 0).numpy())


def draw_keypoints(image, keypoints, labels, visible, show_all=False):
    image_t = image_to_tensor(image)
    d = image_t.size()[1] // 300
    bboxes = [(xy[1] - d, xy[0] - d, xy[1] + d, xy[0] + d) for xy in keypoints]
    colors = [COLORS[o.lower()] for o in labels]
    if not show_all:
        bboxes = [o for i, o in enumerate(bboxes) if visible[i]]
        colors = [o for i, o in enumerate(colors) if visible[i]]
    bboxes = torch.tensor(bboxes, dtype=torch.float)
    image_t = utils.draw_bounding_boxes(
        image_t, bboxes, labels=labels, colors=colors, fill=True
    )
    return PIL.Image.fromarray(image_t.permute(1, 2, 0).numpy())
