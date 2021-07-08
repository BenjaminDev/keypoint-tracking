from pathlib import Path
from typing import DefaultDict, List, Tuple, Dict
from PIL import Image
import OpenEXR
import Imath
import numpy
import OpenEXR
import Imath
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from skimage.measure import label, regionprops
from scipy import ndimage
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

from pydantic import BaseModel

# @dataclass(frozen=False)
# class Keypoints():
#     RIGHT_SHOE_FRONT: int =1
#     RIGHT_SHOE_TOP: int =2
#     RIGHT_SHOE_OUTER_SIDE: int =3
#     RIGHT_SHOE_INNER_SIDE: int =4
#     RIGHT_SHOE_BACK: int =5
#     RIGHT_SHOE_ANKLE: int =6
#     LEFT_SHOE_BACK: int = 7
#     LEFT_SHOE_ANKLE: int =8
#     LEFT_SHOE_OUTER_SIDE: int =9
#     LEFT_SHOE_TOP: int =10
#     LEFT_SHOE_INNER_SIDE: int =11
#     LEFT_SHOE_FRONT: int = 12
#     BODY: int =13
#     RIGHT_SHOE: int =14
#     LEFT_SHOE: int =15
#     lookup: Dict = field(default_factory=lambda:{
#     1: "RIGHT_SHOE_FRONT",
#     2: "RIGHT_SHOE_TOP",
#     3: "RIGHT_SHOE_OUTER_SIDE",
#     4: "RIGHT_SHOE_INNER_SIDE",
#     5: "RIGHT_SHOE_BACK",
#     6: "RIGHT_SHOE_ANKLE",
#     7:"LEFT_SHOE_BACK",
#     8:"LEFT_SHOE_ANKLE",
#     9:"LEFT_SHOE_OUTER_SIDE",
#     10:"LEFT_SHOE_TOP",
#     11:"LEFT_SHOE_INNER_SIDE",
#     12:"LEFT_SHOE_FRONT",
#     13:"BODY",
#     14:"RIGHT_SHOE",
#     15:"LEFT_SHOE"
#     })
#     @staticmethod
#     def labels()->List[str]:
#         # TODO: make it better
#         labels = list(Keypoints.__dataclass_fields__.keys())
#         labels.sort()
#         return labels
#     @staticmethod
#     def parts():
#         return (Keypoints.BODY, Keypoints.LEFT_SHOE, Keypoints.RIGHT_SHOE)
from collections import namedtuple
KeypointInfo = namedtuple("KeypointInfo", ["RIGHT_SHOE_FRONT","RIGHT_SHOE_TOP", "RIGHT_SHOE_OUTER_SIDE", "RIGHT_SHOE_INNER_SIDE", "RIGHT_SHOE_BACK", "RIGHT_SHOE_ANKLE","LEFT_SHOE_BACK","LEFT_SHOE_ANKLE","LEFT_SHOE_OUTER_SIDE","LEFT_SHOE_TOP","LEFT_SHOE_INNER_SIDE","LEFT_SHOE_FRONT","BODY","RIGHT_SHOE","LEFT_SHOE"], defaults=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
Keypoints = KeypointInfo()
class MetaData(BaseModel):
    source: Dict[str, str]
    keypoints: List[Tuple[int, int]]
    keypoint_labels: List[str] = Keypoints._fields
    visible: List[bool] = [False]*len(Keypoints._fields)
    bounding_boxes: Dict[str, Tuple[int, int, int,int]]





# {
#     "source":
#     {
#         "main": "TRAIN_v004_MAIN_.0101.exr",
#         "object_id": "TRAIN_v004_ID_.0101.exr"
#     },
#     "keypoint_labels":["left_shoe_back", "left_shoe_back" ......]
#     "keypoints":
#     [    [1302, 785],
#         [0,0],
#         [1360, 906],
#         [0,0],
#     ],
#     "visible" :[1,0,1,0,1,1,1,1,1........]
#     "bounding_boxes":
#     {
#         "left_shoe": [754, 1009, 1205, 1384],
#         "right_shoe": [1093, 1343, 1201, 1385],
#         "body": [713, 1315, 0, 1384]
#     }
# }

def exr_channel_to_np(exr_file_discriptor: OpenEXR.InputFile, size:Tuple[int, int, int], channel_name:str)->List[np.array]:

    channel_str = exr_file_discriptor.channel(channel_name, FLOAT)

    channel = np.frombuffer(channel_str, dtype = np.float32).reshape(size[1],-1)

    return channel
# Old not correct
# def EncodeToSRGB(v):
#     return(np.where(v<=0.0031308,v * 12.92, 1.055*(v**(1.0/2.4)) - 0.055))


def EncodeToSRGB(v):
    return np.where(v <= 0.0031308, (v * 12.92) * 255, (1.055 * (v ** (1.0 / 2.4)) - 0.055) * 255)


def exr2rgb(exr_path):
    exr_file_discriptor = OpenEXR.InputFile(exr_path)
    dw = exr_file_discriptor.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    channels = []
    channel_names = ['R','G','B']
    for channel_name in channel_names:
        channel  = exr_channel_to_np(exr_file_discriptor, size, channel_name)
        channels.append(EncodeToSRGB(channel))
    return np.dstack(channels), exr_file_discriptor.header()
from collections import defaultdict

def mask_to_bounding_boxes(mask):
    bounding_boxes = defaultdict(tuple)
    props = [o for o in regionprops(mask.astype('int')) if o.label in Keypoints]
    for prop in props:
        bounding_boxes[Keypoints._fields[prop.label-1]] = prop.bbox
    return bounding_boxes

def mask_to_keypoints(mask)->Tuple[List[Tuple[int, int]], List[bool]]:
    keypoints = []
    visible = []
    for kp_number in Keypoints:
        kp = ndimage.measurements.center_of_mass((mask == kp_number).astype('int'))
        visible.append(not any([np.isnan(o) for o in kp]))
        if any([np.isnan(o) for o in kp]):
            kp=(0,0)
        keypoints.append(kp)


    return [(int(o[0]), int(o[1])) for o in keypoints], visible

import json
def read_meta(file_path:Path):
    with open(file_path, mode='r') as fp:
        MetaData(json.load(fp))