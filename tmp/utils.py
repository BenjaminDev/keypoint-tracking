from pathlib import Path
from typing import DefaultDict, List, Tuple, Dict
from PIL import Image
import OpenEXR
import Imath
import PIL
from matplotlib import colors
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
    bounding_boxes: Dict[str, Tuple[int, int, int,int]] #  (xmin, ymin, xmax, ymax)





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
        ymin, xmin, ymax, xmax = prop.bbox
        bounding_boxes[Keypoints._fields[prop.label-1]] = (xmin, ymin, xmax, ymax)
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
        return MetaData(**json.load(fp))

from torchvision import utils
import torch
from torchvision import transforms as T
from torchvision.transforms import ToTensor
image_to_tensor = T.Compose([
    T.ToTensor(),
    T.ConvertImageDtype(torch.uint8)
    ])
def draw_keypoints(image, bounding_boxes:Dict):
    image_t = image_to_tensor(image)
    bboxes = torch.tensor([bbox for bbox in bounding_boxes.values()], dtype=torch.float)
    labels = list(bounding_boxes.keys())
    colors = [COLORS[o.lower()] for o in bounding_boxes.keys()]
    print(bboxes)
    print(labels)
    image_t = utils.draw_bounding_boxes(image_t.type(torch.uint8), bboxes, labels=labels, colors=colors)
    return PIL.Image.fromarray(image_t.permute(1, 2, 0).numpy())


# def test_draw_boxes(self):
#     img = torch.full((3, 100, 100), 255, dtype=torch.uint8)
#     boxes = torch.tensor([[0, 0, 20, 20], [0, 0, 0, 0],
#                             [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)
#     labels = ["a", "b", "c", "d"]
#     colors = ["green", "#FF00FF", (0, 255, 0), "red"]
#     result = utils.draw_bounding_boxes(img, boxes, labels=labels, colors=colors)

#     path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fakedata", "draw_boxes_util.png")
#     if not os.path.exists(path):
#         Image.fromarray(result.permute(1, 2, 0).numpy()).save(path)




# import sys
# import cairo
# import json
# import math
# import matplotlib.colors

# pngFilePath = sys.argv[1]
# jsonFilePath = sys.argv[2]
# pngOutFilePath = sys.argv[3]

# image = cairo.ImageSurface.create_from_png(pngFilePath)


# f = open(jsonFilePath, 'r')
# data = json.load(f)
# f.close()

# context = cairo.Context(image)

COLORS = {
    'left_shoe_front'       : '#F40001',
    'left_shoe_back'        : '#FF5700',
    'left_shoe_inner_side'  : '#FFB300',
    'left_shoe_outer_side'  : '#35D900',
    'left_shoe_top'         : '#00DBB4',
    'left_shoe_ankle'       : '#00C7FF',
    'right_shoe_front'      : '#F40001',
    'right_shoe_back'       : '#FF5700',
    'right_shoe_inner_side' : '#FFB300',
    'right_shoe_outer_side' : '#35D900',
    'right_shoe_top'        : '#00DBB4',
    'right_shoe_ankle'      : '#00C7FF',
    'left_shoe'             : '#F7B500',
    'right_shoe'            : '#FA6400',
    'body'                  : '#FFFFFF',
}

# for (name, (xmin,xmax,ymin,ymax)) in data["bounding_boxes"].items():
#     if name == "body":
#         continue
#     rgb = matplotlib.colors.to_rgb(COLORS[name])
#     context.rectangle(xmin,ymin,xmax-xmin,ymax-ymin)
#     context.set_source_rgba(rgb[0], rgb[1], rgb[2], 1)
#     context.set_line_width(2)
#     context.stroke()

# for (name, (x,y)) in data["keypoints"].items():
#     rgb = matplotlib.colors.to_rgb(COLORS[name])
#     context.arc(x, y, 4, 0, 2*math.pi)
#     context.set_source_rgba(rgb[0], rgb[1], rgb[2], 1)
#     context.fill()

# image.write_to_png(pngOutFilePath)