from typing import List, Tuple, Dict
from PIL import Image
import OpenEXR
import Imath
import numpy
import OpenEXR
import Imath
from PIL import Image
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

from pydantic import BaseModel

@dataclass(frozen=False)
class Keypoints():
    LEFT_SHOE_FRONT: int = 12
    LEFT_SHOE_BACK: int = 7
    LEFT_SHOE_INNER_SIDE: int =11
    LEFT_SHOE_OUTER_SIDE: int =9
    LEFT_SHOE_TOP: int =10
    LEFT_SHOE_ANKLE: int =8
    RIGHT_SHOE_FRONT: int =1
    RIGHT_SHOE_BACK: int =5
    RIGHT_SHOE_INNER_SIDE: int =4
    RIGHT_SHOE_OUTER_SIDE: int =3
    RIGHT_SHOE_TOP: int =2
    RIGHT_SHOE_ANKLE: int =6
    LEFT_SHOE: int =15
    RIGHT_SHOE: int =14
    BODY: int =13

    @staticmethod
    def labels()->List[str]:
        # TODO: make it better
        labels = list(Keypoints.__dataclass_fields__.keys())
        labels.sort()
        return labels

class MetaData(BaseModel):
    source: Dict[str, str]
    keypoints: List[Tuple[int, int]]
    keypoint_labels: List[str] = Keypoints.labels()
    visible: List[bool] = [False]*len(Keypoints.labels())
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






def encode_to_srgb(x):
    """
    https://en.wikipedia.org/wiki/SRGB
    """
    a = 0.055
    return numpy.where(x <= 0.0031308,
                       x * 12.92,
                       (1 + a) * pow(x, 1 / 2.4) - a)

def exr_to_srgb(exrfile):
    array, header = exr_to_array(exrfile)
    result = encode_to_srgb(array) * 255.
    present_channels = ["R", "G", "B", "A"][:result.shape[2]]
    channels = "".join(present_channels)
    return Image.fromarray(result.astype('uint8'), channels), header

def exr_to_array(exrfile):
    file = OpenEXR.InputFile(exrfile)
    dw = file.header()['dataWindow']

    channels = file.header()['channels'].keys()
    channels_list = list()
    for c in ('R', 'G', 'B', 'A'):
        if c in channels:
            channels_list.append(c)

    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    color_channels = file.channels(channels_list, FLOAT)
    channels_tuple = [numpy.frombuffer(channel, dtype='f') for channel in color_channels]
    res = numpy.dstack(channels_tuple)
    return res.reshape(size + (len(channels_tuple),)), file.header()
from pathlib import Path
def exr_channel_to_np(exr_file_discriptor: OpenEXR.InputFile, size:Tuple[int, int, int], channel:str)->List[np.array]:
    '''
    See:
    https://excamera.com/articles/26/doc/intro.html
    http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    '''

    channel_str = exr_file_discriptor.channel(channel, FLOAT)

    channel = np.frombuffer(channel_str, dtype = np.float32).reshape(size[1],-1)

    return channel

def EncodeToSRGB(v):
    return(np.where(v<=0.0031308,v * 12.92, 1.055*(v**(1.0/2.4)) - 0.055))

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

def mask_to_meta(mask):
    keypoints = []
    for kp_name in Keypoints.labels():
        keypoints.append(ndimage.measurements.center_of_mass((mask == Keypoints.LEFT_SHOE).astype('int')))
        

    meta = MetaData()