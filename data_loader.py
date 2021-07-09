import argparse
import numpy as np
from PIL import Image
from pydantic.types import Json
from utils import (
    exr2rgb,
    Keypoints,
    MetaData,
    exr_channel_to_np,
    mask_to_keypoints,
    mask_to_bounding_boxes,
    read_meta,
)
from pathlib import Path
from scipy import ndimage
import OpenEXR
