
import argparse
# import json
import os
# import warnings
# from datetime import datetime
from pathlib import Path

# import onnxruntime
# import coremltools
# import coremltools as ct
# import cv2
# import numpy as np
# import PIL
import torch
# from coremltools import models
# from coremltools.models import pipeline
# from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

from train import Keypointdetector
# from utils import Keypoints, draw_keypoints
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processes exr file from Wearfits and generates a folder of images and json meta data files."
    )
    parser.add_argument(
        "--mpath",
        dest="model_path",
        type=Path,
        help="Path to the pytorch lightning model .ckpt or a coreml .mlmodel. See --trace",
    )
    parser.add_argument(
        "--mdir-dst",
        dest="model_dir_dst",
        type=Path,
        help="Path to the pytorch lightning model .ckpt or a coreml .mlmodel. See --trace",
    )
    parser.add_argument(
        "--im-width",
        dest="input_image_width",
        type=int,
        help="width of the image to process",
        default=160
    )
    parser.add_argument(
        "--im-height",
        dest="input_image_height",
        type=int,
        help="height of the image to process",
        default=160
    )
    parser.add_argument(
       '--trace', action='store_true',
        dest="trace",
        help="Specify if tracing is needed. Otherwise expects --mdir to point to a .mlmodel",
    )
    parser.add_argument('--no-trace',
                    action='store_false',
                    dest='trace')
    args = parser.parse_args()
input_size = (args.input_image_width, args.input_image_height)
args.model_dir_dst.mkdir(parents=True, exist_ok=True)
# test_image_path = Path("test_data/frame_001.jpg")
# assert test_image_path.is_file(), "cannot find test image file."
manual_decoded_keypoints = []
if args.trace:
    if args.model_path.suffix != ".ckpt":
        raise ValueError("--trace is enabled and mdir is not pointing to a .ckpt file.")
    traceable_model = Keypointdetector.load_from_checkpoint(os.fsdecode(args.model_path), output_image_size=input_size, inferencing=True).eval()
    input_batch = torch.rand(1, 3, *input_size)
    trace = torch.jit.trace(traceable_model, input_batch)
    torch.jit.save(trace, os.fsdecode(args.model_dir_dst/"traced_model.pt"))