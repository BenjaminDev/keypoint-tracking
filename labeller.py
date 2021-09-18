from datetime import time
import coremltools as ct
from pathlib import Path
from coremltools import models
from tqdm import tqdm
from utils import draw_keypoints, Keypoints, image_to_tensor
import PIL
from torch import nn
import torch
from PIL import Image
from kornia.geometry import subpix
import os
from torchvision.utils import draw_bounding_boxes
import argparse
from matplotlib import cm
import numpy as np
from pathlib import Path
from random import sample
from utils import (MetaData, draw_bounding_box, draw_keypoints, exr2rgb,
                   exr_channel_to_np, mask_to_bounding_boxes,
                   mask_to_keypoints, read_meta)
import shutil
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show mlcore model predictions"
    )
    parser.add_argument(
        "--mpath",
        dest="model_path",
        type=Path,
        help="Path to the mlcore model",
    )
    parser.add_argument(
        "--ddir",
        dest="data_dir",
        type=Path,
        help="Path to the .png files",
    )


    args = parser.parse_args()
    # ml_model_path = Path(args.mpath)

    spec = ct.utils.load_spec(os.fsdecode(args.model_path))
    model = ct.models.MLModel(spec)

    image_files = list(args.data_dir.glob("*.png"))
    (args.data_dir/"inspect").mkdir(exist_ok=True)

    label_names = [Keypoints._fields[o-1] for o in range(12)]
    i=0
    for fn in tqdm(image_files):
        i+=1
        # breakpoint()
        image = PIL.Image.open(fn).resize((160, 160))
        y_hat = model.predict({"input0:0" : image})["output0"]
        # breakpoint()
        # y_hat=torch.tensor(y_hat).squeeze(0).permute(2,1,0)
        y_hat = y_hat.transpose(0,3,2,1)
        y_hat = torch.tensor(np.concatenate([y_hat[:,:6,...], y_hat[:,7:13,...]])).reshape(1,12,160,160)
        # breakpoint()
        keypoints=subpix.spatial_soft_argmax.spatial_soft_argmax2d(y_hat, normalized_coordinates=False).squeeze(0)
        manual_decoded_keypoints=[]
        input_image = PIL.Image.open(fn)
        h, w = input_image.size
        for kp in keypoints:
            manual_decoded_keypoints.append(((kp[1].item()/160.0)*h, (kp[0].item()/160.0)*w))
        # for j in range(12): #range(y_hat.shape[0]):
        #     manual_decoded_keypoints.append((y_hat[j]==torch.max(y_hat[j])).nonzero()[0].tolist())
        annotated_image = draw_keypoints(input_image, manual_decoded_keypoints, labels=label_names, show_labels=False, show_all=True)
        annotated_image.save(f"{os.fsdecode(args.data_dir/'inspect')}/{i}_tmp.png")
        source = {
            "main": "",
            "object_id": f"",
            "image": f"{fn}",
        }

        meta = MetaData(
            source=source,
            keypoints=manual_decoded_keypoints,
            visible=[True]*len(keypoints),
            bounding_boxes=[],
        )

        # Write meta data to destination directory
        with open(f"{os.fsdecode(args.data_dir)}/{fn.stem}.json", mode="w") as fp:
            fp.write(meta.json(indent=4))
    from subprocess import check_call
    # ffmpeg -framerate 25 -i 'TRAIN_v004_.0%3d_out.png' -c:v libx264 -pix_fmt yuv420p out.mp4
    # out = check_call(["ffmpeg -framerate 25 -i  /Volumes/external/wf/data/ARSession/out_yolov5/%d_tmp.png -c:v libx264 -pix_fmt yuv420p /Volumes/external/wf/data/ARSession/out_yolov5/out_new.mp4"], shell=True)