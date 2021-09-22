import argparse
import os
import shutil
from datetime import time
from pathlib import Path
from random import sample

import coremltools as ct
import numpy as np
import PIL
import torch
from coremltools import models
from matplotlib import cm
from PIL import Image
from torch import nn
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from utils import Keypoints, draw_keypoints, image_to_tensor

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
    parser.add_argument(
        "--mtype",
        dest="model_type",
        type=str,
        choices=("yolov5", "heatmap", "heatmap-only", "ref"),
        help="type of model - todo: infer from outputs",
    )

    args = parser.parse_args()
    ml_model_path = Path("outputs/37uxpjcy/Pipeline.mlmodel")

    spec = ct.utils.load_spec(os.fsdecode(args.model_path))
    model = ct.models.MLModel(spec)

    image_files = list(args.data_dir.glob("*.png"))


    if args.model_type == "heatmap":
        label_names = [Keypoints._fields[o-1] for o in range(12)]
        i=0
        for fn in tqdm(image_files):
            i+=1
            # breakpoint()
            image = PIL.Image.open(fn).resize((480, 480))
            outputs = model.predict({"input" : image})
            kps=outputs["coordinates"]
            annotated_image = draw_keypoints(image, kps, labels=label_names, show_labels=True, show_all=True)
            annotated_image.save(f"/Volumes/external/wf/data/ARSession/out/{i}_tmp.png")
    elif args.model_type == "heatmap-only":
        label_names = [Keypoints._fields[o-1] for o in range(12)]
        i=0
        for fn in tqdm(image_files):
            i+=1
            # breakpoint()
            image = PIL.Image.open(fn).resize((480, 480))
            heatmaps = model.predict({"input" : image})["4260"]
            m = nn.Sigmoid()
            with torch.no_grad():
                heatmaps = m(torch.tensor(heatmaps)).numpy()
            heatmap = Image.fromarray(np.uint8(cm.viridis(heatmaps.max(axis=0)[0]) * 255))
            # breakpoint()
            # kps=outputs["coordinates"]
            # annotated_image = draw_keypoints(image, kps, labels=label_names, show_labels=True, show_all=True)
            heatmap.save(f"/Volumes/external/wf/data/ARSession/out/{i}_tmp_heat.png")
            # annotated_image.save(f"/Volumes/external/wf/data/ARSession/out/{i}_tmp.png")
    elif args.model_type == "ref":
        label_names = [Keypoints._fields[o-1] for o in range(12)]
        i=0
        for fn in tqdm(image_files):
            i+=1
            # breakpoint()
            image = PIL.Image.open(fn).resize((160, 160))
            y_hat = model.predict({"input0:0" : image})["output0"]
            # breakpoint()
            y_hat=torch.tensor(y_hat).squeeze(0).permute(2,1,0)
            m = nn.Sigmoid()
            with torch.no_grad():
                heatmaps = m(y_hat)
            heatmap = Image.fromarray(np.uint8(cm.viridis(y_hat[:12].max(axis=0)[0]) * 255))
            heatmap.save(f"/Volumes/external/wf/data/ARSession/out/{i}_tmp_heat_ref.png")
            manual_decoded_keypoints=[]
            for j in range(12): #range(y_hat.shape[0]):
                manual_decoded_keypoints.append((y_hat[j]==torch.max(y_hat[j])).nonzero()[0].tolist())
            annotated_image = draw_keypoints(image, manual_decoded_keypoints, labels=label_names, show_labels=False, show_all=True)
            annotated_image.save(f"/Volumes/external/wf/data/ARSession/out_ref/{i}_tmp.png")

    elif args.model_type == "yolov5":

        for i, fn in tqdm(enumerate(image_files)):
            image = PIL.Image.open(fn).resize((320, 320))
            outputs = model.predict({"image" : image, "confidenceThreshold": 0.002})
            image = image_to_tensor(image)
            _, h, w = image.shape
            bboxes = []
            # breakpoint()
            for bbox in  outputs['coordinates']:
                x, y,  relw, relh = bbox
                bboxes.append([ int(x * w) + int(w*relw)/2, int(y * h) + int(h*relh)/2, int(x * w)- int(w*relw)/2, int(y * h)- int(h*relh)/2])
            annotated_image = draw_bounding_boxes(image, torch.tensor(bboxes))
            PIL.Image.fromarray(annotated_image.permute(1, 2, 0).numpy()).save(f"/Volumes/external/wf/data/ARSession/out_yolov5/{i}_tmp.png")
    from subprocess import check_call

    # ffmpeg -framerate 25 -i 'TRAIN_v004_.0%3d_out.png' -c:v libx264 -pix_fmt yuv420p out.mp4
    out = check_call(["ffmpeg -framerate 25 -i  /Volumes/external/wf/data/ARSession/out_yolov5/%d_tmp.png -c:v libx264 -pix_fmt yuv420p /Volumes/external/wf/data/ARSession/out_yolov5/out_new.mp4"], shell=True)