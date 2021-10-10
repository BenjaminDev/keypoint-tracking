from datetime import time
import coremltools as ct
from pathlib import Path
# from coremltools import models
from tqdm import tqdm
from utils import draw_keypoints, Keypoints, image_to_tensor
import PIL
from torch import nn
import torch
from PIL import Image
from joblib import Parallel, delayed
from loky import set_loky_pickler
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
import subprocess
import shlex
def convert_video_to_frames(video_path:Path,height:int=480)->Path:
    output_dir = video_path.parent/video_path.stem
    output_dir.mkdir(exist_ok=True)
    video_path = shutil.move(os.fsdecode(video_path), os.fsdecode(output_dir/f"{video_path.stem}{video_path.suffix}"))
    cmd = f"ffmpeg -i \"{os.fsdecode(video_path)}\" -r 25 -vf \"scale=-1:{height}\" \"{os.fsdecode(output_dir)}/frame_%03d.png\""
    subprocess.check_call(shlex.split(cmd))
    return output_dir

def archive_frames(output_dir:Path):
    folder_name = output_dir.stem
    cmd = f"tar -czf {folder_name}.tar --exclude=\"{folder_name}/inspect\" {folder_name}"
    subprocess.check_call(shlex.split(cmd), cwd=output_dir.parent)


def pred_data_test(i, fn, args, network_input_key, network_output_key):
    return pred_data(i, fn, args, network_input_key, network_output_key,model_test,annoate_path="annotated/test")

def pred_data_ref(i, fn, args, network_input_key, network_output_key):
    return pred_data(i, fn, args, network_input_key, network_output_key,model_ref,annoate_path="annotated/ref")

def pred_data(i, fn, args, network_input_key, network_output_key, model, annoate_path):
    # spec = ct.utils.load_spec(os.fsdecode(model_path))
    # model = ct.models.MLModel(spec)
    image = PIL.Image.open(fn).resize((160, 160))
    y_hat = model.predict({network_input_key : image})[network_output_key]
    # breakpoint()
    # y_hat=torch.tensor(y_hat).squeeze(0).permute(2,1,0)
    y_hat = y_hat.transpose(0,3,2,1)
    # breakpoint()
    if model.author == "Boulderama": # currently boulderama models are 6 keypoint models.
        y_hat = torch.tensor(np.ascontiguousarray(y_hat, dtype=np.float32))
        keypoints = []

        for i in range(12):
            keypoints.append(
                (y_hat[0][i] == torch.max(y_hat[0][i])).nonzero()[0].tolist()[::-1]
            )
    else:
        y_hat = torch.tensor(np.concatenate([y_hat[:,:6,...], y_hat[:,7:13,...]])).reshape(1,12,160,160)
        keypoints=subpix.spatial_soft_argmax.spatial_soft_argmax2d(y_hat, normalized_coordinates=False).squeeze(0)
    # breakpoint()
    manual_decoded_keypoints=[]
    input_image = PIL.Image.open(fn)
    h, w = input_image.size
    for kp in keypoints:
        if model.author == "Boulderama":
            manual_decoded_keypoints.append(((kp[1]/160.0)*h, (kp[0]/160.0)*w))
        else:
            manual_decoded_keypoints.append(((kp[1].item()/160.0)*h, (kp[0].item()/160.0)*w))
    # for j in range(12): #range(y_hat.shape[0]):
    #     manual_decoded_keypoints.append((y_hat[j]==torch.max(y_hat[j])).nonzero()[0].tolist())
    if args.annotate:
        annotated_image = draw_keypoints(input_image, manual_decoded_keypoints, labels=label_names, show_labels=False, show_all=True)
        annotated_image.save(f"{os.fsdecode(args.data_dir/annoate_path)}/{fn.stem}.png")
    source = {
        "main": "",
        "object_id": f"{model.version or 'ref'}",
        "image": f"{fn}",
    }

    meta = MetaData(
        source=source,
        keypoints=manual_decoded_keypoints,
        visible=[True]*len(keypoints),
        bounding_boxes=[],
    )

    # Write meta data to destination directory
    with open(f"{os.fsdecode(args.data_dir/annoate_path)}/{fn.stem}.json", mode="w") as fp:
        fp.write(meta.json(indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare 2 models"
    )
    parser.add_argument(
        "--mpath-ref",
        dest="model_path_ref",
        type=Path,
        help="Path to the mlcore model",
    )
    parser.add_argument(
        "--mpath-test",
        dest="model_path_test",
        type=Path,
        help="Path to the mlcore model",
    )
    parser.add_argument(
        "--video",
        dest="video_path",
        type=Path,
        help="Path to video file",
    )
    parser.add_argument(
        "--images",
        dest="image_path",
        type=Path,
        help="Path to folder of .png's",
    )
    parser.add_argument(
       '--annotate', action='store_true',
        dest="annotate",
        default=True,
        help="Creates a inspect folder with annotated images.",
    )


    args = parser.parse_args()
    # ml_model_path = Path(args.mpath)

    spec_ref = ct.utils.load_spec(os.fsdecode(args.model_path_ref))
    model_ref = ct.models.MLModel(spec_ref)
    spec_test = ct.utils.load_spec(os.fsdecode(args.model_path_test))
    model_test = ct.models.MLModel(spec_test)
    data_dirs=[]
    if args.video_path:
        for video_path in args.video_path.glob("*.MOV"):
            print(video_path)
            data_dirs.append(convert_video_to_frames(video_path))
    if args.image_path:
        data_dirs.append(args.image_path)
    for data_dir in data_dirs:
        image_files = list(data_dir.glob("*.png"))
        (data_dir/"annotated/ref").mkdir(exist_ok=True, parents=True)
        (data_dir/"annotated/test").mkdir(exist_ok=True, parents=True)
        args.data_dir = data_dir

        label_names = [Keypoints._fields[o-1] for o in range(12)]
        i=0

        n_jobs = os.cpu_count() if not os.environ.get("PRE_DEBUG", False) else 1
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(pred_data_test)(i, fn,args, network_input_key="input",network_output_key="heatmaps") for i, fn in enumerate(tqdm(image_files)))
        Parallel(n_jobs=n_jobs, prefer="threads")(delayed(pred_data_ref)(i, fn,args, network_input_key="input0:0",network_output_key="output0") for i, fn in enumerate(tqdm(image_files)))
        # archive_frames(args.data_dir)

    # for fn in tqdm(image_files):
    #     i+=1
        # breakpoint()
    # from subprocess import check_call
    # ffmpeg -framerate 25 -i 'TRAIN_v004_.0%3d_out.png' -c:v libx264 -pix_fmt yuv420p out.mp4
    # out = check_call(["ffmpeg -framerate 25 -i  /Volumes/external/wf/data/ARSession/out_yolov5/%d_tmp.png -c:v libx264 -pix_fmt yuv420p /Volumes/external/wf/data/ARSession/out_yolov5/out_new.mp4"], shell=True)