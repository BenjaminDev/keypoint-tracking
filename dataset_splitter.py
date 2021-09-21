
import argparse
from pathlib import Path
from random import sample
import shutil
import os
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed

def verify_image(image_file):
    image = cv2.imread(os.fsdecode(image_file))
    if image is None:
        print (f"BROKEN FILE: {os.fsdecode(image_file)}")
        raise AssertionError(f"remove the image and json file: {os.fsdecode(image_file)}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processes exr file from Wearfits and generates a folder of images and json meta data files."
    )
    parser.add_argument(
        "--src",
        dest="src_dir",
        type=Path,
        help="Path to the source directroy containing the .png and .json files.",
    )
    parser.add_argument(
        "--pct_val",
        dest="pct_val",
        type=float,
        help="Percentage of data points to add to validation set.",
        default=0.2
    )
    parser.add_argument(
        "--pct_test",
        dest="pct_test",
        type=float,
        help="Percentage of data points to add to test set.",
        default=0.1
    )
    parser.add_argument(
        "--ext",
        dest="ext",
        choices=[".png", ".jpg"],
        default=".png",
        help="Specify image format of output.",
    )
    args = parser.parse_args()
    image_files = sorted(list(args.src_dir.glob(f"*{args.ext}")))
    json_files = sorted(list(args.src_dir.glob(f"*.json")))
    for json_file, image_file in tqdm(zip(json_files, image_files), total=len(image_files)):
        if json_file.stem != image_file.stem:
            raise Exception(f"we have a big problem! meta and image file mismatch: {json_file.stem}.json not paired with {image_file.stem}.png")


    n_jobs = os.cpu_count() if not os.environ.get("PRE_DEBUG", False) else 1
    Parallel(n_jobs=n_jobs)(delayed(verify_image)(image_file) for image_file in tqdm(image_files))
    total = len(image_files)
    total_num_val_and_test = int(total*(args.pct_val+args.pct_test))
    total_num_val = int(total*(args.pct_val))

    sample_indices_val_and_test = sample(range(0, total), total_num_val_and_test)
    sample_indices_val = sample_indices_val_and_test[:total_num_val]
    sample_indices_test = sample_indices_val_and_test[total_num_val:]
    if (args.src_dir/"val").exists() and any((args.src_dir/"val").iterdir()):
        print("Val dir exists we don't want to move more....")
        exit(0)
    if (args.src_dir/"test").exists() and any((args.src_dir/"test").iterdir()):
        print("Test dir exists we don't want to move more....")
        exit(0)
    (args.src_dir/"val").mkdir(exist_ok=True)
    (args.src_dir/"test").mkdir(exist_ok=True)
    for sample_idx  in sample_indices_val:
        image_file = image_files[sample_idx]
        json_file = json_files[sample_idx]
        if json_file.stem != image_file.stem:
            raise Exception(f"we have a big problem! meta and image file mismatch: {json_file.stem} != {image_file.stem}")
        shutil.move(image_file, image_file.parent/'val'/f"{image_file.stem}{image_file.suffix}")
        shutil.move(json_file, json_file.parent/'val'/f"{json_file.stem}{json_file.suffix}")

    for sample_idx  in sample_indices_test:
        image_file = image_files[sample_idx]
        json_file = json_files[sample_idx]
        if json_file.stem != image_file.stem:
            raise Exception("we have a big problem! meta and image file mismatch")
        shutil.move(image_file, image_file.parent/'test'/f"{image_file.stem}{image_file.suffix}")
        shutil.move(json_file, json_file.parent/'test'/f"{json_file.stem}{json_file.suffix}")
