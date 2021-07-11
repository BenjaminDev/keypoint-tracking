import argparse
import os
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image
from pydantic.types import Json
from scipy import ndimage
from tqdm import tqdm

from utils import (MetaData, draw_bounding_box, draw_keypoints, exr2rgb,
                   exr_channel_to_np, mask_to_bounding_boxes,
                   mask_to_keypoints, read_meta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-processes exr file from Wearfits and generates a folder of images and json meta data files."
    )
    parser.add_argument(
        "--src",
        dest="src_dir",
        type=Path,
        help="Path to the source directroy containing the .exr files.",
    )
    parser.add_argument(
        "--dst",
        dest="dst_dir",
        type=Path,
        help="Path to the destination directory containing the images and meta data files",
    )
    parser.add_argument(
        "--num",
        dest="num",
        type=int,
        default=-1,
        help="Number of file to process. Defaults to -1 implying all files in src_dir",
    )
    parser.add_argument(
        "--ext",
        dest="ext",
        choices=[".png", ".jpg"],
        default=".png",
        help="Specify image format of output.",
    )
    args = parser.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    for i, exr_file in enumerate(tqdm(list(args.src_dir.glob("*MAIN*.exr")))):
        if args.num < i and args.num != -1:
            break
        # Read "MAIN" exr file
        exr_file_discriptor = OpenEXR.InputFile(os.fsdecode(exr_file))
        dw = exr_file_discriptor.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Get RGB and header
        image, header = exr2rgb(exr_file_discriptor, size)
        # Get Visible keypoints
        mask = exr_channel_to_np(exr_file_discriptor, size, "ObjectID")
        _, visible = mask_to_keypoints(mask)
        bounding_boxes = mask_to_bounding_boxes(mask)
        # Read "ID" exr file
        meta_file = os.fsdecode(exr_file).replace("MAIN", "ID")
        exr_file_discriptor = OpenEXR.InputFile(os.fsdecode(meta_file))
        dw = exr_file_discriptor.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        # Get all keypoints
        mask = exr_channel_to_np(exr_file_discriptor, size, "ObjectID")
        keypoints, _ = mask_to_keypoints(mask)

        # Set source
        source = {
            "main": f"{exr_file.stem}{exr_file.suffix}",
            "object_id": f"{Path(meta_file).stem}{Path(meta_file).suffix}",
            "image": f"{exr_file.stem}{args.ext}",
        }
        # Form metadata
        meta = MetaData(
            source=source,
            keypoints=keypoints,
            visible=visible,
            bounding_boxes=bounding_boxes,
        )

        # Reduces dynamic range to 8bit.
        img = Image.fromarray(np.clip(image, a_min=0, a_max=255).astype("uint8"))
        # Write image to destination directory
        img.save(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}{args.ext}")
        # Write meta data to destination directory
        with open(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}.json", mode="w") as fp:
            fp.write(meta.json(indent=4))

        # Read back and annotate to verify
        (args.dst_dir / "annotated").mkdir(parents=True, exist_ok=True)
        read_meta_back = read_meta(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}.json")
        img = draw_bounding_box(img, read_meta_back.bounding_boxes)
        draw_keypoints(
            img,
            read_meta_back.keypoints,
            read_meta_back.keypoint_labels,
            read_meta_back.visible,
            show_all=True,
        ).save(f"{os.fsdecode(args.dst_dir)}/annotated/anno_{exr_file.stem}{args.ext}")
