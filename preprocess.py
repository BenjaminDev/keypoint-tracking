import argparse
import numpy as np
from PIL import Image
from pydantic.types import Json
from utils import exr2rgb, Keypoints, MetaData, exr_channel_to_np, mask_to_keypoints, mask_to_bounding_boxes, read_meta
from pathlib import Path
from scipy import ndimage
import OpenEXR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-processes exr file from Wearfits and generates a folder of images and json meta data files.")
    parser.add_argument(
        "--src",
        dest="src_dir",
        type=Path,
        help="Path to the source directroy containing the .exr files.",
    )
    parser.add_argument(
        "--dst", dest="dst_dir", type=Path, help="Path to the destination directory containing the images and meta data files"
    )
    parser.add_argument(
        "--ext", dest="ext", choices=[".png", ".jpg"],
        default=".png",
        help="Specify image format of output.",
    )
    args = parser.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    import os
    for exr_file in args.src_dir.glob("*MAIN*.exr"):
        print(exr_file.absolute())
        # Get RGB and header
        image, header = exr2rgb(os.fsdecode(exr_file))
        # Get Visible keypoints
        exr_file_discriptor = OpenEXR.InputFile(os.fsdecode(exr_file))
        dw = exr_file_discriptor.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        mask = exr_channel_to_np(exr_file_discriptor, size,"ObjectID")
        _, visible = mask_to_keypoints(mask)
        bounding_boxes = mask_to_bounding_boxes(mask)


        meta_file = os.fsdecode(exr_file).replace("MAIN", "ID")
        exr_file_discriptor = OpenEXR.InputFile(os.fsdecode(exr_file))
        dw = exr_file_discriptor.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        mask = exr_channel_to_np(exr_file_discriptor, size,"ObjectID")
        keypoints, _ = mask_to_keypoints(mask)

        # Set source
        source={
                "main": f"{exr_file.stem}{exr_file.suffix}",
                "object_id": f"{Path(meta_file).stem}{Path(meta_file).suffix}",
                "image": f"{exr_file.stem}{args.ext}"
            }
        # Form metadata
        meta = MetaData(source=source, keypoints=keypoints, visible=visible, bounding_boxes=bounding_boxes)

        # Reduces dynamic range to 8bit.
        img = Image.fromarray(np.clip(image, a_min=0, a_max=255).astype("uint8"))
        # Write image to destination directory
        img.save(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}{args.ext}")
        # Write meta data to destination directory
        with open(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}.json", mode="w") as fp:
            fp.write(meta.json(indent=4))
        read_meta_back = read_meta(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}.json")
        
    breakpoint()


