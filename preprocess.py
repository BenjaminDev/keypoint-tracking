import argparse
import numpy as np
from PIL import Image
from utils import exr_channels_to_np, exr_to_srgb, exr2rgb, Keypoints, MetaData, exr_channel_to_np
from pathlib import Path
from scipy import ndimage


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
        "--ext", dest="ext", choices=["png", "jpg"],
        default="png",
        help="Specify image format of output.",
    )
    args = parser.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    import os
    for exr_file in args.src_dir.glob("*MAIN*.exr"):
        print(exr_file.absolute())
        image, header = exr2rgb(os.fsdecode(exr_file))
        mask = exr_channel_to_np(os.fsdecode(exr_file), "ObjectID")
        ndimage.measurements.center_of_mass((mask == Keypoints.LEFT_SHOE).astype('int'))
        # Image.fromarray(image)
        # Reduces dynamic range to 8bit.
        img = Image.fromarray((image*255/np.max(image)).astype("uint8"))
        img.save(f"{os.fsdecode(args.dst_dir)}/{exr_file.stem}.{args.ext}")
        break
    breakpoint()


