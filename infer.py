import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import torch
from tqdm import tqdm

from train import Keypointdetector
from utils import Keypoints, draw_keypoints, load_image, sn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a folder of images through the Keypoint detection maodel"
    )
    parser.add_argument(
        "--src",
        dest="src_dir",
        type=Path,
        help="Path to the source directroy containing the .png files.",
    )
    parser.add_argument(
        "--dst",
        dest="dst_dir",
        type=Path,
        help="Path to the destination directroy containing the .png files.",
    )
    parser.add_argument(
        "--mdir", dest="model_dir", type=Path, help="Path to model file.",
    )
    parser.add_argument(
        "--imsize",
        dest="image_size",
        type=int,
        help="Size of the input image.",
        default=480,
    )
    parser.add_argument(
        "--cuda", dest="cuda", type=bool, help="Size of the input image.", default=False
    )
    # parser.add_argument(
    #     "--bs",
    #     dest
    # )

    args = parser.parse_args()
    args.dst_dir.mkdir(exist_ok=True, parents=True)
    input_size = (args.image_size, args.image_size)
    model = Keypointdetector.load_from_checkpoint(
        args.model_dir, output_image_size=input_size
    )
    tfms = A.Compose(
        [
            A.Resize(*input_size),
            A.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        ],
    )
    model.eval()
    if args.cuda:
        model.cuda()
    frames = list(Path(args.src_dir).glob("*.png"))
    MEAN = 255 * torch.tensor([0.485, 0.456, 0.406])
    STD = 255 * torch.tensor([0.229, 0.224, 0.225])

    from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

    class InferenceDataset(Dataset):
        def __init__(self, src_dir):
            """"""
            self.src_dir = src_dir
            self.image_files = sorted(list(Path(self.src_dir).glob("*.png")))

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            image = cv2.imread(os.fsdecode(self.image_files[index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            trasformed = tfms(image=image)
            x = torch.tensor(trasformed["image"], dtype=torch.float32).permute(2, 0, 1)
            return x, self.image_files[index].stem

    BATCH_SIZE = 8
    ds = InferenceDataset(src_dir=args.src_dir)
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=False
    )

    for batch, filenames in tqdm(dl):
        if args.cuda:
            batch = batch.to("cuda")
        with torch.no_grad():
            y_hats = model(batch)
        batch = batch.cpu()
        y_hats = y_hats.cpu()
        for x, y_hat, fn in zip(batch, y_hats, filenames):
            keypoints = []
            for i in range(y_hat.shape[0]):
                keypoints.append(
                    (y_hat[i] == torch.max(y_hat[i])).nonzero()[0].tolist()[::-1]
                )
            label_names = [Keypoints._fields[o] for o in range(12)]

            x = x * STD[:, None, None] + MEAN[:, None, None]
            res = draw_keypoints(
                x.squeeze(0),
                keypoints,
                labels=label_names,
                show_labels=True,
                show_all=True,
            )
            res.save(f"{os.fsdecode(args.dst_dir)}/{fn}_out.png")
