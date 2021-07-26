
from train import Keypointdetector
from utils import load_image, Keypoints, draw_keypoints, sn
from pathlib import Path
from tqdm import tqdm
import torch
import albumentations as A
import cv2
import os
input_size = (480, 480)
model = Keypointdetector.load_from_checkpoint("/mnt/vol_c/models/wf/3taowhhg/checkpoints/epoch=51-step=11908.ckpt", output_image_size=input_size)
# vgfbe5ql
print(model.learning_rate)
# prints the learning_rate you used in this checkpoint
# x=load_image("/mnt/vol_b/data/VIDEO-2021-07-01-10-32-57/frame_00232.png", image_size)
tfms = A.Compose(
            [
                A.Resize(*input_size),
                A.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std= torch.tensor([0.229, 0.224, 0.225])),
            ],
)
model.eval()
model.cuda()
frames = list(Path("/mnt/vol_b/data/mockup_002/").glob("*.png"))
MEAN = 255*torch.tensor([0.485, 0.456, 0.406])
STD = 255*torch.tensor([0.229, 0.224, 0.225])

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
class InferenceDataset(Dataset):

    def __init__(self, src_dir):
        """"""
        self.src_dir=src_dir
        self.image_files = sorted(list(Path(self.src_dir).glob("*.png")))
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(os.fsdecode(self.image_files[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trasformed = tfms(image=image)
        x = torch.tensor(trasformed["image"],dtype=torch.float32).permute(2, 0, 1)
        return x, self.image_files[index].stem

BATCH_SIZE=8
ds=InferenceDataset(src_dir="/mnt/vol_b/data/mockup_002/")
dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=False)

for batch, filenames in tqdm(dl):

    # x=load_image(frame, input_size)
    # image = cv2.imread(os.fsdecode(frame))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #
    batch = batch.to('cuda')
    with torch.no_grad():
        y_hats = model(batch)
    batch = batch.cpu()
    y_hats = y_hats.cpu()
    for x, y_hat, fn in zip(batch, y_hats, filenames):
        keypoints=[]
        for i in range(y_hat.shape[0]):
            keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
        label_names = [Keypoints._fields[o] for o in range(12)]

        x = x * STD[:, None, None] + MEAN[:, None, None]
        res = draw_keypoints(
            x.squeeze(0), keypoints, labels=label_names, show_labels=True, show_all=True)
        res.save(f"/mnt/vol_b/outputs/mockup_002_new/{fn}_out.png")