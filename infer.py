
from train import Keypointdetector
from utils import load_image, Keypoints, draw_keypoints, sn
import torch
image_size = (2*480, 2*480)
model = Keypointdetector.load_from_checkpoint("/mnt/vol_b/models/keypoints/wf/285vbodm/checkpoints/epoch=93-step=3290.ckpt", output_image_size=image_size)

print(model.learning_rate)
# prints the learning_rate you used in this checkpoint
# x=load_image("/mnt/vol_b/data/VIDEO-2021-07-01-10-32-57/frame_00232.png", image_size)
x=load_image("/mnt/vol_b/clean_data/tmp2/val/TRAIN_v004_MAIN_.0000.png", image_size)

model.eval()


y_hat = model(x).squeeze(0)


keypoints=[]
for i in range(y_hat.shape[0]):
    keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
label_names = [Keypoints._fields[o] for o in range(12)]
res = draw_keypoints(
    x.squeeze(0), keypoints, labels=label_names, show_labels=True, show_all=True)
res.save(f"out.png")