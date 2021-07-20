

from train import Keypointdetector
from utils import load_image, Keypoints, draw_keypoints, sn
import torch
import numpy as np
import onnxruntime
image_size = (480, 480)
model = Keypointdetector.load_from_checkpoint("/mnt/vol_c/epoch=8-step=2061.ckpt", output_image_size=image_size)
# vgfbe5ql

filepath = 'model.onnx'
# 1vzt4h2g
input_sample = torch.randn(( 64,3,3,3))
x=load_image("/mnt/vol_c/clean_data/tmp/TRAIN_v004_MAIN_.0000.png", image_size)
breakpoint()
model.to_onnx(filepath, x, export_params=True, opset_version=11)



ort_session = onnxruntime.InferenceSession(filepath)
input_name = ort_session.get_inputs()[0].name
breakpoint()
ort_inputs = {input_name: x}
ort_outs = ort_session.run(None, ort_inputs)
print(model.learning_rate)
breakpoint()
# prints the learning_rate you used in this checkpoint
# x=load_image("/mnt/vol_b/data/VIDEO-2021-07-01-10-32-57/frame_00232.png", image_size)

model.eval()


y_hat = model(x).squeeze(0)


keypoints=[]
for i in range(y_hat.shape[0]):
    keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
label_names = [Keypoints._fields[o] for o in range(12)]
res = draw_keypoints(
    x.squeeze(0), keypoints, labels=label_names, show_labels=True, show_all=True)
res.save(f"out.png")