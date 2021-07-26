

from train import Keypointdetector
from utils import load_image, Keypoints, draw_keypoints, sn
import torch
import numpy as np
import onnxruntime
import coremltools as ct
import cv2
from pathlib import Path
import os
import albumentations as A
breakpoint()
input_size = (480, 480)
tfms = A.Compose(
            [
                A.Resize(*input_size),
                A.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std= torch.tensor([0.229, 0.224, 0.225])),
            ],
)

traceable_model = Keypointdetector.load_from_checkpoint("/mnt/vol_c/epoch=8-step=2061.ckpt", output_image_size=input_size).eval()
frame = Path("/mnt/vol_b/training_data/clean/0002-reference-video-2/source/v001/TRAIN_v004_.0599.png")
image = cv2.imread(os.fsdecode(frame))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
trasformed = tfms(image=image)
x = torch.tensor(trasformed["image"],dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


input_batch = x

trace = torch.jit.trace(traceable_model, input_batch)

mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

mlmodel.save("keypoint_no_metadata.mlmodel")

# filepath = 'model.onnx'
# # 1vzt4h2g
# input_sample = torch.randn(( 64,3,3,3))
# x=load_image("/mnt/vol_c/clean_data/tmp/TRAIN_v004_MAIN_.0000.png", image_size)
# breakpoint()
# model.to_onnx(filepath, x, export_params=True, opset_version=11)



# ort_session = onnxruntime.InferenceSession(filepath)
# input_name = ort_session.get_inputs()[0].name
# breakpoint()
# ort_inputs = {input_name: x}
# ort_outs = ort_session.run(None, ort_inputs)
# print(model.learning_rate)
# breakpoint()
# # prints the learning_rate you used in this checkpoint
# # x=load_image("/mnt/vol_b/data/VIDEO-2021-07-01-10-32-57/frame_00232.png", image_size)

# model.eval()


# y_hat = model(x).squeeze(0)


# keypoints=[]
# for i in range(y_hat.shape[0]):
#     keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
# label_names = [Keypoints._fields[o] for o in range(12)]
# res = draw_keypoints(
#     x.squeeze(0), keypoints, labels=label_names, show_labels=True, show_all=True)
# res.save(f"out.png")