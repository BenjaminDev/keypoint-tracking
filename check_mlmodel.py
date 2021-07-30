import coremltools as ct
from pathlib import Path
from coremltools import models
from tqdm import tqdm
from utils import draw_keypoints, Keypoints
import PIL
import os
ml_model_path = Path("outputs/beta_v0.2/Pipeline.mlmodel")

spec = ct.utils.load_spec(os.fsdecode(ml_model_path))
model = ct.models.MLModel(spec)

image_files = list(Path("/Volumes/external/wf/data/ARSession/frames").glob("*.png"))
label_names = [Keypoints._fields[o-1] for o in range(12)]
i=0
# for fn in tqdm(image_files):
#     i+=1
#     # breakpoint()
#     image = PIL.Image.open(fn).resize((480, 480))
#     outputs = model.predict({"input" : image})
#     kps=outputs["coordinates"]
#     annotated_image = draw_keypoints(image, kps, labels=label_names, show_labels=True, show_all=True)
#     annotated_image.save(f"/Volumes/external/wf/data/ARSession/out/{i}_tmp.png")

from subprocess import check_call
# ffmpeg -framerate 25 -i 'TRAIN_v004_.0%3d_out.png' -c:v libx264 -pix_fmt yuv420p out.mp4
out = check_call(["ffmpeg -framerate 25 -i  /Volumes/external/wf/data/ARSession/out/%d_tmp.png -c:v libx264 -pix_fmt yuv420p /Volumes/external/wf/data/ARSession/out/out.mp4"])