import coremltools as ct
from pathlib import Path
from coremltools import models
from tqdm import tqdm
from utils import draw_keypoints, Keypoints
import PIL
ml_model_path = "/Users/benjamin/projects/keypoint-tracking/Test.mlmodel"

spec = ct.utils.load_spec(ml_model_path)
model = ct.models.MLModel(spec)

image_files = list(Path("/Volumes/external/wf/data/ARSession").glob("*.png"))
label_names = [Keypoints._fields[o-1] for o in range(12)]
i=0
for fn in tqdm(image_files):
    i+=1
    # breakpoint()
    image = PIL.Image.open(fn).resize((480, 480))
    outputs = model.predict({"input" : image})
    kps=outputs["coordinates"]
    annotated_image = draw_keypoints(image, kps, labels=label_names, show_labels=True, show_all=True)
    annotated_image.save(f"/Volumes/external/wf/data/ARSession/out/{i}_tmp.png")