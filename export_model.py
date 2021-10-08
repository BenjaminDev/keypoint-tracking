import argparse
import json
import os

# import warnings
from datetime import datetime
from pathlib import Path

# import onnxruntime
# import coremltools
import coremltools as ct

# import cv2
# import numpy as np
import PIL
import torch

from train import Keypointdetector
from utils import Keypoints, draw_keypoints

# from coremltools import models
# from coremltools.models import pipeline
# from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Does 2 things:
            1) Converts a model from pytorch_lightning checkpoint .ckpt to a pytorch traced model .pt (run on linux)
            2) Converts a model from pytorch traced .pt to a coreml model. (run on macos)
        """
    )
    parser.add_argument(
        "--mpath",
        dest="model_path",
        type=Path,
        help="Path to the pytorch lightning model .ckpt or a coreml .mlmodel. See --trace",
    )
    parser.add_argument(
        "--mdir-dst",
        dest="model_dir_dst",
        type=Path,
        help="Path to the pytorch lightning model .ckpt or a coreml .mlmodel. See --trace",
    )
    parser.add_argument(
        "--im-width",
        dest="input_image_width",
        type=int,
        help="width of the image to process",
        default=480,
    )
    parser.add_argument(
        "--im-height",
        dest="input_image_height",
        type=int,
        help="height of the image to process",
        default=480,
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        dest="trace",
        help="Specify if tracing is needed. Otherwise expects --mdir to point to a .mlmodel",
    )
    parser.add_argument("--no-trace", action="store_false", dest="trace")
    args = parser.parse_args()
input_size = (args.input_image_width, args.input_image_height)
args.model_dir_dst.mkdir(parents=True, exist_ok=True)
test_image_path = Path("test_data/frame_001.jpg")
assert test_image_path.is_file(), "cannot find test image file."
manual_decoded_keypoints = []
if args.trace:
    if args.model_path.suffix != ".ckpt":
        raise ValueError("--trace is enabled and mdir is not pointing to a .ckpt file.")
    traceable_model = Keypointdetector.load_from_checkpoint(
        os.fsdecode(args.model_path), output_image_size=input_size, inferencing=True
    ).eval()
    input_batch = torch.rand(1, 3, *input_size)
    trace = torch.jit.trace(traceable_model, input_batch)
    scale = 1.0 / (0.226 * 255.0)
    red_bias = -(0.485 * 255.0) * scale
    green_bias = -(0.456 * 255.0) * scale
    blue_bias = -(0.406 * 255.0) * scale

    mlmodel = ct.convert(
        trace,
        inputs=[
            ct.ImageType(
                name="input",
                shape=input_batch.shape,
                bias=[red_bias, green_bias, blue_bias],
                scale=scale,
            )
        ],
    )

    mlmodel.save(os.fsdecode(args.model_dir_dst / f"heatmap_only.mlmodel"))


existing_model = ct.utils.load_spec(
    os.fsdecode(args.model_dir_dst / f"heatmap_only.mlmodel")
)  
heatmap_model = ct.models.MLModel(existing_model)
image = PIL.Image.open(test_image_path).resize(input_size)
outputs = heatmap_model.predict({"input": image})
output_name = list(outputs.keys())
assert len(output_name) == 1

y_hat = torch.tensor(outputs[output_name[0]]).squeeze(0)

for i in range(y_hat.shape[0]):
    manual_decoded_keypoints.append(
        (y_hat[i] == torch.max(y_hat[i])).nonzero()[0].tolist()[::-1]
    )

label_names = [Keypoints._fields[o] for o in range(12)]
res = draw_keypoints(
    image, manual_decoded_keypoints, labels=label_names, show_labels=True, show_all=True
)
res.save(os.fsdecode(args.model_dir_dst / f"manual_decode_out.png"))


builder = ct.models.neural_network.NeuralNetworkBuilder(spec=existing_model)
################
def add_output(builder, name):
    out = builder.spec.description.output.add()
    out.name = name
    out.type.multiArrayType.MergeFromString(b"")
    out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE


input_name = "4260"

builder.add_reduce_max(
    "x_maxes", input_name=input_name, output_name="x_maxes", axes=[2], keepdims=True
)
builder.add_argmax("x_args", input_name="x_maxes", output_name="x_args", axis=3)

builder.add_reduce_max(
    "y_maxes", input_name=input_name, output_name="y_maxes", axes=[3], keepdims=True
)
builder.add_argmax("y_args", input_name="y_maxes", output_name="y_args", axis=2)

builder.add_squeeze(
    "ys_abs", input_name="y_args", output_name="ys_abs", squeeze_all=True
)
builder.add_elementwise(
    "ys",
    input_names=["ys_abs"],
    output_name="ys",
    mode="MULTIPLY",
    alpha=1.0 / input_size[0],
)
builder.add_squeeze(
    "xs_abs", input_name="x_args", output_name="xs_abs", squeeze_all=True
)
builder.add_elementwise(
    "xs",
    input_names=["xs_abs"],
    output_name="xs",
    mode="MULTIPLY",
    alpha=1.0 / input_size[1],
)
builder.add_concat_nd(
    "coordinates_flat",
    input_names=["xs", "ys"],
    output_name="coordinates_flat",
    axis=0,
    interleave=True,
)
builder.add_reshape_static(
    "coordinates",
    input_name="coordinates_flat",
    output_name="coordinates",
    output_shape=(12, 2),
)


del builder.spec.description.output[0]  # Remove the heatmap outputs
out = builder.spec.description.output.add()
out.name = "coordinates"
out.type.multiArrayType.MergeFromString(b"")
out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE

builder.spec.description.metadata.shortDescription = "Shoe keypoint detector"
builder.spec.description.metadata.author = "Boulderama"
builder.spec.description.metadata.versionString = (
    f"{args.model_path.stem} - {datetime.now().isoformat()}"
)

adjusted_model = ct.models.MLModel(builder.spec)
adjusted_model.save(os.fsdecode(args.model_dir_dst / f"Pipeline.mlmodel"))

# Load the final model and validate the outputs
spec = ct.utils.load_spec(os.fsdecode(args.model_dir_dst / f"Pipeline.mlmodel"))
model = ct.models.MLModel(spec)
image = PIL.Image.open(test_image_path).resize(input_size)
outputs = model.predict({"input": image})

kps = outputs["coordinates"]

annotated_image = draw_keypoints(
    image, kps, labels=label_names, show_labels=True, show_all=True
)
annotated_image.save(os.fsdecode(args.model_dir_dst / "Pipeline_model_output.png"))
# Final check!
for x_rel_y_rel, x_md_abs_y_md_abs in zip(kps, manual_decoded_keypoints):
    x_rel, y_rel = x_rel_y_rel
    x_md_abs, y_md_abs = x_md_abs_y_md_abs
    assert ((x_rel * input_size[0]) - x_md_abs) < 1.0
    assert ((y_rel * input_size[1]) - y_md_abs) < 1.0
print("That should be good. Check the test outputs but all should be good!")
