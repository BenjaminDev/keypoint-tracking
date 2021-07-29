
import coremltools
import warnings
from coremltools import models
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from matplotlib.colors import BoundaryNorm
warnings.simplefilter("default")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
from train import Keypointdetector
from utils import load_image, Keypoints, draw_keypoints, sn
import torch

import json
import numpy as np
# import onnxruntime
import coremltools as ct
import cv2
from pathlib import Path
import os
import PIL
import albumentations as A

input_size = (480, 480)
tfms = A.Compose(
            [
                A.Resize(*input_size),
                A.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std= torch.tensor([0.229, 0.224, 0.225])),
            ],
)
if False:
    traceable_model = Keypointdetector.load_from_checkpoint("/Users/benjamin/projects/keypoint-tracking/epoch=51-step=11908.ckpt", output_image_size=input_size, inferencing=True).eval()
    # traceable_model = Keypointdetector(output_image_size=input_size, inferencing=True).eval()
    frame = Path("/Users/benjamin/projects/keypoint-tracking/TRAIN_v004_.0599.png")
    image = cv2.imread(os.fsdecode(frame))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    trasformed = tfms(image=image)
    # x = torch.tensor(trasformed["image"],dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    x = torch.rand(1, 3, 480, 480)
    from coremltools.models import pipeline

    input_batch = x

    # scripted_model = torch.jit.script(traceable_model)
    trace = torch.jit.trace(traceable_model, input_batch)
    scale = 1.0 / (0.226 * 255.0)

    red_bias = -(0.485 * 255.0) * scale
    green_bias = -(0.456 * 255.0) * scale
    blue_bias = -(0.406 * 255.0) * scale

    mlmodel = ct.convert(
        trace,
        inputs=[ct.ImageType(name="input", shape=input_batch.shape, bias=[red_bias, green_bias, blue_bias], scale=scale)],

        # outputs=ct.
        # compute_precision=ct.precision.FLOAT16) TEST THIS!
    )
    # model = ct.models.MLModel("posenet_no_preview_type.mlmodel")
    # mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "poseEstimation"
    # params_json = {"width_multiplier": 1.0, "output_stride": 16}
    # mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(params_json)
    # model.save("posenet_with_preview_type.mlmodel")
    # labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}

    # mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
    # mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

    # mlmodel.save("SegmentationModel_with_metadata.mlmodel")
    mlmodel.save("Original.mlmodel")
    image = PIL.Image.open("/Volumes/external/wf/data/ARSession/frame_001.png").resize((480,480))
    outputs = mlmodel.predict({"input": image})
    breakpoint()
    y_hat = torch.tensor(outputs["4254"])
    keypoints=[]
    for i in range(y_hat.shape[0]):
        keypoints.append((y_hat[i]==torch.max(y_hat[i])).nonzero()[0].tolist()[::-1])
        label_names = [Keypoints._fields[o] for o in range(12)]

    print(keypoints)
    res = draw_keypoints(
        image, keypoints, labels=label_names, show_labels=True, show_all=True)
    res.save(f"manual_decode_out.png")
    # exit()

existing_model = ct.utils.load_spec("Original.mlmodel") # torch.Size([12, 480, 480])
builder = ct.models.neural_network.NeuralNetworkBuilder(spec=existing_model)
################
def add_output(builder, name):
    out=builder.spec.description.output.add()
    out.name = name
    out.type.multiArrayType.MergeFromString(b"")
    out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE
    
builder.add_reduce_max("x_maxes", input_name="4252", output_name="x_maxes", axes=[2], keepdims=True)
builder.add_argmax("x_args", input_name="x_maxes",output_name="x_args", axis=3)

builder.add_reduce_max("y_maxes", input_name="4252", output_name="y_maxes", axes=[3], keepdims=True)
builder.add_argmax("y_args", input_name="y_maxes",output_name="y_args", axis=2)

builder.add_squeeze("ys", input_name="y_args", output_name="ys",squeeze_all=True)
builder.add_squeeze("xs", input_name="x_args", output_name="xs",squeeze_all=True)
add_output(builder, name="xs")
builder.add_concat_nd("coordinates_flat", input_names=["xs", "ys"], output_name="coordinates_flat", axis=0, interleave=True)
builder.add_reshape_static("coordinates", input_name="coordinates_flat", output_name="coordinates", output_shape=(12,2))
# del builder.spec.description.output[0] # Remove the heatmap outputs
out=builder.spec.description.output.add()
out.name = "coordinates"
out.type.multiArrayType.MergeFromString(b"")
out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE
adjusted_model = ct.models.MLModel(builder.spec)
adjusted_model.save("Test.mlmodel")

spec = ct.utils.load_spec("Test.mlmodel")
model = ct.models.MLModel(spec)
# [[260, 252], [260, 238], [272, 233], [248, 245], [245, 203], [252, 201], [211, 242], [198, 215], [181, 239], [197, 242], [211, 244], [195, 260]]
image = PIL.Image.open("/Volumes/external/wf/data/ARSession/frame_001.png").resize((480,480))
outputs = model.predict({"input": image})
from utils import draw_keypoints, Keypoints
label_names = [Keypoints._fields[o-1] for o in range(12)]

kps=outputs["coordinates"]
breakpoint()
annotated_image = draw_keypoints(image, kps, labels=label_names, show_labels=True, show_all=True)
annotated_image.save("tmp.png")
exit()
# x_maxes = np.max( outputs["4252"][:,:,:,:], axis=2)

###################


builder.add_argmax("argmax_1_points_y",input_name="4252",output_name="argmax_1_points_y", axis=2, keepdims=False)
builder.add_argmax("argmax_2_points_y",input_name="argmax_1_points_y",output_name="argmax_2_points_y", axis=2, keepdims=False)

# builder.add_elementwise("argmax_2_points_y_rel",
#                         input_names=["argmax_2_points_y"],
#                         output_name="argmax_2_points_y_rel",
#                         mode="MULTIPLY",
#                         alpha=1.0/input_size[0]
#                         )
builder.add_argmax("argmax_1_points_x",input_name="4252",output_name="argmax_1_points_x", axis=3, keepdims=False)
builder.add_argmax("argmax_2_points_x",input_name="argmax_1_points_x",output_name="argmax_2_points_x", axis=2, keepdims=False)
# builder.add_elementwise("argmax_2_points_x_rel",
#                         input_names=["argmax_2_points_x"],
#                         output_name="argmax_2_points_x_rel",
#                         mode="MULTIPLY",
#                         alpha=1.0/input_size[0]
#                         )
builder.add_concat_nd("raw_points", input_names=["argmax_2_points_x", "argmax_2_points_y"], output_name="raw_points", axis=1, interleave=True)
builder.add_reshape_static("coordinates", input_name="raw_points", output_name="coordinates", output_shape=(12,2))

del builder.spec.description.output[0] # Remove the heatmap outputs

out=builder.spec.description.output.add()
out.name = "coordinates"
out.type.multiArrayType.MergeFromString(b"")
out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE
# out.type.multiArrayType.shape.extend([12,4])

adjusted_model = ct.models.MLModel(builder.spec)
adjusted_model.save("adjusted_model.mlmodel")

image = PIL.Image.open("/Volumes/external/wf/data/ARSession/frame_001.png").resize((480,480))
outputs = adjusted_model.predict({"input": image})
from utils import draw_keypoints, Keypoints
label_names = [Keypoints._fields[o-1] for o in range(12)]

kps=outputs["coordinates"]
annotated_image = draw_keypoints(image, kps, labels=label_names, show_labels=True, show_all=True)
annotated_image.save("tmp.png")
exit()

builder.add_concat_nd("raw_coordinates_flat", input_names=["raw_points", "raw_points"], output_name="raw_coordinates_flat", axis=1, interleave=True)
builder.add_reshape_static("raw_coordinates", input_name="raw_coordinates_flat", output_name="raw_coordinates", output_shape=(1,12,4))

out=builder.spec.description.output.add()
out.name = "raw_coordinates"
out.type.multiArrayType.MergeFromString(b"")
out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE
out.type.multiArrayType.shape.extend([1,12,4])

builder.add_load_constant("raw_confidence", "raw_confidence",
    np.array([0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3,0.2, 0.3,0.2]),
    shape=(1,12,1)
)

output2_ = builder.spec.description.output.add()
output2_.name = 'raw_confidence'

output2_.type.multiArrayType.MergeFromString(b"")
output2_.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE

builder.add_nms

adjusted_model = ct.models.MLModel(builder.spec)
adjusted_model.save("adjusted_model.mlmodel")
image = PIL.Image.open("/Users/benjamin/projects/keypoint-tracking/TRAIN_v004_.0599.png").resize((480,480))
outputs = adjusted_model.predict({"input": image})
breakpoint()

nms_spec = ct.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

for i in range(2):
	decoder_output = adjusted_model._spec.description.output[i].SerializeToString()
	nms_spec.description.input.add()
	nms_spec.description.input[i].ParseFromString(decoder_output)
	nms_spec.description.output.add()
	nms_spec.description.output[i].ParseFromString(decoder_output)

nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

# output_sizes = [nc, 4]

# for i in range(2):
# 	ma_type = nms_spec.description.output[i].type.multiArrayType
# 	ma_type.shapeRange.sizeRanges.add()
# 	# ma_type.shapeRange.sizeRanges[0].lowerBound = 0
# 	ma_type.shapeRange.sizeRanges[0].upperBound = -1
# 	ma_type.shapeRange.sizeRanges.add()
# 	# ma_type.shapeRange.sizeRanges[1].lowerBound = -1 # output_sizes[i]
# 	ma_type.shapeRange.sizeRanges[1].upperBound = -1 # output_sizes[i]
# 	del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

nms.iouThreshold = 0.1 #default_iou_threshold
nms.confidenceThreshold = 0.001 # default_confidence_threshold

nms.stringClassLabels.vector.extend(["classes"])

nms_model = ct.models.MLModel(nms_spec)
nms_model.save("NMS.mlmodel")



from coremltools.models import datatypes
from coremltools.models.pipeline import *
img_size = 480
input_features = [ ("image", datatypes.Array(3, img_size, img_size)),
                   ("iouThreshold", datatypes.Double()),
                   ("confidenceThreshold", datatypes.Double()) ]
output_features = [ "confidence", "coordinates" ]

pipeline = Pipeline(input_features, output_features)

pipeline.add_model(adjusted_model)
pipeline.add_model(nms_model)
breakpoint()
pipeline.spec.description.input[0].ParseFromString(
    adjusted_model._spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(
    nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(
    nms_model._spec.description.output[1].SerializeToString())

pipeline.spec.specificationVersion = 3

final_model = ct.models.MLModel(pipeline.spec)
final_model.save("Pipeline.mlmodel")
breakpoint()





# out=builder.spec.description.output.add()
# out.name = "y_points"
# out.type.multiArrayType.MergeFromString(b"")
# out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE

# out=builder.spec.description.output.add()
# out.name = "x_points"
# out.type.multiArrayType.MergeFromString(b"")
# out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE

# builder.add_reshape_static("points",input_name="raw_points", output_name="points", output_shape=[1,12,2])


# builder.add_expand_dims()



out=builder.spec.description.output.add()
out.name = "raw_points"
out.type.multiArrayType.MergeFromString(b"")
out.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE
out.type.multiArrayType.shape.extend([1,12,2])
# out.type.multiArrayType.shape.append(2)


raws = np.array([0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3,0.2, 0.3])
# breakpoint()

builder.add_load_constant("raw_coordinates", "raw_coordinates",
    raws,
    shape=(1,12,4)
)
builder.add_load_constant("raw_confidence", "raw_confidence",
    np.array([0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3,0.2, 0.3,0.2]),
    shape=(1,12,1)
)

output2_ = builder.spec.description.output.add()
output2_.name = 'raw_confidence'

output2_.type.multiArrayType.MergeFromString(b"")
output2_.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE


output_ = builder.spec.description.output.add()
output_.name = 'raw_coordinates'

output_.type.multiArrayType.MergeFromString(b"")
output_.type.multiArrayType.dataType = ct.proto.Model_pb2.ArrayFeatureType.DOUBLE
# output2_.type.multiArrayType.shape.append(3 * output_size)
# output2_.type.multiArrayType.shape.append(nc)

# builder.add_reshape()
# breakpoint()
adjusted_model = ct.models.MLModel(builder.spec)
adjusted_model.save("adjusted_model.mlmodel")
image = PIL.Image.open("/Users/benjamin/projects/keypoint-tracking/TRAIN_v004_.0599.png").resize((480,480))
outputs = adjusted_model.predict({"input": image})
# exit()
# print(outputs["points"].shape)



nms_spec = ct.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

# nms_spec.description.input.add()
# nms_spec.description.input[0].name="x_points"
# nms_spec.description.input.add()
# nms_spec.description.input[1].name="y_points"


breakpoint()

for i in range(2):
	decoder_output = adjusted_model._spec.description.output[4+i].SerializeToString()
	nms_spec.description.input.add()
	nms_spec.description.input[i].ParseFromString(decoder_output)
	nms_spec.description.output.add()
	nms_spec.description.output[i].ParseFromString(decoder_output)

nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

# output_sizes = [nc, 4]

for i in range(2):
	ma_type = nms_spec.description.output[i].type.multiArrayType
	ma_type.shapeRange.sizeRanges.add()
	# ma_type.shapeRange.sizeRanges[0].lowerBound = 0
	ma_type.shapeRange.sizeRanges[0].upperBound = -1
	ma_type.shapeRange.sizeRanges.add()
	# ma_type.shapeRange.sizeRanges[1].lowerBound = -1 # output_sizes[i]
	ma_type.shapeRange.sizeRanges[1].upperBound = -1 # output_sizes[i]
	del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

nms.iouThreshold = 0.1 #default_iou_threshold
nms.confidenceThreshold = 0.001 # default_confidence_threshold

nms.stringClassLabels.vector.extend(["classes"])

nms_model = ct.models.MLModel(nms_spec)
nms_model.save("NMS.mlmodel")
breakpoint()

from coremltools.models import datatypes
from coremltools.models.pipeline import *
img_size = 480
input_features = [ ("image", datatypes.Array(3, img_size, img_size)),
                   ("iouThreshold", datatypes.Double()),
                   ("confidenceThreshold", datatypes.Double()) ]
output_features = [ "confidence", "coordinates" ]

pipeline = Pipeline(input_features, output_features)

pipeline.add_model(adjusted_model)
pipeline.add_model(nms_model)
breakpoint()
pipeline.spec.description.input[0].ParseFromString(
    adjusted_model._spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(
    nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(
    nms_model._spec.description.output[1].SerializeToString())

pipeline.spec.specificationVersion = 3

final_model = ct.models.MLModel(pipeline.spec)
final_model.save("Pipeline.mlmodel")
exit()


mlmodel = coremltools.models.MLModel("/Users/benjamin/projects/keypoint-tracking/keypoint.mlmodel")
spec = mlmodel._spec
# layers_to_change = []
# layers_to_change.append(-1)
# for x in layers_to_change:
#     del spec.neuralNetwork.layers[x].squeeze.axes[:]
#     print("fixing")
#     spec.neuralNetwork.layers[x].squeeze.axes.extend([0])

from coremltools.models import datatypes
from coremltools.models import neural_network
from coremltools.models.pipeline import *

input_features = [ ("image", datatypes.Array(1, 3, 480, 480)) ]

output_features = [ "coordinates" ]

pipeline = Pipeline(input_features, output_features)
pipeline.add_model(mlmodel)
# pipeline.add_model(decoder_model)
# pipeline.add_model(nms_model)
final_model = coremltools.models.MLModel(pipeline.spec)
final_model.save("pipemodel.mlmodel")
new_model = coremltools.models.MLModel(spec)
new_model.save("w00t.mlmodel")

coremltools.models.neural_network.NeuralNetworkBuilder()
import PIL
image = PIL.Image.open("/Users/benjamin/projects/keypoint-tracking/TRAIN_v004_.0599.png").resize((480,480))
outputs = new_model.predict({"input": image})

exit()
import numpy as np

mlmodel = coremltools.models.MLModel("/Users/benjamin/projects/keypoint-tracking/keypoint.mlmodel")



#iterates through the network layers and identifies the squeeze layers
for i,layer in enumerate(spec.neuralNetwork.layers):
        if "Squeeze" in layer.name:
            print ("Got one")
            layers_to_change.append(i)

#changes the axes to squeeze along the 0 axis which should be 1 dimensional
#in the converted model




# new_model.predict({"input_ids": np.zeros((1, 256), dtype=np.int32)})



# regModel = pipeline.PipelineRegressor.add_model(mlmodel.get_spec())
# regModel

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