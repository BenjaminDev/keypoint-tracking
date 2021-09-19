import torch
from mmpose.models import HRNet
from mmcv import Config, DictAction
from mmpose.models import HRNet
import torch
from mmcv.utils import build_from_cfg
from mmpose.apis import train_model
from torch import nn
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import pytorch_lightning as pl
from mmpose.models.builder import BACKBONES, HEADS, LOSSES, NECKS, POSENETS

def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)

def build_posenet(cfg):
    """Build posenet."""
    return build(cfg, POSENETS)

img = torch.rand(1, 3, 160, 160)


cfg = Config.fromfile("/mnt/vol_c/code/sketchpad/experiments/hrnet_light_config.py")
model = build_posenet(cfg.model)

load_checkpoint(model, "/mnt/vol_c/code/sketchpad/naive_litehrnet_18_coco_256x192.pth", map_location='cpu')
breakpoint()
# mm = nn.Sequential(*[o for o in model.modules()])
# mm.eval()
# breakpoint()
# mm(img, return_loss=False)
features = model.backbone(img)
# features = model.neck(features)
output_heatmap = model.keypoint_head.inference_model(features, flip_pairs=None)
breakpoint()
class WFHRnet(torch.nn.Module):

    def __init__(self):
        super(WFHRnet, self).__init__()
        self.cfg = Config.fromfile("/mnt/vol_c/code/sketchpad/experiments/hrnet_light_config.py")
        model = build_posenet(self.cfg.model)

        load_checkpoint(model, "/mnt/vol_c/code/sketchpad/naive_litehrnet_18_coco_256x192.pth", map_location='cpu')
        breakpoint()
        self.model = nn.Sequential(*[o for o in model.children()])
    def forward(self, x, **kwargs):
        breakpoint()
        return self.model(x, **kwargs)

from typing import Optional
class FootDetector(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.cfg = Config.fromfile("/mnt/vol_c/code/sketchpad/experiments/hrnet_light_config.py")
        model = build_posenet(self.cfg.model)
        load_checkpoint(model, "/mnt/vol_c/code/sketchpad/naive_litehrnet_18_coco_256x192.pth", map_location='cpu')
        self.backbone = [o for o in model.children()][0]
        self.head = [o for o in model.children()][1]

    def forward(self, x):
        # use forward for inference/predictions
        features = self.backbone(x)
        output_heatmap = self.head(features)
        return output_heatmap

fd=FootDetector()
breakpoint()
# img = torch.rand(1, 3, 160, 160)
out = fd(img)
def infer(model,image):
    model.eval()
    results = []
    with torch.no_grad():
        result = model(return_loss=False, **image)
    results.append(result)

    #     # use the first key as main key to calculate the batch size
    #     batch_size = len(next(iter(data.values())))
    #     for _ in range(batch_size):
    #         prog_bar.update()
    # return results


extra = dict(
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4, ),
        num_channels=(64, )),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(32, 64)),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(32, 64, 128)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(32, 64, 128, 256)))

model = HRNet(extra=extra)
inputs = torch.rand(1, 3, 32, 32)
level_outputs = model.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))

# model = torch.load("/mnt/vol_c/code/sketchpad/naive_litehrnet_18_coco_256x192.pth")
from mmcv.runner import get_dist_info, init_dist, load_checkpoint