# SketchPad
A basic deep learning pipeline for keypoint detection.

# Dev box setup
__System Requirements:__
1. OS 20.04-Ubuntu
2. Python 3.8 or later
3. Working nvidia drivers. Run `nvidia-smi` to check.
4. Weights and Biases accunt. You'll need to sign up [here](https://wandb.ai/login?signup=true)

__Python Env Setup:__

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
2. `conda create -n kpd python=3.8`
3. `conda activate kpd`
4. `(kpd) conda env update --name kpd --file environment.yml`
5. `(kpd) conda install pip`
6. `(kpd) python -c "import torch; print(torch.__version__)"` # check torch version
7. `(kpd) conda list cudatoolkit` # check cuda version. These might differ in nvidia drivers forced your hand. depends on cc
5. `(kpd) pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html`


## Setup for vscode
Make a ssh key pair for accessing the box
```
ssh-keygen -t rsa -f /{some path}/wf_rsa
```
This creates `wf_rsa` and `wf_rsa.pub` Add `wf_rsa.pub` to the keys in genisis cloud.
Add the following to ~/.ssh/config
```
Host wf
 HostName {ip address}
 User ubuntu
 IdentityFile /{some path same as above}/wf_ras
```

## gc-sharp-goldstine box config
1. `conda activate open-mmlab`


## Training
Check the `/experiments/config_1.yaml` for what data sets you using and what batch size.
```
conda activate kpd
cd /mnt/vol_c/code/network

python train.py
```
## Trace Model
```
conda activate kpd
python trace_model.py -h (takes a while as the imports are a mess and imports the world.)

python trace_model.py --mpath /mnt/vol_c/models/wf/1s2psguu/checkpoints/epoch\=8-step\=728.ckpt --mdir-dst /mnt/vol_c/models/wf/1s2psguu/ --trace
# Note: 1s2psguu is the run id from wand
```
This will create a trace_model.pt that can be converted to a mlmodel on macos. Check `manual_decode_out.png`. If this looks okay copy traced_model.pt to laptop to export model.
```
code /mnt/vol_c/models/wf/1s2psguu/manual_decode_out.png
```

## Export model to coreml
Copy `traced_model.pt` from the server.
```
scp wf:/mnt/vol_c/models/wf/{run_id}/traced_model.pt .
```

```
python export_model.py --mpath {path to}/traced_model.pt --mdir-dst {path to folder of outputs} --no-trace
```
Copy the `Pipeline.ml` to the SolvePNP App and see what happens - hope it's good ;)

Note: keep pillow on verion 8.2.0 latest is 8.3.0 but has a breaking change.
https://github.com/pytorch/pytorch/issues/61125