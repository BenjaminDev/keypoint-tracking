# SketchPad
A basic deep learning pipeline for keypoint detection.

# dev box setup
```
sudo apt-get install libcairo2-dev libgirepository1.0-dev
python -m pip install -r requirements.txt
```

## Preprocess data
```
python pr
```
## Setup
Make a ssh key pair for accessing the box
```
ssh-keygen -t rsa -f /{some path}/wf_rsa
```
This creates `wf_rsa` and `wf_rsa.pub` Add `wf_rsa.pub` to the keys in genisis cloud.
Add the following to ~/.ssh/config
```
Host wf
 HostName 147.189.195.23
 User ubuntu
 IdentityFile /{some path same as above}/wf_ras
```
Note: that HostName ip changes so look on cloudgenesis for latest one.
## Training
Check the `/experiments/config_1.yaml` for what data sets you using and what batch size.
``
conda activate open-mmlab
cd /mnt/vol_c/code/sketchpad

python train.py
``
## Trace Model
```
conda activate open-mmlab
python trace_model.py -h (takes a while as the imports are a mess and imports the world.)

python trace_model.py --mpath /mnt/vol_c/models/wf/1s2psguu/checkpoints/epoch\=8-step\=728.ckpt --mdir-dst /mnt/vol_c/models/wf/1s2psguu/ --trace
# Note: 1s2psguu is the run id from wand
```
This will create a trace_model.pt that can be converted to a mlmodel on macos.
run this to see results on a test image. if this looks okay copy traced_model.pt to laptop.
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

export video: `ffmpeg -framerate 25 -i 'TRAIN_v004_.0%3d_out.png' -c:v libx264 -pix_fmt yuv420p out.mp4`
Split video `ffmpeg -i {} -r 25 'frame_%3d.png'`

Note: keep pillow on verion 8.2.0 latest is 8.3.0 but has a breaking change.
https://github.com/pytorch/pytorch/issues/61125