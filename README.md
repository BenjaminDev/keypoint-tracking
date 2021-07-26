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

export video: `ffmpeg -framerate 25 -i 'TRAIN_v004_.0%3d_out.png' -c:v libx264 -pix_fmt yuv420p out.mp4`
Split video `ffmpeg -i {} -r 25 'frame_%3d.png'`

Note: keep pillow on verion 8.2.0 latest is 8.3.0 but has a breaking change.
https://github.com/pytorch/pytorch/issues/61125