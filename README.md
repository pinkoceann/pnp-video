# pnp_video

Code associated to the contribution **Video Restoration with a Deep Plug-and-Play Prior**

### Requirements
  - conda
  - linux
  - NVIDIA GPU with CUDA

### Installation

```conda env create -f pnp_video_conda_env.yml```

Download DRUNet pretrained weights (too large to be stored in this repo) at [https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing) and store `drunet_color.pth` file in `pretrained_models/` folder

### Additional data

Full DAVIS 2017 test set downloadable at [https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip)
