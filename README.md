# pnp-video

Code associated to the contribution [**Video Restoration with a Deep Plug-and-Play Prior**](https://arxiv.org/abs/2209.02854)

### Requirements
  - conda
  - linux
  - NVIDIA GPU with CUDA

### Installation

```conda env create -f pnp_video_conda_env.yml```

Download DRUNet pretrained weights (too large to be stored in this repo) at [https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing) and store `drunet_color.pth` file in `pretrained_models/` folder

### Additional data

Full DAVIS 2017 test set downloadable at [https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip)

## Video Results

**Comparison videos** of the experiments described in the article are downloadable at [https://drive.google.com/drive/folders/1f-YOjAopNj2QPkOiXsRpY8QKcUqAuZft?usp=sharing](https://drive.google.com/drive/folders/1f-YOjAopNj2QPkOiXsRpY8QKcUqAuZft?usp=sharing) (to be replaced by a different solution and/or a Youtube link)
  - deblurring
  - super-resolution
  - demosaicking
  - interpolation of random missing pixels
 
 Download the .mp4 files and look at the videos locally for minimal compression. Feel free to view the sequences at a lower framerate if needed.

## Citation

If you find this work useful in your research or publication, please cite our paper:
```
@article{arxiv.2209.02854,
    title = {Video Restoration with a Deep Plug-and-Play Prior}, 
    author = {Monod, Antoine and Delon, Julie and Tassano, Matias and Almansa, Andrés},
    year = {2022},
    doi = {10.48550/ARXIV.2209.02854},
    url = {https://arxiv.org/abs/2209.02854}  
}
```
