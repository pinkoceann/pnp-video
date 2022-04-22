"""

@author: amonod

"""

import numpy as np
import torch
import torch.nn.functional as F

from models.network_drunet import DRUNet
from models.network_fastdvdnet import FastDVDnet


def pytorch_drunet_video_denoiser(video, model, noise_level, model_device, output_device=torch.device("cpu")):
	"""
	pytorch_denoiser
	Inputs:
		video            noisy image / image sequence
		model            DRUNet model
		noise_level      noise level to be used in the input noise map
		output_device           torch.device("cuda:X") or torch.device("cpu")
	Output:
		denoised_video   denoised video
	"""
	assert isinstance(model, DRUNet), 'invalid model specified'

	# image size
	assert len(video.shape) == 5, 'expected a 5D tensor (B, N, C, H, W)'
	B, N, C, H, W = video.shape
	assert C == 1 or C == 3, 'expected third dimension to be the channel dimension (= 1 / 3)'
 
	denoised_video = torch.empty((B, N, C, H, W), device=output_device)

	# pad to fit the 3 U-Net downsamplings if needed
	pad_H, pad_W = 8 - H % 8 if H % 8 else 0, 8 - W % 8 if W % 8 else 0
	padding = (0, pad_W, 0, pad_H)
	H2, W2, = H + pad_H, W + pad_W
	video_pad = F.pad(video.view(B*N, C, H, W), padding, mode='reflect').view(B, N, C, H2, W2)
	noise_map = noise_level * torch.ones((B, 1, H2, W2), device=model_device)
	for i in range(N):
		frame = video_pad[:, i]
		denoised_video[:, i] = model(torch.cat([frame.to(model_device), noise_map], dim=1))[..., :H, :W].to(output_device)

	return denoised_video


def pytorch_fastdvdnet_video_denoiser(video, model, noise_level, model_device, output_device=torch.device("cpu")):
	"""
	pytorch_denoiser
	Inputs:
		video            noisy image / image sequence
		model            FastDVDnet model
		noise_level      noise level to be used in the input noise map
		output_device           torch.device("cuda:X") or torch.device("cpu")
	Output:
		denoised_video   denoised video
	"""
	assert isinstance(model, FastDVDnet), 'invalid model specified'

	# image size
	assert len(video.shape) == 5, 'expected a 5D tensor (B, N, C, H, W)'
	B, N, C, H, W = video.shape
	assert C == 1 or C == 3, 'expected third dimension to be the channel dimension (= 1 / 3)'

	denoised_video = torch.empty((B, N, C, H, W), device=output_device) 

	# pad by reflecting the first and last two images of the video because fastDVDnet takes 5 input frames
	video_pad = torch.empty((B, N + 4, C, H, W), device=video.device)
	video_pad[:, 2:-2] = video
	video_pad[:, :2] = video[:, 1:3].flip((1,))
	video_pad[:, -2:] = video[:, -3:-1].flip((1,))
	# pad to fit the 2 U-Net downsamplings if needed
	pad_H, pad_W = 4 - H % 4 if H % 4 else 0, 4 - W % 4 if W % 4 else 0
	padding = (0, pad_W, 0, pad_H)
	N2, H2, W2, = N + 4,  H + pad_H, W + pad_W
	video_pad = F.pad(video_pad.view(B*N2, C, H, W), padding, mode='reflect').view(B, N2, C, H2, W2)
	noise_map = noise_level * torch.ones((B, 1, H2, W2), device=model_device)
	for i in range(2, N2 - 2):
		# denoise each group of 5 frames
		frame_seq = video_pad[:, i-2: i+2+1].view(B, -1, H2, W2)
		denoised_video[:, i - 2] = model(frame_seq.to(model_device), noise_map)[..., :H, :W].to(output_device)

	return denoised_video

