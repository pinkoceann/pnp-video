# -*- coding: utf-8 -*-
"""

@author: amonod

"""
import torch
from skimage.metrics import structural_similarity

def pytorch_psnr(tensor1, tensor2, data_range = 1.):
	assert torch.equal(torch.tensor(tensor1.shape), torch.tensor(tensor2.shape))
	mse = ((tensor1 - tensor2) ** 2).mean()
	return 10 * torch.log10(data_range ** 2 / mse)


def ssim_video_batch(tensor1, tensor2, data_range=1., set_gaussian_weights=True, sigma=1.5, use_sample_covariance=False):
	B, N, C, H, W = tensor1.shape
	assert (C >= 1 and C <= 3), 'ERROR: expected tensors of shape B, N, C, H, W'
	assert torch.equal(torch.tensor(tensor1.shape), torch.tensor(tensor2.shape))
	np1, np2 = tensor1.numpy(), tensor2.numpy()
	ssim = 0.
	for b in range(B):
		for n in range(N):
			ssim += structural_similarity(np1[b, n], np2[b, n], data_range=data_range, channel_axis=0, sigma=sigma, use_sample_covariance=use_sample_covariance)
	return ssim / (B * N)


def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict


def get_n_params(model):
	pp=0
	for p in list(model.parameters()):
		nn=1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp
