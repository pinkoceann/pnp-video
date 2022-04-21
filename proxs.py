# -*- coding: utf-8 -*-
"""

@authors: rlaumont, amonod

"""

import numpy as np
import torch
from zhang_utils import utils_sisr

def prox_denoising(x, im_b, sigma, gamma):
	"""
	Proximal Operator for Gaussian denoising:
	f(x) = || x - y ||^2 / (2 sigma^2)
	prox_{gamma f} (x[i]) = (x[i] + y[i]*gamma/sigma^2)/(1+gamma/sigma^2)
	
	Parameters:
		:x - the argument to the proximal operator.
		:im_b - the noisy observation.
		:sigma - the standard deviation of the gaussian noise in y.
		:gamma -  the regularization parameter.
	"""
	s_2 = sigma**2
	a = gamma/s_2
	
	return (a*im_b + x)/(a+1)

def prox_inpainting(x, im_b, mask, sigma, gamma):
	"""
	Proximal Operator for Gaussian inpainting:
	f(x) = || M*x - y ||^2 / (2 sigma^2)
	prox_{gamma f} (x[i]) = (x[i] + y[i]*gamma/sigma^2)/(1+gamma/sigma^2) if M[i]==1
						  = x[i]                                          if M[i]==0
	Parameters:
		:x - the argument to the proximal operator.
		:im_b - the noisy observation.
		:mask - binary image of the same size as x
		:sigma - the standard deviation of the gaussian noise in y.
		:gamma - the regularization parameter.
	"""
	
	if sigma==0:
	  tmp = im_b*mask  + (1-mask)*x
	elif sigma!=0:
	  s_2 = sigma**2
	  a = gamma/s_2
	  tmp = ((a*im_b + x)/(a+1))*mask + (1-mask)*x
	return tmp 


def prox_deblurring(x, im_b, h, sigma, gamma):
	"""
	Proximal Operator for Gaussian deblurring:
	f(x) = || A.x - im_b ||^2 / (2 sigma^2) 
	avec A.x = h*x
	prox_{gamma f} (x[i]) = 1/(1+gamma/sigma^2*h_fft*hc_fft)*(gamma/sigma^2 A.T y[i] +x[i]) 
	Parameters:
		:x - the argument to the proximal operator.
		:im_b - the noisy observation.
		:h_fft - FFT2 of the blurring kernel.
		:sigma - the standard deviation of the gaussian noise in y.
		:gamma - the regularization parameter.
	"""

	s2 = max(0.255/255., sigma)**2  # trick to avoid division by 0 in noiseless case
	a = gamma/s2
	
	h_fft = torch.fft.fft2(h)
	hc_fft = torch.conj(h_fft)
	
	X = a*torch.real(torch.fft.ifft2(hc_fft*torch.fft.fft2(im_b))) + x
	
	return torch.real(torch.fft.ifft2(torch.fft.fft2(X)/(a*h_fft*hc_fft + 1))).float()

def prox_sr(x, FB, FBC, F2B, FBFy, sf, sigma, gamma):
	a = gamma * max(0.255/255., sigma)**2  # trick to avoid division by 0 in noiseless case
	px = utils_sisr.data_solution_video(x, FB, FBC, F2B, FBFy, a, sf)
	return px
