"""

@author: amonod

"""
import torch
import numpy as np

from utils import pytorch_psnr


def torch_admm(init, clean, proxF, proxG, max_iters, print_frequency=10, verbose=False):
	
	psnrs_x = torch.empty((max_iters), device=init.device)
	psnrs_z = torch.empty_like(psnrs_x)
	x_grads, z_grads, x_minus_zs, denoiser_residual = torch.empty_like(psnrs_x), torch.empty_like(psnrs_x), torch.empty_like(psnrs_x), torch.empty_like(psnrs_x)
	# initialization
	z = init.clone()
	x = torch.zeros_like(z)
	u = torch.zeros_like(x)
	for i in range(max_iters):
		x_old = x.clone()
		z_old = z.clone()
		u_old = u.clone()
		x = proxF(z - u)
		z = proxG(x + u)
		u = (u + x - z).clone()
		psnrs_x[i] = pytorch_psnr(x, clean, data_range=1.0)
		psnrs_z[i] = pytorch_psnr(z, clean, data_range=1.0)
		x_grads[i] = ((x - x_old)**2).mean().sqrt()
		z_grads[i] = ((z - z_old)**2).mean().sqrt()  # dual residual
		x_minus_zs[i] = ((x - z)**2).mean().sqrt()  # denoiser residual
		denoiser_residual[i] = ((z - (x + u_old))**2).mean().sqrt()
		if verbose and (i == 0 or(i + 1) % max(1, (max_iters // print_frequency)) == 0):
			print(f"iteration k={i+1: <4}/{max_iters} \t psnr x = {psnrs_x[i]:<6.2f}, psnr z = {psnrs_z[i]:<6.2f} \t ||x_k - x_{{k-1}}|| = {x_grads[i]:.2e} \t ||z_k - z_{{k-1}}|| = {z_grads[i]:.2e} \t ||x_k - z_k|| = {x_minus_zs[i]:.2e} \t ||z_k - (x_k + u_{{k-1}})|| = {denoiser_residual[i]:.2e}")
	return x, psnrs_x, psnrs_z, x_grads, z_grads, x_minus_zs, denoiser_residual


def torch_admm_annealing(init, clean, proxF, proxG, gammaF_range, gammaG_range, max_iters, print_frequency=10, verbose=False):

	psnrs_x = torch.empty((max_iters), device=init.device)
	psnrs_z = torch.empty_like(psnrs_x)
	x_grads, z_grads, x_minus_zs, denoiser_residual = torch.empty_like(psnrs_x), torch.empty_like(psnrs_x), torch.empty_like(psnrs_x), torch.empty_like(psnrs_x)
	# initialization
	z = init.clone()
	x = torch.zeros_like(z)
	u = torch.zeros_like(x)
	for i in range(max_iters):
		x_old = x.clone()
		z_old = z.clone()
		u_old = u.clone()
		x = proxF(z - u, gammaF_range[i])
		z = proxG(x + u, gammaG_range[i])
		u = (u + x - z).clone()
		psnrs_x[i] = pytorch_psnr(x, clean, data_range=1.0)
		psnrs_z[i] = pytorch_psnr(z, clean, data_range=1.0)
		x_grads[i] = ((x - x_old)**2).mean().sqrt()
		z_grads[i] = ((z - z_old)**2).mean().sqrt()  # dual residual
		x_minus_zs[i] = ((x - z)**2).mean().sqrt()  # denoiser residual
		denoiser_residual[i] = ((z - (x + u_old))**2).mean().sqrt()
		if verbose and (i == 0 or(i + 1) % max(1, (max_iters // print_frequency)) == 0):
			print(f"iteration k={i+1: <4}/{max_iters} \t psnr x = {psnrs_x[i]:<6.2f}, psnr z = {psnrs_z[i]:<6.2f} \t gammaF = {gammaF_range[i]}, gammaG = {gammaG_range[i]} \t ||x_k - x_{{k-1}}|| = {x_grads[i]:.2e} \t ||z_k - z_{{k-1}}|| = {z_grads[i]:.2e} \t ||x_k - z_k|| = {x_minus_zs[i]:.2e} \t ||z_k - (x_k + u_{{k-1}})|| = {denoiser_residual[i]:.2e}")
	return x, psnrs_x, psnrs_z, x_grads, z_grads, x_minus_zs, denoiser_residual
