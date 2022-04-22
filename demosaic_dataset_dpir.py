"""

@author: amonod
adapted from https://github.com/cszn/DPIR

"""
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import cv2
import json
from thop.utils import clever_format

from models.network_drunet import DRUNet
from image_datasets import videoDataset
from utils import pytorch_psnr, ssim_video_batch, get_n_params
from video_utils import tensor_to_images

from zhang_utils import utils_mosaic, utils_pnp, utils_image
from demosaic_dataset import mosaic_mask_video
from deblur_dataset_dpir import pytorch_drunet_image_denoiser


def demosaic_video_dataset(model, dataloader, x8=False, init_type="matlab", noise_level=1./255, max_denoiser_level=49./255, max_iters=24, save_frames=False, output_folder=None, device=torch.device("cpu"), verbose=1):
	# set PnP-HQS parameters
	rhos, sigmas = utils_pnp.get_rho_sigma(sigma=max(0.255/255., noise_level), iter_num=max_iters, modelSigma1=max_denoiser_level*255, modelSigma2=max(0.6, noise_level*255.), w=1)
	rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
	if verbose >= 2:
		print(f"denoiser range = [{sigmas[0]:.3f}, {sigmas[-1]:.3f}], sigma={noise_level} => rho range = [{rhos[0]:.3f}, {rhos[-1]:.3f}]")

	vid_names, psnrs_noisy, psnrs_out, ssims_noisy, ssims_out, runtimes = [], [], [], [], [], []
	with torch.no_grad():
		for b, batch in enumerate(dataloader):
				video = batch['video'].to(device)
				B, N, C, H, W = video.shape
				assert B == 1, "this code only works with batch size 1 for now"
				# pad to avoid missing pixels when downscaling
				pad_H, pad_W = 2 - H % 2 if H % 2 else 0, 2 - W % 2 if W % 2 else 0
				padding = (0, pad_W, 0, pad_H)
				video = F.pad(video.view(B*N, C, H, W), padding, mode='reflect').view(B, N, C, H + pad_H, W + pad_W)
				H, W, = H + pad_H, W + pad_W

				# generate the degraded video
				degraded_video, video_mask, bayer_flat, bayer_stack = mosaic_mask_video(video, noise_level, pattern="RGGB")

				# initialize with crudely demosaicked degraded image
				init = torch.empty_like(video)
				if init_type == "matlab":  # matlab demosaicing for initialization
					for n in range(N):
						init[:, n] = utils_mosaic.dm_matlab(bayer_stack[:, n].permute(0, 3, 1, 2))
				elif init_type == "cv2":
					for b in range(B):
						for n in range(N):
							_ = 255 * bayer_flat[b, n].cpu().clamp(0., 1.).numpy().astype(np.uint8)
							# print(f"x shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
							_ = cv2.cvtColor(_, cv2.COLOR_BAYER_BG2RGB_EA)
							# print(f"x shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
							init[b, n] = 1./255 * torch.from_numpy(_).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
				elif init_type == "mosaic":
					# initialize with the degraded video
					init = degraded_video
				restored_video = torch.empty_like(video)

				t0 = time.time()
				for n, img in enumerate(degraded_video[0]):  # for each image
					if verbose >= 3:
						print(f"processing image {n+1}/{N}")

					y = img.unsqueeze(0)  # observation: mosaicked image
					mask = video_mask[:, n]
					x = init[:, n]

					for i in range(max_iters):
						# --------------------------------
						# step 1, closed-form solution
						# --------------------------------

						x = (mask*y+rhos[i].float()*x).div(mask+rhos[i])

						# --------------------------------
						# step 2, denoiser
						# --------------------------------
						if x8:
							x = utils_image.augment_img_tensor4(x, i % 8)
						x = pytorch_drunet_image_denoiser(x, model, sigmas[i], model_device=device, output_device=device)
						if x8:
							if i % 8 == 3 or i % 8 == 5:
								x = utils_image.augment_img_tensor4(x, 8 - i % 8)
							else:
								x = utils_image.augment_img_tensor4(x, i % 8)
					restored_video[0, n] = x[0]

				t_forward = time.time() - t0

				# remove padding and compute psnr
				H, W, = H - pad_H, W - pad_W
				video, degraded_video, init, restored_video, = video[..., :H, :W].cpu(), degraded_video[..., :H, :W].cpu(), init[..., :H, :W].cpu(), restored_video[..., :H, :W].cpu()
				psnr_noisy = pytorch_psnr(video, degraded_video)
				psnr_out = pytorch_psnr(video, restored_video)
				ssim_noisy = ssim_video_batch(video, degraded_video)
				ssim_out = ssim_video_batch(video, restored_video, data_range=1.)
				runtime = t_forward / (B * N)
				if verbose >= 2:
					print(f"video: {batch['video_name'][0]:<18} PSNR/SSIM noisy: {psnr_noisy:<2.2f}/{ssim_noisy:.4f}, PSNR/SSIM out: {psnr_out:<2.2f}/{ssim_out:.4f} \t runtime: {runtime:.3f}s/frame")
				vid_names.append(str(batch['video_name'][0]))
				psnrs_noisy.append(psnr_noisy)
				psnrs_out.append(psnr_out)
				ssims_noisy.append(ssim_noisy)
				ssims_out.append(ssim_out)
				runtimes.append(runtime)
				if save_frames:
					image_folder = os.path.join(output_folder, "images")
					if not os.path.exists(image_folder):
						os.makedirs(image_folder)
					vid_name = batch['video_name'][0]
					denoiser_name = model.__class__.__name__
					tensor_to_images(video.squeeze(), f'{vid_name}_clean', os.path.join(image_folder, f"{vid_name}_clean"))
					tensor_to_images(degraded_video.squeeze(), f'{vid_name}_mosaicked', os.path.join(image_folder, f"{vid_name}_mosaicked_RGGB_sigma{noise_level*255}"))
					tensor_to_images(init.squeeze(), f'{vid_name}_init', os.path.join(image_folder, f"{vid_name}_init_RGGB_sigma{noise_level*255}_{init_type}_init"))
					tensor_to_images(restored_video.squeeze(), f'{vid_name}_restored', os.path.join(image_folder, f"{vid_name}_mosaicked_RGGB_sigma{noise_level*255}_{denoiser_name}_s{int(max_denoiser_level*255)}_DPIR_{max_iters}iters_restored"))
	avg_psnr_noisy = torch.Tensor(psnrs_noisy).mean()
	avg_ssim_noisy = torch.Tensor(ssims_noisy).mean()
	avg_psnr_out = torch.Tensor(psnrs_out).mean()
	avg_ssim_out = torch.Tensor(ssims_out).mean()
	avg_runtime = torch.Tensor(runtimes).mean()

	if verbose >= 1:
		print(f'DPIR model: {model.__class__.__name__:<18} PSNR/SSIM noisy: {avg_psnr_noisy:<2.2f}/{avg_ssim_noisy:.4f}, PSNR/SSIM out: {avg_psnr_out:<2.2f}/{avg_ssim_out:.4f} \t runtime: {avg_runtime:.3f}s/frame\n')

	return vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime


def main(**args):
	t_init = time.time()

	if args['deterministic']:
		seed = 47
		np.random.seed(seed)
		cudnn.benchmark = False
		cudnn.deterministic = True
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.use_deterministic_algorithms(True)

	gpu = args['gpu']
	device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
	print(f"selected device: {device}")

	model_list, path_list = [], []

	for denoiser in args['denoisers']:
		if 'drunet' == denoiser:
			path_list.append("pretrained_models/drunet_color.pth")
			model_list.append(DRUNet())
	tf = transforms.CenterCrop(args['centercrop']) if args['centercrop'] > 0 else None
	dataset = videoDataset(args['dataset_path'], extension=args['extension'], nested_subfolders=args['dataset_depth'], transform=tf, max_video_length=args['max_frames'])
	print(f'created a dataset of {len(dataset)} videos.')
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	out_folder = args['logdir']
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	out_filename = os.path.join(out_folder, f"demosaicking_{args['dataset_name']}_")
	out_filename += "DPIR_"
	for denoiser in args['denoisers']:
		out_filename += denoiser + '_'
	out_filename += "s_"
	for dl in args['max_denoiser_levels']:
		out_filename += str(dl) + '_'
	out_filename += "sigmas_"
	for sigma in args['sigmas']:
		out_filename += str(sigma) + '_'
	out_filename += str(args['max_iters']) + '_iters'
	out_filename += '.json'

	res_dict = {'args': args}

	for i, current_model in enumerate(model_list):
		num_params = clever_format(get_n_params(current_model), "%.4f")
		print(f'Evaluating {current_model.__class__.__name__:<18} number of parameters = {num_params}')
		# instantiate model
		model = current_model.to(device)
		# load trained weights
		model.load_state_dict(torch.load(path_list[i], map_location=device))
		model.eval()
		res_dict[model.__class__.__name__] = {'params': num_params}
		for s in args['max_denoiser_levels']:
			print(f'denoiser std={s}')
			res_dict[model.__class__.__name__][f"s={s}"] = {}
			for sigma in args['sigmas']:
				print(f'sigma={sigma}')
				vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime = demosaic_video_dataset(model, dataloader, x8=args['x8'], init_type=args['init'], noise_level=sigma/255., max_denoiser_level=s/255., max_iters=args['max_iters'], save_frames=args['save_frames'], output_folder=args['logdir'], device=device, verbose=args['verbose'])
				res_dict[model.__class__.__name__][f"s={s}"][f"sigma={sigma}"] = {'avg_psnr_noisy': float(avg_psnr_noisy), 'avg_ssim_noisy': float(avg_ssim_noisy), 'avg_psnr_out': float(avg_psnr_out), 'avg_ssim_out': float(avg_ssim_out), 'avg_runtime': float(avg_runtime)}
				for v, vid_name in enumerate(vid_names):
					res_dict[model.__class__.__name__][f"s={s}"][f"sigma={sigma}"][vid_name] = {'psnr_noisy': float(psnrs_noisy[v]), 'ssim_noisy': float(ssims_noisy[v]), 'psnr_out': float(psnrs_out[v]), 'ssim_out' : float(ssims_out[v]), 'runtime' : runtimes[v]}
				torch.cuda.empty_cache()
				# save res dict frequently just in case
				with open(out_filename, 'w') as handle:
					json.dump(res_dict, handle, indent=4)

		del model

	print(f"\nresults file location: {out_filename}")

	t_final = time.time() - t_init
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(t_final))))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Run PnP restoration of a video dataset")
	# Program-specific arguments
	parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='flag to ensure full reproductibility')
	parser.add_argument('--no-deterministic', dest='deterministic', action='store_false')
	parser.set_defaults(deterministic=True)
	parser.add_argument('--verbose', type=int, default=1, help='verbose level (0 to 3)')
	parser.add_argument('--gpu', dest='gpu', action='store_true', help='run the code on GPU instead of CPU')
	parser.add_argument('--cpu', dest='gpu', action='store_false', help='run the code on CPU instead of GPU')
	parser.set_defaults(gpu=True)
	parser.add_argument("--logdir", type=str, default='./demosaicking_results', help="path to the folder containing the output results")
	parser.add_argument("--save_frames", action='store_true', help="save videos as images")
	# Model parameters
	parser.add_argument("--denoisers", type=str, nargs='+', default=['drunet'], help="selected model ('drunet')")
	parser.add_argument("--max_denoiser_levels", type=float, nargs='+', default=[49], help="maximum noise level applied to the CNN denoiser (between 0 and 255)")
	parser.add_argument('--x8', action='store_true', help='use geometric self-ensemble (flip / rotate input before denoising in one of 8 different ways each iteration)')
	# data parameters
	parser.add_argument("--dataset_path", type=str, default='./data/subset_4', help="path to the folder of the video dataset")
	parser.add_argument("--dataset_name", type=str, default='davis_subset4', help="name of the dataset")
	parser.add_argument("--dataset_depth", type=int, default=1, help="number of nested subfolders in the dataset")
	parser.add_argument("--extension", type=str, default='.jpg', help="file extension ('.jpg' / '.png')")
	parser.add_argument("--centercrop", type=int, default=-1, help="center crop size if any (-1 => full res)")
	parser.add_argument("--max_frames", type=int, default=-1, help="maximum number of frames per video to load (-1 => load all frames)")
	# pnp-admm parameters
	parser.add_argument("--max_iters", type=int, default=40, help="maximum number of pnp-hqs iterations")
	parser.add_argument("--sigmas", type=float, nargs='+', default=[5], help="noise level of the extra AWGN applied during image degradation (between 0 and 255)")
	parser.add_argument("--init", type=str, default='matlab', help="init type ('matlab' / 'cv2' / 'mosaic')")
	argspar = parser.parse_args()

	print("\n### Running DPIR for video demosaicking ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
