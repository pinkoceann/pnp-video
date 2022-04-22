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
import hdf5storage
import json
from thop.utils import clever_format

from models.network_drunet import DRUNet
from image_datasets import videoDataset
from utils import pytorch_psnr, ssim_video_batch, get_n_params
from video_utils import tensor_to_images
from skimage.filters import gaussian

from zhang_utils import utils_sisr, utils_pnp, utils_image, utils_model
from deblur_dataset import blur_video


def pytorch_drunet_image_denoiser(image, model, noise_level, model_device, output_device=torch.device("cpu")):
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
	assert len(image.shape) == 4, 'expected a 4D tensor (B, C, H, W)'
	B, C, H, W = image.shape
	assert C == 1 or C == 3, 'expected third dimension to be the channel dimension (= 1 / 3)'

	denoised_image = torch.empty((B, C, H, W), device=output_device)

	# pad to fit the 3 U-Net downsamplings if needed
	pad_H, pad_W = 8 - H % 8 if H % 8 else 0, 8 - W % 8 if W % 8 else 0
	padding = (0, pad_W, 0, pad_H)
	H2, W2, = H + pad_H, W + pad_W
	image_pad = F.pad(image, padding, mode='reflect')
	noise_map = noise_level * torch.ones((B, 1, H2, W2), device=model_device)
	input = torch.cat([image_pad.to(model_device), noise_map], dim=1)
	output = utils_model.test_mode(model, input, mode=2, refield=32, min_size=256, modulo=16)
	denoised_image = output[..., :H, :W].to(output_device)

	return denoised_image


def deblur_video_dataset(model, dataloader, kernel="randomLevin", x8=False, noise_level=2.55/255, max_denoiser_level=49./255, max_iters=24, save_frames=False, output_folder=None, device=torch.device("cpu"), verbose=1):
	# set PnP-HQS parameters
	rhos, sigmas = utils_pnp.get_rho_sigma(sigma=max(0.255/255., noise_level), iter_num=max_iters, modelSigma1=max_denoiser_level*255, modelSigma2=noise_level*255., w=1)
	rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)
	if verbose >= 2:
		print(f"denoiser range = [{sigmas[0]:.3f}, {sigmas[-1]:.3f}], sigma={noise_level} => rho range = [{rhos[0]:.3f}, {rhos[-1]:.3f}]")

	# set the blur kernel
	if kernel == "uniform":
		kernel_size = 9
		ker = torch.ones((kernel_size, kernel_size), device=device) / (kernel_size**2)
	elif kernel == "gaussian3":
		a = np.zeros((9, 9))
		a[4, 4] = 1
		ker = torch.from_numpy(gaussian(a, sigma=3))
	elif kernel == "gaussian5":
		a = np.zeros((11, 11))
		a[5, 5] = 1
		ker = torch.from_numpy(gaussian(a, sigma=5))
	elif kernel == "gaussian25-1.6":
		a = np.zeros((25, 25))
		a[12, 12] = 1
		ker = torch.from_numpy(gaussian(a, sigma=1.6))
	elif kernel == "randomLevin":
		kernels = hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']

	vid_names, psnrs_noisy, psnrs_out, ssims_noisy, ssims_out, runtimes = [], [], [], [], [], []
	with torch.no_grad():
		for b, batch in enumerate(dataloader):
				video = batch['video'].to(device)
				B, N, C, H, W = video.shape
				assert B == 1, "this code only works with batch size 1 for now"

				# generate the blur
				if kernel == "randomLevin":
					blur = torch.zeros(B, N, 1, H, W, device=device)
					# randomly select one of the 8 blur kernels for each image
					per_frame_kernel_indexes, kernel_list = [], []
					for _ in range(N):  # for each frame
						ker_idx = torch.randint(0, kernels.shape[1], (1,))
						per_frame_kernel_indexes.append(ker_idx)
						ker = torch.from_numpy(kernels[0, ker_idx])
						kernel_list.append(ker)
					# blur = torch.zeros(H, W, device=device)
						li, lj = ker.shape[0], ker.shape[1]
						ci, cj = li // 2, lj // 2
						blur[:, _, :, :ci + 1, :cj + 1] = ker[ci:, cj:]
						blur[:, _, :, :ci + 1, -cj:] = ker[ci:, :cj]
						blur[:, _, :, -ci:, :cj + 1] = ker[:ci, cj:]
						blur[:, _, :, -ci:, -cj:] = ker[:ci, :cj]
				else:
					blur = torch.zeros(H, W, device=device)
					li, lj = ker.shape[0], ker.shape[1]
					ci, cj = li // 2, lj // 2
					blur[:ci + 1, :cj + 1] = ker[ci:, cj:]
					blur[:ci + 1, -cj:] = ker[ci:, :cj]
					blur[-ci:, :cj + 1] = ker[:ci, cj:]
					blur[-ci:, -cj:] = ker[:ci, :cj]
					blur = blur.view(1, 1, 1, H, W)
				blur_fft = torch.fft.fft2(blur)

				# generate the degraded video
				degraded_video = blur_video(video, blur_fft, noise_level)
				init = degraded_video.clone()  # initialize with the degraded video
				restored_video = torch.empty_like(video)

				t0 = time.time()
				for n, img in enumerate(degraded_video[0]):  # for each image
					if verbose >= 3:
						print(f"processing image {n+1}/{N}")
					img_L_tensor = img.unsqueeze(0)
					x = init[0, n].unsqueeze(0)
					k_tensor = kernel_list[n].float().unsqueeze(0).unsqueeze(0).to(device)
					FB, FBC, F2B, FBFy = utils_sisr.pre_calculate(img_L_tensor, k_tensor, sf=1)
					for i in range(max_iters):
						# --------------------------------
						# # step 1: data fidelity term
						# --------------------------------
						tau = rhos[i].float().repeat(1, 1, 1, 1)
						x = utils_sisr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf=1)
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
						# print(f"x shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
					# print(restored_video[0, n].shape)
					restored_video[0, n] = x[0]

				t_forward = time.time() - t0

				# compute psnr
				video, degraded_video, restored_video = video.cpu(), degraded_video.cpu(), restored_video.cpu()
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
					tensor_to_images(degraded_video.squeeze(), f'{vid_name}_blurred', os.path.join(image_folder, f"{vid_name}_blurred_kernel_{kernel}_sigma{noise_level*255}"))
					tensor_to_images(restored_video.squeeze(), f'{vid_name}_restored', os.path.join(image_folder, f"{vid_name}_blurred_kernel_{kernel}_sigma{noise_level*255}_{denoiser_name}_s{int(max_denoiser_level*255)}_DPIR_{max_iters}iters_restored"))
	avg_psnr_noisy = torch.Tensor(psnrs_noisy).mean()
	avg_ssim_noisy = torch.Tensor(ssims_noisy).mean()
	avg_psnr_out = torch.Tensor(psnrs_out).mean()
	avg_ssim_out = torch.Tensor(ssims_out).mean()
	avg_runtime = torch.Tensor(runtimes).mean()

	if verbose >= 1:
		print(f'DPIR model: {model.__class__.__name__:<18} PSNR/SSIM noisy: {avg_psnr_noisy:<2.2f}/{avg_ssim_noisy:.4f}, PSNR/SSIM out: {avg_psnr_out:<2.2f}/{avg_ssim_out:.4f} \t runtime: {avg_runtime:.3f}s/frame\n')

	if kernel == "randomLevin":
		ker = []
		for k in kernels[0]:
			ker.append(k.tolist())
	else:
		ker = ker.cpu().numpy().tolist()

	return vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime, ker


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
	out_filename = os.path.join(out_folder, f"deblurring_{args['dataset_name']}_")
	out_filename += "DPIR_"
	for denoiser in args['denoisers']:
		out_filename += denoiser + '_'
	out_filename += "s_"
	for dl in args['max_denoiser_levels']:
		out_filename += str(dl) + '_'
	out_filename += "kernel_" + args['kernel'] + '_'
	out_filename += "sigmas_"
	for sigma in args['sigmas']:
		out_filename += str(sigma) + '_'
	out_filename += str(args['max_iters']) + '_iters'
	out_filename += '.json'

	res_dict = {'args': args}
	no_kernel = True

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
				vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime, kernel = deblur_video_dataset(model, dataloader, kernel=args['kernel'], x8=args['x8'], noise_level=sigma/255., max_denoiser_level=s/255., max_iters=args['max_iters'], save_frames=args['save_frames'], output_folder=args['logdir'], device=device, verbose=args['verbose'])
				if no_kernel:
					res_dict['kernel'] = kernel
					no_kernel = False
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
	parser.add_argument("--logdir", type=str, default='./sr_results', help="path to the folder containing the output results")
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
	parser.add_argument("--max_iters", type=int, default=8, help="maximum number of pnp-hqs iterations")
	parser.add_argument("--sigmas", type=float, nargs='+', default=[2.55], help="noise level of the extra AWGN applied during image degradation (between 0 and 255)")
	parser.add_argument("--kernel", type=str, default='randomLevin', help="blur kernel (randomLevin / uniform-9 / gaussian-9-3 / gaussian11-5 / gaussian25-1.6)")
	argspar = parser.parse_args()

	print("\n### Running DPIR for video deblurring ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
