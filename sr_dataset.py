import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json
import cv2
from thop.utils import clever_format

from models.network_drunet import DRUNet
from models.network_fastdvdnet import FastDVDnet
from image_datasets import videoDataset
from utils import pytorch_psnr, ssim_video_batch, get_n_params
from video_utils import tensor_to_images
from graph_utils import save_graph, save_graph2
from skimage.filters import gaussian

from video_denoiser import pytorch_dncnn_video_denoiser, pytorch_ffdnet_video_denoiser, pytorch_drunet_video_denoiser, pytorch_fastdvdnet_video_denoiser
from proxs import prox_sr
from torch_optim import torch_admm
from zhang_utils import utils_sisr


def lr_video(video, kernel_fft, sf, noise_level=5./255):
	video_fft = torch.fft.fft2(video)
	# generate the blurry video
	blurry_video = torch.real(torch.fft.ifft2(kernel_fft*video_fft)).float()
	# downsample
	degraded_video = blurry_video[..., 0::sf, 0::sf]
	# generate the noise
	if noise_level > 0:
		noise = torch.normal(mean=torch.zeros_like(degraded_video), std=noise_level)
		degraded_video = degraded_video + noise
	return degraded_video


def sr_video_dataset(model, dataloader, kernel="gaussian25-1.6", scale_factor=2, noise_level=0./255, admm_alpha=0.025, admm_denoiser_level=20./255, admm_iters=20, save_frames=False, save_graphs=False, output_folder=None, device=torch.device("cpu"), verbose=1):

	alpha_epsilon = admm_alpha / admm_denoiser_level**2
	if verbose >= 2:
		print(f"admm_denoiser_level = {admm_denoiser_level}, admm_alpha = {admm_alpha} => alpha/epsilon = {alpha_epsilon:.5f}")

	# define the deep denoiser Ds as plug and play prior
	if isinstance(model, DRUNet):
		Ds = lambda x: pytorch_drunet_video_denoiser(x, model, admm_denoiser_level, model_device=device, output_device=device)
	elif isinstance(model, FastDVDnet):
		Ds = lambda x: pytorch_fastdvdnet_video_denoiser(x, model, admm_denoiser_level, model_device=device, output_device=device)


	# set the blur kernel
	if kernel=="uniform-9":
		kernel_size = 9
		ker = torch.ones((kernel_size, kernel_size), device=device) / (kernel_size**2)
	elif kernel=="gaussian-9-3":
		a = np.zeros((9,9))
		a[4,4] = 1
		ker = torch.from_numpy(gaussian(a, sigma=3))
	elif kernel=="gaussian11-5":
		a = np.zeros((11,11))
		a[5,5] = 1
		ker = torch.from_numpy(gaussian(a, sigma=5))
	elif kernel=="gaussian25-1.6":
		a = np.zeros((25,25))
		a[12,12] = 1
		ker = torch.from_numpy(gaussian(a, sigma=1.6))
	elif kernel=="gaussian25-3.2":
		a = np.zeros((25,25))
		a[12,12] = 1
		ker = torch.from_numpy(gaussian(a, sigma=3.2))

	vid_names, psnrs_noisy, psnrs_out, ssims_noisy, ssims_out, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, runtimes = [], [], [], [], [], [], [], [], [], [], [], []
	with torch.no_grad():
		for b, batch in enumerate(dataloader):
				video = batch['video'].to(device)
				B, N, C, H, W = video.shape
				# pad to avoid missing pixels when downscaling
				pad_H, pad_W = 	scale_factor - H % scale_factor if H % scale_factor else 0, scale_factor - W % scale_factor if W % scale_factor else 0
				padding = (0, pad_W, 0, pad_H)
				video = F.pad(video.view(B*N, C, H, W), padding, mode='reflect').view(B, N, C, H + pad_H, W + pad_W)
				H, W, = H + pad_H, W + pad_W

				# generate the blur
				blur = torch.zeros(H, W, device=device)
				li, lj = ker.shape[0], ker.shape[1]
				ci, cj = li // 2, lj //2
				blur[:ci + 1, :cj + 1] = ker[ci:, cj:]
				blur[:ci + 1, -cj:] = ker[ci:, :cj]
				blur[-ci:, :cj + 1] = ker[:ci, cj:]
				blur[-ci:, -cj:] = ker[:ci, :cj]
				blur = blur.view(1, 1, 1, H, W)
				blur_fft=torch.fft.fft2(blur)
				
				# generate the degraded video
				degraded_video = lr_video(video, blur_fft, scale_factor, noise_level)

				# define the proximal operator of the data term
				FB, FBC, F2B, FBFy = utils_sisr.pre_calculate_video(degraded_video, blur, scale_factor)
				proxF = lambda x : prox_sr(x, FB, FBC, F2B, FBFy, scale_factor, noise_level, alpha_epsilon)

				# initialize with bicubically upsampled degraded video
				video_cv2 = degraded_video.cpu().permute(0,1,3,4,2).numpy()
				init_np = torch.zeros_like(video).cpu().permute(0,1,3,4,2).numpy()
				for _ in range(B):
					for n in range(N):
						joj = video_cv2[_, n]
						jij = cv2.resize(joj, (W, H), interpolation=cv2.INTER_CUBIC)
						init_np[_, n] = jij
				init = torch.from_numpy(init_np).permute(0,1,4,2,3).to(device)
				t0 = time.time()
				restored_video, psnr_x_iters, psnr_z_iters, x_grad_iters, z_grad_iters, x_minus_z_iters, dr_iters = torch_admm(init=init, clean=video, proxF=proxF, proxG=Ds, max_iters=admm_iters, verbose=True if verbose >=3 else False)
				t_forward = time.time() - t0

				# remove padding and compute psnr
				H, W, = H - pad_H, W - pad_W
				video, degraded_video, init, restored_video, = video[..., :H, :W].cpu(), degraded_video.cpu(), init[..., :H, :W].cpu(), restored_video[..., :H, :W].cpu()
				psnr_noisy = pytorch_psnr(video, init)
				psnr_out = pytorch_psnr(video, restored_video)
				ssim_noisy = ssim_video_batch(video, init)
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
				psnrs_x_iters.append(psnr_x_iters)
				psnrs_z_iters.append(psnr_z_iters)
				x_grads_iters.append(x_grad_iters)
				z_grads_iters.append(z_grad_iters)
				x_minus_zs_iters.append(x_minus_z_iters)
				drs_iters.append(dr_iters)
				if save_frames:
					image_folder = os.path.join(output_folder, "images")
					if not os.path.exists(image_folder):
						os.makedirs(image_folder)
					vid_name = batch['video_name'][0]
					denoiser_name = model.__class__.__name__
					tensor_to_images(video.squeeze(), f'{vid_name}_clean', os.path.join(image_folder, f"{vid_name}_clean"))
					tensor_to_images(degraded_video.squeeze(), f'{vid_name}_lr', os.path.join(image_folder, f"{vid_name}_lr_sf{scale_factor}_kernel_{kernel}_sigma{noise_level*255}"))
					tensor_to_images(init.squeeze(), f'{vid_name}_bicubic', os.path.join(image_folder, f"{vid_name}_bicubic_sf{scale_factor}_kernel_{kernel}_sigma{noise_level*255}"))
					tensor_to_images(restored_video.squeeze(), f'{vid_name}_restored', os.path.join(image_folder, f"{vid_name}_sr_sf{scale_factor}_kernel_{kernel}_sigma{noise_level*255}_{denoiser_name}_alpha{admm_alpha}_s{int(admm_denoiser_level*255)}_{admm_iters}iters_restored"))
				if save_graphs:
					graph_folder = os.path.join(output_folder, "graphs")
					if not os.path.exists(graph_folder):
						os.makedirs(graph_folder)
					vid_name = batch['video_name'][0]
					denoiser_name = model.__class__.__name__
					graph_subfolder = f"{os.path.join(graph_folder, f'{vid_name}_lr_sf{scale_factor}_kernel_{kernel}_sigma{noise_level*255}_{denoiser_name}_alpha{admm_alpha}_s{int(admm_denoiser_level*255)}_{admm_iters}iters')}"
					if not os.path.exists(graph_subfolder):
						os.makedirs(graph_subfolder)
					save_graph2(np.arange(len(psnr_x_iters)), psnr_x_iters.cpu().numpy(), psnr_z_iters.cpu().numpy(), graph_subfolder, "psnr_iter", title=None, xlabel="iterations", legend=["x_k", "z_k"])
					save_graph(np.arange(len(x_grad_iters)), x_grad_iters.cpu().numpy(), graph_subfolder, "x_grad_iter", title=None, xlabel="iterations")
					save_graph(np.arange(len(z_grad_iters)), z_grad_iters.cpu().numpy(), graph_subfolder, "z_grad_iter", title=None, xlabel="iterations")
					save_graph(np.arange(len(x_minus_z_iters)), x_minus_z_iters.cpu().numpy(), graph_subfolder, "x_minus_z_iter", title=None, xlabel="iterations")
					save_graph(np.arange(len(dr_iters)), dr_iters.cpu().numpy(), graph_subfolder, "denoiser_residual_iter", title=None, xlabel="iterations")
	avg_psnr_noisy = torch.Tensor(psnrs_noisy).mean()
	avg_ssim_noisy = torch.Tensor(ssims_noisy).mean()
	avg_psnr_out = torch.Tensor(psnrs_out).mean()
	avg_ssim_out = torch.Tensor(ssims_out).mean()
	avg_runtime = torch.Tensor(runtimes).mean()

	if verbose >= 1:
		print(f'model: {model.__class__.__name__:<18} PSNR/SSIM noisy: {avg_psnr_noisy:<2.2f}/{avg_ssim_noisy:.4f}, PSNR/SSIM out: {avg_psnr_out:<2.2f}/{avg_ssim_out:.4f} \t runtime: {avg_runtime:.3f}s/frame\n')

	return vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime, ker.cpu().numpy()



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

	gpu=args['gpu']
	device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
	print(f"selected device: {device}")

	model_list, path_list = [], []

	for denoiser in args['denoisers']:
		if 'dncnn' == denoiser:
			path_list.append("pretrained_models/dncnn_color_blind.pth")
			model_list.append(DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R'))
		elif 'ffdnet' == denoiser:
			path_list.append("pretrained_models/ffdnet_color.pth")
			model_list.append(FFDNet(in_nc=3, out_nc=3, nc=96, nb=12))
		elif 'drunet' == denoiser:
			path_list.append("pretrained_models/drunet_color.pth")
			model_list.append(DRUNet())
		elif 'fastdvdnet' == denoiser:
			path_list.append("pretrained_models/fastdvdnet_nodp.pth")
			model_list.append(FastDVDnet(num_input_frames=5))


	tf = transforms.CenterCrop(args['centercrop']) if args['centercrop'] > 0 else None
	dataset = videoDataset(args['dataset_path'], extension=args['extension'], nested_subfolders=args['dataset_depth'], transform=tf, max_video_length=args['max_frames'])
	print(f'created a dataset of {len(dataset)} videos.')
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	out_folder = args['logdir']
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	out_filename = os.path.join(out_folder, f"sr_{args['dataset_name']}_")
	for denoiser in args['denoisers']:
		out_filename += denoiser + '_'
	out_filename += "s_"
	for dl in args['denoiser_levels']:
		out_filename += str(dl) + '_'
	out_filename += "kernel_" +	args['kernel'] + '_'
	out_filename += "sf_"
	for sf in args['scale_factors']:
		out_filename += str(sf) + '_'
	out_filename += "sigmas_"
	for sigma in args['sigmas']:
		out_filename += str(sigma) + '_'
	out_filename += "alphas_"
	for alpha in args['alphas']:
		out_filename += str(alpha) + '_'
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
		for s in args['denoiser_levels']:
			print(f'denoiser std={s}')
			res_dict[model.__class__.__name__][f"s={s}"] = {}
			for sf in args['scale_factors']:
				print(f'scale factor={sf}')
				res_dict[model.__class__.__name__][f"s={s}"][f"sf={sf}"] = {}
				for sigma in args['sigmas']:
					print(f'sigma={sigma}')
					res_dict[model.__class__.__name__][f"s={s}"][f"sf={sf}"][f"sigma={sigma}"] = {}
					for alpha in args['alphas']:
						print(f'alpha={alpha}')
						vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime, kernel = sr_video_dataset(model, dataloader, kernel=args['kernel'], scale_factor=sf, noise_level=sigma/255., admm_alpha=alpha, admm_denoiser_level=s/255., admm_iters=args['max_iters'], save_frames=args['save_frames'], save_graphs=args['save_graphs'], output_folder=args['logdir'], device=device, verbose=args['verbose'])
						if no_kernel:
							res_dict['kernel'] = kernel.tolist()
							no_kernel = False
						res_dict[model.__class__.__name__][f"s={s}"][f"sf={sf}"][f"sigma={sigma}"][f"alpha={alpha}"] = {'avg_psnr_noisy': float(avg_psnr_noisy), 'avg_ssim_noisy': float(avg_ssim_noisy), 'avg_psnr_out': float(avg_psnr_out), 'avg_ssim_out': float(avg_ssim_out), 'avg_runtime': float(avg_runtime)}
						for v, vid_name in enumerate(vid_names):
							res_dict[model.__class__.__name__][f"s={s}"][f"sf={sf}"][f"sigma={sigma}"][f"alpha={alpha}"][vid_name] = {'psnr_noisy': float(psnrs_noisy[v]), 'ssim_noisy': float(ssims_noisy[v]), 'psnr_out': float(psnrs_out[v]), 'ssim_out' : float(ssims_out[v]), 'runtime' : runtimes[v], 'psnr_x_iters': psnrs_x_iters[v].tolist(), 'psnr_z_iters': psnrs_z_iters[v].tolist(), 'x_grad_iters': x_grads_iters[v].tolist(), 'z_grad_iters': z_grads_iters[v].tolist(), 'x_minus_z_iters': x_minus_zs_iters[v].tolist(), 'dr_iters': drs_iters[v].tolist()}

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
	parser.add_argument("--save_graphs", action='store_true', help="save graphs as images")
	#Model parameters
	parser.add_argument("--denoisers", type=str, nargs='+', default=['fastdvdnet'], help="selected model ('fastdvdnet' / 'drunet')")	parser.add_argument("--denoiser_levels", type=float, nargs='+', default=[20.], help="noise level applied to the CNN denoiser (between 0 and 255)")
	#data parameters
	parser.add_argument("--dataset_path", type=str, default='./data/subset_4', help="path to the folder of the video dataset")
	parser.add_argument("--dataset_name", type=str, default='davis_subset4', help="name of the dataset")
	parser.add_argument("--dataset_depth", type=int, default=1, help="number of nested subfolders in the dataset")
	parser.add_argument("--extension", type=str, default='.jpg', help="file extension ('.jpg' / '.png')")
	parser.add_argument("--centercrop", type=int, default=-1, help="center crop size if any (-1 => full res)")
	parser.add_argument("--max_frames", type=int, default=-1, help="maximum number of frames per video to load (-1 => load all frames)")
	#pnp-admm parameters
	parser.add_argument("--max_iters", type=int, default=20, help="maximum number of pnp-admm iterations")
	parser.add_argument("--alphas", type=float, nargs='+', default=[0.025], help="admm alpha parameter")
	parser.add_argument("--sigmas", type=float, nargs='+', default=[0.], help="noise level of the extra AWGN applied during image degradation (between 0 and 255)")
	parser.add_argument("--kernel", type=str, default='gaussian25-1.6', help="blur kernel (uniform-9 / gaussian-9-3 / gaussian11-5 / gaussian25-1.6)")
	parser.add_argument("--scale_factors", type=int, nargs='+',  default=[2], help="downsampling / upsampling scale factor for SR")
	argspar = parser.parse_args()


	print("\n### Running video PnP-ADMM SR ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))