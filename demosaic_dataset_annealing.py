"""

@author: amonod

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
from models.network_fastdvdnet import FastDVDnet
from image_datasets import videoDataset
from utils import pytorch_psnr, ssim_video_batch, get_n_params
from video_utils import tensor_to_images
from graph_utils import save_graph, save_graph2

from video_denoiser import pytorch_dncnn_video_denoiser, pytorch_ffdnet_video_denoiser, pytorch_drunet_video_denoiser, pytorch_fastdvdnet_video_denoiser
from proxs import prox_inpainting
from torch_optim import torch_admm_annealing
from demosaic_dataset import mosaic_mask_video
from zhang_utils.utils_mosaic import dm_matlab


def demosaic_video_dataset_annealing(model, dataloader, init_type="mosaic", noise_level=0./255, admm_alpha=0.25, max_admm_denoiser_level=30./255, min_admm_denoiser_level=5./255, admm_iters=200, save_frames=False, save_graphs=False, output_folder=None, device=torch.device("cpu"), verbose=1):

	# denoiser_range = np.linspace(max_admm_denoiser_level, min_admm_denoiser_level, admm_iters)
	denoiser_range = np.logspace(np.log10(max_admm_denoiser_level), np.log10(min_admm_denoiser_level), admm_iters)
	epsilon_alpha_range = denoiser_range**2 / admm_alpha

	if verbose >= 2:
		print(f"denoiser range = [{denoiser_range[0]:.3f}, {denoiser_range[-1]:.3f}], alpha={admm_alpha} => epsilon/alpha range = [{epsilon_alpha_range[0]:.3f}, {epsilon_alpha_range[-1]:.3f}]")

	# define the deep denoiser Ds as plug and play prior
	if isinstance(model, DnCNN):
		Ds = lambda x, denoiser_level: pytorch_dncnn_video_denoiser(x, model, denoiser_level, model_device=device, output_device=device)
	elif isinstance(model, FFDNet):
		Ds = lambda x, denoiser_level: pytorch_ffdnet_video_denoiser(x, model, denoiser_level, model_device=device, output_device=device)
	elif isinstance(model, DRUNet):
		Ds = lambda x, denoiser_level: pytorch_drunet_video_denoiser(x, model, denoiser_level, model_device=device, output_device=device)
	elif isinstance(model, FastDVDnet):
		Ds = lambda x, denoiser_level: pytorch_fastdvdnet_video_denoiser(x, model, denoiser_level, model_device=device, output_device=device)

	vid_names, psnrs_noisy, psnrs_out, ssims_noisy, ssims_out, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, runtimes = [], [], [], [], [], [], [], [], [], [], [], []
	with torch.no_grad():
		for b, batch in enumerate(dataloader):
				video = batch['video'].to(device)
				B, N, C, H, W = video.shape
				# pad to avoid missing pixels when downscaling
				pad_H, pad_W = 	2 - H % 2 if H % 2 else 0, 2 - W % 2 if W % 2 else 0
				padding = (0, pad_W, 0, pad_H)
				video = F.pad(video.view(B*N, C, H, W), padding, mode='reflect').view(B, N, C, H + pad_H, W + pad_W)
				H, W, = H + pad_H, W + pad_W
				# generate the degraded video
				degraded_video, mask, bayer_flat, bayer_stack  = mosaic_mask_video(video, noise_level, pattern="RGGB")

				# define the proximal operator of the data term
				proxF = lambda x, gamma : prox_inpainting(x, degraded_video, mask, noise_level, gamma)

				if init_type == "mosaic":
					# initialize with the degraded video
					init = degraded_video
				elif init_type == "matlab":
					# initialize with crudely demosaicked degraded video
					init = torch.empty_like(degraded_video)
					for n, img in enumerate(degraded_video[0]):
						demosaicked_img = dm_matlab(bayer_stack[:, n].permute(0, 3, 1, 2))
						init[0, n] = demosaicked_img
				elif init_type == "cv2":
					for b in range(B):
						for n in range(N):
							_ = 255 * bayer_flat[b, n].cpu().clamp(0., 1.).numpy().astype(np.uint8)
							_ = cv2.cvtColor(_, cv2.COLOR_BAYER_BG2RGB_EA)
							init[b, n] = 1./255 * torch.from_numpy(_).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)

				t0 = time.time()
				restored_video, psnr_x_iters, psnr_z_iters, x_grad_iters, z_grad_iters, x_minus_z_iters, dr_iters = torch_admm_annealing(init=init, clean=video, proxF=proxF, proxG=Ds, gammaF_range=epsilon_alpha_range, gammaG_range=denoiser_range, max_iters=admm_iters, verbose=True if verbose >=3 else False)
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
					tensor_to_images(degraded_video.squeeze(), f'{vid_name}_mosaicked', os.path.join(image_folder, f"{vid_name}_mosaicked_RGGB_sigma{noise_level*255}"))
					tensor_to_images(init.squeeze(), f'{vid_name}_init', os.path.join(image_folder, f"{vid_name}_mosaicked_RGGB_sigma{noise_level*255}_{init_type}_init"))
					tensor_to_images(restored_video.squeeze(), f'{vid_name}_restored', os.path.join(image_folder, f"{vid_name}_mosaicked_RGGB_sigma{noise_level*255}_{denoiser_name}_alpha{admm_alpha}_s{int(max_admm_denoiser_level*255)}to{int(min_admm_denoiser_level*255)}_{admm_iters}iters_restored"))
				if save_graphs:
					graph_folder = os.path.join(output_folder, "graphs")
					if not os.path.exists(graph_folder):
						os.makedirs(graph_folder)
					vid_name = batch['video_name'][0]
					denoiser_name = model.__class__.__name__
					graph_subfolder = f"{os.path.join(graph_folder, f'{vid_name}_mosaicked_RGGB_sigma{noise_level*255}_{denoiser_name}_alpha{admm_alpha}_s{int(max_admm_denoiser_level*255)}to{int(min_admm_denoiser_level*255)}_{admm_iters}iters')}"
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

	return vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime



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
		if 'drunet' == denoiser:
			path_list.append("pretrained_models/drunet_color.pth")
			model_list.append(DRUNet())
		elif 'fastdvdnet' == denoiser:
			path_list.append("pretrained_models/fastdvdnet_nodp.pth")
			model_list.append(FastDVDnet(num_input_frames=5))net(num_input_frames=5))


	tf = transforms.CenterCrop(args['centercrop']) if args['centercrop'] > 0 else None
	dataset = videoDataset(args['dataset_path'], extension=args['extension'], nested_subfolders=args['dataset_depth'], transform=tf, max_video_length=args['max_frames'])
	print(f'created a dataset of {len(dataset)} videos.')
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	out_folder = args['logdir']
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	out_filename = os.path.join(out_folder, f"demosaicking_anneal_{args['dataset_name']}_")
	for denoiser in args['denoisers']:
		out_filename += denoiser + '_'
	for dl in args['max_denoiser_levels']:
		out_filename += str(dl) + '_'
	out_filename += 'to' + str(args['min_denoiser_level']) + '_'
	out_filename += "sigmas_"
	for sigma in args['sigmas']:
		out_filename += str(sigma) + '_'
	out_filename += "alphas_"
	for alpha in args['alphas']:
		out_filename += str(alpha) + '_'
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
			print(f'max denoiser std={s}')
			res_dict[model.__class__.__name__][f"s={s}"] = {}
			for sigma in args['sigmas']:
				print(f'sigma={sigma}')
				res_dict[model.__class__.__name__][f"s={s}"][f"sigma={sigma}"] = {}
				for alpha in args['alphas']:
					print(f'alpha={alpha}')
					vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime = demosaic_video_dataset_annealing(model, dataloader, init_type=args['init'], noise_level=sigma/255., admm_alpha=alpha, max_admm_denoiser_level=s/255., min_admm_denoiser_level=args['min_denoiser_level']/255., admm_iters=args['max_iters'], save_frames=args['save_frames'], save_graphs=args['save_graphs'], output_folder=args['logdir'], device=device, verbose=args['verbose'])
					res_dict[model.__class__.__name__][f"s={s}"][f"sigma={sigma}"][f"alpha={alpha}"] = {'avg_psnr_noisy': float(avg_psnr_noisy), 'avg_ssim_noisy': float(avg_ssim_noisy), 'avg_psnr_out': float(avg_psnr_out), 'avg_ssim_out': float(avg_ssim_out), 'avg_runtime': float(avg_runtime)}
					for v, vid_name in enumerate(vid_names):
						res_dict[model.__class__.__name__][f"s={s}"][f"sigma={sigma}"][f"alpha={alpha}"][vid_name] = {'psnr_noisy': float(psnrs_noisy[v]), 'ssim_noisy': float(ssims_noisy[v]), 'psnr_out': float(psnrs_out[v]), 'ssim_out' : float(ssims_out[v]), 'runtime' : runtimes[v], 'psnr_x_iters': psnrs_x_iters[v].tolist(), 'psnr_z_iters': psnrs_z_iters[v].tolist(), 'x_grad_iters': x_grads_iters[v].tolist(), 'z_grad_iters': z_grads_iters[v].tolist(), 'x_minus_z_iters': x_minus_zs_iters[v].tolist(), 'dr_iters': drs_iters[v].tolist()}
					# res_dict[model.__class__.__name__][prob] = {'PSNR/SSIM noisy': [psnr_noisy, ssim_noisy], 'PSNR/SSIM out': [psnr_out, ssim_out]}
					# average_runtime += torch.Tensor(runtimes).mean()
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
	parser.add_argument("--save_graphs", action='store_true', help="save graphs as images")
	#Model parameters
	parser.add_argument("--denoisers", type=str, nargs='+', default=['fastdvdnet'], help="selected model ('fastdvdnet' / 'drunet')")	parser.add_argument("--max_denoiser_levels", type=float, nargs='+', default=[50.], help="max noise level applied to the CNN denoiser (between 0 and 255)")
	parser.add_argument("--min_denoiser_level", type=float, default=5., help="min noise level applied to the CNN denoiser (between 0 and 255)")
	#data parameters
	parser.add_argument("--dataset_path", type=str, default='./data/subset_4', help="path to the folder of the video dataset")
	parser.add_argument("--dataset_name", type=str, default='davis_subset4', help="name of the dataset")
	parser.add_argument("--dataset_depth", type=int, default=1, help="number of nested subfolders in the dataset")
	parser.add_argument("--extension", type=str, default='.jpg', help="file extension ('.jpg' / '.png')")
	parser.add_argument("--centercrop", type=int, default=-1, help="center crop size if any (-1 => full res)")
	parser.add_argument("--max_frames", type=int, default=-1, help="maximum number of frames per video to load (-1 => load all frames)")
	#pnp-admm parameters
	parser.add_argument("--max_iters", type=int, default=100, help="maximum number of pnp-admm iterations")
	parser.add_argument("--sigmas", type=float, nargs='+', default=[5], help="noise level of the extra AWGN applied during image degradation (between 0 and 255)")
	parser.add_argument("--alphas", type=float, nargs='+', default=[1.], help="admm alpha parameter")
	parser.add_argument("--init", type=str, default='mosaic', help="init type ('mosaic' / 'matlab' / 'cv2')")
	argspar = parser.parse_args()


	print("\n### Running video PnP-ADMM demosaicking annealing###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
