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
from thop.utils import clever_format

from models.network_drunet import DRUNet
from models.network_fastdvdnet import FastDVDnet
from image_datasets import videoDataset
from utils import pytorch_psnr, ssim_video_batch, get_n_params
from video_utils import tensor_to_images
from graph_utils import save_graph, save_graph2

from video_denoiser import pytorch_drunet_video_denoiser, pytorch_fastdvdnet_video_denoiser
from proxs import prox_inpainting
from torch_optim import torch_admm


def mask_video(video, missing_pixels=0.5, mask_channels_equally=False, noise_level=5/255.):
	B, N, C, H, W = video.shape
	# generate the mask
	if mask_channels_equally:
		mask = torch.ones((B, C, N * H * W), device=video.device)
		masked_indexes = torch.randperm(N * H * W)[:round(N * H * W * missing_pixels)]
		mask[:, :, masked_indexes] = 0
		mask = mask.view((B, C, N, H, W)).permute(0, 2, 1, 3, 4)
	else:
		mask = torch.ones((B, N * C * H * W), device=video.device)
		masked_indexes = torch.randperm(N * C * H * W)[:round(N * C * H * W * missing_pixels)]
		mask[:, masked_indexes] = 0
		mask = mask.view((B, N, C, H, W))
	# generate the degraded video
	degraded_video = mask * video
	if noise_level > 0:
		noise = torch.normal(mean=torch.zeros_like(video), std=noise_level)
		degraded_video += noise
	return degraded_video, mask


def interpolate_video_dataset(model, dataloader, missing_pixels=.9, noise_level=0./255, mask_channels_equally=False, admm_alpha=2.25, admm_denoiser_level=30./255, admm_iters=200, save_frames=False, save_graphs=False, output_folder=None, device=torch.device("cpu"), verbose=1):

	epsilon_alpha = admm_denoiser_level**2 / admm_alpha
	if verbose >= 2:
		print(f"admm_denoiser_level = {admm_denoiser_level}, admm_alpha = {admm_alpha} => epsilon/alpha = {epsilon_alpha:.5f}")

	# define the deep denoiser Ds as plug and play prior
	if isinstance(model, DRUNet):
		Ds = lambda x: pytorch_drunet_video_denoiser(x, model, admm_denoiser_level, model_device=device, output_device=device)
	elif isinstance(model, FastDVDnet):
		Ds = lambda x: pytorch_fastdvdnet_video_denoiser(x, model, admm_denoiser_level, model_device=device, output_device=device)

	vid_names, psnrs_noisy, psnrs_out, ssims_noisy, ssims_out, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, runtimes = [], [], [], [], [], [], [], [], [], [], [], []
	with torch.no_grad():
		for b, batch in enumerate(dataloader):
				video = batch['video'].to(device)
				B, N, C, H, W = video.shape
				# generate the degraded video
				degraded_video, mask = mask_video(video, missing_pixels, mask_channels_equally, noise_level)

				# define the proximal operator of the data term
				proxF = lambda x : prox_inpainting(x, degraded_video, mask, noise_level, epsilon_alpha)

				init = degraded_video  # initialize with the degraded video
				t0 = time.time()
				restored_video, psnr_x_iters, psnr_z_iters, x_grad_iters, z_grad_iters, x_minus_z_iters, dr_iters = torch_admm(init=init, clean=video, proxF=proxF, proxG=Ds, max_iters=admm_iters, verbose=True if verbose >= 3 else False)
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
					tensor_to_images(degraded_video.squeeze(), f'{vid_name}_masked', os.path.join(image_folder, f"{vid_name}_masked_p{missing_pixels}_sigma{noise_level*255}"))
					tensor_to_images(restored_video.squeeze(), f'{vid_name}_restored', os.path.join(image_folder, f"{vid_name}_masked_p{missing_pixels}_sigma{noise_level*255}_{denoiser_name}_alpha{admm_alpha}_s{int(admm_denoiser_level*255)}_{admm_iters}iters_restored"))
				if save_graphs:
					graph_folder = os.path.join(output_folder, "graphs")
					if not os.path.exists(graph_folder):
						os.makedirs(graph_folder)
					vid_name = batch['video_name'][0]
					denoiser_name = model.__class__.__name__
					graph_subfolder = f"{os.path.join(graph_folder, f'{vid_name}_masked_p{missing_pixels}_sigma{noise_level*255}_{denoiser_name}_alpha{admm_alpha}_s{int(admm_denoiser_level*255)}_{admm_iters}iters')}"
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

	gpu = args['gpu']
	device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
	print(f"selected device: {device}")

	model_list, path_list = [], []

	for denoiser in args['denoisers']:
		if 'drunet' == denoiser:
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
	out_filename = os.path.join(out_folder, f"inpainting_{args['dataset_name']}_")
	for denoiser in args['denoisers']:
		out_filename += denoiser + '_'
	out_filename += "s_"
	for dl in args['denoiser_levels']:
		out_filename += str(dl) + '_'
	out_filename += "probs_"
	for prob in args['probs']:
		out_filename += str(prob) + '_'
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
		for s in args['denoiser_levels']:
			print(f'denoiser std={s}')
			res_dict[model.__class__.__name__][f"s={s}"] = {}
			for p, prob in enumerate(args['probs']):
				print(f'prob={prob}')
				res_dict[model.__class__.__name__][f"s={s}"][f"prob={prob}"] = {}
				for sigma in args['sigmas']:
					print(f'sigma={sigma}')
					res_dict[model.__class__.__name__][f"s={s}"][f"prob={prob}"][f"sigma={sigma}"] = {}
					for alpha in args['alphas']:
						print(f'alpha={alpha}')
						vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, psnrs_x_iters, psnrs_z_iters, x_grads_iters, z_grads_iters, x_minus_zs_iters, drs_iters, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime = interpolate_video_dataset(model, dataloader, missing_pixels=prob, noise_level=sigma/255., mask_channels_equally=args['mask_channels_equally'], admm_alpha=alpha, admm_denoiser_level=s/255., admm_iters=args['max_iters'], save_frames=args['save_frames'], save_graphs=args['save_graphs'], output_folder=args['logdir'], device=device, verbose=args['verbose'])
						res_dict[model.__class__.__name__][f"s={s}"][f"prob={prob}"][f"sigma={sigma}"][f"alpha={alpha}"] = {'avg_psnr_noisy': float(avg_psnr_noisy), 'avg_ssim_noisy': float(avg_ssim_noisy), 'avg_psnr_out': float(avg_psnr_out), 'avg_ssim_out': float(avg_ssim_out), 'avg_runtime': float(avg_runtime)}
						for v, vid_name in enumerate(vid_names):
							res_dict[model.__class__.__name__][f"s={s}"][f"prob={prob}"][f"sigma={sigma}"][f"alpha={alpha}"][vid_name] = {'psnr_noisy': float(psnrs_noisy[v]), 'ssim_noisy': float(ssims_noisy[v]), 'psnr_out': float(psnrs_out[v]), 'ssim_out' : float(ssims_out[v]), 'runtime' : runtimes[v], 'psnr_x_iters': psnrs_x_iters[v].tolist(), 'psnr_z_iters': psnrs_z_iters[v].tolist(), 'x_grad_iters': x_grads_iters[v].tolist(), 'z_grad_iters': z_grads_iters[v].tolist(), 'x_minus_z_iters': x_minus_zs_iters[v].tolist(), 'dr_iters': drs_iters[v].tolist()}

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
	parser.add_argument("--logdir", type=str, default='./inpainting_results', help="path to the folder containing the output results")
	parser.add_argument("--save_frames", action='store_true', help="save videos as images")
	parser.add_argument("--save_graphs", action='store_true', help="save graphs as images")
	# Model parameters
	parser.add_argument("--denoisers", type=str, nargs='+', default=['fastdvdnet'], help="selected model ('fastdvdnet' / 'drunet')")
	parser.add_argument("--denoiser_levels", type=float, nargs='+', default=[30.], help="noise level applied to the CNN denoiser (between 0 and 255)")
	# data parameters
	parser.add_argument("--dataset_path", type=str, default='./data/subset_4', help="path to the folder of the video dataset")
	parser.add_argument("--dataset_name", type=str, default='davis_subset4', help="name of the dataset")
	parser.add_argument("--dataset_depth", type=int, default=1, help="number of nested subfolders in the dataset")
	parser.add_argument("--extension", type=str, default='.jpg', help="file extension ('.jpg' / '.png')")
	parser.add_argument("--centercrop", type=int, default=-1, help="center crop size if any (-1 => full res)")
	parser.add_argument("--max_frames", type=int, default=-1, help="maximum number of frames per video to load (-1 => load all frames)")
	# pnp-admm parameters
	parser.add_argument("--max_iters", type=int, default=200, help="maximum number of pnp-admm iterations")
	parser.add_argument("--alphas", type=float, nargs='+', default=[2.25], help="admm alpha parameter")
	parser.add_argument("--sigmas", type=float, nargs='+', default=[0.], help="noise level of the extra AWGN applied during image degradation (between 0 and 255)")
	parser.add_argument("--probs", type=float, nargs='+', default=[.9], help="percentage of missing pixels in the interpolation problem")
	parser.add_argument("--mask_channels_equally", action='store_true', help="mask all channels in an image the same way")

	argspar = parser.parse_args()

	print("\n### Running video PnP-ADMM interpolation ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
