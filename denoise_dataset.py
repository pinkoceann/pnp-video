import argparse
import torch
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

from video_denoiser import pytorch_dncnn_video_denoiser, pytorch_ffdnet_video_denoiser, pytorch_drunet_video_denoiser, pytorch_fastdvdnet_video_denoiser
from video_utils import tensor_to_images


def denoise_video_dataset(model, dataloader, noise_level=40./255, save_frames=False, output_folder=None, device=torch.device("cpu"), verbose=1):

	vid_names, psnrs_noisy, psnrs_out, ssims_noisy, ssims_out, runtimes = [], [], [], [], [], []
	with torch.no_grad():
		for b, batch in enumerate(dataloader):
				video = batch['video']
				B, N, C, H, W = video.shape
				assert B == 1, "this code only works with batch size 1 for now"
				# add AWGN
				noise = torch.normal(mean=torch.zeros_like(video), std=noise_level)
				noisy_video = video + noise
				t0 = time.time()
				if isinstance(model, DnCNN):
					denoised_video = pytorch_dncnn_video_denoiser(noisy_video, model, noise_level, model_device=device, output_device=device)
				elif isinstance(model, FFDNet):
					denoised_video = pytorch_ffdnet_video_denoiser(noisy_video, model, noise_level, model_device=device, output_device=device)
				elif isinstance(model, DRUNet):
					denoised_video = pytorch_drunet_video_denoiser(noisy_video, model, noise_level, model_device=device, output_device=device)
				elif isinstance(model, FastDVDnet):
					denoised_video = pytorch_fastdvdnet_video_denoiser(noisy_video, model, noise_level, model_device=device, output_device=device)
				t_forward = time.time() - t0
				# compute psnr
				video, noisy_video, denoised_video = video.cpu(), noisy_video.cpu(), denoised_video.cpu()
				psnr_noisy = pytorch_psnr(video, noisy_video)
				psnr_out = pytorch_psnr(video, denoised_video)
				ssim_noisy = ssim_video_batch(video, noisy_video)
				ssim_out = ssim_video_batch(video, denoised_video, data_range=1.)
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
					tensor_to_images(noisy_video.squeeze(), f'{vid_name}_noisy', os.path.join(image_folder, f"{vid_name}_noisy_sigma{noise_level*255}"))
					tensor_to_images(denoised_video.squeeze(), f'{vid_name}_denoised', os.path.join(image_folder, f"{vid_name}_noisy_sigma{noise_level*255}_{denoiser_name}_restored"))
	avg_psnr_noisy = torch.Tensor(psnrs_noisy).mean()
	avg_ssim_noisy = torch.Tensor(ssims_noisy).mean()
	avg_psnr_out = torch.Tensor(psnrs_out).mean()
	avg_ssim_out = torch.Tensor(ssims_out).mean()
	avg_runtime = torch.Tensor(runtimes).mean()

	if verbose >= 1:
		print(f'model: {model.__class__.__name__:<18} PSNR/SSIM noisy: {avg_psnr_noisy:<2.2f}/{avg_ssim_noisy:.4f}, PSNR/SSIM out: {avg_psnr_out:<2.2f}/{avg_ssim_out:.4f} \t runtime: {avg_runtime:.3f}s/frame\n')

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
	out_filename = os.path.join(out_folder, f"denoising_{args['dataset_name']}_")
	for denoiser in args['denoisers']:
		out_filename += denoiser + '_'
	out_filename += "sigmas_"
	for sigma in args['sigmas']:
		out_filename += str(sigma) + '_'
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
		for sigma in args['sigmas']:
			print(f'sigma={sigma}')
			res_dict[model.__class__.__name__][f"sigma={sigma}"] = {}
			vid_names, psnrs_noisy, ssims_noisy, psnrs_out, ssims_out, runtimes, avg_psnr_noisy, avg_ssim_noisy, avg_psnr_out, avg_ssim_out, avg_runtime = denoise_video_dataset(model, dataloader, noise_level=sigma/255., save_frames=args['save_frames'], output_folder=args['logdir'], device=device, verbose=args['verbose'])
			res_dict[model.__class__.__name__][f"sigma={sigma}"] = {'avg_psnr_noisy': float(avg_psnr_noisy), 'avg_ssim_noisy': float(avg_ssim_noisy), 'avg_psnr_out': float(avg_psnr_out), 'avg_ssim_out': float(avg_ssim_out), 'avg_runtime': float(avg_runtime)}
			for v, vid_name in enumerate(vid_names):
				res_dict[model.__class__.__name__][f"sigma={sigma}"][vid_name] = {'psnr_noisy': float(psnrs_noisy[v]), 'ssim_noisy': float(ssims_noisy[v]), 'psnr_out': float(psnrs_out[v]), 'ssim_out' : float(ssims_out[v]), 'runtime' : runtimes[v]}
			torch.cuda.empty_cache()
			# save res dict frequently just in case
			with open(out_filename, 'w') as handle:
				json.dump(res_dict, handle, indent=4)
		del model

	print(f"\nresults file location: {out_filename}")

	t_final = time.time() - t_init
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(t_final))))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Run the denoiser")
	# Program-specific arguments
	parser.add_argument('--deterministic', dest='deterministic', action='store_true', help='flag to ensure full reproductibility')
	parser.add_argument('--no-deterministic', dest='deterministic', action='store_false')
	parser.set_defaults(deterministic=True)
	parser.add_argument('--verbose', type=int, default=1, help='verbose level (0 to 3)')
	parser.add_argument('--gpu', dest='gpu', action='store_true', help='run the code on GPU instead of CPU')
	parser.add_argument('--cpu', dest='gpu', action='store_false', help='run the code on CPU instead of GPU')
	parser.set_defaults(gpu=True)
	parser.add_argument("--logdir", type=str, default='./denoising_results', help="path to the folder containing the output results")
	parser.add_argument("--save_frames", action='store_true', help="save videos as images")
	#Model parameters
	parser.add_argument("--denoisers", type=str, nargs='+', default=['fastdvdnet'], help="selected model ('fastdvdnet' / 'drunet')")	#data parameters
	parser.add_argument("--dataset_path", type=str, default='./data/subset_4', help="path to the folder of the video dataset")
	parser.add_argument("--dataset_name", type=str, default='davis_subset4', help="name of the dataset")
	parser.add_argument("--dataset_depth", type=int, default=1, help="number of nested subfolders in the dataset")
	parser.add_argument("--extension", type=str, default='.jpg', help="file extension ('.jpg' / '.png')")
	parser.add_argument("--centercrop", type=int, default=-1, help="center crop size if any (-1 => full res)")
	parser.add_argument("--max_frames", type=int, default=-1, help="maximum number of frames per video to load (-1 => load all frames)")
	parser.add_argument("--sigmas", type=float, nargs='+', default=[10.], help="noise level of the extra AWGN applied during image degradation (between 0 and 255)")

	argspar = parser.parse_args()


	print("\n### Running video denoising ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))