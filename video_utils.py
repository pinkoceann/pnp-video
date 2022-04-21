"""

@author: amonod

"""
import os
import glob
import cv2
import numpy as np

from PIL import Image


def tensor_to_video(video_tensor, video_path, video_format='.avi', fps=30, codec_name='avc1'):
	# expects N, C, H, W tensor, normalized 
	N, C, H, W = video_tensor.shape
	assert C == 1 or C == 3
	video = video_tensor.clamp(0., 1.).permute((0, 2, 3, 1)).cpu().numpy()
	image_list = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in video[:]]
	# define video codec
	codec = cv2.VideoWriter_fourcc(*codec_name)
	video = cv2.VideoWriter(video_path, fourcc=codec, fps=fps, frameSize=(W, H))
	for img in imgs:
		video.write(img)
	video.release()


def tensor_to_images(video_tensor, video_name, out_folder, image_format='.png'):
	# expects N, C, H, W tensor, normalized 
	N, C, H, W = video_tensor.shape
	assert C == 1 or C == 3
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	video = 255. * video_tensor.clamp(0., 1.).permute((0, 2, 3, 1)).cpu().numpy()
	image_list = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in video[:]]
	for i, img in enumerate(image_list):
		img_filename=os.path.join(out_folder, f"{video_name}_{i:03}{image_format}")
		cv2.imwrite(img_filename, img)


def images_to_video(in_folder, out_folder=None, video_name=None, image_format='.png', video_format='.mp4', fps=30, codec_name='mp4v'):
	if out_folder is None:
		out_folder = in_folder
	if video_name is None:
		video_name = os.path.basename(in_folder)
	vid_path = os.path.join(out_folder, f'{video_name}{video_format}')
	imgs = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(in_folder, f'*{image_format}')))]
	# define video codec
	codec = cv2.VideoWriter_fourcc(*codec_name)
	video = cv2.VideoWriter(vid_path, fourcc=codec, fps=fps, frameSize=(imgs[0].shape[1], imgs[0].shape[0]))
	for img in imgs:
		video.write(img)
	video.release()

def images_to_gif(in_folder, out_folder=None, gif_name=None, image_format='.png', fps=30, codec_name='avc1'):
	if out_folder is None:
		out_folder = in_folder
	if gif_name is None:
		gif_name = os.path.basename(in_folder)
	framerate_ms = 1. / fps * 1000
	gif_path = os.path.join(out_folder, f'{gif_name}.gif')
	img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(in_folder, f'*{image_format}')))]
	img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, duration=framerate_ms, loop=0)


if __name__ == "__main__":
	folder = "./sr_results/images"
	sublist = sorted(glob.glob(os.path.join(folder, '*')))
	for s, sub in enumerate(sublist):
		print(f"processing video {s+1}/{len(sublist)} \t {sub}")
		images_to_video(sub)
	print(sublist)