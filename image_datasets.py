"""

@author: amonod

"""
import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2


class ImageDataset(Dataset):
	"""Dataset of images."""
	def __init__(self, dataset_path, extension='.jpg', nested_subfolders=0, normalize_data=True, transform=None):
		"""
		Args:
			datasetPath (string): Path to folder containing all images.
			num_frames_per_burst(int): number of frames kept in the burst (stick to 5 for now)
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		# Look for all images in the appropriate amount of subfolders
		substr = dataset_path
		for i in range(nested_subfolders):
			substr = os.path.join(substr, '*')
		self.impaths = sorted(glob.glob(os.path.join(substr, '*' + extension)))
		self.normalize_data = normalize_data
		self.transform = transform

	@staticmethod
	def open_image(filepath):
		img = cv2.imread(filepath)
		# from HxWxC to CxHxW, RGB image
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
		return img

	@staticmethod
	def normalize(image):
		assert np.min(image) >= 0 and np.max(image) <= 255, 'Invalid data range for a 8 bit image'
		return image.astype(np.float32) / 255.

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()
		img = self.open_image(self.impaths[index])
		if self.normalize_data:
			img = self.normalize(img)
		sample = torch.from_numpy(img)
		if self.transform:
			sample = self.transform(sample)
		return {'image_data': sample, 'filepath': self.impaths[index]}

	def __len__(self):
		return len(self.impaths)


class burstDataset(ImageDataset):
	"""Dataset of sequence of image bursts."""
	def __init__(self, dataset_path, extension='.jpg', nested_subfolders=1, normalize_data=True, transform=None, frames_per_burst=3, bursts_per_seq=1, step=3):
		super().__init__(dataset_path, extension, nested_subfolders, normalize_data, transform)
		if bursts_per_seq == 1:
			step = 1
		# Look for all sequences in the appropriate amount of subfolders (one sequence = one bottom level folder)
		impaths = []
		substr = dataset_path
		for i in range(nested_subfolders):
			substr = os.path.join(substr, '*')
		seq_dirs = sorted(glob.glob(substr))
		for i, seq_dir in enumerate(seq_dirs):
			# look for all images within that sequence
			seq_paths = sorted(glob.glob(os.path.join(seq_dir, '*' + extension)))
			# create as many bursts of frames_per_burst frames as possible according to the step size
			start_idx = np.arange(0, len(seq_paths) - frames_per_burst, step)
			if bursts_per_seq != -1:
				assert bursts_per_seq <= len(start_idx), 'ERROR: number of bursts per folder is too large'
				start_idx = start_idx[:bursts_per_seq]
			stop_idx = start_idx + frames_per_burst
			for i in range(len(start_idx)):
				impaths.append(seq_paths[start_idx[i]:stop_idx[i]])
		self.impaths = impaths

	def __getitem__(self, index):
		burst = []
		for impath in self.impaths[index]:
			img = self.open_image(impath)
			burst.append(img)
		burst = np.stack(burst, axis=0)
		if self.normalize_data:
			burst = self.normalize(burst)
		sample = torch.from_numpy(burst)
		if self.transform:
			sample = self.transform(sample)
		return {'burst': sample, 'filepaths': self.impaths[index]}

	def __len__(self):
		return len(self.impaths)


class videoDataset(ImageDataset):
	"""Dataset of where each item is a video stored as a sequence of image files."""
	def __init__(self, dataset_path, extension='.jpg', nested_subfolders=1, normalize_data=True, transform=None, max_video_length=-1):
		super().__init__(dataset_path, extension, nested_subfolders, normalize_data, transform)

		# Look for all videos in the appropriate amount of subfolders (one video = one bottom level folder)
		vidpaths = []
		vidnames = []
		substr = dataset_path
		for i in range(nested_subfolders):
			substr = os.path.join(substr, '*/')
		vid_dirs = sorted(glob.glob(substr))
		for i, vid_dir in enumerate(vid_dirs):
			# look for all images within that video
			image_paths = sorted(glob.glob(os.path.join(vid_dir, '*' + extension)))
			if max_video_length != -1:
				assert max_video_length > 0, 'ERROR: invalid max number of frames specified'
				stop_idx = min(len(image_paths), max_video_length)
			else:
				stop_idx = len(image_paths)
			vidpaths.append(image_paths[0:stop_idx])
			vidnames.append(os.path.basename(os.path.dirname(vid_dir)))
		self.vidpaths = vidpaths
		self.vidnames = vidnames

	def __getitem__(self, index):
		video = []
		for impath in self.vidpaths[index]:
			img = self.open_image(impath)
			video.append(img)
		video = np.stack(video, axis=0)
		if self.normalize_data:
			video = self.normalize(video)
		sample = torch.from_numpy(video)
		if self.transform:
			sample = self.transform(sample)
		return {'video': sample, 'video_name': self.vidnames[index], 'filepaths': self.impaths[index]}

	def __len__(self):
		return len(self.vidpaths)
