import matplotlib.pyplot as plt
import numpy as np
from os.path import join

def save_graph(x, y, dest_folder, file_name, title=None, xlabel=None, ylabel=None, legend=None):
	plt.figure()
	plt.plot(x, y, '--o')
	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	if legend:
		plt.legend(legend)
	plt.tight_layout()
	plt.savefig(f"{join(dest_folder, file_name)}.png")
	plt.close()

def save_graph2(x, y1, y2, dest_folder, file_name, title=None, xlabel=None, ylabel=None, legend=None):
	plt.figure()
	plt.plot(x, y1, '--o')
	plt.plot(x, y1, '--+')
	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
	if legend:
		plt.legend(legend)
	plt.tight_layout()
	plt.savefig(f"{join(dest_folder, file_name)}.png")
	plt.close()