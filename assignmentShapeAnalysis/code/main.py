import os
import numpy as np 
import matplotlib.pyplot as plt 
import h5py

from functions import *

def solve(data, save_plot_dir = '../results/hand', data_dir = None):

	# Shape of Data is Nxnxd. N images. n points per each image pointset. d dimensions of the points.

	os.makedirs(save_plot_dir, exist_ok=True)
	N = data.shape[0]
	colors = np.random.rand(N,3)
	
	# -------------------------- Part d --------------------------- #
	for i in range(N):
		plt.plot(data[i,:,0], data[i,:,1], 'o', color=colors[i])
	plt.title('Plot of all initital pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'initial-all-data-scatter.png'))

	for i in range(N):
		plt.plot(data[i,:,0], data[i,:,1], color=colors[i])
		plt.plot([data[i,0,0], data[i,-1,0]], [data[i,0,1], data[i,-1,1]], color=colors[i])
	plt.savefig(os.path.join(save_plot_dir, 'initial-all-data-polyline.png'))
	plt.clf()

	# --------------------------Part e----------------------------- #
	mean, z_aligned = find_mean_case1(data)
	for i in range(N):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.4)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.title('findd shape mean (in black), together with all the aligned pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-aligned-data.png'))
	plt.clf()    

	# -------------------------- Case 2 --------------------------- #
	mean2, z_aligned2 = find_mean_case2(data)
	for i in range(N):
		plt.plot(z_aligned2[i,:,0], z_aligned2[i,:,1], 'o', alpha=0.4)
	plt.plot(mean2[0,:,0], mean2[0,:,1], 'ko-')
	plt.plot([mean2[0,-1,0], mean2[0,0,0]], [mean2[0,-1,1], mean2[0,0,1]], 'ko-')

	plt.title('findd shape mean (in black), together with all the aligned pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-aligned-data2.png'))
	plt.clf()
	# ------------------------------------------------------------- #

	# -------------------------- Part f --------------------------- #

	cov_matrix = find_covariance_matrix(z_aligned, mean) # ndxnd matrix
	eig_values, eig_vecs = np.linalg.eig(cov_matrix)

	idx = eig_values.argsort()[::-1]
	eig_values = eig_values[idx]
	eig_vecs = eig_vecs[:,idx]

	plt.plot(np.real(eig_values[::-1]))
	plt.title('Eigenvalues (in y axis) plot (sorted in ascending order)')
	plt.savefig(os.path.join(save_plot_dir, 'eigen-values.png'))
	plt.clf()

	# -------------------------- Case 2 --------------------------- #
	cov_matrix_2 = find_covariance_matrix(z_aligned2, mean2) # ndxnd matrix
	eig_values_2, eig_vecs_2 = np.linalg.eig(cov_matrix_2)

	idx_2 = eig_values_2.argsort()[::-1]
	eig_values_2 = eig_values_2[idx_2]
	eig_vecs_2 = eig_vecs_2[:,idx_2]

	plt.plot(np.real(eig_values_2[::-1]))
	plt.title('Eigenvalues (in y axis) plot (sorted in ascending order)')
	plt.savefig(os.path.join(save_plot_dir, 'eigen-values2.png'))
	plt.clf()

	# -------------------------- Part g  Case 1--------------------------- #

	def get_modes_of_variation(i):

		var_plus = mean + 3*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		var_minus = mean - 3*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		return var_plus, var_minus

	for i in range(N):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-', label='Mean')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	var_1_plus, var_1_minus = get_modes_of_variation(0)

	plt.plot(var_1_plus[0,:,0], var_1_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_1_plus[0,-1,0], var_1_plus[0,0,0]], [var_1_plus[0,-1,1], var_1_plus[0,0,1]], 'ro-')

	plt.plot(var_1_minus[0,:,0], var_1_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_1_minus[0,-1,0], var_1_minus[0,0,0]], [var_1_minus[0,-1,1], var_1_minus[0,0,1]], 'bo-')

	plt.title('1st Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-first-mode.png'))
	plt.clf()

	var_2_plus, var_2_minus = get_modes_of_variation(1)

	for i in range(N):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-', label='Mean')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.plot(var_2_plus[0,:,0], var_2_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_2_plus[0,-1,0], var_2_plus[0,0,0]], [var_2_plus[0,-1,1], var_2_plus[0,0,1]], 'ro-')

	plt.plot(var_2_minus[0,:,0], var_2_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_2_minus[0,-1,0], var_2_minus[0,0,0]], [var_2_minus[0,-1,1], var_2_minus[0,0,1]], 'bo-')

	plt.title('2nd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-second-mode.png'))
	plt.clf()

	var_3_plus, var_3_minus = get_modes_of_variation(2)

	for i in range(N):
		plt.plot(z_aligned[i,:,0], z_aligned[i,:,1], 'o', alpha=0.15)
	plt.plot(mean[0,:,0], mean[0,:,1], 'ko-', label='Mean')
	plt.plot([mean[0,-1,0], mean[0,0,0]], [mean[0,-1,1], mean[0,0,1]], 'ko-')

	plt.plot(var_3_plus[0,:,0], var_3_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_3_plus[0,-1,0], var_3_plus[0,0,0]], [var_3_plus[0,-1,1], var_3_plus[0,0,1]], 'ro-')
	plt.plot(var_3_minus[0,:,0], var_3_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_3_minus[0,-1,0], var_3_minus[0,0,0]], [var_3_minus[0,-1,1], var_3_minus[0,0,1]], 'bo-')

	plt.title('3rd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-third-mode.png'))
	plt.clf()

	# ---------------------------Part g case 2 -------------------------------------- #

	def get_modes_of_variation_2(i):
    
		var_plus = mean2 + 3*np.sqrt(np.real(eig_values_2[i]))*np.real(eig_vecs_2[:,i]).reshape(mean.shape)
		var_minus = mean2 - 3*np.sqrt(np.real(eig_values_2[i]))*np.real(eig_vecs_2[:,i]).reshape(mean.shape)
		return var_plus, var_minus

	for i in range(N):
		plt.plot(z_aligned2[i,:,0], z_aligned2[i,:,1], 'o', alpha=0.15)
	plt.plot(mean2[0,:,0], mean2[0,:,1], 'ko-', label='Mean')
	plt.plot([mean2[0,-1,0], mean2[0,0,0]], [mean2[0,-1,1], mean2[0,0,1]], 'ko-')

	var_1_plus, var_1_minus = get_modes_of_variation_2(0)

	plt.plot(var_1_plus[0,:,0], var_1_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_1_plus[0,-1,0], var_1_plus[0,0,0]], [var_1_plus[0,-1,1], var_1_plus[0,0,1]], 'ro-')

	plt.plot(var_1_minus[0,:,0], var_1_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_1_minus[0,-1,0], var_1_minus[0,0,0]], [var_1_minus[0,-1,1], var_1_minus[0,0,1]], 'bo-')

	plt.title('1st Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-first-mode_2.png'))
	plt.clf()

	var_2_plus, var_2_minus = get_modes_of_variation_2(1)

	for i in range(N):
		plt.plot(z_aligned2[i,:,0], z_aligned2[i,:,1], 'o', alpha=0.15)
	plt.plot(mean2[0,:,0], mean2[0,:,1], 'ko-', label='Mean')
	plt.plot([mean2[0,-1,0], mean2[0,0,0]], [mean2[0,-1,1], mean2[0,0,1]], 'ko-')

	plt.plot(var_2_plus[0,:,0], var_2_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_2_plus[0,-1,0], var_2_plus[0,0,0]], [var_2_plus[0,-1,1], var_2_plus[0,0,1]], 'ro-')

	plt.plot(var_2_minus[0,:,0], var_2_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_2_minus[0,-1,0], var_2_minus[0,0,0]], [var_2_minus[0,-1,1], var_2_minus[0,0,1]], 'bo-')

	plt.title('2nd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-second-mode_2.png'))
	plt.clf()

	var_3_plus, var_3_minus = get_modes_of_variation_2(2)

	for i in range(N):
		plt.plot(z_aligned2[i,:,0], z_aligned2[i,:,1], 'o', alpha=0.15)
	plt.plot(mean2[0,:,0], mean2[0,:,1], 'ko-', label='Mean')
	plt.plot([mean2[0,-1,0], mean2[0,0,0]], [mean2[0,-1,1], mean2[0,0,1]], 'ko-')

	plt.plot(var_3_plus[0,:,0], var_3_plus[0,:,1], 'ro-', label='Mean + 3 S.D')
	plt.plot([var_3_plus[0,-1,0], var_3_plus[0,0,0]], [var_3_plus[0,-1,1], var_3_plus[0,0,1]], 'ro-')
	plt.plot(var_3_minus[0,:,0], var_3_minus[0,:,1], 'bo-', label='Mean - 3 S.D')
	plt.plot([var_3_minus[0,-1,0], var_3_minus[0,0,0]], [var_3_minus[0,-1,1], var_3_minus[0,0,1]], 'bo-')

	plt.title('3rd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-third-mode_2.png'))
	plt.clf()






# Question1 
data_path = '../data/ellipses2D.mat'
f = h5py.File(data_path)
for k, v in f.items():
	data = np.array(v)
print(data[0][1])

solve(data, save_plot_dir = '../results/ellipses')

# Question2
data_path2 = '../data/hands2D.mat'
g = h5py.File(data_path2)
for k, v in g.items():
	data2 = np.array(v)

solve(data2, save_plot_dir = '../results/hand')