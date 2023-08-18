from math import *
import numpy as np 

def align_pointsets_case2(weights, z1, z2):

    n = z1.shape[1]
    z1 = z1[0]
    X1, X2, Z, Y1,Y2, C1,C2, W = 0, 0, 0, 0, 0, 0,0, 0
    for k in range(n):
        X1 = X1 + weights[k]*z1[k][0]
        X2 = X2 + weights[k]*z2[k][0]
        Y1 = Y1 + weights[k]*z1[k][1]
        Y2 = Y2 + weights[k]*z2[k][1]

        Z = Z + weights[k]*(z2[k][0]*z2[k][0] + z2[k][1]*z2[k][1])
        W = W + weights[k]
        C1 = C1 + weights[k]*(z1[k][0]*z2[k][0] + z1[k][1]*z2[k][1])
        C2 = C2 + weights[k]*(z1[k][1]*z2[k][0] - z1[k][0]*z2[k][1])
    M = np.array([[X2,-1*Y2,W,0],[Y2,X2,0,W],[Z,0,X2,Y2],[0,Z,-1*Y2,X2]])
    N = np.array([X1,Y1,C1,C2])

    P = np.matmul(np.linalg.inv(M), N.T)
    ax = P[0]
    ay = P[1]
    sR = np.array([[ax, -1*ay],[ay, ax]])
    T = np.array([P[2],P[3]])

    return sR, T.reshape(2,1)

def find_weights(pointsets):
    z = pointsets
    N = z.shape[0]
    n = z.shape[1]
    d = 2
    # find weights for each point
    weights = np.zeros(n)

    for k in range(n):
        variances = np.zeros(n)
        for l in range(n):
            distances = np.zeros(N)
            for p in range(N):
                distances[p] = sqrt((z[p][k][0]-z[p][l][0])*(z[p][k][0]-z[p][l][0]) + (z[p][k][1]-z[p][l][1])*(z[p][k][1]-z[p][l][1]))
            variance = np.var(distances)
            variances[l] = variance
        weights[k] = 1/np.sum(variances)

    return weights

def find_optimal_rotation(z1, z2):

	X = z1 # (Nx2)
	Y = z2 # (2xN)
	# import pdb; pdb.set_trace()

	U, S, Vt = np.linalg.svd(np.matmul(X, Y.T))

	R = np.matmul(Vt.T, U.T)

	if np.linalg.det(R) == 1:
		return R 
	else:
		M = np.eye(U.shape[0])
		M[-1,-1] = -1
		R = np.matmul(Vt.T, np.matmul(M, U.T))
		return R

def find_preshape_space(z):
    
	centroid = np.mean(z, axis=1, keepdims=True)

	# Put points on the same hyperplane
	z_prime = z - centroid

	z_prime_norm = np.linalg.norm(z_prime, axis=(1,2), keepdims=True)

	# Put points on the same hypersphere
	z_preshape = z_prime/z_prime_norm 

	return z_preshape

def find_mean_case1(pointsets):

	z = pointsets
	z_mean = np.expand_dims(z[0], axis=0)

	prev_z_mean = z_mean

	while True:

		# For a given Mean, Find optimal transformation parameters
		# import pdb; pdb.set_trace()

		z = find_preshape_space(z)
		z_mean = find_preshape_space(z_mean)

		for i in range(z.shape[0]):
			R = find_optimal_rotation(z[i], z_mean[0])
			z[i] = np.matmul(R, z[i])

		# Find mean for a given theta
		z_mean = np.mean(z, axis=0, keepdims=True)
		z_mean = z_mean/np.linalg.norm(z_mean)

		if np.linalg.norm(prev_z_mean-z_mean) < 0.00001:
			break 

		prev_z_mean = z_mean

	return z_mean, z

def find_mean_case2(pointsets):

	z = pointsets
	z_mean = np.expand_dims(z[0], axis=0)

	prev_z_mean = z_mean
	weights = find_weights(z)

	while True:
        

		# For a given Mean, Find optimal transformation parameters
        
		z_mean = find_preshape_space(z_mean)

		for i in range(z.shape[0]):
			sR, T = align_pointsets_case2(weights,z_mean,z[i])
			X = np.dot(sR, z[i].T)
			z[i] = (X + T).T

		# Find mean for a given theta
		z_mean = np.mean(z, axis=0, keepdims=True)
		z_mean = z_mean/np.linalg.norm(z_mean)
		print(np.linalg.norm(prev_z_mean-z_mean))

		if np.linalg.norm(prev_z_mean-z_mean) < 0.00001:
			break 

		prev_z_mean = z_mean

	return z_mean, z

def find_covariance_matrix(Z, mean):
	# mean - 1xnxd
	# z - Nxnxd

	N = Z.shape[0]
	z_vec_dim = Z.shape[1]*Z.shape[2]
	mean = mean.reshape((1,z_vec_dim))
	mean = mean.T
	Z = Z.reshape((N,z_vec_dim))
	Z = Z.T
	cov_matrix = np.cov(Z)

	return cov_matrix





