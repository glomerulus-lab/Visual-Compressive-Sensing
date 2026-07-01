import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

sys.path.append('..')
from src.compress_sensing import *
from src.utility import *
from A_experiments.theta_exp_improved import *


'''
Finding dimensions of theta.
'''

# Find the singular values of theta
measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
theta_V1 = generate_design_matrix(measurement_matrix_V1)

U, S_V1, V = np.linalg.svd(theta_V1)

measurement_matrix_pix, pix_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
theta_pix = generate_design_matrix(measurement_matrix_pix)

U, S_pix, V = np.linalg.svd(theta_pix)

measurement_matrix_gauss, gauss_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
theta_gauss = generate_design_matrix(measurement_matrix_gauss)

U, S_gauss, V = np.linalg.svd(theta_gauss)

rank_V1 = np.count_nonzero(S_V1) 
rank_pix = np.count_nonzero(S_pix)
rank_gauss = np.count_nonzero(S_gauss)

#Find the dimension of V1, pix, gauss
n = 20
dim_arr_V1 = np.zeros(n)
dim_arr_pix = np.zeros(n)
dim_arr_gauss = np.zeros(n)

for i in range(n):
    # Generate theta for each obs type:
    measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)

    U, S_V1, V = np.linalg.svd(theta_V1)

    measurement_matrix_pix, pix_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
    theta_pix = generate_design_matrix(measurement_matrix_pix)

    U, S_pix, V = np.linalg.svd(theta_pix)

    measurement_matrix_gauss, gauss_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)

    U, S_gauss, V = np.linalg.svd(theta_gauss)

    # Find the dimension of theta and log it for later plotting: 
    p_V1 = S_V1 / np.sum(S_V1)
    dim_V1 = 1 / np.sum(p_V1 * p_V1)
    dim_arr_V1[i] = dim_V1

    p_pix = S_pix / np.sum(S_pix)
    dim_pix = 1 / np.sum(p_pix * p_pix)
    dim_arr_pix[i] = dim_pix

    p_gauss = S_gauss / np.sum(S_gauss)
    dim_gauss = 1 / np.sum(p_gauss * p_gauss)
    dim_arr_gauss[i] = dim_gauss

    
# Plot the dimensions of theta for comparison
plt.figure()
plt.boxplot( [dim_arr_V1, dim_arr_pix, dim_arr_gauss], tick_labels=['V1','Pix', "Gaussian"])

plt.ylabel("Dimension")
plt.title("Dimensions of Theta over " + str(n) + " runs")
plt.show()