import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.compress_sensing import *
from src.utility import *
from PIL import Image, ImageOps
import sys
from A_experiments.theta_exp_improved import *

'''
Exploring the col_norms as histograms.
'''

def dot_product_matrix_mod(img_arr, obs_type, num_cell, cell_size = None, blob_size = None, center = None):
    '''
    Create an array of dot products between columns
    Modified - normalize after matrix multiplication
        To see if 

    Parameters
    ----------

    img_arr: numpy_array
        (n, m) shape image containing array of pixels.

    observation: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].
    '''

    if obs_type == 'V1':
        measurement_matrix, Y = generate_V1_observation(img_arr, num_cell, cell_size, blob_size, center)
        #col_norms = np.linalg.norm(measurement_matrix, axis = 0)
        design_matrix = generate_design_matrix(measurement_matrix)
    if obs_type == "pixel":
        measurement_matrix, Y = generate_pixel_observation(img_arr, num_cell)
        design_matrix = generate_design_matrix(measurement_matrix)
    if obs_type == "gaussian":
        measurement_matrix, Y = generate_gaussian_observation(img_arr, num_cell)
        design_matrix = generate_design_matrix(measurement_matrix)

    
    
     
    # M = design_matrix.T @ design_matrix
    # col_norms = np.linalg.norm(M, axis=0)
    # M = M / col_norms
    # np.fill_diagonal(M, 0)
    col_norms = np.linalg.norm(design_matrix, axis=0)
    x = design_matrix / col_norms
    M = x.T @ x 
    np.fill_diagonal(M, 0) 

    #plt.hist(col_norms, 200)
    #plt.title(obs_type)
    #x = design_matrix / col_norms
    return np.abs(M), col_norms

blob_size = 2
plt.figure()
v1_dot, v1_norms = dot_product_matrix_mod(small_img_arr_gray, "V1", num_cell_300, cell_size, blob_size)
#bins = np.linspace(0, 0.35,200)
bins=200
plt.hist(v1_dot.flatten(), bins, cumulative = False, density = True, label = "v1")

pix_dot, pix_norms = dot_product_matrix_mod(small_img_arr_gray, "pixel", num_cell_300, cell_size, blob_size)
plt.hist(pix_dot.flatten(), bins, cumulative = False, density = True, label = "pixel")

gauss_dot, gauss_norms = dot_product_matrix_mod(small_img_arr_gray, "gaussian", num_cell_300, cell_size, blob_size)
plt.hist(gauss_dot.flatten(), bins, cumulative = False, density = True, label = "gauss")

plt.xlabel("Dot Product")
plt.ylabel("Frequency")
plt.legend()
plt.yscale('log')
plt.show()
plt.savefig("Dot_Hist.svg") # saves a level above

plt.figure()
plt.hist(v1_norms.flatten(), bins, cumulative = False, density = True, label = "v1")
plt.hist(pix_norms.flatten(), bins, cumulative = False, density = True, label = "pixel")
plt.hist(gauss_norms.flatten(), bins, cumulative = False, density = True, label = "gauss")
plt.xlabel("Col Norm")
plt.ylabel("Frequency")
plt.legend()
plt.yscale('log')
plt.show()
plt.savefig("Norms Hist")
