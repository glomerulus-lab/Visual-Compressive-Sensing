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
If some col norms are close to 0, the system can't see that frequency and dividing
    by that col norm makes dot products blow up.
For some epsilon, drop all columns with norms less than epsilon.

'''

def compute_mutual_coherence(design_matrix, epsilon = 0) :
    '''
    Compute mutual coherence for generic given matrix

    Parameters
    ----------

    design_matrix: array_like
        matrix with more than one column

    epsilon: int
        max col norm we want to be kept

    The how:
    1. normalize columns of A (divide each by its norm):
       collect n = columns, m = rows
       for each column n, compute col_norm = sqrt(a_n1^2 + a_n2^2 ... + a_nm^2)
            for each a in n, a = a/col_norm
       A is now an array of normalized columns
    2. find max dot product between columns of A = mutual coherence
        create array total_dot
        for each column x in A
            create array dot = dot products between col x with every column after it
            add dot to total_dot
        return max(total_dot)

    Mod:
    1. if the column's norm is less than epsilon, replace the column with 0s

    '''
    rows, columns = design_matrix.shape
    col_norms = np.linalg.norm(design_matrix, axis=0)

    #should we be doing this for every type or just V1?
    for i in range(columns):
        if col_norms[i] >= epsilon:
            # print(str(i))
            # print(str(col_norms[i]))
            design_matrix[:,i] = 0

    x = design_matrix / col_norms
    M = x.T @ x 
    np.fill_diagonal(M, 0) 
    return np.abs(M).flatten().max() 


def mutual_coherence_matrix_mod(img_arr, n, num_cell, obs_type, cell_size = None, blob_size = None, center = None, epsilon = 0) :
    '''
    Returns a list of n computed mutual coherence(MC) values for given image and observation type
    
    Parameters
    ----------

    img_arr: array_like
        I(n, m) shape image containing array of pixels

    n: int 
        how many MC should be collected from one image, 
        with purpose of averaging and comparing

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.

    obs_type: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1']. 

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training. 
    
    The how:    col_norms = np.linalg.norm(design_matrix, axis=0)
    x = design_matrix / (col_norms + epsilon)
    M = x.T @ x 
    np.fill_diagonal(M, 0) 
    1. Create array M, will be our final list of MCs
    2. for n times, generate design_matrix and compute mutual coherence depending on ovserbation type
        add each MC value to M
    3. return M - to be plotted
    
    '''

    M = np.zeros(n)
    i = 0
    for i in range(n):
        if obs_type == 'V1':
            measurement_matrix, Y = generate_V1_observation(img_arr, num_cell, cell_size, blob_size, center)
            design_matrix = generate_design_matrix(measurement_matrix)
            #M[i] = compute_mutual_coherence(sort_design_matrix(design_matrix))
            M[i] = compute_mutual_coherence(design_matrix, epsilon)
        if obs_type == "pixel":
            measurement_matrix, Y = generate_pixel_observation(img_arr, num_cell)
            design_matrix = generate_design_matrix(measurement_matrix)
            M[i] = compute_mutual_coherence(design_matrix, epsilon)
        if obs_type == "gaussian":
            measurement_matrix, Y = generate_gaussian_observation(img_arr, num_cell)
            design_matrix = generate_design_matrix(measurement_matrix)
            M[i] = compute_mutual_coherence(design_matrix, epsilon)
    return M

epsilon = 20
num = 5
#Plot Modded Mutual Coherence
v1_mc = mutual_coherence_matrix_mod(small_img_arr_gray, num, num_cell_300,  "V1", blob_size, cell_size, epsilon = epsilon)
pix_mc = mutual_coherence_matrix_mod(small_img_arr_gray, num, num_cell_300, "pixel", epsilon = epsilon)
gaus_mc = mutual_coherence_matrix_mod(small_img_arr_gray, num,num_cell_300, "gaussian", epsilon = epsilon)
all_mc = [v1_mc, pix_mc, gaus_mc]
fig = plt.figure()
fig.suptitle(str(num) + " MC per Type, Epsilon = " + str(epsilon), fontsize=14)
ax = fig.add_subplot()
ax.boxplot(all_mc, tick_labels=['V1', 'pixel','Gaussian'])
plt.show()
plt.savefig("Less than Epsilon Mc")