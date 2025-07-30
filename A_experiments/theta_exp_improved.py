import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

sys.path.append('..')
from src.compress_sensing import *
from src.utility import *

'''
Big question associated with this folder: 
    Why do V1 observations make for better image reconstructions than pixel or gaussian?

Current Thoughts:
    V1 mutual coherence is very bad, but reconstructions remain better
        Currently investigating why
    Hypothesis (Hyp): V1 observations end up with areas with values 0 and close to 0, so
        when we create the dot product matricies and normalize them, the dot products are being
        divided by something close to 0, 'blowing them up'

W = measurement_matrix
U = basis_matrix
theta = A = design_matrix
'''

small_img = "tree_part1.jpg"
big_img="peppers.png"
method = 'dct'
observation="pixel"
mode = '-c'
alpha=0.1
num_cell_100 = 100
num_cell_300 = 300
cell_size = 7    # receptive field size (200;.001 like gaussian)
blob_size = 2  # formerly sparse_freq            
num = 20

## For wavelet variable
lv= 2
dwt_type= 'db2'


plt.ion()

#Load Images:
# Represent image as numpy array to make it easier to process
small_img_arr = process_image(small_img, mode)
small_img_arr_gray = process_image(small_img, False) #change from 'gray' to False
big_img_arr = process_image(big_img, mode)
big_img_arr_gray = process_image(big_img, False) #change from 'gray' to False


def generate_design_matrix(measurement_matrix):
    '''
    Generate design_matrix for given weight matrix

    Parameters
    ----------

    measurement_matrix: array_like
        Lists of weighted data
    '''

    num_cell, n, m = measurement_matrix.shape
    design_matrix = fft.dctn(measurement_matrix.reshape(num_cell, n, m), norm = 'ortho', axes = [1, 2])
    design_matrix = design_matrix.reshape(num_cell, n * m) # PASS INTO COHERENCE FUNCTION 
    return design_matrix

def compute_mutual_coherence(design_matrix) :
    '''
    Compute mutual coherence for generic given matrix

    Parameters
    ----------

    design_matrix: array_like
        matrix with more than one column

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

    '''
    col_norms = np.linalg.norm(design_matrix, axis=0)
    x = design_matrix / col_norms 
    M = x.T @ x 
    np.fill_diagonal(M, 0) 
    return np.abs(M).flatten().max() 

def dot_product_matrix(img_arr, obs_type, num_cell, cell_size = None, blob_size = None, center = None):
    '''
    Create an array of dot products between columns

    Parameters
    ----------

    img_arr: numpy_array
        (n, m) shape image containing array of pixels.

    obs_type: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''

    if obs_type == 'V1':
        measurement_matrix, Y = generate_V1_observation(img_arr, num_cell, cell_size, blob_size, center)
        design_matrix = generate_design_matrix(measurement_matrix)
    if obs_type == "pixel":
        measurement_matrix, Y = generate_pixel_observation(img_arr, num_cell)
        design_matrix = generate_design_matrix(measurement_matrix)
    if obs_type == "gaussian":
        measurement_matrix, Y = generate_gaussian_observation(img_arr, num_cell)
        design_matrix = generate_design_matrix(measurement_matrix)

    col_norms = np.linalg.norm(design_matrix, axis=0)
    x = design_matrix / col_norms 
    M = x.T @ x
    np.fill_diagonal(M, 0)
    return np.abs(M)

def mutual_coherence_matrix(img_arr, n, num_cell, obs_type, cell_size = None, blob_size = None, center = None) :
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
    
    The how:
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
            M[i] = compute_mutual_coherence((design_matrix))
        if obs_type == "pixel":
            measurement_matrix, Y = generate_pixel_observation(img_arr, num_cell)
            design_matrix = generate_design_matrix(measurement_matrix)
            M[i] = compute_mutual_coherence(design_matrix)
        if obs_type == "gaussian":
            measurement_matrix, Y = generate_gaussian_observation(img_arr, num_cell)
            design_matrix = generate_design_matrix(measurement_matrix)
            M[i] = compute_mutual_coherence(design_matrix)
    return M

def sort_design_matrix(design_matrix):
    '''
    Sorts design_matrix frequencies (?)
    '''

    arr = np.arange(30)
    kx = np.tile(arr, 30)
    ky = np.repeat(arr, 30)

    ksum = kx**2 + ky**2
    perm = np.argsort(ksum) 
    return design_matrix[:, perm]

def high_freq_table(img_arr, obs_type, num_cell, cell_size = None, blob_size = None, center = None):
    '''
    Creates a table identifying which DCT basis frequencies are where

    Parameters
    ----------

    img_arr:
        (n, m) shape image containing array of pixels

    obs_type: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''

    M = dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, blob_size, center)
    M_vec = np.ravel(M) # dot product matrix as a vector
    #M_coords will be the coordinates of each coherence in M, which
    #we want to keep track of bc that'll tell us where the largest coherences are
    n = M.shape[0]
    M_coords = np.unravel_index(range(n**2), (n,n))
    arr = np.arange(30)
    kx = np.tile(arr, 30)
    ky = np.repeat(arr, 30)
    perm = np.argsort(M_vec)[::-1] # tells how to sort, reversed to decrease
    M_vec[perm] # decreasing
    i = M_coords[0][perm]
    j = M_coords[1][perm]
    # look into kx, ky to map i -> (kx, ky)
    i_coords=[[]for s in range(i.shape[0])]
    j_coords=[[]for s in range(j.shape[0])]
    for m in range(i.shape[0]):
        i_coords[m] = [kx[i[m]],ky[i[m]]] #assigns kx,ky coordinates to elements in i
    for m in range(j.shape[0]):
        j_coords[m] = [kx[j[m]],ky[j[m]]] #assigns kx,ky coordinates to elements in j
    i_arr = [i,i_coords]
    j_arr = [j,j_coords]
    sorted = pd.DataFrame({
        "i": i,
        "j": j,
        "(kx_i,ky_i)": i_coords,
        "(kx_j,ky_j)": j_coords,
        "MC": M_vec[perm]
    })
    return sorted


def generate_coeff_vector(img_arr, num_cell, cell_size, blob_size):
    '''
    Generates the coeffiecient vector for frequencies present in img

    Parameters
    ----------

    img_arr:
        (n, m) shape image containing array of pixels

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''

    n, m = img_arr.shape
    c = fft.dctn(img_arr, norm = 'ortho', axes = [0, 1])    
    return c   #.reshape(n*m,1)

def generate_ctDc(img_arr, obs_type, num_cell, norm = 2, diagonal = 0, cell_size = None, blob_size = None, center = None):
    '''
    Returns coefficient matrix * dot product matrix based on norm and what 
    values the diagonal is set to

    Parameters
    ----------

    img_arr: array_like
        (n, m) shape image containing array of pixels

    obs_type: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    norm: int
        np.linalg.norm(coeffs) ** norm
        norm type to divide by

    diagonal: int
        Number to replace diagonal values with in dot vector
        0: will return metric without altering diagonal
        diagonal >0: will return metric having replaced dot_vec diagonal

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''
    coeffs= generate_coeff_vector(small_img_arr_gray,num_cell,cell_size,blob_size).flatten()
    dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, blob_size, center)
    # dot_vec = np.linalg.inv(dot_vec)
    
    if diagonal >= 1:
        metric = np.fill_diagonal(dot_vec, diagonal)
        metric = coeffs.T @ dot_vec @ coeffs / np.linalg.norm(coeffs) ** norm
        return metric
    else:
        metric = coeffs.T @ dot_vec @ coeffs / np.linalg.norm(coeffs) ** norm
        return metric
    
def generate_Dc(img_arr, obs_type, num_cell, norm = 1, cell_size = None, blob_size = None, center = None):
    '''
    Returns dot product matrix * coefficient matrix based on norm

    Parameters
    ----------

    img_arr: array_like
        I(n, m) shape image containing array of pixels

    observation: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    norm: int
        np.linalg.norm(coeffs) ** norm
        norm type to divide by

    diagonal: int
        Number to replace diagonal values with in dot vector
        0: will return metric without altering diagonal
        diagonal >0: will return metric having replaced dot_vec diagonal

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''

    coeffs= generate_coeff_vector(small_img_arr_gray,num_cell,cell_size,blob_size).flatten()
    dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, blob_size, center)

    if norm <= 0:
        return np.linalg.norm(dot_vec @ coeffs, np.inf)
    else:
        return np.linalg.norm(dot_vec @ coeffs, norm)

