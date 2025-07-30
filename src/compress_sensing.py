import numpy as np
import sys
import os
from structured_random_features.src.models.weights import V1_weights

# Packages for dct, dwt and fitting data
from scipy import fftpack as fft
import pywt
from pywt import wavedecn
from sklearn.linear_model import Lasso

import warnings
from sklearn.exceptions import ConvergenceWarning
from src.utility import *

# Packages for images
from PIL import Image, ImageOps


# Generate General Variables
def generate_Y(W, img_arr):
    ''' 
    Generate sample y vector variable for data reconstruction using constant
    matrix W (containing open indices).
    Function does inner product W matrix with image array to find sample y 
    vector
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights/sample_size, n*m) shape array. Lists of weighted data
        
    img : array_like
        (n, m) shape image containing array of pixels
    
    Returns
    ----------
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
    
    '''
    num_cells, n, m = W.shape
    W = W.reshape(num_cells, n*m)
    y = W @ img_arr.reshape(n * m, 1)
    return y

def generate_V1_weights(num_cell, dim, cell_size, blob_size, center = None):
    # Store generated V1 cells in W
    n, m = dim
    W = V1_weights(num_cell, dim, cell_size, blob_size, center)
    # Resize W to shape (num_cell, height of image, width of image) for 
    # fetching into function
    W = W.reshape(num_cell, n, m)
    return W

def generate_V1_observation(img_arr, num_cell, cell_size, blob_size, center = None):
    ''' 
    Automatically generates variables needed for 
    data reconstruction using V1 weights.
    
    Parameters
    ----------    
    img_arr : array_like
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
    
    Returns
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
    
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
    '''

    dim = np.asanyarray(img_arr).shape[:2]
    W = generate_V1_weights(num_cell, dim, cell_size, blob_size, center)
    # Retrieve y from W @ imgArr
    y = generate_Y(W, img_arr)
    return W, y


def generate_pixel_weights(num_cell, n, m, rand_index):
    W = np.eye(n * m)[rand_index, :] * np.sqrt(n * m)
    W = W.reshape(num_cell, n, m)
    return W

# Generate pixel Variables
def generate_pixel_observation(img_arr, num_cell) :
    ''' 
    Generate random pixel arrays with its indices length of sample size.
        
    Parameters
    ----------
    img_arr : array_like
        (n, m) sized data array
    
    num_cell : int
        Number of sample data to be collected
    
    Returns
    ----------
    C3D : array like
        (sample_size, n, m) shape array that only has one index open 
        that corresponds to y vector per each (n, m) shape array.
    
    y : vector
        Actual value of randomly selected indices
    '''
    
    n, m = img_arr.shape[:2]
    rand_index = np.random.randint(0, n * m, num_cell)
    W = generate_pixel_weights(num_cell, n, m, rand_index)
    y = generate_Y(W, img_arr)
    return W, y

# Generate Gaussian Weights
def generate_gaussian_weights(num_cell, n, m):
    W = np.random.randn(num_cell, n, m)
    return W

def generate_gaussian_observation(img_arr, num_cell):
    ''' 
    Generate 3 dimensional arrays. 
    Creates arrays of randomly generated gaussian 
    2 dimensional arrays as a weight W.
    
    Parameters
    ----------
    img_arr : array_like
        (n, m) sized data array.
        
    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
        
    Returns
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of gaussian weighted data.
    
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image.
    '''
    n, m = img_arr.shape[:2]
    W = generate_gaussian_weights(num_cell, n, m)
    y = generate_Y(W, img_arr)
    return W, y

# Error Calculation by Frobenius Norm
def error_calculation(img_arr, reconst):
    ''' 
    Compute mean error per each data using frobenius norm.
        
    Parameters
    ----------
    img_arr : array_like
        (n, m) shape array of actual dataset.
    
    reconst : array_like
        (n, m) shape array of reconstructed array.
    
    Returns
    ----------
    error : float
        Computed normalized error value per each pixel.
    '''
    if (len(img_arr.shape) == 3):
        n, m, d = img_arr.shape
    else: 
        n, m = img_arr.shape

    error = np.linalg.norm(img_arr - reconst) / np.sqrt(m * n)
    return error

# Reconstruction (Current Methods: Fourier Base Transform, Wavelet Transform)
def fourier_reconstruct(W, y, alpha, sample_sz, n, m, fit_intercept) :
    ''' 
    Reconstruct signals through cosine transform.
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data.
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image.
        
    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.
        Defaults to 1 / 50 if not passed as an arg.

    sample_sz : int
        Number of sample collected.
    
    n : int
        Height of each data.
    
    m : int
        Width of each data.
    
    fit_intercept : bool
        default set to false to prevent 
        LASSO function to calculate intercept for model.
        
    Returns
    ----------
    img : array_like
        (n, m) shape array. Reconstructed image pixel array.
    '''
    
    # Ignore convergence warning to allow convergence warning not filling up 
    # all spaces when testing
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    theta = fft.dctn(W.reshape(sample_sz, n, m), norm = 'ortho', axes = [1, 2])
    
    theta = theta.reshape(sample_sz, n * m)
    #print(theta.shape)
    ## Initialize Lasso and Fit data
    mini = Lasso(alpha = alpha, fit_intercept = fit_intercept)
    mini.fit(theta, y)
    ## Retrieve sparse vector s
    s = mini.coef_
    #    print(s.shape)
    img = fft.idctn(s.reshape(n, m), norm='ortho', axes=[0,1])
    return img

def wavelet_reconstruct(W, y, alpha, sample_sz, n, m,
                        fit_intercept, dwt_type, lv) :
    ''' 
    Reconstruct signals through wavelet transform.
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data.
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image.
        
    alpha : float
        Penalty for fitting data onto LASSO function to search 
        for significant coefficents.
    
    sample_sz : int
        Number of sample collected.
    
    n : int
        Height of each data.
    
    m : int
        Width of each data.
    
    fit_intercept : bool
        Default set to false to prevent LASSO function to 
        calculate intercept for model.
    
    dwt_type : String
        type of dwt method to be used
        ex)'haar', 'db1', 'db2', ...
        
    lv : int
        Generate level of signal frequencies when dwt is used.
        
    Returns
    ----------
    img : array_like
        (n, m) shape array. Reconstructed image pixel array.
    '''
    
    # Ignore convergence warning to allow convergence warning not filling up 
    # all spaces when testing
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    dwt_sample = wavedecn(W[0], wavelet = dwt_type, level = lv, mode = 'zero')
    coeff, coeff_slices, coeff_shapes = pywt.ravel_coeffs(dwt_sample)
    theta = np.zeros((len(W), len(coeff)))
    theta[0, :] = coeff 

    # Loop the wavedecn to fill theta
    for i in range(sample_sz):
        theta_i = wavedecn(W[i], wavelet= dwt_type, level = lv, mode = 'zero')
        theta[i, :] = pywt.ravel_coeffs(theta_i)[0]

    mini = Lasso(alpha = alpha, fit_intercept = fit_intercept)
    mini.fit(theta, y)
    s = mini.coef_
    #print(s.shape)

    # print(s.shape)
    s_unravel = pywt.unravel_coeffs(s, coeff_slices, coeff_shapes)
#    print(s_unravel)
    #   print(coeff_slices)
    #  print(coeff_shapes)
    
    img = pywt.waverecn(s_unravel, dwt_type, mode = 'zero')
    #    print(img.shape)
    return img

def generate_observations(img_arr, num_cell, observation, cell_size = None,
                          blob_size = None):
    ''' 
    Helper function to generate observations using the specified technique.
    
    Parameters
    ----------
    img_arr : numpy_array
        (n, m) shape image containing array of pixels.
          
    num_cell : int
        Number of blobs that will be used to be determining 
        which pixels to grab and use.
    
    observation : String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how 
        frequently opened and closed area would appear. 
        Affect the data training.
    
    Returns
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data.
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image.
   
    '''
    # Check if the cell_size and blob_size is none while it is conduction 
    # V1 observation
    if (observation.lower() == "v1" and (cell_size == None
                                         or blob_size == None)) :
        print(f"For {observation} observation, both cell_size"+
              " and blob_size parameters are required")
        sys.exit(0)
    if (type(num_cell) == str):
        print(num_cell)
        print(type(num_cell))
        sys.exit(0)
    if (observation.lower() == "v1"):
        W, y = generate_V1_observation(img_arr, num_cell, cell_size, blob_size)
    elif (observation.lower() == "gaussian"):
        W, y = generate_gaussian_observation(img_arr, num_cell)
    elif (observation.lower() == "pixel"):
        W, y = generate_pixel_observation(img_arr, num_cell)
    else:
        print("This obervation technique is currently not supported")
        print("Please use valid observation: ['pixel', 'gaussian', 'V1']")
    return W, y

def reconstruct(W, y, alpha = None, fit_intercept = False, method = 'dct',
                lv = 4, dwt_type = 'db2'):
    ''' 
    Reconstruct gray-scaled image using sample data fitting into LASSO model.
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data.
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image.
        
    alpha : float
        Penalty for fitting data onto LASSO function to search 
        for significant coefficents.
    
    fit_intercept : bool
        Default set to false to prevent LASSO function 
        to calculate intercept for model.
    
    method : String
        Currently supporting dct and dwt (discrete cosine/wavelet transform).
        Default set to dct
        
    lv : int
        Generate level of signal frequencies when dwt is used
        Default set to 4
        
    dwt_type : String
        type of dwt method to be used
        Default set to db2
        
    Returns
    ----------
    img : array_like
        (n, m) shape array. Reconstructed image pixel array
        
    '''
    
    num_cell, n, m = W.shape
      
    if alpha == None :
        alpha = 1 * 50 / num_cell
        
    if fit_intercept:
        raise Exception("fit_intercept = True not implemented")
    
    if (method == 'dct') :
        img = fourier_reconstruct(W, y, alpha, num_cell, n, m, fit_intercept)
    elif (method == 'dwt') :
        img = wavelet_reconstruct(W, y, alpha, num_cell, n, m,
                                  fit_intercept, dwt_type, lv)

        # Reform the image using sparse vector s with inverse discrete cosine
        
    if fit_intercept:
        reform += mini.intercept_ # not sure this is right
    
    #return reformed img
    return img

def color_experiment(img_arr, num_cell, cell_size = None, blob_size = None,
                     alpha = None, fit_intercept = False, method = 'dct',
                     observation = 'pixel', lv = 4, dwt_type = 'db2', W = None, rand_index=None) :
    ''' 
    Reconstruct colored (RGB) image with sample data.
    
    Parameters
    ----------
    img_arr : numpy_array
          (n, m) shape image containing array of pixels.
          
    num_cell : int
        Number of blobs that will be used to be determining which 
        pixels to grab and use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently opened and 
        closed area would appear. Affect the data training.
      
    alpha : float
        Penalty for fitting data onto LASSO function to search 
        for significant coefficents.
    
    fit_intercept : bool
        Parameter for LASSO function. Automatically adjust intercept 
        calculated by LASSO, but it is recommended to set if False.
        
    method : String
        Determines whether the function will be based on 
        discrete cosine transform (dct) or discrete wavelet transform (dwt). 
        Default set up to dct.
    
    observation : String
        Observation technique that are going to be used to collect samples. 
        Default set up to 'pixel'.
        Supported observation : ['pixel', 'gaussian', 'V1'].
    
    lv : int
        Determines level of frequency details for wavelet transform. 
        Not used for dct.
    
    dwt_type : String
        Determines types of wavelet transform when dwt is used for its method.
        Not used for dct.

    Returns
    ----------
    img : numpy_array
        (n * m) shape array containing reconstructed RGB image array pixels.
    '''

    i = 0
    dim = img_arr[:,:,i].shape
    n, m = dim
    
    # Computes num_cell if the num_cell is in between 0~1.
    # Get percentage of num_cell with its img dim (num_cell * n * m)
    if (num_cell < 1):
        num_cell = int(round(num_cell * n * m))
        
    #alpha parameter is dependent on the number of cell if alpha is not specified
    if alpha == None :
        alpha = 1 * 50 / num_cell

    img = np.zeros(img_arr.shape)

    # with same V1 cells generated, reconstruct images for each of 3 rgb 
    # arrays and append to img
    while (i < 3):
        img_arr_pt = img_arr[:,:,i]
        img_arr_pt_dim = img_arr_pt.shape
        n_pt, m_pt = img_arr_pt_dim
        
        if W is not None:
            y = generate_Y(W, img_arr_pt)
        
        else :
            W, y = generate_observations(img_arr_pt, num_cell, observation,
                                     cell_size, blob_size)
            
        if (method == 'dct') :
            reconst = reconstruct(W, y, alpha, method = method)
        else :
            reconst = reconstruct(W, y, alpha, method = method,
                                  lv = lv, dwt_type = dwt_type)
        img[:,:,i] = reconst
        i+=1
        
    # Fix any over/underestimated pixel between 0~255
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.round(img).astype(int)
    return img


def large_img_experiment(img_arr, num_cell, cell_size = None,
                         blob_size = None, filter_dim = (16, 16),
                         alpha = None, method = 'dct', observation = 'pixel',
                         lv = 2, dwt_type = 'db2', fixed_weights = False,
                         color = False) :
    ''' 
    Allows to reconstruct any size of signal data since regular reconstruct 
    function can only deal with small size of data. 
    For filter reconstruction function, it can reconstruct any size of data 
    as the function will break data into several parts and use reconstruction 
    on each part.
    
    Parameters
    ----------
    img_arr : numpy_array
          (n, m) or (n, m, d) shape image containing array of pixels.
          
    num_cell : int
        Number of blobs that will be used to be determining which pixels to use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        Default set to None as only V1 obervation requires it
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. Affect the data training.
        Default set to None as only V1 obervation requires it
    
    filter_dim : tuple
        Determines area of data that are going to be reconstructed, until
        all image is reconstructed. Zero padding technique used.
        Default set to (30, 30)
    
    alpha : float
        Penalty for fitting data onto LASSO function to search 
        for significant coefficents. 
        Default set to None, and if the parameter is None, it computes 
        alpha value that would fit to the data reconstruction depending on
        its size.
    
    fit_intercept : bool
        Parameter for LASSO function. Automatically adjust intercept 
        calculated by LASSO, but it is recommended to set if False.
        
    method : String
        Determines whether the function will be based on 
        discrete cosine transform (dct) or discrete wavelet transform (dwt). 
        Default set up to dct.
    
    observation : String
        Observation technique that are going to be used to collet sample 
        for reconstruction. 
        Default set up to 'pixel'.
        Supported observation : ['pixel', 'gaussian', 'V1']
            pixel: Select num_cell amount of random data point throughout data
            gaussian: Give weights to individual data, and select num_cell 
                      amount of random data point throughout data
            V1: Select random data that are observed by the num_cell amount of 
                neurocomputation model of visual cortex cells
    
    lv : int
        Determines level of frequency details for wavelet transform. 
        Not used for dct.
        Default set to level 2
    
    dwt_type : String
        Determines types of wavelet transform when dwt is used for its method.
        Not used for dct.
        Default set to 'db2'

    fixed_weight : bool
        Decide if reconstruction for each data part is going to use 
        same weight or random weight.
        Default set up to be False.
    
    color : bool
        Indicates if the image working on is color image or black/white image.
        Default set to be False.
        Possible colors are [True, False].
    
    Returns
    ----------
    img : numpy_array
        (n * m) shaped or (n * m * d) array containing reconstructed 
        grayscale/RGB image array pixels.
    '''
                       
    # Create Filter with size of filter_dim
    filt = np.zeros(filter_dim)
    filt_n, filt_m = filter_dim
    
    # Computes num_cell if the num_cell is in between 0~1.
    # Get percentage of num_cell with its filter dim (num_cell * filt_n * filt_m)
    if (num_cell < 1):
        num_cell = int(round(num_cell * filt_n * filt_m))
    
    # Edge handling case when user wants black/white reconstruction, but passed
    # in image array has rgb channel. Grayscale the image if so.
    if (not color and len(img_arr.shape) == 3):
        img_arr = np.asarray(ImageOps.grayscale(Image.fromarray(img_arr)))
        
    #alpha parameter is dependent on the number of cell if alpha is not specified
    if (alpha == None) :
        alpha = 1 * 50 / num_cell
    
    # Retrieve image dimension
    if color :
        n, m, d = img_arr.shape
    else:
        n, m = dim = img_arr.shape

    # If user wants to test the fixed weights. Default set to Fixed
    W = None
    rand_index = None
    if fixed_weights == True:
        dim = (filt_n, filt_m)
        # Store generated V1 cells in W
        if observation == 'V1':
            W = generate_V1_weights(num_cell, dim, cell_size, blob_size)
            # W = V1_weights(num_cell, dim, cell_size, blob_size)
            # W = W.reshape(num_cell, filt_n, filt_m)
        elif observation == 'pixel':
            n, m = img_arr.shape[:2]
            rand_index = np.random.randint(0, filt_n * filt_m, num_cell)
            # W = np.eye(filt_n * filt_m)[rand_index, :] * np.sqrt(filt_n * filt_m)
            # W = W.reshape(num_cell, filt_n, filt_m)
            W = generate_pixel_weights(num_cell, filt_n, filt_m, rand_index)
        elif observation == 'gaussian':
            # W = np.random.randn(num_cell, filt_n, filt_m)
            W = generate_gaussian_weights(num_cell, filt_n, filt_m)

    
    # Compute the size of the dimension once zero padding is applied
    padding_n = compute_zero_padding_dimension(n, filt_n)
    padding_m = compute_zero_padding_dimension(m, filt_m)
    
    # Process image with adding zero paddings
    if color :
        img_arr_padded = np.zeros((padding_n, padding_m, d))
        img_arr_padded[:n, :m, :] = img_arr
    else:
        img_arr_padded = np.zeros((padding_n, padding_m))
        img_arr_padded[:n, :m] = img_arr
    
    i = 1 # counter
    # Array that saves each part of completed reconstruced array
    result = np.zeros(img_arr.shape) 
    cur_n, cur_m = (0, 0)
    # Computes number of reconstruction batches to be done 
    # base on filter dimension 
    num_work = (padding_n * padding_m) // (filt_n * filt_m)
    
    #    print(filter_dim)
    #   print(alpha)
    #  print(num_cell)
    # For each batch of data, run reconstruction 
    for pt in range(num_work):
        # Keep track over height of the batches. If working column
        # (cur_m) exceeds/equals padded column, reset the cur_m and 
        # update row (cur_n) 
        if (cur_m >= padding_m) :
            cur_n += filt_n
            cur_m = 0
        # print(f"{pt} of {num_work}")
        # Compute the end of column for each batch, which is also
        # going to be the start point of next batch
        nxt_m = cur_m + filt_m
        # Reconstruct all 3 rgb channels if color
        if color :
            # Retrieve batch
            img_arr_pt = img_arr_padded[cur_n : (cur_n + filt_n), cur_m : nxt_m, :]
            
            reconst = color_experiment(
                img_arr_pt,  
                num_cell, 
                cell_size, 
                blob_size, 
                alpha = alpha, 
                method = method, 
                observation = observation,
                lv = lv, 
                dwt_type = dwt_type,
                W = W, 
                rand_index=rand_index)
            img_arr_padded[cur_n : (cur_n + filt_n), cur_m : nxt_m, :] = reconst
        else:    
            img_arr_pt = img_arr_padded[cur_n : (cur_n + filt_n), cur_m : nxt_m]
            
            # if fixed_weights == True, there is pre-set weight, so just compute y
            if fixed_weights == True:
                y = generate_Y(W, img_arr_pt)
            # else, all W is randomized for each batch of reconstruction
            else :
                W, y = generate_observations(img_arr_pt, num_cell, observation,
                                         cell_size, blob_size)

            W_model = W.reshape(num_cell, filt_n, filt_m)    
            #print(W.shape)
            #print(y.shape)
            if (method == 'dct'):
                reconst = reconstruct(W, y, alpha, method = method)
            else :
                reconst = reconstruct(W, y, alpha, method = method, lv = lv,
                                      dwt_type = dwt_type)
            img_arr_padded[cur_n : (cur_n + filt_n), cur_m : nxt_m] = reconst
        cur_m = nxt_m

        i+=1
    result = img_arr_padded[:n, :m, :d] \
        if color else img_arr_padded[:n,:m]
    # Fix any over/underestimated pixel between 0~255
    result[result < 0] = 0
    result[result > 255] = 255
    img = np.round(result).astype(int)
    
    return img
