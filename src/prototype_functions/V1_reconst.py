import numpy as np
import sys
sys.path.append("../")
from structured_random_features.models.weights import V1_weights

# Packages for fft and fitting data
from scipy import fftpack as fft
from sklearn.linear_model import Lasso

def generate_Y(W, img):
    ''' Generate sample y vector variable for data reconstruction using constant matrix W (containing open indices). Function does inner product W matrix with image array to find sample y vector, 
    
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
    
    
    num_cell = W.shape[0]
    n, m = img.shape
    W = W.reshape(num_cell, n*m)
    y = W @ img.reshape(n * m, 1)
    return y

def generate_V1_variables(num_cell, cell_size, sparse_freq, img):
    ''' Automatically generates variables needed for data reconstruction using V1 weights.
    
    Parameters
    ----------
    num_cell : int
        Number of blobs that will be used to be determining which pixles to grab and use
    
    cell_size : int
        Determines field size of opened and closed blob of data. Affect the data training
        
    sparse_freq : int
        Determines filed frequency on how frequently opened and closed area would appear. Affect the data training
    
    img : array_like
          (n, m) shape image containing array of pixels
    
    Returns
    ----------
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
    
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
    
    '''
    # Get size of image
    dim = np.asanyarray(img).shape
    n, m = dim
    # Store generated V1 cells in W
    W = V1_weights(num_cell, dim, cell_size, sparse_freq) 
    
    # Retrieve y from W @ imgArr
    y = W @ img.reshape(n*m, 1)

    # Resize W to shape (num_cell, height of image, width of image) for fetching into function
    W = W.reshape(num_cell, dim[0], dim[1])
    return W, y

# Depending on basis, make dwt or fft works
def reconstruct(W, y, alpha = None, fit_intercept = False,):
    ''' Reconstruct gray-scaled image using sample data fitting into LASSO model
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
        
    alpha : float
        Penalty for fitting data onto LASSO function to search for significant coefficents
    
    fit_intercept : bool
        default set to false to prevent LASSO function to calculate intercept for model
    
    Returns
    ----------
    theta : array_like
        (num_V1_weights/sample_size, n * m) shape. Data after discrete fourier transform applied 
    
    reformed : array_like
        (n, m) shape array. Reconstructed image pixel array
        
    s : vector
        (num_V1_weights/sample_size, 1) shape. Coefficient value generated from fitting data to LASSO. Contains significant values with most of vector zeroed out.
    '''
    
    sample_sz, n, m = W.shape
    
    
    if alpha == None :
        alpha = 1 * 50 / num_cell
        
    if fit_intercept:
        raise Exception("fit_intercept = True not implemented")
    
    ## WΨ
    #W_reshaped = W.reshape(sample_sz, n, m)
#     Possible dwt solution
#     theta = np.zeros(sample_sz, n * m)
#     for i in range(sample_sz):
#         theta_i = dwt2(W[i, :, :])
#         theta[i, :] = pywt.ravel_coeffs(theta_i)
    
    theta = fft.dctn(W.reshape(sample_sz, n, m), norm = 'ortho', axes = [1, 2])
    theta = theta.reshape(sample_sz, n * m)

    ## Initialize Lasso and Fit data
    mini = Lasso(alpha = alpha, fit_intercept = fit_intercept)
    mini.fit(theta, y)
    
    ## Retrieve sparse vector s
    s = mini.coef_
    
    # Reform the image using sparse vector s with inverse descrete cosine
    reform = fft.idctn(s.reshape(n, m), norm='ortho', axes=[0,1])
    if fit_intercept:
        reform += mini.intercept_ # not sure this is right
    
    #return theta, reformed img, sparse vectors
    return theta, reform, s

def color_reconstruct(img_arr, num_cell, cell_size, sparse_freq, alpha = None) :
    ''' Reconstruct colored (RGB) image with sample data
    
    Parameters
    ----------
    img_arr : numpy_array
          (n, m) shape image containing array of pixels
          
    num_cell : int
        Number of blobs that will be used to be determining which pixles to grab and use
    
    cell_size : int
        Determines field size of opened and closed blob of data. Affect the data training
        
    sparse_freq : int
        Determines filed frequency on how frequently opened and closed area would appear. Affect the data training
      
    alpha : float
        Penalty for fitting data onto LASSO function to search for significant coefficents

    Returns
    ----------
    final : numpy_array
        (n * m) shape array containing reconstructed RGB image array pixels.
    
    '''
    
    if alpha == None :
        alpha = 1 * 50 / num_cell
    i = 0
    dim = img_arr[:,:,i].shape

    W = V1_weights(num_cell, dim, cell_size, sparse_freq) 
    final = np.zeros(img_arr.shape)

    # with same V1 cells generated, reconstruct images for each of 3 rgb arrays and append to final
    while (i < 3):
        img_arr_pt = img_arr[:,:,i]
        img_arr_pt_dim = img_arr_pt.shape
        n_pt, m_pt = img_arr_pt_dim
        y = generate_Y(W, img_arr_pt)
        W_model = W.reshape(num_cell, n_pt, m_pt)
        theta, reconst, s = reconstruct(W_model, y, alpha)
        final[:,:,i] = reconst
        i+=1
        
    final = np.round(final).astype(int)
    final[final < 0] = 0
    final[final > 255] = 255
    final = final.astype(int)
    return final

        
