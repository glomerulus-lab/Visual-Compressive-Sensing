import numpy as np
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform


def sensilla_covariance_matrix(dim, sampling_rate, duration, lowcut, highcut, decay_coef=np.inf, scale=1):
    '''
    Generates the (dim x dim) covariance matrix for Gaussain Process inspired by the STAs 
    of mechanosensory neurons in insect halteres. Decaying sinusoids.
    
    $$k(t, t') = \mathbb{E}[w(t)^T w(t')] =  \sum_{j=0}^{dim-1} \lambda_j \cos{\dfrac{i 2 \pi j (t-t')}{dim}} * exp((- \|t- N - 1\| + \|t'- N - 1\|) / decay_coef ** 2) $$
    $$ \lambda_j = \begin{cases} 1 & lowcut \leq highcut \\ 0 & otherwise \end{cases}$$

    Parameters
    ----------

    dim: int
        dimension of each random weight
        
    sampling_rate : int
        Sampling rate of the weights in Hz
    
    lowcut: int
        low end of the frequency band in Hz

    highcut : int
        high end of the frequency band in Hz
        
    decay_coef : float, default=np.inf
        controls the window of the weights in seconds
        With default value, the weights do not decay
    
    scale: float
        Normalization factor for Tr norm of cov matrix
    
    Returns
    -------
    C : array-like of shape (dim, dim) 
        Covariance matrix w/ Tr norm = scale * dim
    '''


    assert dim == int(sampling_rate * duration), "The dim of weights does not match sampling rate * duration"

    # time grid
    grid = np.arange(0, duration, 1 / sampling_rate)
    yy, xx = np.meshgrid(grid, grid)
    diff = xx - yy

    # cosine part
    low_idx = int(duration * lowcut)
    high_idx = int(duration * highcut)
    C_cos = np.zeros((dim, dim))
    for k in range(low_idx, high_idx):
        C_cos += np.cos(2 * np.pi * k * diff / duration)

    # exponential part
    C_exp = np.exp(((xx - duration) + (yy - duration)) / decay_coef)

    # final covariance matrix
    C = C_cos * C_exp 
    C *= (scale * dim / np.trace(C))
    C += 1e-5 * np.eye(dim)
    return C

def sensilla_weights(num_weights, dim, sampling_rate, duration, lowcut, highcut, decay_coef=np.inf, scale=1, seed=None):
    """
    Generates random weights with tuning similar to mechanosensory 
    neurons found in insect halteres and wings.

    Parameters
    ----------

    num_weights: int
        Number of random weights

    dim : int
        dim of each random weight

    sampling_rate : int
        Sampling rate of the weights
    
    lowcut: int
        low end of the frequency band in Hz

    highcut : int
        high end of the frequency band in Hz
        
    decay_coef : float, default=np.inf
        controls the window of the weights in seconds
        With default value, the weights do not decay
    
    seed : int, default=None
        Used to set the seed when generating random weights.
    
    Returns
    -------

    W : array-like of shape (num_weights, dim)
        Matrix of Random weights.
    """
    assert dim == int(sampling_rate * duration), "The dim of weights does not match sampling rate * duration"
    np.random.seed(seed)
    C = sensilla_covariance_matrix(dim, sampling_rate, duration, lowcut, highcut, decay_coef, scale)
    W = np.random.multivariate_normal(np.zeros(dim), cov=C, size=num_weights)
    return W


def V1_covariance_matrix(dim, size, spatial_freq, center, scale=1):
    """
    Generates the covariance matrix for Gaussian Process with non-stationary 
    covariance. This matrix will be used to generate random 
    features inspired from the receptive-fields of V1 neurons.

    C(x, y) = exp(-|x - y|/(2 * spatial_freq))^2 * exp(-|x - m| / (2 * size))^2 * exp(-|y - m| / (2 * size))^2

    Parameters
    ----------

    dim : tuple of shape (2, 1)
        Dimension of random features.

    size : float
        Determines the size of the random weights 

    spatial_freq : float
        Determines the spatial frequency of the random weights  
    
    center : tuple of shape (2, 1)
        Location of the center of the random weights.

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    Returns
    -------

    C : array-like of shape (dim[0] * dim[1], dim[0] * dim[1])
        covariance matrix w/ Tr norm = scale * dim[0] * dim[1]
    """

    x = np.arange(dim[0])
    y = np.arange(dim[1])
    yy, xx = np.meshgrid(y, x)
    grid = np.column_stack((xx.flatten(), yy.flatten()))

    a = squareform(pdist(grid, 'sqeuclidean'))
    b = la.norm(grid - center, axis=1) ** 2
    c = b.reshape(-1, 1)
    C = np.exp(-a / (2 * spatial_freq ** 2)) * np.exp(-b / (2 * size ** 2)) * np.exp(-c / (2 * size ** 2)) \
        + 1e-5 * np.eye(dim[0] * dim[1])
    C *= scale * dim[0] * dim[1] / np.trace(C)
    return C


def classical_covariance_matrix(dim, scale=1):
    """
    Generates the covariance matrix for Gaussian Process with identity covariance. 
    This matrix will be used to generate random weights that are traditionally used 
    in kernel methods.

    C(x, y) = \delta_{xy}

    Parameters
    ----------

    dim: int or tuple (2, 1)
        dimension of each weight
        int for time-series, tuple for images 

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    Returns
    -------

    C : array-like of shape (dim, dim) or (dim[0] * dim[1], dim[0] * dim[1])
        covariance matrix w/ Tr norm = scale * dim[0] * dim[1]
    """
    if type(dim) is tuple:
        C = np.eye(dim[0] * dim[1]) * scale

    elif type(dim) is int:
        C = np.eye(dim) * scale
    return C


def shift_pad(img, y_shift, x_shift):
    '''
    Given an image, we shift every pixel by x_shift and y_shift. We zero pad the portion
    that ends up outside the original frame. We think of the origin of the image
    as its top left. The co-ordinate frame is the matrix kind, where (a, b) means
    ath row and bth column.
    
    Parameters
    ----------
    img: array-like
        image to shift
        
    y_shift: int
        Pixel shift in the vertical direction
        
    x_shift: int
        Pixel shift in the horizontal direction
    
    Returns
    -------
    img_shifted: array-like with the same shape as img
        Shifted and zero padded image

    '''
    img_shifted = np.roll(img, x_shift, axis=1)
    img_shifted = np.roll(img_shifted, y_shift, axis=0)
    
    if y_shift > 0:
        img_shifted[:y_shift, :] = 0
    if y_shift < 0:
        img_shifted[y_shift:, :] = 0
    if x_shift > 0:
        img_shifted[:, :x_shift] = 0
    if x_shift < 0:
        img_shifted[:, x_shift:] = 0
    return img_shifted
    

def V1_weights(num_weights, dim, size, spatial_freq, center=None, scale=1, seed=None):
    """
    Generate random weights inspired by the tuning properties of the 
    neurons in Primary Visual Cortex (V1).

    If a value is given for the center, all generated weights have the same center
    If value is set to None, the centers randomly cover the RF space

    Parameters
    ----------

    num_weights : int
        Number of random weights

    dim : tuple of shape (2,1)
        dim of each random weights
    
    size : float
        Determines the size of the random weights

    spatial_freq : float
        Determines the spatial frequency of the random weights 

    center: tuple of shape (2, 1), default = None
        Location of the center of the random weights
        With default value, the centers uniformly cover the RF space

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    seed : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (num_weights, dim[0] * dim[1])
        Matrix of random weights

    """
    np.random.seed(seed)
    if center == None: # centers uniformly cover the visual field
        # first generate centered weights
        c = (int(dim[0]/ 2), int(dim[1]/2)) # center of the visual field
        C = V1_covariance_matrix(dim, size, spatial_freq, c, scale) 

        W_centered = np.random.multivariate_normal(mean=np.zeros(dim[0] * dim[1]), cov=C, size=num_weights)
        W_centered = W_centered.reshape(-1, dim[0], dim[1])
        
        # shift around to uniformly cover the visual field
        centers = np.random.randint((dim[0], dim[1]), size=(num_weights, 2))
        shifts = centers - c
        W = np.zeros_like(W_centered)
        for i, [y_shift, x_shift] in enumerate(shifts):
            W[i] = shift_pad(W_centered[i], y_shift, x_shift)
        W = W.reshape(-1, dim[0] * dim[1])

    elif center is not None:
        C = V1_covariance_matrix(dim, size, spatial_freq, center, scale)

        W = np.random.multivariate_normal(mean=np.zeros(dim[0] * dim[1]), cov=C, size=num_weights)
        
    return W


def classical_weights(num_weights, dim, scale=1, seed=None):
    """"
    Generates classical random weights with identity covariance W ~ N(0, I).

    Parameters
    ----------

    num_weights : int
        Number of random weights

    dim : int
        dimension of each random weight

    scale : float, default=1
        Normalization factor for Tr norm of cov matrix
    
    seed : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : array-like of shape (num_weights, dim) or (num_weights, dim[0] * dim[1])
        Matrix of random weights.
    """
    C = classical_covariance_matrix(dim, scale)
    if type(dim) is tuple:
        W = np.random.multivariate_normal(mean=np.zeros(dim[0] * dim[1]), cov=C, size=num_weights)
    elif type(dim) is int:
        W = np.random.multivariate_normal(mean=np.zeros(dim), cov=C, size=num_weights)
    return W


def V1_weights_for_plotting(num_weights, dim, size, spatial_freq, center, scale=1, random_state=None):
    """
    Generates random weights for one given center by sampling a 
    non-stationary Gaussian Process. 
    
    Note: This is only used for plotting because it fixes the random normal 
    vectors. We can vary the covariance params and see the effects. For 
    classification, use V1_weighs function above. 

    Parameters
    ----------

    num_weights : int
        Number of random weights

    dim : tuple of shape (2,1)
        dim of each random weights
    
    size : float
        Determines the size of the random weights

    spatial_freq : float
        Determines the spatial frequency of the random weights 

    center: tuple of shape (2, 1)
        Location of the center of the random weights

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix

    seed : int, default=None
        Used to set the seed when generating random weights.

    Returns
    -------

    W : (array-like) of shape (num_weights, dim)
        Random weights
    """
    np.random.seed(random_state) 
    K = V1_covariance_matrix(dim, size, spatial_freq, center, scale=1)
    L = la.cholesky(K)
    W = np.dot(L, np.random.randn(dim[0] * dim[1], num_weights)).T
    return W

