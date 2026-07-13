import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys
import matplotlib.cm as cm

from src.compress_sensing import *
from src.utility import *
from .plots.extract_patches import *
from .plots.exp_constants import *

'''
Big question associated with this folder: 
    Why do V1 observations make for better image reconstructions than pixel or gaussian?

Current Thoughts:
    V1 mutual coherence is very bad, but reconstructions remain better
        Currently investigating why
    Hypothesis (Hyp): V1 observations end up with areas with values 0 and close to 0, so
        when we create the dot product matricies and normalize them, the dot products are being
        divided by something close to 0, 'blowing them up'
    Spectrum
        Look at the SVD of design matrix

W = measurement_matrix
U = basis_matrix
A = design_matrix
'''



def generate_design_matrix(measurement_matrix):
    '''
    Generate design_matrix for given weight matrix

    Parameters
    ----------

    measurement_matrix : array_like
        Weights on pixels that generate observations.

    Returns
    -------

    design_matrix : numpy_array
        (num_cell, n * m) matrix, the DCT of each measurement row
        flattened, ready to pass into the coherence functions.
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

    design_matrix : array_like
        matrix with more than one column

    Returns
    -------

    float
        The mutual coherence: the largest absolute normalized
        off-diagonal dot product between columns of design_matrix.

    Notes
    -----

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

    img_arr : numpy_array
        (n, m) shape image containing array of pixels.

    obs_type : String
        Observation technique that are going to be used to
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be
        determining which pixels to grab and use.

    cell_size : int, optional
        Determines field size of opened and closed blob of data.
        Affect the data training.

    blob_size : int, optional
        Determines filed frequency on how frequently
        opened and closed area would appear.
        Affect the data training.

    center : array_like, optional
        Center of the receptive fields for V1 observations.

    Returns
    -------

    numpy_array
        (num_cell, num_cell) matrix of absolute normalized dot products
        between columns of the design matrix, with the diagonal zeroed.
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

    img_arr : array_like
        I(n, m) shape image containing array of pixels

    n : int
        how many MC should be collected from one image,
        with purpose of averaging and comparing

    num_cell : int
        Number of blobs that will be used to be
        determining which pixels to grab and use.

    obs_type : String
        Observation technique that are going to be used to
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    cell_size : int, optional
        Determines field size of opened and closed blob of data.
        Affect the data training.

    blob_size : int, optional
        Determines filed frequency on how frequently
        opened and closed area would appear.
        Affect the data training.

    center : array_like, optional
        Center of the receptive fields for V1 observations.

    Returns
    -------

    numpy_array
        (n,) array holding one mutual coherence value per run,
        for averaging and comparison across observation types.

    Notes
    -----

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
    Sorts design_matrix by frequencies

    Parameters
    ----------

    design_matrix : array_like
        (num_cell, n * m) design matrix whose columns index DCT frequencies.

    Returns
    -------

    numpy_array
        The design_matrix with its columns reordered by ascending
        radial frequency (kx**2 + ky**2).
    '''

    arr = np.arange(30)
    kx = np.tile(arr, 30)
    ky = np.repeat(arr, 30)

    ksum = kx**2 + ky**2
    perm = np.argsort(ksum) 
    return design_matrix[:, perm]

def high_freq_table(img_arr, obs_type, num_cell, cell_size = None,
                    blob_size = None, center = None):
    '''
    Creates a table identifying which DCT basis frequencies are where

    Parameters
    ----------

    img_arr : array_like
        (n, m) shape image containing array of pixels

    obs_type : String
        Observation technique that are going to be used to
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be
        determining which pixels to grab and use.

    cell_size : int, optional
        Determines field size of opened and closed blob of data.
        Affect the data training.

    blob_size : int, optional
        Determines filed frequency on how frequently
        opened and closed area would appear.
        Affect the data training.

    center : array_like, optional
        Center of the receptive fields for V1 observations.

    Returns
    -------

    pandas.DataFrame
        Rows sorted by descending mutual coherence, with columns
        i, j (column indices into the dot product matrix),
        (kx_i, ky_i), (kx_j, ky_j) (their DCT frequency coordinates),
        and MC (the coherence value).
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
    Generates the coeffiecient vector for frequencies present in img_arr

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
    -------

    numpy_array
        (n, m) array of 2D DCT coefficients of img_arr.
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

    img_arr : array_like
        (n, m) shape image containing array of pixels

    obs_type : String
        Observation technique that are going to be used to
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be
        determining which pixels to grab and use.

    norm : int, optional
        np.linalg.norm(coeffs) ** norm
        norm type to divide by

    diagonal : int, optional
        Number to replace diagonal values with in dot vector
        0: will return metric without altering diagonal
        diagonal >0: will return metric having replaced dot_vec diagonal

    cell_size : int, optional
        Determines field size of opened and closed blob of data.
        Affect the data training.

    blob_size : int, optional
        Determines filed frequency on how frequently
        opened and closed area would appear.
        Affect the data training.

    center : array_like, optional
        Center of the receptive fields for V1 observations.

    Returns
    -------

    float
        The scalar metric coeffs.T @ dot_vec @ coeffs divided by
        np.linalg.norm(coeffs) ** norm.
    '''
    coeffs= generate_coeff_vector(img_arr,num_cell,cell_size,blob_size).flatten()
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

    img_arr : array_like
        I(n, m) shape image containing array of pixels

    obs_type : String
        Observation technique that are going to be used to
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be
        determining which pixels to grab and use.

    norm : int, optional
        np.linalg.norm(coeffs) ** norm
        norm type to divide by

    cell_size : int, optional
        Determines field size of opened and closed blob of data.
        Affect the data training.

    blob_size : int, optional
        Determines filed frequency on how frequently
        opened and closed area would appear.
        Affect the data training.

    center : array_like, optional
        Center of the receptive fields for V1 observations.

    Returns
    -------

    float
        The norm of dot_vec @ coeffs: the infinity norm when norm <= 0,
        otherwise the given p-norm.
    '''

    coeffs= generate_coeff_vector(img_arr,num_cell,cell_size,blob_size).flatten()
    dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, blob_size, center)

    if norm <= 0:
        return np.linalg.norm(dot_vec @ coeffs, np.inf)
    else:
        return np.linalg.norm(dot_vec @ coeffs, norm)


def MC_box_plot(num_runs, num_cell):
    '''
    mutual coherence (MC) plot

    Parameters
    ----------

    num_runs : int
        Number of mutual coherence samples to draw per observation type.

    num_cell : int
        Number of blobs that will be used to be
        determining which pixels to grab and use.

    Returns
    -------

    None
        Saves a box plot comparing V1, pixel, and gaussian mutual
        coherence to MC_box_plot_<num_runs>_runs_<num_cell>_blobs.svg.
    '''
    # mutual coherence for each observation type
    mc_v1 = mutual_coherence_matrix(small_img_arr_gray, n=num_runs, num_cell=num_cell, obs_type='V1', cell_size=cell_size, blob_size=blob_size)
    mc_pixel = mutual_coherence_matrix(small_img_arr_gray, n=num_runs, num_cell=num_cell, obs_type='pixel')
    mc_gauss = mutual_coherence_matrix(small_img_arr_gray, n=num_runs, num_cell=num_cell, obs_type='gaussian')

    data = [mc_v1, mc_pixel, mc_gauss]

    plt.figure(figsize=(8,6))
    plt.boxplot(data, labels=['V1', 'Pixel', 'Gaussian'])
    plt.ylabel("Mutual Coherence")
    plt.title(f"Mutual Coherence ({num_runs} runs, {num_cell} blobs)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"MC_box_plot_{num_runs}_runs_{num_cell}_blobs.svg")

def design_matrix_from_patch(patch, obs_type, num_cell):
    """
    design matric per patch

    Parameters
    ----------

    patch : ndarray
        2D image patch.

    obs_type : str
        Measurement strategy — ['V1', 'pixel', or 'gaussian'].

    num_cell : int
        Number of observations to generate.

    Returns
    -------

    ndarray or None
        Design matrix, or None if obs_type is unrecognized.
    """
    if obs_type == "V1":
        W, _ = generate_V1_observation(patch, num_cell, CELL_SIZE, BLOB_SIZE, None)
    elif obs_type == "pixel":
        W, _ = generate_pixel_observation(patch, num_cell)
    elif obs_type == "gaussian":
        W, _ = generate_gaussian_observation(patch, num_cell)
    else:
        return None
    return generate_design_matrix(W)


def mutual_coherence_runs(patch, obs_type, num_cell, n_runs):
    """
    Repeatedly compute mutual coherence for a single patch to build a distribution
    over the randomness in the measurement process.

    Parameters
    ----------

    patch : ndarray
        2D image patch.

    obs_type : str
        Measurement strategy — ['V1', 'pixel', or 'gaussian'].

    num_cell : int
        Number of observations per run.

    n_runs : int
        Number of independent MC samples to collect.

    Returns
    -------

    ndarray
        Array of shape (n_runs,) containing one mutual coherence value per run.
    """
    M = np.zeros(n_runs)
    for i in range(n_runs):
        A = design_matrix_from_patch(patch, obs_type, num_cell)
        M[i] = compute_mutual_coherence(A)
    return M


def MC_box_plot_for_patch(patch_idx):
    """
    Plot a box plot comparing the mutual coherence distributions of V1, Pixel, and
    Gaussian measurements for a single patch, over NUM_MC_RUNS repeated samples.

    Parameters
    ----------

    patch_idx : int
        Index into the global 'patches' list.

    Returns
    -------

    None
        Saves the box plot to MC_patch_<patch_idx>.svg and shows it.
    """
    patch = patches[patch_idx]

    mc_v1   = mutual_coherence_runs(patch, "V1", NUM_CELL, NUM_MC_RUNS)
    mc_pix  = mutual_coherence_runs(patch, "pixel",NUM_CELL, NUM_MC_RUNS)
    mc_gauss = mutual_coherence_runs(patch, "gaussian", NUM_CELL, NUM_MC_RUNS)

    plt.figure(figsize=(7, 5))
    plt.boxplot([mc_v1, mc_pix, mc_gauss], labels=["V1", "Pixel", "Gaussian"])
    plt.ylabel("Mutual Coherence")
    plt.title(f"Mutual Coherence (Patch {patch_idx}, {NUM_CELL} obs)")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"MC_patch_{patch_idx}.svg")
    plt.show()


def dot_product_matrix_from_patch(patch, obs_type, num_cell):
    """
    Normalized absolute dot-product matrix used in mutual coherence.

    The columns of the design matrix are normalized to unit vectors,
    then their outer product is computed, the diagonal (=1) is set to 0,
    and we return the absolute value.

    Parameters
    ----------

    patch : ndarray
        2D image patch.

    obs_type : str
        Measurement strategy — one of 'V1', 'pixel', or 'gaussian'.

    num_cell : int
        Number of observations to generate.

    Returns
    -------

    ndarray
        Dot product matrix.
    """
    A = design_matrix_from_patch(patch, obs_type, num_cell)
    col_norms = np.linalg.norm(A, axis=0)
    A_hat = A / col_norms
    M = A_hat.T @ A_hat
    np.fill_diagonal(M, 0)
    return np.abs(M)


def plot_dot_products_for_patch(patch_idx):
    """
    Plot dot products per patch.

    Parameters
    ----------

    patch_idx : int
        Index into the 'patches' list.

    Returns
    -------

    None
        Saves the heatmaps to DotProducts_patch_<patch_idx>.svg and shows them.
    """
    patch = patches[patch_idx]

    mats = [
        dot_product_matrix_from_patch(patch, "V1",      NUM_CELL),
        dot_product_matrix_from_patch(patch, "pixel",   NUM_CELL),
        dot_product_matrix_from_patch(patch, "gaussian", NUM_CELL),
    ]
    titles = ["V1", "Pixel", "Gaussian"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title in zip(axs, mats, titles):
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Column index")

    # Single shared colorbar attached to the rightmost subplot
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    plt.suptitle(f"Dot Products – Patch {patch_idx}", y=0.98)
    plt.savefig(f"DotProducts_patch_{patch_idx}.svg")
    plt.show()


def dot_product_histogram_for_patch(patch_idx, bins=50):
    """
    Dot product histograms

    Parameters
    ----------

    patch_idx : int
        Index into the global 'patches' list.

    bins : int, optional
        Number of histogram bins. Default 50.

    Returns
    -------

    None
        Saves the histogram to DotProductHist_patch_<patch_idx>.svg and shows it.
    """
    patch = patches[patch_idx]
    colors = ["#2196F3", "#FF6F00", "#43A047"]  # blue, amber, green

    obs_types = ["V1", "pixel", "gaussian"]
    labels    = ["V1", "Pixel", "Gaussian"]

    plt.figure(figsize=(7, 5))

    for obs, label, color in zip(obs_types, labels, colors):
        dot = dot_product_matrix_from_patch(patch, obs, NUM_CELL)
        plt.hist(dot.flatten(), bins, density=True,
                 alpha=0.5,
                 color=color,
                 edgecolor=color,
                 linewidth=1.2,
                 label=label)

    plt.xlabel("Dot Product")
    plt.ylabel("Density")
    plt.title(f"Dot Products – Patch {patch_idx}")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"DotProductHist_patch_{patch_idx}.svg")
    plt.show()


def dot_product_histograms_all_patches(patches, patch_idxs, num_cell, bins=50, filename="DotProductHist_all_patches.svg"):
    """
    Dot product historgrams for all patches.

    Parameters
    ----------

    patches : list[ndarray]
        Full list of extracted image patches.

    patch_idxs : list[int]
        Patch indices to include.

    num_cell : int
        Number of observations per patch.

    bins : int, optional
        Number of histogram bins. Default 50.

    filename : str, optional
        Output path for the saved SVG. Default
        "DotProductHist_all_patches.svg".

    Returns
    -------

    None
        Saves the grid of histograms to `filename`.
    """
    obs_types = ["V1", "pixel", "gaussian"]
    labels    = ["V1", "Pixel", "Gaussian"]
    colors    = ["#2196F3", "#FF6F00", "#43A047"]  # blue, amber, green

    n_patches = len(patch_idxs)
    n_cols = 2
    n_rows = (n_patches + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), sharey=True)
    axes = axes.flatten()

    for i, patch_idx in enumerate(patch_idxs):
        ax = axes[i]
        patch = patches[patch_idx]

        for obs, label, color in zip(obs_types, labels, colors):
            dot = dot_product_matrix_from_patch(patch, obs, num_cell)
            ax.hist(dot.flatten(), bins, density=True,
                    alpha=0.5,
                    color=color,
                    edgecolor=color,
                    linewidth=1.2,
                    label=label)

        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Coherence")
        if col == 0:
            ax.set_ylabel("Density")
        if col != 0:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.set_yscale("log")
        ax.set_title(f"Patch {patch_idx}")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    for ax in axes[n_patches:]:
        fig.delaxes(ax)

    fig.suptitle("Dot Product Distributions", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg", dpi=300)
    plt.close()


def plot_dot_products_all_patches(patches, patch_idxs, num_cell, filename="DotProducts_all_patches.svg"):
    """
    Dot products for all patches.

    Parameters
    ----------

    patches : list[ndarray]
        Full list of extracted image patches.

    patch_idxs : list[int]
        Patch indices to include as rows.

    num_cell : int
        Number of observations per patch.

    filename : str, optional
        Output path for the saved SVG. Default
        "DotProducts_all_patches.svg".

    Returns
    -------

    None
        Saves the grid of dot product heatmaps to `filename`.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable # for color bar

    obs_types = ["V1", "pixel", "gaussian"]
    titles    = ["V1", "Pixel", "Gaussian"]

    n_rows = len(patch_idxs)
    n_cols = len(obs_types)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, patch_idx in enumerate(patch_idxs):
        patch = patches[patch_idx]
        for col, obs in enumerate(obs_types):
            mat = dot_product_matrix_from_patch(patch, obs, num_cell)
            ax  = axes[row, col]
            im  = ax.imshow(mat, cmap="viridis")

            if row == 0:
                ax.set_title(titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\nColumn index", fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel("Column index", fontsize=12)
            if col != 0:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # add a per-row colorbar to the right of the last column
        divider = make_axes_locatable(axes[row, -1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    fig.suptitle("Dot Products", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg", dpi=300)
    plt.close()


def MC_box_plot_all_patches(patches, patch_idxs, filename="MC_all_patches.svg"):
    """
    MC box plot for all patches.

    Parameters
    ----------

    patches : list[ndarray]
        Full list of extracted image patches.

    patch_idxs : list[int]
        Patch indices to include.

    filename : str, optional
        Output path for the saved SVG. Default "MC_all_patches.svg".

    Returns
    -------

    None
        Saves the grid of mutual coherence box plots to `filename`.
    """
    n_patches = len(patch_idxs)
    n_cols = 2
    n_rows = (n_patches + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, patch_idx in enumerate(patch_idxs):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row, col]

        patch    = patches[patch_idx]
        mc_v1    = mutual_coherence_runs(patch, "V1", NUM_CELL, NUM_MC_RUNS)
        mc_pix   = mutual_coherence_runs(patch, "pixel", NUM_CELL, NUM_MC_RUNS)
        mc_gauss = mutual_coherence_runs(patch, "gaussian", NUM_CELL, NUM_MC_RUNS)

        ax.boxplot([mc_v1, mc_pix, mc_gauss], labels=["V1", "Pixel", "Gaussian"])
        if col == 0:
            ax.set_ylabel("Mutual Coherence")
        if col != 0:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.set_title(f"Patch {patch_idx}")
        ax.grid(alpha=0.4)

    total_axes = n_rows * n_cols
    for empty_idx in range(n_patches, total_axes):
        axes[empty_idx // n_cols, empty_idx % n_cols].axis('off')

    fig.suptitle("Mutual Coherence", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg", dpi=300)
    plt.close()
