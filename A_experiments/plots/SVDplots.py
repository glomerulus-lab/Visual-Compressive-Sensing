import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

sys.path.append('../')
from src.compress_sensing import *
from src.utility import *
from plots.theta_exp_improved import *
from plots.extract_patches import *
from plots.exp_constants import *


NUM_OBS_LIST = [256]

img = process_image("barbara.bmp", color=False)
patches = extract_patches(img, PATCH_SIZE)


def compute_patch_singular_values(patch, num_cell):
    """
    Compute the singular values of the design matrix

    Args:
        patch (ndarray): 
            2D image patch to measure.
        num_cell (int): 
            Number of observations.

    Returns:
        tuple[ndarray, ndarray, ndarray]: 
            Singular value arrays (S_v1, S_pix, S_gauss),
    """
    # V1
    W_v1, _ = generate_V1_observation(patch, num_cell, CELL_SIZE, BLOB_SIZE, None)
    theta_v1 = generate_design_matrix(W_v1)
    _, S_v1, _ = np.linalg.svd(theta_v1, full_matrices=False)

    # Pixel
    W_pix, _ = generate_pixel_observation(patch, num_cell)
    theta_pix = generate_design_matrix(W_pix)
    _, S_pix, _ = np.linalg.svd(theta_pix, full_matrices=False)

    # Gaussian
    W_gauss, _ = generate_gaussian_observation(patch, num_cell)
    theta_gauss = generate_design_matrix(W_gauss)
    _, S_gauss, _ = np.linalg.svd(theta_gauss, full_matrices=False)

    return S_v1, S_pix, S_gauss


def plot_SVD(num_obs, patches, savefile):
    """
    Plot the singular value spectra of for all patches

    Args:
        num_obs (int): 
            number observations
        patches (list[ndarray]): 
            Full list of extracted image patches.
        savefile (str): 
            Output path for the saved figure.

    TODO: Print pixel values for the largest singular value step to check for constant-valued patches.
    TODO: (for Kameron) Explore plotting as a distribution with methods in the same column.
    """
    n_patches = len(PATCH_IDXS)
    ncols = int(np.ceil(n_patches / 2))
    nrows = 2
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 8), sharey=True)
    axes = axes.flatten()
 
    for i, patch_idx in enumerate(PATCH_IDXS):
 
        S_V1_256_, S_pix_256_, S_gauss_256_ = compute_patch_singular_values(patches[patch_idx], 256)
 
        axes[i].plot(np.arange(1, num_obs+1), S_V1_256_,    "o", label="V1")
        axes[i].plot(np.arange(1, num_obs+1), S_pix_256_,   "x", label="Pix")
        axes[i].plot(np.arange(1, num_obs+1), S_gauss_256_, "+", label="Gauss")
 
        axes[i].set_title(f"Patch {patch_idx}")
 
        row = i // ncols
        col = i % ncols
 
        if row == nrows - 1:
            axes[i].set_xlabel("Index")
        if col == 0:
            axes[i].set_ylabel("Singular Value")
        if col != 0:
            axes[i].tick_params(axis='y', which='both', left=False, labelleft=False)
        axes[i].legend()
 
    for j in range(n_patches, len(axes)):
        fig.delaxes(axes[j])
 
    plt.suptitle("Singular Value Spectra of Theta", fontsize=16)
    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()


def plot_SVD_single_patch(num_obs, patches, savefile):
    """
    Plot the singular value spectra of Theta for single patch.

    Args:
        num_obs (int): 
            Number of observations.
        patches (list[ndarray]): 
            Full list of extracted image patches.
        savefile (str): 
            Output path for the saved figure.
    """
    patch_idx = 58 # TODO: change for other patches

    S_V1_256_, S_pix_256_, S_gauss_256_ = compute_patch_singular_values(patches[patch_idx], 256)

    plt.figure(figsize=(8, 6))

    plt.plot(np.arange(1, num_obs+1), S_V1_256_,    "o", label="V1")
    plt.plot(np.arange(1, num_obs+1), S_pix_256_,   "x", label="Pix")
    plt.plot(np.arange(1, num_obs+1), S_gauss_256_, "+", label="Gauss")

    plt.title(f"SVD - Patch {patch_idx}")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()

plot_SVD(256, patches, "SVD_256_patches.svg")
plot_SVD_single_patch(256, patches, "SVD_patch_58.svg")