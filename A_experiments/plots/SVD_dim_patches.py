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

'''
PLOTS: 
    - Dimension of Theta
''' 


barbara = process_image("barbara.bmp", color=False)
patches = extract_patches(barbara, PATCH_SIZE)

# preallocate result arrays: rows = patches, cols = independent runs.
dim_arr_V1    = np.zeros((len(PATCH_IDXS), NUM_RUNS))
dim_arr_pix   = np.zeros((len(PATCH_IDXS), NUM_RUNS))
dim_arr_gauss = np.zeros((len(PATCH_IDXS), NUM_RUNS))

for p_idx, patch_idx in enumerate(PATCH_IDXS):
    patch = patches[patch_idx]

    for run in range(NUM_RUNS):
        # V1 measurements
        meas_V1, y_V1 = generate_V1_observation(patch, N_OBS, CELL_SIZE, BLOB_SIZE, None)
        theta_V1 = generate_design_matrix(meas_V1)
        _, S_V1, _ = np.linalg.svd(theta_V1)
        p_V1 = S_V1 / np.sum(S_V1)
        dim_arr_V1[p_idx, run] = 1 / np.sum(p_V1 * p_V1)

        # Pixel measurements
        meas_pix, y_pix = generate_pixel_observation(patch, N_OBS)
        theta_pix = generate_design_matrix(meas_pix)
        _, S_pix, _ = np.linalg.svd(theta_pix)
        p_pix = S_pix / np.sum(S_pix)
        dim_arr_pix[p_idx, run] = 1 / np.sum(p_pix * p_pix)

        # Gaussian measurements
        meas_gauss, y_gauss = generate_gaussian_observation(patch, N_OBS)
        theta_gauss = generate_design_matrix(meas_gauss)
        _, S_gauss, _ = np.linalg.svd(theta_gauss)
        p_gauss = S_gauss / np.sum(S_gauss)
        dim_arr_gauss[p_idx, run] = 1 / np.sum(p_gauss * p_gauss)

# MUTIPLE PATCHES
n_rows = 2
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharey=True)
axes = axes.flatten()

for i, patch_idx in enumerate(PATCH_IDXS):
    ax = axes[i]
    ax.boxplot(
        [dim_arr_V1[i], dim_arr_pix[i], dim_arr_gauss[i]],
        labels=["V1", "Pixel", "Gaussian"]
    )
    ax.set_title(f"Patch {patch_idx}")

    col = i % n_cols
    if col == 0:
        ax.set_ylabel("Dimension of Theta")
    if col != 0:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.grid(alpha=0.3, axis='y')

# remove any unused subplot cells if PATCH_IDXS has fewer than 4 entries
for ax in axes[len(PATCH_IDXS):]:
    fig.delaxes(ax)

fig.suptitle(f"Dimensions of Theta over {NUM_RUNS} runs", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Dimensions_all_patches.svg", format="svg", dpi=300)
plt.close()

# SINGLE PATCHES
patch_idx = 58 # TODO: change this per patch

fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot(
    [dim_arr_V1[0], dim_arr_pix[0], dim_arr_gauss[0]],
    labels=["V1", "Pixel", "Gaussian"]
)
ax.set_title(f"Dimensions of Theta over {NUM_RUNS} runs - Patch {patch_idx}")
ax.set_ylabel("Dimension of Theta")
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"Dimensions_patch_{patch_idx}.svg", format="svg", dpi=300)
plt.show()