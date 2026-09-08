# %load_ext autoreload
# %autoreload 2

import numpy as np
from matplotlib import pyplot as plt

from .plots import paper_plots
from .plots.paper_plots import process_image, extract_patches
from .plots.exp_constants import *

dim = (PATCH_SIZE, PATCH_SIZE)
center = (PATCH_SIZE / 2, PATCH_SIZE / 2)

ALG = 'bp'
idx = 58

# Sweep number of observations from 16 to 1024 (powers of two).
# N_OBS_SWEEP = np.geomspace(16, 1024, num=7).round().astype(int)
N_OBS_SWEEP = np.linspace(200, 1024, 10).round().astype(int)

barbara = process_image("barbara.bmp", color=False)
patches = extract_patches(barbara, PATCH_SIZE)

methods = ['V1', 'Gaussian']
l2_error = {method: [] for method in methods}

for n_obs in N_OBS_SWEEP:
    print(f"Running {ALG} on patch {idx} of {IMAGE_FILE} with {n_obs} observations")
    # Overload N_OBS from exp_constants.py; run_selected_patches reads it as a
    # global out of the paper_plots module namespace.
    paper_plots.N_OBS = int(n_obs)
    results = paper_plots.run_selected_patches(patches, [idx], center=center, algorithm=ALG)
    patch_results = results[idx]
    for method in methods:
        err = patch_results[method]['error']  # per-component squared error
        l2_error[method].append(np.sqrt(np.sum(err)))

plt.figure()
for method in methods:
    plt.loglog(N_OBS_SWEEP, l2_error[method], marker='o', label=method)
plt.xlabel('Number of observations')
plt.ylabel(r'$\ell_2$ error')
plt.title(f'Reconstruction error vs. number of observations ({ALG.upper()}, patch {idx})')
plt.legend()
plt.show()
