import sys
import matplotlib.pyplot as plt
import sys
import pandas as pd

from src.compress_sensing import *
from src.utility import *
from ..core import *
from .extract_patches import *
from .exp_constants import *
from .util import *

'''
PLOTS: 
    - est pcs vs true pcs scatter 
    - est pcs vs rank 
    - pc error
    - pcs as images 
    - coefficient hists
''' 

def compute_patch_results(patch, n, cell_size, blob_size, alpha, algorithm=ALG):
    """
    Get necessary data for plots.

    Args:
        patch (ndarray): 
            2D image patch to reconstruct.
        n (int): 
            Number of observations
        cell_size (int), blob_size (int): 
            V1 parameters
            s and f in the paper
        alpha (float): 
            LASSO penalty

    Returns:
        dict: Nested dictionary with top-level keys:
            "coeffs_true" (ndarray): Ground-truth DCT coefficient vector for the patch.
            "V1" / "Pixel" / "Gaussian" (dict): Per-method results, each containing:
                "U"             (ndarray): Left singular vectors of the design matrix.
                "S"             (ndarray): Singular values of the design matrix.
                "Vh"            (ndarray): Right singular vectors (PCs) of the design matrix.
                "reconstruction" (ndarray): Reconstructed patch from estimated coefficients.
                "est_coeffs"    (ndarray): Estimated DCT coefficient vector.
                "p_est"         (ndarray): Estimated coefficients projected onto PCs.
                "p_true"        (ndarray): True coefficients projected onto PCs.
                "error"         (ndarray): Per-component squared error (p_true - p_est)^2.
    """
    # true coefs of theta
    coeffs_true = generate_coeff_vector(patch, n, cell_size, blob_size)

    # V1 - SVD
    measurement_matrix_V1, V1_y = generate_V1_observation(patch, n, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)
    U_V1, S_V1, Vh_V1 = np.linalg.svd(theta_V1)

    # V1 - estimated coeffs
    reconst_v1 = reconstruct(measurement_matrix_V1, V1_y, alpha, algorithm=algorithm)
    coeffs_est_V1 = generate_coeff_vector(reconst_v1, n, cell_size, blob_size)

    # Project both true and estimated coefficients onto the V1 principal components
    p_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()
    p_true_V1 = Vh_V1 @ coeffs_true.flatten()

    err_V1 = (p_true_V1 - p_est_V1) ** 2

    # Pixel - SVD
    measurement_matrix_pix, pixel_y = generate_pixel_observation(patch, n)
    theta_pix = generate_design_matrix(measurement_matrix_pix)
    U_pix, S_pix, Vh_pix = np.linalg.svd(theta_pix)

    # Pixel - estimated coeffs
    reconst_pix = reconstruct(measurement_matrix_pix, pixel_y, alpha, algorithm=algorithm)
    coeffs_est_pix = generate_coeff_vector(reconst_pix, n, cell_size, blob_size)

    # Project both true and estimated coefficients onto the Pixel principal components
    p_est_pix = Vh_pix @ coeffs_est_pix.flatten()
    p_true_pix = Vh_pix @ coeffs_true.flatten()

    err_pix = (p_true_pix - p_est_pix) ** 2

    # Gauss - SVD
    measurement_matrix_gauss, gaussian_y = generate_gaussian_observation(patch, n)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)
    U_gauss, S_gauss, Vh_gauss = np.linalg.svd(theta_gauss)

    # Gauss - estimated coeffs
    reconst_gauss = reconstruct(measurement_matrix_gauss, gaussian_y, alpha, algorithm=algorithm)
    coeffs_est_gauss = generate_coeff_vector(reconst_gauss, n, cell_size, blob_size)

    # Project both true and estimated coefficients onto the Gaussian principal components
    p_est_gauss = Vh_gauss @ coeffs_est_gauss.flatten()
    p_true_gauss = Vh_gauss @ coeffs_true.flatten()

    err_gauss = (p_true_gauss - p_est_gauss) ** 2

    return {
        "coeffs_true": coeffs_true,

        "V1": {
            "U" : U_V1,
            "S" : S_V1,
            "Vh": Vh_V1,
            "reconstruction": reconst_v1,
            "est_coeffs" : coeffs_est_V1,
            "p_est": p_est_V1,
            "p_true": p_true_V1,
            "error" : err_V1,
        },
        "Pixel": {
            "U" : U_pix,
            "S" : S_pix,
            "Vh": Vh_pix,
            "reconstruction": reconst_pix,
            "est_coeffs" : coeffs_est_pix,
            "p_est": p_est_pix,
            "p_true": p_true_pix,
            "error" : err_pix,
        },
        "Gaussian": {
            "U" : U_gauss,
            "S" : S_gauss,
            "Vh": Vh_gauss,
            "reconstruction": reconst_gauss,
            "est_coeffs" : coeffs_est_gauss,
            "p_est": p_est_gauss,
            "p_true": p_true_gauss,
            "error" : err_gauss,
        }
    }

def get_results(img):
    """
    Extract all patches from an image and compute reconstruction results for each.

    Args:
        img (ndarray): 
            Grayscale image array to extract patches from.

    Returns:
        list[dict]: 
            One result dictionary per patch, each in the format returned
            by compute_patch_results.
    """
    patches = extract_patches(img, PATCH_SIZE)
    all_results = []
    for patch in patches:
        res = compute_patch_results(patch, N_OBS, CELL_SIZE, BLOB_SIZE, ALPHA)
        all_results.append(res)
    return all_results

def run_selected_patches(patches, patch_idxs):
    """
    Run reconstruction on a specific subset of patches identified by index.

    Args:
        patches (list[ndarray]): 
            Full list of extracted image patches.
        patch_idxs (list[int]): 
            Indices into 'patches' to process.

    Returns:
        dict[int, dict]: 
            Mapping from patch index to its compute_patch_results dictionary.
    """
    results = {}

    for idx in patch_idxs:
        print(f"Running patch {idx}")
        results[idx] = compute_patch_results(
            patches[idx],
            N_OBS,
            CELL_SIZE,
            BLOB_SIZE,
            ALPHA
        )

    return results

def pc_per_method(results, num_obs, patch_idx, vector="est"):
    """
    Scatter plot of estimated principal component magnitudes by rank for each
    measurement method.

    Args:
        results (dict): Nested results dict keyed by num_obs.
        num_obs (int): Observation count key used to index into 'results'.
        patch_idx (int): Patch index used for the plot title and output filename.
        vector (str): "est" or "true", default = "est"
    """
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, methods):
        if vector == "est":
            components = np.abs(results[num_obs][method]["p_est"])
            ylabel_str = "True Principal Component"
        elif vector == "true":
            components = np.abs(results[num_obs][method]["p_true"])
            ylabel_str = "Estimated Principal Component"
        else:
            raise Exception(f"Invalid argument results = {results}")
        ax.scatter(range(len(components)), components, s=10, color='skyblue')

        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel_str)
        ax.set_yscale('log')
        ax.set_title(f"{method} {ylabel_str}")

    plt.suptitle(f"Principal Component - Patch {patch_idx}")
    plt.tight_layout()
    plt.savefig(f"pc_per_method_patch_{patch_idx}_{vector}.svg", dpi=300)
    plt.close()

def plot_cdf_error(ax, err, label):
    """
    Plot the CDF of per-component squared errors.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        err (array-like): Per-component squared error values.
        label (str): Legend label for the plotted line.
    """
    err = np.array(err)
    sorted_err = np.sort(err)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf, label=label)

def pc_scatter_plots_all_patches(results, patch_idxs, filename, cmap='cool'):
    """
    Plot true vs estimated principal component scatter plots for multiple patches in
    a single figure.

    Args:
        results (dict[int, dict]): 
            Results dict keyed by patch index, as returned by run_selected_patches.
        patch_idxs (list[int]): 
            Ordered list of patch indices to include as rows.
        filename (str): 
            Output path for the saved SVG file.
        cmap (str): 
            Matplotlib colormap name for PC rank colouring. Default 'cool'.
    """
    n_rows = len(patch_idxs)
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, 5 * n_rows),
        sharey=True,
        layout="constrained"
    )

    # axes is always 2D even for a single patch
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # global min and max for y=x 
    global_min = np.inf
    global_max = -np.inf
    for patch_idx in patch_idxs:
        for method in methods:
            est = np.abs(results[patch_idx][method]["p_est"])
            true = np.abs(results[patch_idx][method]["p_true"])
            combined = np.concatenate([est, true])
            combined = combined[combined > 0]
            global_min = min(global_min, combined.min())
            global_max = max(global_max, combined.max())

    for row, patch_idx in enumerate(patch_idxs):
        sc = None
        for col, method in enumerate(methods):
            ax = axes[row, col]

            est = results[patch_idx][method]["p_est"]
            true = results[patch_idx][method]["p_true"]

            sc = ax.scatter(
                np.abs(est), np.abs(true),
                c=np.arange(len(est)),
                s=30, cmap=cmap, alpha=0.5
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            ax.plot([global_min, global_max], [global_min, global_max], '--', color='gray')
            ax.set_aspect('equal', adjustable='box')

            if row == 0:
                ax.set_title(method, fontsize=18)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\nTrue PC", fontsize=18)
            if row == n_rows - 1:
                ax.set_xlabel("Estimated PC", fontsize=18)
            if col != 0:
                ax.yaxis.set_visible(False)

        # add a colorbar to the rightmost subplot in each row
        fig.colorbar(sc, ax=axes[row, 2], shrink=0.8, label="PC rank")

    fig.suptitle("True vs Estimated Principal Components", fontsize=20)
    plt.savefig(filename, format="svg")
    plt.close()
    
def pc_per_method_all_patches(results, patch_idxs, filename, vector="est"):
    """
    Scatter plot of estimated PC magnitudes by rank for multiple patches and all three
    methods, arranged in a grid with patches as rows and methods as columns.

    Args:
        results (dict[int, dict]): 
            Results dict keyed by patch index.
        patch_idxs (list[int]): 
            Ordered list of patch indices to include as rows.
        filename (str): 
            Output path for the saved SVG file.
        vector (str):
            Optional, should be "est" or "true"
    """
    n_rows = len(patch_idxs)
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, patch_idx in enumerate(patch_idxs):

        for col, method in enumerate(methods):
            ax = axes[row, col]

            if vector == "est":
                components = np.abs(results[patch_idx][method]["p_est"])
                ylabel_str = "Estimated Principal Component"
            elif vector == "true":
                components = np.abs(results[patch_idx][method]["p_true"])
                ylabel_str = "True Principal Component"
            else:
                raise Exception(f"Invalid argument results = {results}")
            ranks = np.arange(1, len(components) + 1)

            ax.scatter(ranks, components, s=10, color='skyblue')

            ax.set_yscale('log')
            if row == 0:
                ax.set_title(method, fontsize=18)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\n {ylabel_str}", fontsize=18)
            if row == n_rows - 1: 
                ax.set_xlabel("Rank", fontsize=18)
            if col != 0:
                ax.yaxis.set_visible(False)

    fig.suptitle(ylabel_str, fontsize=20, x=0.55)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, format="svg")
    plt.close()

def error_all_patches(results, patch_idxs, filename):
    """
    Plot cumulative squared error by component index for all three methods across
    multiple patches.

    Args:
        results (dict[int, dict]): 
            Results dict keyed by patch index.
        patch_idxs (list[int]): 
            Ordered list of patch indices to plot.
        filename (str): 
            Output path for the saved SVG file.
    """
    n_rows = (len(patch_idxs) + 1) // 2
    n_cols = 2
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    
    for i, patch_idx in enumerate(patch_idxs):
        ax = axes[i]
        for method in methods:
            cumsum_err(ax, results[patch_idx][method]["error"], method)

        # for cumsum:
        ax.set_xscale('linear')
        ax.set_yscale('linear')

        # for ranked:
        # ax.set_xscale("linear")
        # ax.set_yscale("log")

        # for cdf: 
        # ax.set_xscale("log")
        # ax.set_yscale("linear")

        # for rolling mean: 
        # ax.set_xscale("log")
        # ax.set_yscale("log")

        ax.set_title(f"Patch {patch_idx}", fontsize=16)

        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Index", fontsize=14)
        if col == 0:
            ax.set_ylabel("Cumulative Squared Error", fontsize=14)

        ax.legend()

    # Remove axes for any unused grid cells
    for ax in axes[len(patch_idxs):]:
        fig.delaxes(ax)

    fig.suptitle("Error per Component", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg")
    plt.close()

def coeff_vectors_hist_all_patches(results, patch_idxs, filename):
    """
    Plot coefficient magnitude histograms for multiple patches in a grid, with one
    row per patch and columns for V1, Pixel, Gaussian estimated, and true coefficients.
    Bin ranges use the 99th percentile.

    Args:
        results (dict[int, dict]): 
            Results dict keyed by patch index.
        patch_idxs (list[int]): 
            Ordered list of patch indices to include as rows.
        filename (str):
            Output path for the saved SVG file.
    """
    n_rows = len(patch_idxs)
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True)
    
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, patch_idx in enumerate(patch_idxs):
        coeff_vectors = [
            ("V1 Estimated", results[patch_idx]["V1"]["est_coeffs"].flatten()),
            ("Pixel Estimated", results[patch_idx]["Pixel"]["est_coeffs"].flatten()),
            ("Gaussian Estimated", results[patch_idx]["Gaussian"]["est_coeffs"].flatten()),
            ("True", results[patch_idx]["coeffs_true"].flatten()),
        ]
        all_abs = np.concatenate([np.abs(c) for _, c in coeff_vectors])
        upper = np.percentile(all_abs, 99)
        bins = np.linspace(0, upper, 50)

        for col, (label, coeffs) in enumerate(coeff_vectors):
            ax = axes[row, col]
            ax.hist(np.abs(coeffs), bins=bins, edgecolor="black", color=f"C{col}")
            ax.set_yscale("log")
            ax.grid(True, which='major', linestyle='--', alpha=0.4)

            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\nCount", fontsize=12)
            if col != 0:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.text(0.5, 0.01, "Coefficient Magnitude", ha="center", fontsize=12)
    plt.suptitle("Frequency of Coefficients", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="svg", dpi=300)
    plt.close()

def coeff_vectors_cdf_all_patches(results, patch_idxs, filename):
    """
    Plot the CDF of absolute coefficient values for multiple patches in a
    two-column grid, comparing true vs estimated coefficients across all three methods.

    Args:
        results (dict[int, dict]): 
            Results dict keyed by patch index.
        patch_idxs (list[int]): 
            Ordered list of patch indices to plot.
        filename (str): 
            Output path for the saved SVG file.
    """
    n_rows = (len(patch_idxs) + 1) // 2
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    
    for i, patch_idx in enumerate(patch_idxs):
        ax = axes[i]
        coeff_vectors = [
            ("True", results[patch_idx]["coeffs_true"].flatten()),
            ("V1 Estimated", results[patch_idx]["V1"]["est_coeffs"].flatten()),
            ("Pixel Estimated", results[patch_idx]["Pixel"]["est_coeffs"].flatten()),
            ("Gaussian Estimated", results[patch_idx]["Gaussian"]["est_coeffs"].flatten()),
        ]

        for label, coeffs in coeff_vectors:
            abs_coeffs = np.sort(np.abs(coeffs))
            cdf = np.arange(1, len(abs_coeffs) + 1) / len(abs_coeffs)
            ax.plot(abs_coeffs, cdf, label=label)

        ax.set_xscale('log')
        ax.set_title(f"Patch {patch_idx}", fontsize=16)

        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Absolute Coefficient Value", fontsize=14)
        if col == 0:
            ax.set_ylabel("CDF", fontsize=14)
        if col != 0:
            ax.yaxis.set_visible(False)
        ax.legend()

    # remove axes for any unused grid cells
    for ax in axes[len(patch_idxs):]:
        fig.delaxes(ax)

    fig.suptitle("CDF of Coefficients", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg")
    plt.close()

barbara = process_image("barbara.bmp", color=False)
patches = extract_patches(barbara, PATCH_SIZE)
# show_patches_grid(patches)
results = run_selected_patches(patches, PATCH_IDXS)

# Theory?
#good_patch = 58
bad_patch = 233#169

pstar = results[bad_patch]['V1']['p_true']
p = results[bad_patch]['V1']['p_est']
err = results[bad_patch]['V1']['error']
svs = results[bad_patch]['V1']['S']
svs = np.pad(svs, (0, len(p) - len(svs)), 'constant', constant_values=(0,0))
plt.semilogy(pstar,'.')

U = results[bad_patch]['V1']['U']
S = results[bad_patch]['V1']['S']
Vh = results[bad_patch]['V1']['Vh']

A = (U * S) @ Vh[:len(S),:]

z = results[bad_patch]['V1']['est_coeffs'].flatten()
zstar = results[bad_patch]['coeffs_true'].flatten()
s = z.copy()
s[np.abs(s) < 1e-12] = 0
s[s > 0] = 1
s[s < 0] = -1
s = s
Vh = results[bad_patch]['V1']['Vh']
coeffs_nonzero = np.abs(s) > 0
Vh_nonzero = Vh[:, coeffs_nonzero]
Vh_zero = Vh[:, np.logical_not(coeffs_nonzero)]

KKT_gap1 = ALPHA * s[coeffs_nonzero] - (A[:, coeffs_nonzero].T @ A @ (zstar - z)) / len(S)
KKT_gap0 = ALPHA - np.abs((A[:, np.logical_not(coeffs_nonzero)].T @ A @ (zstar - z)) / len(S))

KKT_pgap1 = ALPHA * s[coeffs_nonzero] - ((Vh_nonzero.T)[:, :len(S)] @ (S**2 * (pstar - p)[:len(S)])) / len(S)
KKT_pgap0 = ALPHA - np.abs(((Vh_zero.T)[:, :len(S)] @ (S**2 * (pstar - p)[:len(S)]))) / len(S)

assert np.allclose(KKT_gap1, KKT_pgap1)
assert np.allclose(KKT_gap0, KKT_pgap0)

R_eps = Vh_nonzero.T
R_epsc = Vh_zero.T

plt.semilogy(zstar ** 2)

# R_eps @ Sigma @ U.T @ A_eps @ (zstar_eps - z_eps) = n lambda s_eps
KKT_gap1 = ALPHA * s[coeffs_nonzero] - (A[:, coeffs_nonzero].T @ A @ (zstar - z)) / len(S)



# TODO: run for single patch results
# for patch_idx, patch_results in results.items():
#     results = {256: patch_results}
#     pc_per_method(results, 256, patch_idx)
#     pc_scatter_plots(results, 256, f"PC_scatter_patch_{patch_idx}.svg", patch_idx)
#     compare_smoothed_errors(results, [256], f"smoothed_error_cdf_patch_{patch_idx}.svg", patch_idx)
#     plot_top_pcs(results, num_obs=256, num_pcs=3,
#                     title=f"Principal Components per Method  - Patch {patch_idx}",
#                     fileName=f"pc_top3_images_256_patch_{patch_idx}.png", 
#     )
#     coeff_vectors_hist(results, 256, patch_idx)
#     coeff_vectors_cdf(results, 256, patch_idx)

# TODO: run for all patches
pc_scatter_plots_all_patches(results, PATCH_IDXS, "all_patches_pc_scatter.svg")
pc_per_method_all_patches(results, PATCH_IDXS,"all_patches_true_pc_per_method.svg", vector="true")
pc_per_method_all_patches(results, PATCH_IDXS,"all_patches_est_pc_per_method.svg", vector="est")
error_all_patches(results, PATCH_IDXS, "all_patches_error_cumsum.svg")
coeff_vectors_hist_all_patches(results, PATCH_IDXS, "all_patches_coeffs_hist_full_y.svg" )
coeff_vectors_cdf_all_patches(results, PATCH_IDXS, "all_patches_coeffs_cdf_lim.svg")
