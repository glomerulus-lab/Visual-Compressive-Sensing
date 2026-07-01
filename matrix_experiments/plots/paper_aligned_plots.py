import sys
import matplotlib.pyplot as plt
import sys
import pandas as pd

sys.path.append('../')
from src.compress_sensing import *
from src.utility import *
from plots.theta_exp_improved import *
from plots.extract_patches import *
from plots.exp_constants import *

'''
PLOTS: 
    - est pcs vs true pcs scatter 
    - est pcs vs rank 
    - pc error
    - pcs as images 
    - coefficient hists
''' 

def compute_patch_results(patch, n, cell_size, blob_size, alpha):
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
                "a_est"         (ndarray): Estimated coefficients projected onto PCs.
                "a_true"        (ndarray): True coefficients projected onto PCs.
                "error"         (ndarray): Per-component squared error (a_true - a_est)^2.
    """
    # true coefs of theta
    coeffs_true = generate_coeff_vector(patch, n, cell_size, blob_size)

    # V1 - SVD
    measurement_matrix_V1, V1_y = generate_V1_observation(patch, n, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)
    U_V1, S_V1, Vh_V1 = np.linalg.svd(theta_V1)

    # V1 - estimated coeffs
    reconst_v1 = reconstruct(measurement_matrix_V1, V1_y, alpha)
    coeffs_est_V1 = generate_coeff_vector(reconst_v1, n, cell_size, blob_size)

    # Project both true and estimated coefficients onto the V1 principal components
    a_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()
    a_true_V1 = Vh_V1 @ coeffs_true.flatten()

    err_V1 = (a_true_V1 - a_est_V1) ** 2

    # Pixel - SVD
    measurement_matrix_pix, pixel_y = generate_pixel_observation(patch, n)
    theta_pix = generate_design_matrix(measurement_matrix_pix)
    U_pix, S_pix, Vh_pix = np.linalg.svd(theta_pix)

    # Pixel - estimated coeffs
    reconst_pix = reconstruct(measurement_matrix_pix, pixel_y, alpha)
    coeffs_est_pix = generate_coeff_vector(reconst_pix, n, cell_size, blob_size)

    # Project both true and estimated coefficients onto the Pixel principal components
    a_est_pix = Vh_pix @ coeffs_est_pix.flatten()
    a_true_pix = Vh_pix @ coeffs_true.flatten()

    err_pix = (a_true_pix - a_est_pix) ** 2

    # Gauss - SVD
    measurement_matrix_gauss, gaussian_y = generate_gaussian_observation(patch, n)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)
    U_gauss, S_gauss, Vh_gauss = np.linalg.svd(theta_gauss)

    # Gauss - estimated coeffs
    reconst_gauss = reconstruct(measurement_matrix_gauss, gaussian_y, alpha)
    coeffs_est_gauss = generate_coeff_vector(reconst_gauss, n, cell_size, blob_size)

    # Project both true and estimated coefficients onto the Gaussian principal components
    a_est_gauss = Vh_gauss @ coeffs_est_gauss.flatten()
    a_true_gauss = Vh_gauss @ coeffs_true.flatten()

    err_gauss = (a_true_gauss - a_est_gauss) ** 2

    return {
        "coeffs_true": coeffs_true,

        "V1": {
            "U" : U_V1,
            "S" : S_V1,
            "Vh": Vh_V1,
            "reconstruction": reconst_v1,
            "est_coeffs" : coeffs_est_V1,
            "a_est": a_est_V1,
            "a_true": a_true_V1,
            "error" : err_V1,
        },
        "Pixel": {
            "U" : U_pix,
            "S" : S_pix,
            "Vh": Vh_pix,
            "reconstruction": reconst_pix,
            "est_coeffs" : coeffs_est_pix,
            "a_est": a_est_pix,
            "a_true": a_true_pix,
            "error" : err_pix,
        },
        "Gaussian": {
            "U" : U_gauss,
            "S" : S_gauss,
            "Vh": Vh_gauss,
            "reconstruction": reconst_gauss,
            "est_coeffs" : coeffs_est_gauss,
            "a_est": a_est_gauss,
            "a_true": a_true_gauss,
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

barbara = process_image("barbara.bmp", color=False)
patches = extract_patches(barbara, PATCH_SIZE)
# show_patches_grid(patches)
results = run_selected_patches(patches, PATCH_IDXS)

def pc_scatter_plots(results, num_obs, filename, patch_idx, cmap='cool'):
    """
    Plot true vs estimated principal component magnitudes as scatter plots for a
    single patch.

    Args:
        results (dict): 
            results dict from compute_patch_results
        num_obs (int): 
            Observation count key used to index into 'results'.
        filename (str): 
            Output path for the saved file.
        patch_idx (int): 
            Patch index used for the y-axis label.
        cmap (str): 
            Matplotlib colormap name for PC rank colouring. Default 'cool'.

    """
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    # shared axis range across all methods for y=x line
    global_min = np.inf
    global_max = -np.inf
    for method in methods:
        est = np.abs(results[num_obs][method]["a_est"])
        true = np.abs(results[num_obs][method]["a_true"])
        combined = np.concatenate([est, true])
        combined = combined[combined > 0]  # exclude zeros before taking log
        global_min = min(global_min, combined.min())
        global_max = max(global_max, combined.max())

    for col, (ax, method) in enumerate(zip(axes, methods)):
        est = results[num_obs][method]["a_est"]
        true = results[num_obs][method]["a_true"]

        sc = ax.scatter(np.abs(est), np.abs(true), c=np.arange(len(est)), s=30, cmap=cmap, alpha=0.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        # y=x line
        ax.plot([global_min, global_max], [global_min, global_max], '--', color='gray')
        ax.set_aspect('equal', adjustable='box')

        ax.set_title(method, fontsize=18)
        ax.set_xlabel("Estimated PC", fontsize=14)
        if col == 0:
            ax.set_ylabel(f"Patch {patch_idx}\n\nTrue PC", fontsize=18)
        else:
            ax.yaxis.set_visible(False)

    fig.colorbar(sc, ax=axes[2], shrink=0.8, label="PC rank")
    fig.suptitle(f"True vs Estimated Principal Components", fontsize=20)
    plt.savefig(filename, format="png")
    plt.close()

def pc_per_method(results, num_obs, patch_idx):
    """
    Scatter plot of estimated principal component magnitudes by rank for each
    measurement method.

    Args:
        results (dict): Nested results dict keyed by num_obs.
        num_obs (int): Observation count key used to index into 'results'.
        patch_idx (int): Patch index used for the plot title and output filename.
    """
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, methods):

        components = np.abs(results[num_obs][method]["a_est"]) # TODO: Change this to "a_true" to get true pc plot (bumps)
        ax.scatter(range(len(components)), components, s=10, color='skyblue')

        ax.set_xlabel("Rank")
        ax.set_ylabel("True Principal Component")
        ax.set_yscale('log')
        ax.set_title(f"{method} True Principal Component")

    plt.suptitle(f"Principal Component - Patch {patch_idx}")
    plt.tight_layout()
    plt.savefig(f"pc_per_method_patch_{patch_idx}.svg", dpi=300)
    plt.close()

def plot_smoothed_error(ax, err, label):
    """
    Plot a rolling-mean smoothed version of the per-component squared error.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        err (array-like): Per-component squared error values.
        label (str): Legend label for the plotted line.
    """
    df = pd.DataFrame({"Index": range(len(err)), "Error": err})
    # Rolling mean (window of 15 components)
    df["Smoothed_Error"] = df["Error"].rolling(15, min_periods=1).mean()

    ax.plot(df["Index"], df["Smoothed_Error"], label=label)

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

def cumsum_err(ax, err, label):
    """
    Plot the cumulative sum of per-component squared errors against component index.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        err (array-like): Per-component squared error values.
        label (str): Legend label for the plotted line.

    Returns:
        None: Cumulative sum line is drawn onto 'ax' in place.
    """
    err = np.array(err)
    cumsum_err = np.cumsum(err)
    x = np.arange(1, len(err) + 1)
    ax.plot(x, cumsum_err, label=label)


def compare_smoothed_errors(results, num_obs_list, filename, patch_idx):
    """
    Plot squared errors for all three methods.

    Args:
        results (dict): Nested results dict keyed by num_obs.
        num_obs_list (list[int]): Observation count keys to plot, one subplot each.
        filename (str): Output path for the saved figure.
        patch_idx (int): Patch index used in subplot titles.
    """
    n_obs = len(num_obs_list)
    fig, axes = plt.subplots(1, n_obs, figsize=(8*n_obs, 6))

    # make sure axes is always iterable even with a single subplot
    if n_obs == 1:
        axes = [axes]

    for ax, num_obs in zip(axes, num_obs_list):
        for method in ["V1", "Pixel", "Gaussian"]:
            cumsum_err(ax, results[num_obs][method]["error"], method)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_title(f"Error per Component - Patch {patch_idx}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Squared Error")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_first_pc(results, num_obs, cmap="gray", title=None, figsize=(12, 4), fileName=None):
    """
    Display the first principal component as a image for
    each method.

    Args:
        results (dict): 
            Nested results dict keyed by num_obs.
        num_obs (int): 
            Observation count key used to index into 'results'.
        cmap (str): 
            Matplotlib colormap for imshow. Default 'gray'.
        title (str): 
            Overall figure title. Default None.
        figsize (tuple): 
            Figure size in inches as (width, height). Default (12, 4).
        fileName (str): 
            Output path for the saved figure.

    """
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)
    
    plt.figure(figsize=figsize)
    
    for i, method in enumerate(methods):
        # First row of Vh corresponds -> largest PC
        pc_dct = results[num_obs][method]["Vh"][0, :].reshape(32, 32)
        # Convert back to pixel space
        pc = fft.idctn(pc_dct, norm='ortho', axes=[0, 1])
        ax = plt.subplot(1, n_methods, i+1)
        ax.imshow(pc, cmap=cmap)
        ax.axis("off")
        ax.set_title(f'{method} Last PC', fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300)
    plt.close()

def plot_top_pcs(results, num_obs, num_pcs=3, cmap="gray", title=None, figsize=(12, 8), fileName=None):
    """
    Display the top-k principal components as spatial images in a grid, with
    rows for each measurement method and columns for each PC rank.

    Args:
        results (dict): 
            Nested results dict keyed by num_obs.
        num_obs (int): 
            Observation count key used to index into `results`.
        num_pcs (int): 
            Number of top PCs to display per method. Default 3.
        cmap (str): 
            Matplotlib colormap for imshow. Default 'gray'.
        title (str): 
            Overall figure title. Default None.
        figsize (tuple): 
            Figure size in inches as (width, height). Default (12, 8).
        fileName (str): 
            Output path for the saved figure.
    """
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)

    plt.figure(figsize=figsize)

    for row, method in enumerate(methods):
        Vh = results[num_obs][method]["Vh"]

        for col in range(num_pcs):
            # Each row of Vh is a right singular vector (pc in DCT space)
            pc_dct = Vh[col, :].reshape(PATCH_SIZE, PATCH_SIZE)
            pc = fft.idctn(pc_dct, norm='ortho', axes=[0, 1])

            ax = plt.subplot(n_methods, num_pcs, row * num_pcs + col + 1)
            ax.imshow(pc, cmap=cmap)
            ax.axis("off")

            ax.set_title(f"PC {col + 1}", fontsize=10)

            # method name on the leftmost column only
            if col == 0:
                ax.annotate(method, xy=(-0.25, 0.5), xycoords="axes fraction", rotation=90, ha="right", va="center", fontsize=12)
                
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fileName, dpi=300)
    plt.close()

def coeff_vectors_hist(results, num_obs, patch_idx):
    """
    Plot histograms of coefficient magnitudes for the true and all three estimated
    coefficient vectors of a single patch. 
    Prints the count of near-zero coefficients.

    Args:
        results (dict): 
            Nested results dict keyed by num_obs.
        num_obs (int): 
            Observation count key used to index into `results`.
        patch_idx (int): 
            Patch index used for the plot title and output filename.
    """
    plt.figure(figsize=(16, 4))

    coeff_vectors = [
        ("V1 Estimated", results[num_obs]["V1"]["est_coeffs"].flatten()),
        ("Pixel Estimated", results[num_obs]["Pixel"]["est_coeffs"].flatten()),
        ("Gaussian Estimated", results[num_obs]["Gaussian"]["est_coeffs"].flatten()),
        ("True", results[num_obs]["coeffs_true"].flatten()),
    ]
    
    # 99th percentile
    all_abs = np.concatenate([np.abs(c) for _, c in coeff_vectors])
    upper = np.percentile(all_abs, 99)
    bins = np.linspace(0, upper, 50)

    for i, (label, coeffs) in enumerate(coeff_vectors):
        ax = plt.subplot(1, 4, i + 1)
        ax.hist(np.abs(coeffs), bins=bins, edgecolor="black", color='C'+str(i))
        ax.set_xlabel("Coefficient Magnitude")
        ax.set_ylabel("Number of Coefficients")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    plt.suptitle(f"Coefficient Histograms - Patch {patch_idx}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"coeff_histograms_{num_obs}_patch_{patch_idx}.svg", dpi=300)
    plt.close()

    print(f"\nNumber of coefficients <0.1 and <0.5 ({num_obs} Obs):")
    for label, coeffs in coeff_vectors:
        less_than_01 = np.sum(np.abs(coeffs) < 0.1)
        less_than_05 = np.sum(np.abs(coeffs) < 0.5)
        print(f"{label:15s}  <0.1: {less_than_01:4d},  <0.5: {less_than_05:4d}")

def coeff_vectors_cdf(results, num_obs, patch_idx):
    """
    Plot the CDF of absolute coefficient values for the true and estimated
    coefficient vectors of a single patch on a log x-axis.

    Args:
        results (dict): 
            Nested results dict keyed by num_obs.
        num_obs (int): 
            Observation count key used to index into `results`.
        patch_idx (int): 
            Patch index used for the plot title and output filename.
    """
    plt.figure(figsize=(6, 5))

    coeff_vectors = [
        ("True", results[num_obs]["coeffs_true"].flatten()),
        ("V1 Estimated", results[num_obs]["V1"]["est_coeffs"].flatten()),
        ("Pixel Estimated", results[num_obs]["Pixel"]["est_coeffs"].flatten()),
        ("Gaussian Estimated", results[num_obs]["Gaussian"]["est_coeffs"].flatten()),
    ]

    for label, coeffs in coeff_vectors:
        abs_coeffs = np.sort(np.abs(coeffs))
        cdf = np.arange(1, len(abs_coeffs) + 1) / len(abs_coeffs)
        plt.plot(abs_coeffs, cdf, label=label)

    plt.xscale("log")
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("CDF")
    plt.title(f"CDF of Coefficients - Patch {patch_idx}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"coeff_cdf_{num_obs}_patch_{patch_idx}.svg", dpi=300)
    plt.close()

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
            est = np.abs(results[patch_idx][method]["a_est"])
            true = np.abs(results[patch_idx][method]["a_true"])
            combined = np.concatenate([est, true])
            combined = combined[combined > 0]
            global_min = min(global_min, combined.min())
            global_max = max(global_max, combined.max())

    for row, patch_idx in enumerate(patch_idxs):
        sc = None
        for col, method in enumerate(methods):
            ax = axes[row, col]

            est = results[patch_idx][method]["a_est"]
            true = results[patch_idx][method]["a_true"]

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
    
def pc_per_method_all_patches(results, patch_idxs, filename):
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
    """
    n_rows = len(patch_idxs)
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, patch_idx in enumerate(patch_idxs):

        for col, method in enumerate(methods):
            ax = axes[row, col]

            components = np.abs(results[patch_idx][method]["a_est"])
            ranks = np.arange(1, len(components) + 1)

            ax.scatter(ranks, components, s=10, color='skyblue')

            ax.set_yscale('log')
            if row == 0:
                ax.set_title(method, fontsize=18)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\n Estimated Principal Components", fontsize=18)
            if row == n_rows - 1: 
                ax.set_xlabel("Rank", fontsize=18)
            if col != 0:
                ax.yaxis.set_visible(False)

    fig.suptitle("Estimated Principal Components", fontsize=20, x=0.55)
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
# pc_scatter_plots_all_patches(results, PATCH_IDXS, "all_patches_pc_scatter.svg")
# pc_per_method_all_patches(results, PATCH_IDXS,"all_patches_true_pc_per_method.svg")
# error_all_patches(results, PATCH_IDXS, "all_patches_error_cumsum.svg")
# coeff_vectors_hist_all_patches(results, PATCH_IDXS, "all_patches_coeffs_hist_full_y.svg" )
# coeff_vectors_cdf_all_patches(results, PATCH_IDXS, "all_patches_coeffs_cdf_lim.svg")