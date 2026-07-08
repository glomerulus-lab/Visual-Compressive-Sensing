import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.compress_sensing import *

'''
Plotting helpers shared by pc_plots.py (whole-image) and paper_aligned_plots.py
(per-patch). Each takes an optional `patch_idx` where the original functions
diverged only on patch-vs-whole-image labeling/filenames; PC-image reshaping
is derived from the data itself rather than a hardcoded 30/32 patch size.
'''


def plot_smoothed_error(ax, err, label):
    '''
    Plot a rolling-mean smoothed version of the per-component squared error.
    '''
    df = pd.DataFrame({"Index": range(len(err)), "Error": err})
    # rolling mean (window of 15 components) to smooth the curve
    df["Smoothed_Error"] = df["Error"].rolling(15, min_periods=1).mean()

    ax.plot(df["Index"], df["Smoothed_Error"], label=label)


def cumsum_err(ax, err, label):
    '''
    Plot the cumulative sum of per-component squared errors against component index.
    '''
    err = np.array(err)
    cumsum_err = np.cumsum(err)
    x = np.arange(1, len(err) + 1)
    ax.plot(x, cumsum_err, label=label)


def compare_smoothed_errors(results, num_obs_list, filename, patch_idx=None):
    '''
    Plot cumulative squared error for V1/Pixel/Gaussian, one subplot per num_obs.
    '''
    n_obs = len(num_obs_list)
    fig, axes = plt.subplots(1, n_obs, figsize=(8 * n_obs, 6))

    # make sure axes is always iterable even with a single subplot
    if n_obs == 1:
        axes = [axes]

    title_suffix = f" - Patch {patch_idx}" if patch_idx is not None else ""

    for ax, num_obs in zip(axes, num_obs_list):
        for method in ["V1", "Pixel", "Gaussian"]:
            cumsum_err(ax, results[num_obs][method]["error"], method)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_title(f"Error per Component{title_suffix}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Squared Error")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def pc_scatter_plots(results, num_obs, filename, patch_idx=None, cmap='cool'):
    '''
    Plot true vs estimated principal component magnitudes as scatter plots.
    '''
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

    y_label_prefix = f"Patch {patch_idx}\n\n" if patch_idx is not None else ""

    for col, (ax, method) in enumerate(zip(axes, methods)):
        est = results[num_obs][method]["a_est"]
        true = results[num_obs][method]["a_true"]

        sc = ax.scatter(np.abs(est), np.abs(true), c=np.arange(len(est)), s=30, cmap=cmap, alpha=0.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.plot([global_min, global_max], [global_min, global_max], '--', color='gray')
        ax.set_aspect('equal', adjustable='box')

        ax.set_title(method, fontsize=18)
        ax.set_xlabel("Estimated PC", fontsize=14)
        if col == 0:
            ax.set_ylabel(f"{y_label_prefix}True PC", fontsize=18)
        else:
            ax.yaxis.set_visible(False)

    fig.colorbar(sc, ax=axes[2], shrink=0.8, label="PC rank")
    fig.suptitle("True vs Estimated Principal Components", fontsize=20)
    plt.savefig(filename, format="png")
    plt.close()


def plot_first_pc(results, num_obs, cmap="gray", title=None, figsize=(12, 4), fileName=None):
    '''
    Display the first principal component as an image for each method.
    '''
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)

    plt.figure(figsize=figsize)

    for i, method in enumerate(methods):
        pc_row = results[num_obs][method]["Vh"][0, :]
        side = int(round(np.sqrt(pc_row.shape[0])))
        pc_dct = pc_row.reshape(side, side)
        pc = fft.idctn(pc_dct, norm='ortho', axes=[0, 1])
        ax = plt.subplot(1, n_methods, i + 1)
        ax.imshow(pc, cmap=cmap)
        ax.axis("off")
        ax.set_title(f'{method} First PC', fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300)
    plt.close()


def plot_top_pcs(results, num_obs, num_pcs=3, cmap="gray", title=None, figsize=(12, 8), fileName=None):
    '''
    Display the top-k principal components as images, one row per method.
    '''
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)

    plt.figure(figsize=figsize)

    for row, method in enumerate(methods):
        Vh = results[num_obs][method]["Vh"]
        side = int(round(np.sqrt(Vh.shape[1])))

        for col in range(num_pcs):
            pc_dct = Vh[col, :].reshape(side, side)
            pc = fft.idctn(pc_dct, norm='ortho', axes=[0, 1])

            ax = plt.subplot(n_methods, num_pcs, row * num_pcs + col + 1)
            ax.imshow(pc, cmap=cmap)
            ax.axis("off")

            ax.set_title(f"PC {col + 1}", fontsize=10)

            if col == 0:
                ax.annotate(method, xy=(-0.25, 0.5), xycoords="axes fraction", rotation=90, ha="right", va="center", fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fileName, dpi=300)
    plt.close()


def coeff_vectors_hist(results, num_obs, patch_idx=None):
    '''
    Plot histograms of coefficient magnitudes for true and estimated coefficient
    vectors, and print counts of near-zero coefficients.
    '''
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
        ax.hist(np.abs(coeffs), bins=bins, edgecolor="black", color='C' + str(i))
        ax.set_xlabel("Coefficient Magnitude")
        ax.set_ylabel("Number of Coefficients")
        ax.set_title(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    title_suffix = f" - Patch {patch_idx}" if patch_idx is not None else ""
    filename_suffix = f"_patch_{patch_idx}" if patch_idx is not None else ""

    plt.suptitle(f"Coefficient Histograms{title_suffix}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"coeff_histograms_{num_obs}{filename_suffix}.svg", dpi=300)
    plt.close()

    print(f"\nNumber of coefficients <0.1 and <0.5 ({num_obs} Obs):")
    for label, coeffs in coeff_vectors:
        less_than_01 = np.sum(np.abs(coeffs) < 0.1)
        less_than_05 = np.sum(np.abs(coeffs) < 0.5)
        print(f"{label:15s}  <0.1: {less_than_01:4d},  <0.5: {less_than_05:4d}")


def coeff_vectors_cdf(results, num_obs, patch_idx=None):
    '''
    Plot the CDF of absolute coefficient values for true and estimated
    coefficient vectors on a log x-axis.
    '''
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
    title_suffix = f" - Patch {patch_idx}" if patch_idx is not None else ""
    filename_suffix = f"_patch_{patch_idx}" if patch_idx is not None else ""
    plt.title(f"CDF of Coefficients{title_suffix}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"coeff_cdf_{num_obs}{filename_suffix}.svg", dpi=300)
    plt.close()
