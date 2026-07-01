import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys
import seaborn as sns
import pandas as pd


sys.path.append('..')
from src.compress_sensing import *
from src.utility import *
from A_experiments.theta_exp_improved import *

def compute_results(num_obs):

    # true coefs of theta
    coeffs_true = generate_coeff_vector( small_img_arr_gray, num_obs, cell_size, blob_size)

    # V1 - SVD
    measurement_matrix_V1, V1_y = generate_V1_observation(small_img_arr_gray, num_obs, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)
    U_V1, S_V1, Vh_V1 = np.linalg.svd(theta_V1)

    # V1 - estimated coeffs
    reconst_v1 = reconstruct(measurement_matrix_V1, V1_y, alpha)
    coeffs_est_V1 = generate_coeff_vector(reconst_v1, num_obs, cell_size, blob_size)

    # V1 - PCs
    a_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()
    a_true_V1 = Vh_V1 @ coeffs_true.flatten()

    err_V1 = (a_true_V1 - a_est_V1) ** 2

    # Pixel - SVD
    measurement_matrix_pix, pixel_y = generate_pixel_observation(small_img_arr_gray, num_obs)
    theta_pix = generate_design_matrix(measurement_matrix_pix)
    U_pix, S_pix, Vh_pix = np.linalg.svd(theta_pix)

    # Pixel - estimated coeffs
    reconst_pix = reconstruct(measurement_matrix_pix, pixel_y, alpha)
    coeffs_est_pix = generate_coeff_vector(reconst_pix, num_obs, cell_size, blob_size)

    # Pixel - PCs
    a_est_pix = Vh_pix @ coeffs_est_pix.flatten()
    a_true_pix = Vh_pix @ coeffs_true.flatten()

    err_pix = (a_true_pix - a_est_pix) ** 2

    # Gauss - SVD
    measurement_matrix_gauss, gaussian_y = generate_gaussian_observation(small_img_arr_gray, num_obs)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)
    U_gauss, S_gauss, Vh_gauss = np.linalg.svd(theta_gauss)

    # Gauss - estimated coeffs
    reconst_gauss = reconstruct(measurement_matrix_gauss, gaussian_y, alpha)
    coeffs_est_gauss = generate_coeff_vector(reconst_gauss, num_obs, cell_size, blob_size)

    # Gauss - PCs
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

# compute results for 100 and 300 obs
results = { # index into results like so: results[100]["V1"]["a_est"]
    100: compute_results(100),
    300: compute_results(300),
}

def pc_scatter_plots(results, num_obs, filename, cmap='cool'):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, ["V1", "Pixel", "Gaussian"]):
        # get estimate and true PCs
        est = results[num_obs][method]["a_est"]
        true = results[num_obs][method]["a_true"]

        # make scatter plot
        sc = ax.scatter(np.abs(est), np.abs(true), c=np.arange(len(est)), s = 30, cmap=cmap, alpha=0.5)

        # y = x line
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        low = max(xmin, ymin) # start at largest of the 2 mins, so it doesn't go below
        high = min(xmax, ymax) # end at smallest of 2 maxima -> doesn't go beyond
        ax.plot([low, high], [low, high], '--', color='gray')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{method} vs True ({num_obs} obs)")
        ax.set_xlabel(f"{method} Principal Component")
        ax.set_ylabel("True Principal Component")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('PC rank', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# pc_scatter_plots(results, 100, "PC_scatter_100.svg")
# pc_scatter_plots(results, 300, "PC_scatter_300.svg")

def plot_smoothed_error(ax, err, label):
    df = pd.DataFrame({"Index": range(len(err)), "Error": err})
    # rolling mean (window of 150 components) to smooth the curve
    df["Smoothed_Error"] = df["Error"].rolling(15, min_periods=1).mean()

    ax.plot(df["Index"], df["Smoothed_Error"], label=label)


def compare_smoothed_errors(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, num_obs in zip(axes, [100, 300]):
        for method in ["V1", "Pixel", "Gaussian"]:
            plot_smoothed_error(ax, results[num_obs][method]["error"], method)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Error per Component ({num_obs} Observations)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Squared Error")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# compare_smoothed_errors(results, "smoothed_error_log_100_300.svg")

'''
PCs as pics
'''

# top of each method
def plot_first_pc(results, num_obs, cmap="gray", title=None, figsize=(12, 4), fileName=None):
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)
    
    plt.figure(figsize=figsize)
    
    for i, method in enumerate(methods):
        pc_dct = results[num_obs][method]["Vh"][0, :].reshape(30, 30)
        pc = fft.idctn(pc_dct, norm = 'ortho', axes = [0, 1])
        ax = plt.subplot(1, n_methods, i+1)
        ax.imshow(pc, cmap=cmap)
        ax.axis("off")
        ax.set_title(f'{method} First PC', fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300)
    plt.close()

# plot_first_pc(results, num_obs=100,
#               title="First Principal Component (30 x 30) (100 Obs)",
#               fileName="pc_first_images_100.png")

# plot_first_pc(results, num_obs=300,
#               title="First Principal Component (30 x 30) (300 Obs)",
#               fileName="pc_first_images_300.png")

# top 3 of each method
def plot_top_pcs(results, num_obs, num_pcs=3, cmap="gray", title=None, figsize=(12, 8), fileName=None):
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)

    plt.figure(figsize=figsize)

    for row, method in enumerate(methods):
        Vh = results[num_obs][method]["Vh"]

        for col in range(num_pcs):
            pc_dct = Vh[col, :].reshape(30, 30)
            pc = fft.idctn(pc_dct, norm = 'ortho', axes = [0, 1])

            ax = plt.subplot(n_methods, num_pcs, row * num_pcs + col + 1)
            ax.imshow(pc, cmap=cmap)
            ax.axis("off")

            # PC label
            ax.set_title(f"PC {col + 1}", fontsize=10)

            # method label
            if col == 0:
                ax.annotate(method, xy=(-0.25, 0.5), xycoords="axes fraction", rotation=90, ha="right", va="center", fontsize=12)
                
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fileName, dpi=300)
    plt.close()

# plot_top_pcs(results, num_obs=100, num_pcs=3,
#                 title="Top 3 Principal Components per Method (30×30) (100 obs)",
#                 fileName="pc_top3_images_100_labeled.png"
# )
# plot_top_pcs(results, num_obs=300, num_pcs=3,
#                 title="Top 3 Principal Components per Method (30×30) (300 obs)",
#                 fileName="pc_top3_images_300_labeled.png"
# )

'''
Sparcity of coeffs vectors - histogram of entries in coeffs vectors
'''
def coeff_vectors_hist(results, num_obs):
    plt.figure(figsize=(16, 4))

    # labels and coeffs
    coeff_vectors = [
        ("True", results[num_obs]["coeffs_true"].flatten()),
        ("V1 Estimated", results[num_obs]["V1"]["est_coeffs"].flatten()),
        ("Pixel Estimated", results[num_obs]["Pixel"]["est_coeffs"].flatten()),
        ("Gaussian Estimated", results[num_obs]["Gaussian"]["est_coeffs"].flatten()),
    ]
    max_val = max(np.max(np.abs(coeffs)) for _, coeffs in coeff_vectors)
    bins = np.linspace(0, max_val, 50)

    for i, (label, coeffs) in enumerate(coeff_vectors):
        ax = plt.subplot(1, 4, i + 1)
        ax.hist(np.abs(coeffs), bins=bins, edgecolor="black", color='C'+str(i))
        ax.set_xlabel("Absolute Coefficient Value")
        ax.set_ylabel("Number of Coefficients")
        ax.set_title(label)
        ax.set_ylim(0, 12)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Coefficient Histograms ({num_obs} observations)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"coeff_histograms_{num_obs}.svg", dpi=300)
    plt.close()

    print(f"\nNumber of coefficients <0.1 and <0.5 ({num_obs} Obs):")
    for label, coeffs in coeff_vectors:
        less_than_01 = np.sum(np.abs(coeffs) < 0.1)
        less_than_05 = np.sum(np.abs(coeffs) < 0.5)
        print(f"{label:15s}  <0.1: {less_than_01:4d},  <0.5: {less_than_05:4d}")

coeff_vectors_hist(results, 100)
coeff_vectors_hist(results, 300)

# cdf of coeffs
def coeff_vectors_cdf(results, num_obs):
    plt.figure(figsize=(6, 5))

    # labels, coefficient vectors
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

    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("CDF")
    plt.title(f"CDF of Coefficients ({num_obs} Observations)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"coeff_cdf_{num_obs}.svg", dpi=300)
    plt.close()

# coeff_vectors_cdf(results, 100)
# coeff_vectors_cdf(results, 300)
