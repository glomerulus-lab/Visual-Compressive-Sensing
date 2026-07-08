import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys
import seaborn as sns
import pandas as pd


from src.compress_sensing import *
from src.utility import *
from ..core import *
from .util import *

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

# pc_scatter_plots, plot_smoothed_error, compare_smoothed_errors, plot_first_pc,
# plot_top_pcs, coeff_vectors_hist, and coeff_vectors_cdf live in .util now,
# shared with plots/paper_aligned_plots.py.

# pc_scatter_plots(results, 100, "PC_scatter_100.svg")
# pc_scatter_plots(results, 300, "PC_scatter_300.svg")

# compare_smoothed_errors(results, [100, 300], "smoothed_error_log_100_300.svg")

# plot_first_pc(results, num_obs=100,
#               title="First Principal Component (30 x 30) (100 Obs)",
#               fileName="pc_first_images_100.png")

# plot_first_pc(results, num_obs=300,
#               title="First Principal Component (30 x 30) (300 Obs)",
#               fileName="pc_first_images_300.png")

# plot_top_pcs(results, num_obs=100, num_pcs=3,
#                 title="Top 3 Principal Components per Method (30×30) (100 obs)",
#                 fileName="pc_top3_images_100_labeled.png"
# )
# plot_top_pcs(results, num_obs=300, num_pcs=3,
#                 title="Top 3 Principal Components per Method (30×30) (300 obs)",
#                 fileName="pc_top3_images_300_labeled.png"
# )

coeff_vectors_hist(results, 100)
coeff_vectors_hist(results, 300)

# coeff_vectors_cdf(results, 100)
# coeff_vectors_cdf(results, 300)
