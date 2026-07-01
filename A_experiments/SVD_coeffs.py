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


'''
Compare the Fourier coefficients to singular values of theta.
'''
# Find the true coefficients of theta
coeffs_true = generate_coeff_vector(small_img_arr_gray, num_cell_300, cell_size, blob_size)
U_c_true, S_c_true, Vh_c_true = np.linalg.svd(coeffs_true)

# Find the singular values of theta
measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
theta_V1 = generate_design_matrix(measurement_matrix_V1)

U_V1, S_V1, Vh_V1 = np.linalg.svd(theta_V1)

measurement_matrix_pix, pixel_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
theta_pix = generate_design_matrix(measurement_matrix_pix)

U_pix, S_pix, Vh_pix = np.linalg.svd(theta_pix)

measurement_matrix_gauss, gaussian_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
theta_gauss = generate_design_matrix(measurement_matrix_gauss)

U_gauss, S_gauss, Vh_gauss = np.linalg.svd(theta_gauss)

# Find the estimated coefficients for each observation type
reconst_gray_300_v1 = reconstruct(measurement_matrix_V1, V1_y_300, alpha)
coeffs_est_V1 = generate_coeff_vector(reconst_gray_300_v1, num_cell_300, cell_size, blob_size)


reconst_gray_300_pix = reconstruct(measurement_matrix_pix, pixel_y_300, alpha)
coeffs_est_pix = generate_coeff_vector(reconst_gray_300_pix, num_cell_300, cell_size, blob_size)


reconst_gray_300_gauss = reconstruct(measurement_matrix_gauss, gaussian_y_300, alpha)
coeffs_est_gauss = generate_coeff_vector(reconst_gray_300_gauss, num_cell_300, cell_size, blob_size)


# find principal components from coefficient vectors for each obs type
#U_c_V1, S_c_V1, Vh_c_V1 = np.linalg.svd(coeffs_est_V1)
a_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()
a_true_V1 = Vh_V1 @ coeffs_true.flatten()

#U_c_pix, S_c_pix, Vh_c_pix = np.linalg.svd(coeffs_est_pix)
a_est_pix = Vh_pix @ coeffs_est_pix.flatten()
a_true_pix = Vh_pix @ coeffs_true.flatten()

#U_c_gauss, S_c_gauss, Vh_c_gauss = np.linalg.svd(coeffs_est_gauss)
a_est_gauss = Vh_gauss @ coeffs_est_gauss.flatten()
a_true_gauss = Vh_gauss @ coeffs_true.flatten()


# compare squared error for principal components and raw pixel values

squared_error_V1 = (a_true_V1 - a_est_V1) ** 2
mse_V1 = np.mean(squared_error_V1)


im = process_image(small_img,False)
true_pixels = np.array(im)
true_pixels = true_pixels.reshape(1,900)

v1_pixels = reconst_gray_300_v1.reshape(1,900)

pixel_squared_error = np.empty(900)
for i in range(900):
    pixel_squared_error[i] = (true_pixels[0,i] - v1_pixels[0,i]) ** 2

mean_pixel_error = np.mean(pixel_squared_error)

# print("PC mse_V1:", mse_V1)
# print("Pixel mse:", mean_pixel_error)


def make_scatter(est, true, xlabel, title, filename, figsize=(8, 8), dpi=200, marker_size=30, cmap='cool', alpha=0.5):
    plt.figure(figsize=figsize, dpi=dpi)
    sc = plt.scatter(np.abs(est), np.abs(true), 
                     c=np.arange(len(est)), s=marker_size,
                     cmap=cmap, alpha=alpha)
    
    plt.colorbar(sc).set_label('PC rank', rotation=270, labelpad=15) # add label to colorbar

    plt.xlabel(f"{xlabel} Principal Component")
    plt.ylabel("True Principal Component")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)

    # add y=x line
    # visibal range on axes
    xmin, xmax = plt.xlim() # min, max of x
    ymin, ymax = plt.ylim() # min, max of y
    low = max(xmin, ymin) # start at largest of 2 mins, so it doesn't go below
    high = min(xmax, ymax) # end at smallest of 2 maxima -> doesn't go beyond
    plt.plot([low, high], [low, high])
    
    plt.savefig(filename)
    plt.close()


make_scatter(a_est_V1,   a_true_V1, "V1", "V1 vs True (300 samples)", "V1_vs_True_300_YlOrRd.png", alpha=0.5)
make_scatter(a_est_pix,  a_true_pix, "Pixel", "Pixel vs True (300 samples)", "Pixel_vs_True_300_YlOrRd.png")
make_scatter(a_est_gauss, a_true_gauss, "Gaussian", "Gaussian vs True (300 samples)", "Gaussian_vs_True_300_YlOrRd.png")

def make_combined_scatter(est_list, true_list, labels, titles, filename, figsize=(18, 6), dpi=200, marker_size=30, cmap='cool', alpha=0.5):
    """
    est_list: list of estimated arrays [a_est_V1, a_est_pix, a_est_gauss]
    true_list: list of true arrays [a_true_V1, a_true_pix, a_true_gauss]
    labels: list of x-axis labels ["V1", "Pixel", "Gaussian"]
    titles: list of subplot titles
    """
    n = len(est_list)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    for i in range(n):
        est = est_list[i]
        true = true_list[i]
        ax = axes[i]

        sc = ax.scatter(np.abs(est), np.abs(true), 
                        c=np.arange(len(est)), s=marker_size,
                        cmap=cmap, alpha=alpha)
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('PC rank', rotation=270, labelpad=15)

        ax.set_xlabel(f"{labels[i]} Principal Component")
        ax.set_ylabel("True Principal Component")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(titles[i])

        # y=x line
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        low = max(xmin, ymin)
        high = min(xmax, ymax)
        ax.plot([low, high], [low, high], color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Call the function
make_combined_scatter(
    est_list=[a_est_V1, a_est_pix, a_est_gauss],
    true_list=[a_true_V1, a_true_pix, a_true_gauss],
    labels=["V1", "Pixel", "Gaussian"],
    titles=["V1 vs True (300 samples)", "Pixel vs True (300 samples)", "Gaussian vs True (300 samples)"],
    filename="Combined_scatter_300.svg",
    alpha=0.5
)

def scatter_PCs(components, xlabel, title):
    plt.figure(figsize=(10,8), dpi=200)
    sc = plt.scatter([i for i in range(900)], components)

    plt.xlabel("Rank")
    plt.ylabel(f"{xlabel} Principal Component")
    #plt.xscale('log')
    plt.yscale('log')
    plt.title(title)

#scatter_PCs(a_est_V1, "V1", "V1 (300 samples)")
#scatter_PCs(a_true_V1, "V1 True", "V1 True (300 samples)")
#scatter_PCs(a_est_pix, "Pixel", "Pixel (300 samples)")
#scatter_PCs(a_est_gauss, "Gaussian", "Gaussian (300 samples)")


def plot_errors(errors, xlabel,title):
    plt.figure(figsize=(10,8), dpi=200)
    sc = plt.plot([i for i in range(900)], errors)

    plt.xlabel(f"{xlabel} Squared Error")
    plt.ylabel("Index")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)

plt.figure(figsize=(10,8), dpi=200)
plt.plot([i for i in range(900)], (a_true_V1 - a_est_V1) ** 2, label="V1")
plt.plot([i for i in range(900)], (a_true_pix - a_est_pix) ** 2, label="Pixel")
plt.plot([i for i in range(900)], (a_true_gauss - a_est_gauss) ** 2, label="Gaussian")

plt.xlabel("Squared Error")
plt.legend()
plt.ylabel("Index")
plt.xscale('log')
plt.yscale('log')
plt.title("Error Per Component (300 Samples)")
#plt.savefig("error_300.png")


# compute squared error
err_v1 = (a_true_V1 - a_est_V1) ** 2
err_pix = (a_true_pix - a_est_pix) ** 2
err_gauss = (a_true_gauss - a_est_gauss) ** 2

# wide data frame
wide_df = pd.DataFrame({
    'Index': range(900),
    'V1': err_v1,
    'Pixel': err_pix,
    'Gaussian': err_gauss
})

# line plot is optimized for "long" format for plotting multiple lines: one column for the category (Method) and one for the values (Squared Error)
# .melt reshapes table from wide -> long. Each row is a single observation w/ its method label
df_long = wide_df.melt(id_vars='Index', var_name='Method', value_name='Squared Error')

# rolling mean (window of 150 components) to smooth the curve
# groupby -> each observation type is seperate
# min_periods=1 prevents NaN values at the start of the series.
df_long['Smoothed Error'] = df_long.groupby('Method')['Squared Error'].transform(
    lambda x: x.rolling(window=150, min_periods=1).mean()
)

plt.figure(figsize=(10, 8), dpi=200)
sns.lineplot(data=df_long, x='Index', y='Smoothed Error', hue='Method')

plt.xscale('log')
plt.yscale('log')
plt.title("Error Per Component (300 Obs)")
plt.xlabel("Index")
plt.ylabel("Squared Error")

plt.savefig("smoothed_error_plot_300.svg", bbox_inches='tight')

import matplotlib.image as mpimg
img1 = mpimg.imread("smoothed_error_plot_300.png")
img2 = mpimg.imread("smoothed_error_plot_300.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # adjust size as needed

# Show first image
axes[0].imshow(img1)
axes[0].axis('off')  # hide axes

# Show second image
axes[1].imshow(img2)
axes[1].axis('off')

plt.tight_layout()
plt.savefig("combined_smoothed_error.svg", dpi=200)
plt.show()


# '''
# Reconstructions as pics

# '''

reconstructed_images = {
    "V1": reconst_gray_300_v1,
    "Pixel": reconst_gray_300_pix,
    "Gaussian": reconst_gray_300_gauss
}

plt.figure(figsize=(12,4))

for i, (label, img_flat) in enumerate(reconstructed_images.items()):
    plt.subplot(1, 3, i+1)
    plt.imshow(img_flat.reshape(30,30), cmap='gray')
    plt.title(f'Reconstructed ({label})')
    plt.axis('off')

plt.suptitle("Reconstructed Images (300 obs)", fontsize=15)
plt.tight_layout()
#plt.savefig("reconstructed_images_300.png")
plt.close()

'''
PCs as pics

'''
pcs = {
    "V1": Vh_V1[0, :],
    "Pixel": Vh_pix[0, :],
    "Gaussian": Vh_gauss[0, :]
}

plt.figure(figsize=(12,4))
for i, (label, pc) in enumerate(pcs.items()):
    plt.subplot(1, 3, i+1)
    plt.imshow((pc.reshape(30,30)), cmap='gray')
    plt.title(f'{label} First PC')
    plt.axis('off')

plt.suptitle("First Principal Component (30 x 30) (300 Obs)", fontsize=15)
plt.tight_layout()
#plt.savefig("pc_first_images_300.png", dpi=300)
plt.close()

pcs_top3 = {
    "V1": Vh_V1[:3, :],      # first 3 PCs
    "Pixel": Vh_pix[:3, :],
    "Gaussian": Vh_gauss[:3, :]
}

plt.figure(figsize=(12, 8))

methods = list(pcs_top3.keys())
num_pcs = 3

for row, method in enumerate(methods):
    for col in range(num_pcs):
        pc = pcs_top3[method][col, :].reshape(30, 30)
        ax = plt.subplot(len(methods), num_pcs, row*num_pcs + col + 1)
        ax.imshow((pc), cmap='gray')
        ax.axis('off')
        ax.set_title(f"PC {col+1}", fontsize=10)  # label for each pc
        if row == 0:
            ax.set_title(f"PC {col+1}", fontsize=10)
        
        # method label
        if col == 0:
            ax.annotate(method, xy=(-0.2, 0.5), xycoords='axes fraction',
                        rotation=90, ha='right', va='center',
                        fontsize=12,)

plt.suptitle("Top 3 Principal Components per Method (30x30) (300 Obs)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig("pc_top3_images_300_labeled.png", dpi=300)
plt.close()


'''
Sparcity of coeffs vectors - histogram of entries in coeffs vectors
'''

coeff_vectors = {
    "True": coeffs_true.flatten(),
    "V1 Estimated": coeffs_est_V1.flatten(),
    "Pixel Estimated": coeffs_est_pix.flatten(),
    "Gaussian Estimated": coeffs_est_gauss.flatten()
}

bins = np.linspace(0, 1.0, 50)  # linear bins

plt.figure(figsize=(16, 4))

for i, (label, coeffs) in enumerate(coeff_vectors.items()):
    # counts, bin_edges = np.histogram(np.abs(coeffs), bins=bins)
    # print(f"{label:20s} total count in bins = {counts.sum()}")
    plt.subplot(1, 4, i + 1)
    plt.hist(np.abs(coeffs), bins=bins, edgecolor='black', color='C'+str(i))
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Number of Coefficients")
    plt.title(label)
    plt.ylim(0, 12)  # changed y-axis range
    plt.grid(alpha=0.3)

plt.suptitle("Coefficient Histograms (300 Obs)", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig("coeff_histograms_fixed_300.png", dpi=300)
plt.close()

print("Number of coefficients <0.1 and <0.5 (300 Obs):")
for label, coeffs in coeff_vectors.items():
    less_than_01 = np.sum(np.abs(coeffs) < 0.1)
    less_than_05 = np.sum(np.abs(coeffs) < 0.5)
    print(f"{label:15s}  <0.1: {less_than_01:4d},  <0.5: {less_than_05:4d}")


plt.figure(figsize=(6, 5))

for label, coeffs in coeff_vectors.items():
    abs_coeffs = np.sort(np.abs(coeffs))
    cdf = np.arange(1, len(abs_coeffs) + 1) / len(abs_coeffs)
    plt.plot(abs_coeffs, cdf, label=label)

plt.xlabel("Absolute Coefficient Value")
plt.ylabel("CDF")
plt.title("CDF of Coefficients (300 Obs)")
plt.legend()
plt.grid(alpha=0.3)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
#plt.savefig("coeff_cdf_zoom_300.png", dpi=300)
plt.close()


def plot_smoothed_error(a_true_V1, a_est_V1, 
                        a_true_pix, a_est_pix, 
                        a_true_gauss, a_est_gauss,
                        title="Error Per Component", filename="smoothed_error.svg"):

    # compute squared errors
    err_v1 = (a_true_V1 - a_est_V1) ** 2
    err_pix = (a_true_pix - a_est_pix) ** 2
    err_gauss = (a_true_gauss - a_est_gauss) ** 2

    # wide dataframe
    wide_df = pd.DataFrame({
        'Index': range(len(err_v1)),
        'V1': err_v1,
        'Pixel': err_pix,
        'Gaussian': err_gauss
    })

    # long format + rolling mean
    df_long = wide_df.melt(id_vars='Index', var_name='Method', value_name='Squared Error')
    df_long['Smoothed Error'] = df_long.groupby('Method')['Squared Error'].transform(
        lambda x: x.rolling(window=150, min_periods=1).mean()
    )

    # plot
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=df_long, x='Index', y='Smoothed Error', hue='Method')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Index")
    plt.ylabel("Squared Error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def compare_errors_100_300(data_100, data_300, filename="combined_errors.svg"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Unpack data tuples
    a_true_V1_100, a_est_V1_100, a_true_pix_100, a_est_pix_100, a_true_gauss_100, a_est_gauss_100 = data_100
    a_true_V1_300, a_est_V1_300, a_true_pix_300, a_est_pix_300, a_true_gauss_300, a_est_gauss_300 = data_300

    # Plot 100 obs
    plt.sca(axes[0])
    plot_smoothed_error(a_true_V1_100, a_est_V1_100,
                        a_true_pix_100, a_est_pix_100,
                        a_true_gauss_100, a_est_gauss_100,
                        title="Error (100 Obs)", filename=None)  # filename=None prevents saving inside function

    # Plot 300 obs
    plt.sca(axes[1])
    plot_smoothed_error(a_true_V1_300, a_est_V1_300,
                        a_true_pix_300, a_est_pix_300,
                        a_true_gauss_300, a_est_gauss_300,
                        title="Error (300 Obs)", filename=None)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
