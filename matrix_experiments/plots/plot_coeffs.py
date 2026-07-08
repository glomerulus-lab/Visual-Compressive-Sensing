import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

from src.compress_sensing import *
from src.utility import *
from ..core import *

'''
Plot the coefficient vectors for an image and its V1 reconstruction.
'''

# Whole-image analysis parameters (this script's own setup, not read from core.py)
SMALL_IMG = "tree_part1.jpg"
NUM_CELL_300 = 300
CELL_SIZE = 7    # receptive field size (200;.001 like gaussian)
BLOB_SIZE = 2    # formerly sparse_freq
ALPHA = 0.1


def main():
    small_img_arr_gray = process_image(SMALL_IMG, color=False)

    coeffs_true = generate_coeff_vector(small_img_arr_gray, NUM_CELL_300, CELL_SIZE, BLOB_SIZE)
    bins = 200

    V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, NUM_CELL_300, CELL_SIZE, BLOB_SIZE, None)
    reconst_gray_300_v1 = reconstruct(V1_W_300, V1_y_300, ALPHA)
    coeffs_est_v1 = generate_coeff_vector(reconst_gray_300_v1, NUM_CELL_300, CELL_SIZE, BLOB_SIZE)
    plt.figure()
    plt.plot(coeffs_true.flatten(), "o", label = "True")#, bins, cumulative = False, density = True, label = "True")
    plt.plot(coeffs_est_v1.flatten(), "+", label = "Estimated")#, bins, cumulative = False, density = True, label = "Estimated")
    plt.xlabel("Index")
    plt.ylabel("Value")
    #plt.ylim([1e-3, 1e4])
    plt.legend()
    plt.yscale('log')
    plt.title("V1")
    plt.show()
    #plt.savefig("Coeffs Hist")


    pixel_W_300, pixel_y_300 = generate_pixel_observation(small_img_arr_gray, NUM_CELL_300)
    reconst_gray_300_pix = reconstruct(pixel_W_300, pixel_y_300, ALPHA)
    coeffs_est_pix = generate_coeff_vector(reconst_gray_300_pix, NUM_CELL_300, CELL_SIZE, BLOB_SIZE)
    plt.figure()
    plt.plot(coeffs_true.flatten(), "o", label = "True")#, bins, cumulative = False, density = True, label = "True")
    plt.plot(coeffs_est_pix.flatten(), "+", label = "Estimated")#, bins, cumulative = False, density = True, label = "Estimated")
    plt.xlabel("Index")
    plt.ylabel("Value")
    #plt.ylim([1e-3, 1e4])
    plt.legend()
    plt.yscale('log')
    plt.title("Pixel")
    plt.show()


    gaussian_W_300, gaussian_y_300 = generate_gaussian_observation(small_img_arr_gray, NUM_CELL_300)
    reconst_gray_300_gauss = reconstruct(gaussian_W_300, gaussian_y_300, ALPHA)
    coeffs_est_gauss = generate_coeff_vector(reconst_gray_300_gauss, NUM_CELL_300, CELL_SIZE, BLOB_SIZE)
    plt.figure()
    plt.plot(coeffs_true.flatten(), "o", label = "True")#, bins, cumulative = False, density = True, label = "True")
    plt.plot(coeffs_est_pix.flatten(), "+", label = "Estimated")#, bins, cumulative = False, density = True, label = "Estimated")
    plt.xlabel("Index")
    plt.ylabel("Value")
    #plt.ylim([1e-3, 1e4])
    plt.legend()
    plt.yscale('log')
    plt.title("Gaussian")
    plt.show()


    '''
    plt.figure()
    plt.title("True")
    plt.imshow(coeffs_true)
    plt.colorbar()
    plt.savefig("Coeffs True Heat")

    plt.figure()
    plt.title("Estimated")
    plt.imshow(coeffs_est)
    plt.colorbar()
    plt.savefig("Coeffs Est Heat")
    '''


if __name__ == "__main__":
    main()