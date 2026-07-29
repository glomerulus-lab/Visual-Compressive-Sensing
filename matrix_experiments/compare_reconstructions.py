import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

from matrix_experiments.col_norms import NUM_CELL_300
from src.compress_sensing import *
from src.utility import *
from .core import *

'''
Compare reconstructions for small_img_arr_gray with different parameters

Current Parameters:
    V1 center - fixed or random
'''

# Whole-image analysis parameters (this script's own setup, not read from core.py)
SMALL_IMG = "tree_part1.jpg"
NUM_CELL = 512
CELL_SIZE = 7    # receptive field size (200;.001 like gaussian)
BLOB_SIZE = 2    # formerly sparse_freq
ALPHA = 1
ALG = 'bp'


def main():
    small_img_arr_gray = process_image(SMALL_IMG, color=False)

    # ## Reconstruction with 300 number of cells grayscaled
    # obs1_W, obs1_y = generate_V1_observation(small_img_arr_gray, NUM_CELL_300, CELL_SIZE, BLOB_SIZE, (15,15))
    obs1_W, obs1_y = generate_gaussian_observation(small_img_arr_gray, NUM_CELL)
    obs2_W, obs2_y = generate_V1_observation(small_img_arr_gray, NUM_CELL, CELL_SIZE, BLOB_SIZE, None)
    reconst_1 = reconstruct(obs1_W, obs1_y, ALPHA, algorithm=ALG)
    reconst_2 = reconstruct(obs2_W, obs2_y, ALPHA, algorithm=ALG)

    err1 = np.linalg.norm(small_img_arr_gray - reconst_1) ** 2
    err2 = np.linalg.norm(small_img_arr_gray - reconst_2) ** 2
    print(f"Reconstruction error for obs1: {err1:.4f}")
    print(f"Reconstruction error for obs2: {err2:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    fig.suptitle("Reconstruction Comparison")
    ax1.imshow(reconst_1, 'gray')
    ax1.set_title("obs 1")
    ax1.axis("off")
    ax2.imshow(reconst_2, 'gray')
    ax2.set_title("obs 2")
    ax2.axis("off")
    #plt.subplots_adjust(top=1.35)
    plt.show()


if __name__ == "__main__":
    main()
