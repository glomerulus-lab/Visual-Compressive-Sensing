import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

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
NUM_CELL_300 = 300
CELL_SIZE = 7    # receptive field size (200;.001 like gaussian)
BLOB_SIZE = 2    # formerly sparse_freq
ALPHA = 0.1


def main():
    small_img_arr_gray = process_image(SMALL_IMG, color=False)

    # ## Reconstruction with 300 number of cells grayscaled
    V1_W_300_fixedCenter, V1_y_300_fixedCenter = generate_V1_observation(small_img_arr_gray, NUM_CELL_300, CELL_SIZE, BLOB_SIZE, (15,15))
    V1_W_300_randCenter, V1_y_300_randCenter = generate_V1_observation(small_img_arr_gray, NUM_CELL_300, CELL_SIZE, BLOB_SIZE, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

    fig.suptitle("V1 Grascaled Reconstruction")
    ## Reconstruction with 100 number of cells grayscaled
    reconst_gray_300_fixedCenter = reconstruct(V1_W_300_fixedCenter, V1_y_300_fixedCenter, ALPHA)
    ax1.imshow(reconst_gray_300_fixedCenter, 'gray')
    ax1.set_title("{num_cell} number of cells".format(num_cell = NUM_CELL_300))
    ax1.axis("off")

    reconst_gray_300_randCenter = reconstruct(V1_W_300_randCenter, V1_y_300_randCenter, ALPHA)
    ax2.imshow(reconst_gray_300_randCenter, 'gray')
    ax2.set_title("{num_cell} number of cells".format(num_cell = NUM_CELL_300))
    ax2.axis("off")
    #plt.subplots_adjust(top=1.35)
    plt.show()


if __name__ == "__main__":
    main()
