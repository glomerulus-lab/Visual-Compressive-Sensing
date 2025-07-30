import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

sys.path.append('..')
from src.compress_sensing import *
from src.utility import *
from A_experiments.theta_exp_improved import *

'''
Compare reconstructions for small_img_arr_gray with different parameters

Current Parameters:
    V1 center - fixed or random
'''

# ## Reconstruction with 300 number of cells grayscaled
V1_W_300_fixedCenter, V1_y_300_fixedCenter = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, (15,15))
V1_W_300_randCenter, V1_y_300_randCenter = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("V1 Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_300_fixedCenter = reconstruct(V1_W_300_fixedCenter, V1_y_300_fixedCenter, alpha)
ax1.imshow(reconst_gray_300_fixedCenter, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax1.axis("off")

reconst_gray_300_randCenter = reconstruct(V1_W_300_randCenter, V1_y_300_randCenter, alpha)
ax2.imshow(reconst_gray_300_randCenter, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()