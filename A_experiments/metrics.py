import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.compress_sensing import *
from src.utility import *
from PIL import Image, ImageOps
import sys
from A_experiments.theta_exp_improved import *

'''
c = coefficient vector
D = dot product

c^tDc and Dc experiments - measurements of confusion
Measuring how much a coefficient vector affects the dot product matrix in
multiple ways.
'''

# ctDc norm = 2, diag = 1
pix_ctDc_n2_d1 = np.zeros(num)
gauss_ctDc_n2_d1 = np.zeros(num)
V1_ctDc_n2_d1 = np.zeros(num)

# ctDc norm = 2, diag = 0
pix_ctDc_n2_d0 = np.zeros(num)
gauss_ctDc_n2_d0 = np.zeros(num)
V1_ctDc_n2_d0 = np.zeros(num)

# Dc norm = 1
pix_Dc_n1 = np.zeros(num)
gauss_Dc_n1 = np.zeros(num)
V1_Dc_n1 = np.zeros(num)

# Dc norm = 2
pix_Dc_n2 = np.zeros(num)
gauss_Dc_n2 = np.zeros(num)
V1_Dc_n2 = np.zeros(num)

for i in range(num):
    # ctDc norm = 2, diag = 1
    pix_ctDc_n2_d1[i] = generate_ctDc(small_img_arr_gray, "pixel", num_cell_300, diagonal = 1)
    gauss_ctDc_n2_d1[i] = generate_ctDc(small_img_arr_gray, "gaussian", num_cell_300, diagonal = 1)
    V1_ctDc_n2_d1[i] = generate_ctDc(small_img_arr_gray, "V1", num_cell_300, diagonal = 1, cell_size = cell_size, blob_size = blob_size, center = (15,15))

    # ctDc norm = 2, diag = 0
    pix_ctDc_n2_d0[i] = generate_ctDc(small_img_arr_gray, "pixel", num_cell_300, diagonal = 0)
    gauss_ctDc_n2_d0[i] = generate_ctDc(small_img_arr_gray, "gaussian", num_cell_300, diagonal = 0)
    V1_ctDc_n2_d0[i] = generate_ctDc(small_img_arr_gray, "V1", num_cell_300, diagonal = 0, cell_size = cell_size, blob_size = blob_size, center = (15,15))

    # Dc norm = 1
    pix_Dc_n1[i] = generate_Dc(small_img_arr_gray, "pixel", num_cell_300)
    gauss_Dc_n1[i] = generate_Dc(small_img_arr_gray, "gaussian", num_cell_300)
    V1_Dc_n1[i] = generate_Dc(small_img_arr_gray, "V1", num_cell_300, 1, cell_size, blob_size, center = (15,15))

    # Dc norm = 2
    pix_Dc_n2[i] = generate_Dc(small_img_arr_gray, "pixel", num_cell_300, 2)
    gauss_Dc_n2[i] = generate_Dc(small_img_arr_gray, "gaussian", num_cell_300, 2)
    V1_Dc_n2[i] = generate_Dc(small_img_arr_gray, "V1", num_cell_300, 2, cell_size, blob_size, center = (15,15))


# ctDc norm = 2, diag = 1
plt.ion()

all_ctDc = [pix_ctDc_n2_d1,gauss_ctDc_n2_d1,V1_ctDc_n2_d1]
fig = plt.figure()
fig.suptitle("ctDc norm=2 diagonal=2, cell=" + str(cell_size) + " blob=" + str(blob_size), fontsize=10)
ax = fig.add_subplot()
ax.boxplot(all_ctDc, tick_labels=['pixel','Gaussian', "V1"])
plt.show()

# ctDc norm = 2, diag = 0
all_ctDc = [pix_ctDc_n2_d0,gauss_ctDc_n2_d0,V1_ctDc_n2_d0]
fig = plt.figure()
fig.suptitle("ctDc norm=2 diagonal=0, cell=" + str(cell_size) + " blob=" + str(blob_size), fontsize=10)
ax = fig.add_subplot()
ax.boxplot(all_ctDc, tick_labels=['pixel','Gaussian', "V1"])
plt.show()

# Dc norm = 1
all_Dc = [pix_Dc_n1,gauss_Dc_n1,V1_Dc_n1]
fig = plt.figure()
fig.suptitle("Dc norm=1, cell=" + str(cell_size) + " blob=" + str(blob_size), fontsize=10)
ax = fig.add_subplot()
ax.boxplot(all_Dc, tick_labels=['pixel','Gaussian', "V1"])
plt.show()

# Dc norm = 2
all_Dc = [pix_Dc_n2,gauss_Dc_n2,V1_Dc_n2]
fig = plt.figure()
fig.suptitle("Dc norm=2, cell=" + str(cell_size) + " blob=" + str(blob_size), fontsize=10)
ax = fig.add_subplot()
ax.boxplot(all_Dc, tick_labels=['pixel','Gaussian', "V1"])
plt.show()
