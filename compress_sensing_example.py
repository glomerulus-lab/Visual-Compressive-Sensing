import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.compress_sensing import *
from src.utility import *
from PIL import Image, ImageOps

#Preparing parameters needed for the examples:

#Hyperparameter Values
small_img = "tree_part1.jpg"
big_img="peppers.png"
method = 'dct'
observation="pixel"
mode = '-c'
alpha=0.1
num_cell_100 = 100
num_cell_300 = 300
cell_size = 7
blob_size = 2

## For wavelet variable
lv= 2
dwt_type= 'db2'

#Load Images:

# Represent image as numpy array to make it easier to process
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

small_img_arr = process_image(small_img, mode)
ax1.imshow(small_img_arr)
ax1.set_title("{img}".format(img = small_img.split('.')[0]))
ax1.axis('off')
print(small_img_arr.shape)
small_img_arr_gray = process_image(small_img, False) #change from 'gray' to False
ax2.imshow(small_img_arr_gray, 'gray')
print(small_img_arr_gray.shape)
ax2.set_title("{img} grayscaled".format(img = small_img.split('.')[0]))
ax2.axis('off')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
big_img_arr = process_image(big_img, mode)
ax1.imshow(big_img_arr)
ax1.set_title("{img}".format(img = big_img.split('.')[0]))
ax1.axis('off')
print(big_img_arr.shape)
big_img_arr_gray = process_image(big_img, False) #change from 'gray' to False
ax2.imshow(big_img_arr_gray, 'gray')
ax2.set_title("{img} grayscaled".format(img = big_img.split('.')[0]))
ax2.axis('off')
print(big_img_arr_gray.shape)
plt.show()


#Reconstruct Images using Compress_Sensing_Library:

#1)Reconstruction Using Pixel Observation with grayscaled reconstruction Vs Color Reconstruction
# Pixel
pixel_W_100, pixel_y_100 = generate_pixel_observation(small_img_arr_gray, num_cell_100) #change generate_pixel_variables to generate_pixel_observation
pixel_W_300, pixel_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("Pixel Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = reconstruct(pixel_W_100, pixel_y_100, alpha)
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = reconstruct(pixel_W_300, pixel_y_300, alpha)
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
fig.suptitle("Pixel Color Reconstruction")
## Reconstruction with 100 number of cells
reconst_color_100 = color_experiment(small_img_arr, num_cell_100, alpha, observation = 'pixel')
ax1.imshow(reconst_color_100)
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")


## Reconstruction with 300 number of cells
reconst_color_300 = color_experiment(small_img_arr, num_cell_300, alpha, observation = 'pixel')
ax2.imshow(reconst_color_300)
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

# Pixel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("Pixel Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = large_img_experiment(big_img_arr_gray, num_cell_100, cell_size, blob_size, alpha = alpha, observation = 'pixel')
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = large_img_experiment(big_img_arr_gray, num_cell_300, cell_size, blob_size, alpha = alpha, observation = 'pixel')
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
fig.suptitle("Pixel Color Reconstruction")
## Reconstruction with 100 number of cells
reconst_color_100 = large_img_experiment(big_img_arr, num_cell_100, cell_size, blob_size, alpha = alpha, color = True, observation = 'pixel')
ax1.imshow(reconst_color_100)
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")


## Reconstruction with 300 number of cells
reconst_color_300 = large_img_experiment(big_img_arr, num_cell_300, cell_size, blob_size, alpha = alpha, color = True, observation = 'pixel')
ax2.imshow(reconst_color_300)
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()


#2)Reconstruction Using Gaussian Observation with grayscaled reconstruction Vs Color Reconstruction:

# Gaussian observation on small image
gaussian_W_100, gaussian_y_100 = generate_gaussian_observation(small_img_arr_gray, num_cell_100)
gaussian_W_300, gaussian_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("gaussian Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = reconstruct(gaussian_W_100, gaussian_y_100, alpha)
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

# ## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = reconstruct(gaussian_W_300, gaussian_y_300, alpha)
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
## Reconstruction with 100 number of cells
reconst_color_100 = color_experiment(small_img_arr, num_cell_100, alpha, observation = 'gaussian')
ax1.imshow(reconst_color_100)
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")


## Reconstruction with 300 number of cells
reconst_color_300 = color_experiment(small_img_arr, num_cell_300, alpha, observation = 'gaussian')
ax2.imshow(reconst_color_300)
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.show()

# Gaussian observation on big image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("gaussian Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = large_img_experiment(big_img_arr_gray, num_cell_100, cell_size, blob_size, alpha = alpha, observation = 'gaussian')
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = large_img_experiment(big_img_arr_gray, num_cell_300, cell_size, blob_size, alpha = alpha, observation = 'gaussian')
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
fig.suptitle("gaussian Color Reconstruction")
## Reconstruction with 100 number of cells
reconst_color_100 = large_img_experiment(big_img_arr, num_cell_100, cell_size, blob_size, alpha = alpha, color = True, observation = 'gaussian')
ax1.imshow(reconst_color_100)
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")


## Reconstruction with 300 number of cells
reconst_color_300 = large_img_experiment(big_img_arr, num_cell_300, cell_size, blob_size, alpha = alpha, color = True, observation = 'gaussian')
ax2.imshow(reconst_color_300)
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

#3)Reconstruction Using V1 Observation with grayscaled reconstruction Vs Color Reconstruction:

# V1 observation on small image
V1_W_100, V1_y_100 = generate_V1_observation(small_img_arr_gray, num_cell_100, cell_size, blob_size)
V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("V1 Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = reconstruct(V1_W_100, V1_y_100, alpha)
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

# ## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = reconstruct(V1_W_300, V1_y_300, alpha)
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
## Reconstruction with 100 number of cells
reconst_color_100 = color_experiment(small_img_arr, num_cell_100, cell_size = cell_size, blob_size = blob_size, alpha = alpha, observation = 'V1')
ax1.imshow(reconst_color_100)
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")


## Reconstruction with 300 number of cells
reconst_color_300 = color_experiment(small_img_arr, num_cell_300, cell_size = cell_size, blob_size = blob_size, alpha = alpha, observation = 'V1')
ax2.imshow(reconst_color_300)
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.show()

# V1 observation on big image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("V1 Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = large_img_experiment(big_img_arr_gray, num_cell_100, cell_size, blob_size, alpha = alpha, observation = 'V1')
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = large_img_experiment(big_img_arr_gray, num_cell_300, cell_size, blob_size, alpha = alpha, observation = 'V1')
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
fig.suptitle("V1 Color Reconstruction")
## Reconstruction with 100 number of cells
reconst_color_100 = large_img_experiment(big_img_arr, num_cell_100, cell_size, blob_size, alpha = alpha,  color = True, observation = 'V1')
ax1.imshow(reconst_color_100)
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")


## Reconstruction with 300 number of cells
reconst_color_300 = large_img_experiment(big_img_arr, num_cell_300, cell_size, blob_size, alpha = alpha, color = True, observation = 'V1')
ax2.imshow(reconst_color_300)
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()