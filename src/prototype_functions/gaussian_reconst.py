import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from src.V1_reconst import reconstruct, generate_Y
import sys

def gaussian_W(num_cell, img_dim):
    n, m = img_dim
    W = np.zeros((num_cell, n, m))
    for i in range(num_cell):
        W[i, :, :] = np.random.randn(n, m)
    return W

def main():
    # convert input arguments to right format
    file = sys.argv[1]
    alpha_list = sys.argv[2]
    alpha_list = "[1,2,3,4,5]"
    alpha_list = alpha_list.strip('],[').split(',')
    alpha_list = [int(i) for i in alpha_list]
    
    # Open image
    img = Image.open("../images/{image}".format(image = file))
    img = ImageOps.grayscale(img)
    img_arr = np.asarray(img)
    
    # save image dimension
    dim = img_arr.shape
    n, m = dim.shape
    num_cell = int(input("enter number of cells"))
    
    for i in range(len(alpha_list)):
        W = gaussian_W(num_cell, img_dim)
        y = generate_Y(W, img_arr)
        W_rev = W.reshape(num_cell, n, m)
        theta, reconst, s = reconstruct(W_rev, y, 0.03)
        plt.imshow(reconst)
        plt.clim(0, 255)
        plt.show()
        error = np.linalg.norm(img_arr - reconst, 'fro') / np.sqrt(n*m)
        print("error is {error}".format(error = error))
    
    
# main()
