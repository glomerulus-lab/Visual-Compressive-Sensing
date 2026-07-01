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
Try to table the coordinates of the coefficient vectors with the
dot products and frequencies. Do low coefficient frequencies create high
dot products, leading to high MC?

If a frequency's coefficient is low but it creates a high dot product,
then that frequency is not very useful for reconstruction but messes with MC.
Can we discard these frequencies in our MC computation?


Abandoned
'''

def dot_matrix_simple(measurement_matrix):
    design_matrix = generate_design_matrix(measurement_matrix)
    col_norms = np.linalg.norm(design_matrix, axis=0)
    x = design_matrix / col_norms 
    M = x.T @ x
    np.fill_diagonal(M, 0)
    return np.abs(M)



coeffs_true = generate_coeff_vector(small_img_arr_gray, num_cell_300, cell_size, blob_size)

measurement_matrix, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
reconst_gray_300 = reconstruct(measurement_matrix, V1_y_300, alpha)
coeffs_est = generate_coeff_vector(reconst_gray_300, num_cell_300, cell_size, blob_size)

M = dot_vec = dot_matrix_simple(measurement_matrix)
M_vec = np.ravel(M) # dot product matrix as a vector
#M_coords will be the coordinates of each coherence in M, which
#we want to keep track of bc that'll tell us where the largest coherences are
n = M.shape[0]
M_coords = np.unravel_index(range(n**2), (n,n))
arr = np.arange(30)
kx = np.tile(arr, 30)
ky = np.repeat(arr, 30)
perm = np.argsort(M_vec)[::-1] # tells how to sort, reversed to decrease
M_vec[perm] # decreasing
i = M_coords[0][perm]
j = M_coords[1][perm]
# look into kx, ky to map i -> (kx, ky)
i_coords=[[]for s in range(i.shape[0])]
j_coords=[[]for s in range(j.shape[0])]
for m in range(i.shape[0]):
    i_coords[m] = [kx[i[m]],ky[i[m]]] #assigns kx,ky coordinates to elements in i
for m in range(j.shape[0]):
    j_coords[m] = [kx[j[m]],ky[j[m]]] #assigns kx,ky coordinates to elements in j
i_arr = [i,i_coords]
j_arr = [j,j_coords]

'''
How do we sort coeffs so that the entries are related to the (i,j) entries correctly?
'''


data = {
    #"true coeffs": np.abs(coeffs_true).flatten(),
    #"estimated coeffs": np.abs(coeffs_est).flatten(),
    "i": i,
    "j": j,
    "(kx_i,ky_i)": i_coords,
    "(kx_j,ky_j)": j_coords,
    "MC": M_vec[perm]
}

df = pd.DataFrame(data)
print(df)