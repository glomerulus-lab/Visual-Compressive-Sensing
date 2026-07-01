import numpy as np
import numpy.linalg as la
import sys
sys.path.append("../")
import matplotlib.pyplot as plt


# Package for importing image representation
from PIL import Image, ImageOps

from compress_sensing_library import generate_Y, reconstruct
import pandas as pd
import itertools
import dask
from dask.distributed import Client, progress
import time
import os.path

def run_sim(rep, alp, num, sz, freq, img_arr):
    num = int(num)
    img_arr = np.array([img_arr]).squeeze()
    dim = img_arr.shape
    n, m = dim

    # Generate V1 weight with y
    W = V1_weights(num, dim, sz, freq) 
    y = generate_Y(W, img_arr)
    W_model = W.reshape(num, n, m)
    
    # Call function and calculate error
    theta, reform, s = reconstruct(W_model, y, alp)
    error = np.linalg.norm(img_arr - reform, 'fro') / np.sqrt(m*n)
    
    return error, theta, reform, s


def main() :
    # Set up hyperparameters that would affect results
    file = sys.argv[1]
    print(sys.argv[2] == None)
    image_path ='../images/{img}'.format(img = file)
    delay_list = []
    params = []
    alpha = np.logspace(-3, 3, 7)
    rep = np.arange(20)
    num_cell = [50, 100, 200, 500]
    cell_sz = [2, 5, 7]
    sparse_freq = [1, 2, 5]

    image_nm = image_path.split('/')[2].split('.')[0]
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img_arr = np.asarray(img)
    #plt.imshow(img_arr)
    #plt.show()
    save_path = os.path.join("../result/{img_nm}/V1".format(img_nm = image_nm))



    search_list = [rep, alpha, num_cell, cell_sz, sparse_freq]

    # All combinations of hyperparameter to try 
    search = list(itertools.product(*search_list))             
    search_df = pd.DataFrame(search, columns= [ 'rep', 'alp', 'num_cell', 'cell_sz', 'sparse_freq'])
    print(search_df.head())

    # Call dask
    client = Client()

    # counter = 0; # Keep track of number of iteration. Debugging method
    for p in search_df.values:
        delay = dask.delayed(run_sim)(*p, img_arr)
        delay_list.append(delay)

    print('running dask completed')

    futures = dask.persist(*delay_list)
    print('futures completed')
    progress(futures)
    print('progressing futures')

    # Compute the result
    results = dask.compute(*futures)
    print('result computed')
    results_df = pd.DataFrame(results, columns=['error', 'theta', 'reform', 's'])

    # Add error onto parameter
    params_result_df = search_df.join(results_df['error'])

    # save parameter_error data with error_results data
    params_result_df.to_csv(os.path.join(save_path, "param_" + "_".join(str.split(time.ctime().replace(":", "_"))) + ".csv"))
    results_df.to_csv(os.path.join(save_path, "result_" + "_".join(str.split(time.ctime().replace(":", "_"))) + ".csv"))
    print("Execution Complete")
if __name__ == "__main__":
    main()
