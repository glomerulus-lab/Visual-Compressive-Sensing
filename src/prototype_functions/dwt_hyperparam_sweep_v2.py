import numpy as np
import numpy.linalg as la
import sys
sys.path.append("../")
import matplotlib.pyplot as plt

# Package for importing image representation
from PIL import Image, ImageOps

from src.compress_sensing_library import *
import pandas as pd
import itertools
import dask
from dask.distributed import Client, progress
import time
import os.path

def usage():
    print(
        '''Usage
        arg[1] = img_name ex)tree_part1.jpg
        arg[2] = dwt_type ex) harr, db1, db2, etc
        arg[3] = lv ex) [1, 2, 3, 4, 5]
        arg[4] = observation_type ex)pixel, V1, gaussian
        arg[5] = alpha_list ex) [0.001, 0.01, 0.1]
        arg[6] = num_cell list ex) [50, 100, 200, 500]
        arg[7] = cell_size list ex) [1, 2, 4, 8, 10]
        arg[8] = sparse_freq ex) [1, 2, 4, 8, 10]
        '''
    )
    sys.exit(0)

def parse_input(i, argv):
    default = {
        1 : 'tree_part1.jpg',
        2 : 'pixel',
        3 : 'db2',
        4 : [1, 2, 3, 4, 5],
        5 : list(np.logspace(-3, 3, 7)),
        6 : [50, 100, 200, 500],
        7 : list(np.arange(1, 11, 1)),
        8 : list(np.arange(1, 11, 1))
        }
    
    ## Checking statement if default is formed correctly
    #print("In parse_input, default {i} is {default}".format(i = i, default = default.get(i)))
    
    if (i != 1 and (argv[i] == 'n' or argv[i] == '-n')):
        return default.get(i)
    
    if (i == 1):
        assert argv[i] != 'n'
        return argv[i]
    elif (i == 2 or i == 3):
        return argv[i]
    elif (i == 5) :
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        print(lst)
        lst = list(map(lambda x: float(x) , lst))
    else:
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        lst = list(map(lambda x: int(x) , lst))
        
    if ((type(lst) == list or type(lst).__module__ == np.__name__) and len(lst) == 1):
        return lst[0]
    else:
        return lst
    
def process_input(argv) :
    param_dict = {}
    
    if (argv[1] == "help" or argv[1] == "-h"):
        usage()
        sys.exit(0)
        
    elif (len(sys.argv) != 9):
        print("All input required. If you want it as basic, put '-n'. Current input number: {input}".format(input = len(sys.argv)))
        usage()    
    
    # Transform String list argument into readable int list
    for i in range(1, len(argv)):
        param_dict[i] = parse_input(i, argv)
    print(param_dict)
    
    
    img, observation, dwt_type, lv, alpha, num_cell, cell_sz, sparse_freq = param_dict.values()
    print(dwt_type)
    
    #Make sure the data type input is in correct format
    assert type(img) == str
    assert type(observation) == str
    assert type(dwt_type) == str
    assert type(lv) == list or type(lv) == int
    assert (type(alpha) == list or type(alpha) == int or type(alpha) == float)
    assert (type(num_cell) == list or type(num_cell) == int)
    if (observation == 'V1'):
        assert (type(cell_sz) == list or type(cell_sz) == int)
        assert (type(sparse_freq) == list or type(sparse_freq) == float)
    
    return img, observation, dwt_type, lv, alpha, num_cell, cell_sz, sparse_freq
    
def run_sim(method, observation, dwt_type, rep, lv, alpha, num_cell, img_arr):
    dim = img_arr.shape
    n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    rep = int(rep)
    lv = int(lv)
    alpha = float(alpha)
    
    img_arr = np.array([img_arr]).squeeze()
    
    if (observation.lower() == 'pixel') :
        W, y = generate_pixel_variables(img_arr, num_cell)
        
    elif (observation.lower() == 'gaussian') :
        W, y = generate_gaussian_variables(img_arr, num_cell)
    
    else :
        print("This obervation technique is currently not supported\n Please use valid observation: ['pixel', 'gaussian', 'V1']")
    
    # Call function and calculate error
    theta, reconst, s = reconstruct(W, y, alpha, method = method, dwt_type = dwt_type, lv = lv)
    error = error_calculation(img_arr, reconst)
    
#     return error, theta, reconst, s
    return error

def run_sim_V1(method, observation, dwt_type, rep, lv, alpha, num_cell, cell_size, sparse_freq, img_arr):
    dim = img_arr.shape
    n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    lv = int(lv)
    rep = int(rep)
    alpha = float(alpha)
    
    img_arr = np.array([img_arr]).squeeze()
    
    W, y = generate_V1_variables(img_arr, num_cell, cell_size, sparse_freq)
    
    # Call function and calculate error
    theta, reconst, s = reconstruct(W, y, alpha, method = method, dwt_type = dwt_type, lv = lv)
    error = error_calculation(img_arr, reconst)
    
#     return error, theta, reconst, s
    return error

def main() :
    '''
    arg[1] = img_name ex)tree_part1.jpg
    arg[2] = method ex) dct, dwt
    arg[3] = dwt_type ex) n (none for dct), harr, db1, db2, etc
    arg[4] = lv ex) n (none for dct), [1, 2, 3, 4, 5]
    arg[5] = observation_type ex)pixel, V1, gaussian
    arg[6] = alpha_list ex) [0.001, 0.01, 0.1]
    arg[7] = num_cell list ex) [50, 100, 200, 500]
    arg[8] = cell_size list ex) [1, 2, 4, 8, 10]
    arg[9] = sparse_freq ex) [1, 2, 4, 8, 10]
    '''
    # Set up hyperparameters that would affect results
    param_dict = {
        #1 : None, 
        #2 : 'dct',
        #3 : 'pixel', 
        #4 : np.logspace(-3, 3, 7),
        #5 : [50, 100, 200, 500], 
        #6 : [2, 5, 7],
        #7 : [1, 2, 5]
    }
    
    img, observation, dwt_type, lv, alpha_list, num_cell, cell_size, sparse_freq = process_input(sys.argv)
    image_path ='../images/{img}'.format(img = img)
    delay_list = []
    params = []
    method = 'dwt'
    rep = np.arange(20)


    image_nm = image_path.split('/')[2].split('.')[0]
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img_arr = np.asarray(img)
    #plt.imshow(img_arr)
    #plt.show()
    save_path = os.path.join("../result/{img_nm}/V1".format(img_nm = image_nm))

    # TODO: if pixel, change search_list only to use rep, alpha, num_cell
    
    # Call dask
    client = Client()
    
    if (observation.upper() != 'V1') :
        search_list = [rep, lv, alpha_list, num_cell]

        # All combinations of hyperparameter to try 
        search = list(itertools.product(*search_list))             
        search_df = pd.DataFrame(search, columns= [ 'rep', 'lv', 'alp', 'num_cell'])
        print(search_df.head())
        
        # counter = 0; # Keep track of number of iteration. Debugging method
        for p in search_df.values:
            delay = dask.delayed(run_sim)(method, observation, dwt_type, *p, img_arr)
            delay_list.append(delay)
            
    elif (observation.upper() == 'V1'):
        search_list = [rep, lv, alpha_list, num_cell, cell_size, sparse_freq]

        # All combinations of hyperparameter to try 
        search = list(itertools.product(*search_list))             
        search_df = pd.DataFrame(search, columns= [ 'rep', 'lv', 'alp', 'num_cell', 'cell_size', 'sparse_freq'])
        print(search_df.head())
        
        for p in search_df.values:
            delay = dask.delayed(run_sim_V1)(method, observation, dwt_type, *p, img_arr)
            delay_list.append(delay)
            
    else :
        print("The observation {observation} is currently not supported. Please try valid observation type.".format(
            observation = observation))

    print('running dask completed')

    futures = dask.persist(*delay_list)
    print('futures completed')
    progress(futures)
    print('progressing futures')

    # Compute the result
    results = dask.compute(*futures)
    print('result computed')
    results_df = pd.DataFrame(results, columns=['error'])
    param_csv_nm = "param_{dwt_type}_".format(dwt_type = dwt_type) + "_".join(str.split(time.ctime().replace(":", "_")))
    param_path = data_save_path(image_nm, method, observation, param_csv_nm)
#     result_path = data_save_path(image_nm, method, observation, "result_" + "_".join(str.split(time.ctime().replace(":", "_"))))
    # Add error onto parameter
    params_result_df = search_df.join(results_df['error'])
    
    # save parameter_error data with error_results data
    params_result_df.to_csv(param_path)
#     results_df.to_csv(result_path)
    print("Execution Completed and file saved")

    hyperparam_track = data_save_path(image_nm, method, observation, 'hyperparam')
    hyperparam_track = hyperparam_track.split('.')[0] + '.txt'
    f = open(hyperparam_track, 'a+')
    hyperparam_list = list(zip(search_df.columns, search_list))
    f.write(f"{param_csv_nm}\n")
    for hyperparam in hyperparam_list :
        f.write(f"   {hyperparam[0]}: {hyperparam[1]}\n")
    f.write("\n\n")
    f.close()
    print("Saved hyperparameter to the txt")

if __name__ == "__main__":
    main()
