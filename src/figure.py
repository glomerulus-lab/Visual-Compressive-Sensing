import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import time
import os.path
from src.compress_sensing import *
from src.utility import *
from src.args import *
# Package for importing image representation
from PIL import Image, ImageOps
# set size for each font
ticksize = 20 # x, y ticks
labelsize = 20 # x, y label
titlesize = 30 #title
legend_size = 20 # legend text

def show_reconstruction_error(img_arr, reconst, method,
                              observation, num_cell, img_name): 
    ''' 
    Display the reconstructed image along with pixel error and a colorbar.
    
    Parameters
    ----------
    img_arr : numpy array 
        Contains the pixel values for the original image

    reconst : numpy array 
        Containing the pixel values for the reconstructed image

    method : String
        Method used for the reconstruction.
        Possible methods are ['dct', 'dwt']

    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']

    num_cell : Integer
        Number of blobs that will be used to be determining 
        which pixels to use.

    img_name : String
        Name of the original image file (e.g. "Peppers")
    '''

    # setup figures and axes
    # NOTE: changing figsize here requires you to rescale the colorbar as well
    ## --adjust the shrink parameter to fit.
    fig, axis = plt.subplots(1, 2, figsize = (8, 8))
    plt.tight_layout()
    # prepare the reconstruction axis
    axis[0].set_title(f"{observation} Reconst: {num_cell} samples")
    axis[0].axis('off')

    # prepare the observation error axis
    axis[1].set_title(f"{observation} Error: {num_cell} samples")
    axis[1].axis('off')
    
    # calculate error for RGB images
    if (len(img_arr.shape) == 3):
        axis[0].imshow(reconst, vmin = 0, vmax = 255)
        vmax = ((img_arr - reconst)**2).mean(axis = 2)
        vmax = vmax.max() if vmax.max() < 255 else 255
        err = axis[1].imshow(((img_arr - reconst)**2).mean(axis = 2),
                             'Reds', vmin = 0, vmax = vmax)

    # calculate error for Grayscaled images
    else :
        axis[0].imshow(reconst, cmap='gray', vmin = 0, vmax = 255)
        vmax = img_arr - reconst
        vmax = vmax.max() if vmax.max() < 255 else 255
        err = axis[1].imshow((img_arr - reconst), 'Reds', vmin = 0, vmax = vmax)

    print(error_calculation(img_arr, reconst))
    # apply colorbar -- NOTE : if figsize is not (8, 8) then shrink value must be changeed as well
    cbar = fig.colorbar(err, ax=axis, shrink = 0.363, aspect=10)
    cbar.set_label("Error")



def error_vs_num_cell(img, method, pixel_file=None, gaussian_file=None,
                          V1_file=None, data_grab = 'auto', ax = None,
                          algorithm = 'lasso') :
    ''' 
    Generate figure that compares which method gives the best minimum error
    
    Parameters
    ----------
    img : String
        The name of image file.
       
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.
    
    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.
    
    data_grab : String
        With structured path, decides to grab all three data files 
        automatically or manually. Currently not implemented.
        ['auto', 'manual'].
    '''
    img_nm = img.split('.')[0]
    
    if None in [pixel_file, gaussian_file, V1_file] and data_grab == 'manual': 
        print("All observation data file must be given")    
        sys.exit(0)
    
    #Pre-processing data to receive
    filter_dim = '(32, 32)'
    data = process_result_data_new(img, method, 'num_cell', pixel_file, gaussian_file, V1_file, algorithm)
    data = data[data['filter_dim']==filter_dim]
    if method == 'dwt':
        mean_data = data.groupby(['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size', 'lv']).mean()
    elif method == 'dct':
        mean_data = data.groupby(['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size']).mean()
        
    mean_data = mean_data.reset_index()
    #    print(mean_data)
    # optimize alp value for each num cell and type
    # limit data to those alp values
    
    pixel_data = pd.DataFrame()
    gaussian_data = pd.DataFrame()
    V1_data = pd.DataFrame()


    for num_cell in mean_data['num_cell'].unique():
        # get optimal hyperparams for each num cell
        pixel_opt = mean_data[mean_data['type'] == 'pixel']
        pixel_opt = pixel_opt[pixel_opt['num_cell']==num_cell]
        pixel_opt = pixel_opt[pixel_opt['error'] == pixel_opt['error'].min()]
        if method == 'dwt':
            pixel_data = pd.concat([pixel_data, pd.merge(pixel_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'lv'])])
        elif method == 'dct':
            pixel_data = pd.concat([pixel_data, pd.merge(pixel_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim'])])

        gaussian_opt = mean_data[mean_data['type'] == 'gaussian']
        gaussian_opt = gaussian_opt[gaussian_opt['num_cell']==num_cell]
        gaussian_opt = gaussian_opt[gaussian_opt['error'] == gaussian_opt['error'].min()]
        if method == 'dwt':
            gaussian_data = pd.concat([gaussian_data, pd.merge(gaussian_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'lv'])])
        if method == 'dct':
            gaussian_data = pd.concat([gaussian_data, pd.merge(gaussian_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim'])])
            
        V1_opt = mean_data[mean_data['type'] == 'V1']
        V1_opt = V1_opt[V1_opt['num_cell']==num_cell]
        V1_opt = V1_opt[V1_opt['error'] == V1_opt['error'].min()]
        if method == 'dwt':
            V1_data = pd.concat([V1_data, pd.merge(V1_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size', 'lv'])])
        elif method == 'dct':
            V1_data = pd.concat([V1_data, pd.merge(V1_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size'])])

    #    print("V1: ", V1_data)
    ax = plt.gca() if ax is None else ax
    sns.lineplot(V1_data, x='num_cell', y='error_y', errorbar='pi', label='V1', ax=ax)
    sns.lineplot(pixel_data, x='num_cell', y='error_y', errorbar='pi', label='pixel', ax=ax)
    sns.lineplot(gaussian_data, x='num_cell', y='error_y', errorbar='pi', label='Gaussian', ax=ax)
    #print(V1_data)
    img = img.split('.')[0].capitalize()

    # set size for each font
    #ticksize = 30 # x, y ticks
    #labelsize = 30 # x, y label
    #titlesize = 40 #title
    #legend_size = 30 # legend text
    dim = eval(filter_dim)[0]

    ax.set_ylabel('Error', fontsize=labelsize)
    ax.set_xlabel('$n$', fontsize=labelsize)
    ax.tick_params(axis='both', labelsize=ticksize)

    ax.set_title(f'{img}', fontsize=titlesize)
    ax.legend(fontsize=legend_size)
    fig = ax.get_figure()
    width = 18.5
    height = 10.5
    fig.tight_layout()
    #    fig.set_size_inches(width, height)
        
def error_vs_filter_dim(img, method, pixel_file=None, gaussian_file=None,
                          V1_file=None, data_grab = 'auto', ax = None,
                          algorithm = 'lasso') :
    ''' 
    Generate figure that compares which method gives the best minimum error
    
    Parameters
    ----------
    img : String
        The name of image file.
       
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.
    
    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.
    
    data_grab : String
        With structured path, decides to grab all three data files 
        automatically or manually. Currently not implemented.
        ['auto', 'manual'].
    '''
    img_nm = img.split('.')[0]
    
    if None in [pixel_file, gaussian_file, V1_file] and data_grab == 'manual': 
        print("All observation data file must be given")    
        sys.exit(0)
    
    #Pre-processing data to receive
    data = process_result_data_new(img, method, 'filter_dim', pixel_file, gaussian_file, V1_file, algorithm)
    if method == 'dwt':
        mean_data = data.groupby(['type', 'alp', 'num_cell', 'sparse_freq', 'cell_size', 'filter_dim', 'lv']).mean()
    elif method == 'dct':
        mean_data = data.groupby(['type', 'alp', 'num_cell', 'sparse_freq', 'cell_size', 'filter_dim']).mean()
    mean_data = mean_data.reset_index()
    # optimize alp value for each num cell and type
    # limit data to those alp values
    
    pixel_data = pd.DataFrame()
    gaussian_data = pd.DataFrame()
    V1_data = pd.DataFrame()

    n_prop = 0.3125
    #print(mean_data)
    
    for filter_dim in mean_data['filter_dim'].unique():
        n = n_prop * eval(filter_dim)[0] ** 2
        
        # get optimal hyperparams for each num cell
        pixel_opt = mean_data[mean_data['type'] == 'pixel']
        pixel_opt = pixel_opt[pixel_opt['filter_dim']==filter_dim]
        pixel_opt = pixel_opt[pixel_opt['num_cell']==n]
        pixel_opt = pixel_opt[pixel_opt['error'] == pixel_opt['error'].min()]
        if method == 'dwt':
            pixel_data = pd.concat([pixel_data, pd.merge(pixel_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'lv'])])
        elif method == 'dct':
            pixel_data = pd.concat([pixel_data, pd.merge(pixel_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim'])])
            
        gaussian_opt = mean_data[mean_data['type'] == 'gaussian']
        gaussian_opt = gaussian_opt[gaussian_opt['filter_dim']==filter_dim]
        gaussian_opt = gaussian_opt[gaussian_opt['num_cell']==n]
        gaussian_opt = gaussian_opt[gaussian_opt['error'] == gaussian_opt['error'].min()]
        if method == 'dwt':
            gaussian_data = pd.concat([gaussian_data, pd.merge(gaussian_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'lv'])])
        elif method == 'dct':
            gaussian_data = pd.concat([gaussian_data, pd.merge(gaussian_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim'])])

        V1_opt = mean_data[mean_data['type'] == 'V1']
        V1_opt = V1_opt[V1_opt['filter_dim']==filter_dim]
        V1_opt = V1_opt[V1_opt['num_cell']==n]
        V1_opt = V1_opt[V1_opt['error'] == V1_opt['error'].min()]
        if method == 'dwt':
            V1_data = pd.concat([V1_data, pd.merge(V1_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size', 'lv'])])
        elif method == 'dct':
            V1_data = pd.concat([V1_data, pd.merge(V1_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size'])])

            
    ax = plt.gca() if ax is None else ax
    sns.lineplot(V1_data, x='filter_dim', y='error_y', errorbar='pi', label='V1', ax=ax)
    sns.lineplot(pixel_data, x='filter_dim', y='error_y', errorbar='pi', label='pixel', ax=ax)
    sns.lineplot(gaussian_data, x='filter_dim', y='error_y', errorbar='pi', label='Gaussian', ax=ax)
    
    img = img.split('.')[0].capitalize()

    # set size for each font
    #ticksize = 20 # x, y ticks
    #labelsize = 20 # x, y label
    #titlesize = 20 #title
    #legend_size = 20 # legend text
    
    ax.tick_params(axis='both', labelsize=ticksize)

    ax.set_ylabel('Error', fontsize=labelsize)
    ax.set_xlabel('Patch dimension', fontsize=labelsize)
    ax.set_title(f'{img}', fontsize=titlesize)
    ax.legend(fontsize=legend_size)
    fig = ax.get_figure()
    ax.set_xticks(range(3))
    ax.set_xticklabels(['8x8', '16x16', '32x32'])
    width = 18.5
    height = 10.5
    fig.tight_layout()
    #fig.set_size_inches(width, height)
    
def error_vs_alpha(img, method, pixel_file, gaussian_file, V1_file, save = False,
                   ax = None, algorithm = 'lasso'):
    ''' 
    Generate figure that compares various alpha LASSO penalty and how it affects
    the error of the reconstruction among three different observations. 
    
    Parameters
    ----------
    img : String
        Name of the image that is used by sweeped data
        
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.

    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.

    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.

    save : boolean
        Determines if the image will be saved.
    '''


    fixed_cell = 320
    filter_dim = '(32, 32)'
    img_nm = img.split('.')[0]
    if None in [pixel_file, gaussian_file, V1_file]:
        print("Currently all file required")
        sys.exit(0)
    
    if None in [pixel_file, gaussian_file, V1_file] and data_grab == 'manual': 
        print("All observation data file must be given")    
        sys.exit(0)

    #Pre-processing data to receive
    data = process_result_data_new(img, method, 'alp', pixel_file, gaussian_file, V1_file, algorithm)
    data = data[data['num_cell'] == fixed_cell]
    data = data[data['filter_dim'] == filter_dim]
    if method == 'dwt':
        mean_data = data.groupby(['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size', 'lv']).mean()
    elif method == 'dct':
        mean_data = data.groupby(['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size']).mean()
    mean_data = mean_data.reset_index()
    #print(mean_data)


    pixel_data = pd.DataFrame()
    gaussian_data = pd.DataFrame()
    V1_data = pd.DataFrame()

    for alp in mean_data['alp'].unique():
        # get optimal hyperparams for each num cell
        pixel_opt = mean_data[mean_data['type'] == 'pixel']
        pixel_opt = pixel_opt[pixel_opt['alp'] == alp]
        pixel_opt = pixel_opt[pixel_opt['error'] == pixel_opt['error'].min()]
        if method == 'dwt':
            pixel_data = pd.concat([pixel_data, pd.merge(pixel_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'lv'])])
        elif method == 'dct':
            pixel_data = pd.concat([pixel_data, pd.merge(pixel_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim'])])

        gaussian_opt = mean_data[mean_data['type'] == 'gaussian']
        gaussian_opt = gaussian_opt[gaussian_opt['alp'] == alp]
        gaussian_opt = gaussian_opt[gaussian_opt['error'] == gaussian_opt['error'].min()]
        if method == 'dwt':
            gaussian_data = pd.concat([gaussian_data, pd.merge(gaussian_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'lv'])])
        elif method == 'dct':
            gaussian_data = pd.concat([gaussian_data, pd.merge(gaussian_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim'])])

        V1_opt = mean_data[mean_data['type'] == 'V1']
        V1_opt = V1_opt[V1_opt['alp']==alp]
        V1_opt = V1_opt[V1_opt['error'] == V1_opt['error'].min()]
        if method == 'dwt':
            V1_data = pd.concat([V1_data, pd.merge(V1_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size', 'lv'])])
        elif method == 'dct':
            V1_data = pd.concat([V1_data, pd.merge(V1_opt, data, on=['type', 'alp', 'num_cell', 'filter_dim', 'sparse_freq', 'cell_size'])])
            
    ax = plt.gca() if ax is None else ax
    sns.lineplot(V1_data, x='alp', y='error_y', errorbar='pi', label='V1', ax=ax)
    sns.lineplot(pixel_data, x='alp', y='error_y', errorbar='pi', label='pixel', ax=ax)
    sns.lineplot(gaussian_data, x='alp', y='error_y', errorbar='pi', label='Gaussian', ax=ax)

    # set size for each font
    #ticksize = 20 # x, y ticks
    #labelsize = 20 # x, y label
    #titlesize = 20 #title
    #legend_size = 20 # legend text
    dim = eval(filter_dim)[0]

    ax.tick_params(axis='both', labelsize=ticksize)
    ax.set_xlabel(r'Penalty $\alpha$', fontsize=labelsize)
    ax.set_ylabel('Error', fontsize = labelsize)
    img = img.split('.')[0].capitalize()
    ax.set_title(f'{img}', fontsize=titlesize)

    ax.legend(fontsize=legend_size)

    fig = ax.get_figure()
    width = 18.5
    height = 10.5
    #fig.set_size_inches(width, height)
    
    ax.set_xscale('log')
    fig.tight_layout()
    
def colorbar_live_reconst(method, img_name, observation, color, dwt_type, level,
                          alpha, num_cells, cell_size, sparse_freq, fixed_weights,
                          filter_dim):
    '''
    Generates a reconstruction and error figure for desired parameters.

    Parameters
    ---------
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).

    img_name : String
        The name of image file to reconstruct from.
            
    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']
        
    color : bool
        Indicates if the image working on is color image or black/white image
        Possible colors are [True, False]
    
    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    level : int
        Level of signal frequencies for dwt 
        Better to be an integer in between [1, 4].
        
    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cells : int
        Number of blobs that will be used to be determining 
        which pixels to grab and use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. Affect the data training
    '''
    
    #  print(filter_dim)
    #  print(alpha)
    #  print(num_cells)
    img_arr = process_image(img_name, color, False)
    print(f"Image \"{img_name}\" loaded.") 
    reconst = large_img_experiment(
        img_arr, num_cells, cell_size, sparse_freq, filter_dim, alpha, method,
        observation, level, dwt_type, fixed_weights, color) 
    show_reconstruction_error(img_arr, reconst, method, observation,
                   num_cells, img_name.split('.')[0])

def main():
    fig_type, args, save = parse_figure_args()
    if fig_type == 'colorbar' :
      method, img_name, observation, color, dwt_type, level, alpha, num_cells,\
          cell_size, sparse_freq, fixed_weights, filter_dim = args
      colorbar_live_reconst(
          method, img_name, observation, color, dwt_type, level,
          alpha, num_cells, cell_size, sparse_freq, fixed_weights, filter_dim)
      if save:
          save_reconstruction_error(img_name, method, observation)
    elif fig_type == 'num_cell':
        img_name, method, pixel, gaussian, v1, data_grab = args
        error_vs_num_cell(img_name, method, pixel,
                              gaussian, v1, data_grab)
        if save:
            save_num_cell(img_name, pixel, gaussian, v1, method)
    elif fig_type == 'alpha':
        img_name, method, pixel, gaussian, v1, data_grab = args
        error_vs_alpha(img_name, method, pixel, gaussian, v1, data_grab)
        if save:
            save_alpha(img_name, pixel, gaussian, v1, method)
    elif fig_type == 'filter_dim':
        img_name, method, pixel, gaussian, v1, data_grab = args
        error_vs_filter_dim(img_name, method, pixel, gaussian, v1, data_grab)
        if save:
            save_filter_dim(img_name, pixel, gaussian, v1, method)
    
    if not save:
        plt.show()

if __name__ == "__main__":
    main()
