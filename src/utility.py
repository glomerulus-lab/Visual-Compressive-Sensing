import pandas as pd
import seaborn as sns
import time
import os.path
import sys
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
from src.compress_sensing import *

# Package for importing image representation
from PIL import Image, ImageOps

def search_root():
    ''' 
    Search for the root directory of the project
            
    Returns
    ----------
    root : path
        Return absolute paths for the project
    
    '''
    back_path = './'
    root = Path(os.path.abspath('./'))
    while not root.match('*Visual-Compressive-Sensing'):
        back_path += '../'
        root = Path(os.path.abspath(back_path))
    return root

def fig_save_path(img_nm, method, observation, save_nm):
    ''' 
    Gives absolute paths for generated figures to be saved to have organized 
    folder structure. Figure will be saved under figure directory and its format
    is set up to be png file. If folder path does not exist with the parameter 
    given, then create a new path.
    
    Parameters
    ----------
    img_nm : String
        Name of the image file data used 
        
    method : String
        Method used for the reconstruction.
        Possible methods are ['dct', 'dwt']
        
    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']
        
    save_nm : String
        Name of the file that it will be saved to
        
    Returns
    ----------
    path : path
        Return absolute path including its file name
    '''
    # Search for the root path
    root = search_root()
    
    method = method.lower()
    observation = observation.upper() if \
        observation.split('/')[0].upper() == 'V1' else observation.lower()
    img_nm = img_nm.split('.')[0]
    save_nm = save_nm.replace(" ", "_")
    
    if (observation != 'num_cell_error') :
        if (save_nm[-1] != "_") :
            save_nm = save_nm + "_" 
        save_nm = save_nm + "_".join(str.split(time.ctime().replace(":", "_")))
    
        
    fig_path = os.path.join(root, f"figures/{method}/{img_nm}/{observation}")
    Path(fig_path).mkdir(parents=True, exist_ok = True)
    #TODO: add timestamp onto save_nm autometically
    return os.path.join(fig_path, "{save_nm}.svg".format(save_nm = save_nm))

def data_save_path(img_nm, method, observation, save_nm): 
    ''' 
    Gives absolute paths for collected data to be saved to have organized folder
    structure. File will be saved under result directory and its format is set 
    up to be csv file. If folder path does not exist with the parameter given,
    then create a new path
    
    Parameters
    ----------
    img_nm : String
        Name of the image file data used 
        
    method : String
        Method used for the reconstruction.
        Possible methods are ['dct', 'dwt']
        
    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']
        
    save_nm : String
        Name of the file that it will be saved to
        
    Returns
    ----------
    path : path
        Return absolute path including its file name
    '''
    # Search for the root path
    root = search_root()
    
    method = method.lower()
    observation = observation.upper() \
        if observation.split('/')[0].upper() == 'V1'\
        else observation.lower()
    img_nm = img_nm.split('.')[0]
    save_nm = save_nm.replace(" ", "_")
    
    match = re.findall("_hyperparam$", save_nm)
    if (match) : 
        save_nm = save_nm + '.txt'
    else :
        if (save_nm[-1] != "_") :
            save_nm = save_nm + "_" 
        save_nm = save_nm + "_".join(
            str.split(time.ctime().replace(":", "_"))) + '.csv'  
    
    result_path = os.path.join(root, f"result/{method}/{img_nm}/{observation}")
    Path(result_path).mkdir(parents=True, exist_ok = True)
    
    return os.path.join(result_path, save_nm)

def compute_zero_padding_dimension(n, filter_n):
    '''
    Computes zero padding that is needed for the large_img_experiment.
    Bot side of n and filter_n should be matching to get exact size of new
    (n, m) dimension. For example, If user pass n from (n, m) dimension,
    user should also pass filter_n from (filter_n, filter_m) and vice versa
    
    Parameters
    ----------
    n : int
        One side (width/height) of the original image dimension
    
    filter_n : int
        One side (width/height) of the filter dimention
    
    Returns
    ----------
    new_n : int
        Size of the side when the zero padding is applied
    
    '''
    new_n = n
    if n % filter_n != 0 :
        new_n = n + (filter_n - (n % filter_n))
    
    return new_n


def process_image(img, color = False, visibility = False):
    ''' 
    Opens image file with given file name and determines whether image file will
    be color or grayscale image. Show the image if the visibility is set to be 
    True.
    
    Parameters
    ----------
    img : String
        Name of the image file data to be opened.
        
    color : bool
        Indicates if the image working on is color image or black/white image
        Possible colors are [True, False]
        
    visibility : bool
        Determines whether image will be shown to the user or not.
        
    Returns
    ----------
    img_arr : array_like
        Return numpy array of image
    '''

    root = search_root()

    img_path = Image.open(os.path.join(root, 'image/{img}'.format(img=img)))
    if not color:
        img_path = ImageOps.grayscale(img_path)
    if visibility:
        if not color:
            print("Processing grayscale image")
            plt.imshow(img_path, 'gray')
            plt.title("grayscaled_{img}".format(img = img.split(".")[0]))
        else:
            print("Processing color image")
            plt.imshow(img_path)
            plt.title(img.split(".")[0])
        plt.axis('off')
        plt.show()
    img_arr = np.asarray(img_path)
    
    return img_arr

def remove_unnamed_data(data):
    ''' 
    Remove unnecessary data column that stores index.
    
    Parameters
    ----------
    data : pandas dataframe
        Dataframe Storing reconstruction hyperparameters and errors.
            
    Returns
    ----------
    data : pandas dataframe
        Stores reconstruction hyperparameters and errors 
        without unnecessary column.
    '''
    for index in data:
        print(index)
        if (index == 'Unnamed: 0') :
            data.drop('Unnamed: 0', axis = 1, inplace=True)
    return data

def load_dataframe(img_nm, method, pixel_file=None,
                        gaussian_file=None, V1_file=None) :
    root = search_root()
    load_V1 = f"{root}/result/{method}/{img_nm}/V1/{V1_file}"
    load_gaussian = f"{root}/result/{method}/{img_nm}/gaussian/{gaussian_file}"
    load_pixel = f"{root}/result/{method}/{img_nm}/pixel/{pixel_file}"
    
    pixel_df = remove_unnamed_data(pd.read_csv(load_pixel))
    gaussian_df = remove_unnamed_data(pd.read_csv(load_gaussian))
    V1_df = remove_unnamed_data(pd.read_csv(load_V1))

    #    print(len(V1_df))
    return V1_df, gaussian_df, pixel_df

def load_dataframe_new(img_nm, method, pixel_file=None,
                        gaussian_file=None, V1_file=None) :
    root = search_root()
    load_V1 = f"{root}/result/{method}/{img_nm}/V1/{V1_file}"
    load_gaussian = f"{root}/result/{method}/{img_nm}/gaussian/{gaussian_file}"
    load_pixel = f"{root}/result/{method}/{img_nm}/pixel/{pixel_file}"
    
    pixel_df = pd.read_csv(load_pixel, index_col=0)
    gaussian_df = pd.read_csv(load_gaussian, index_col=0)
    V1_df = pd.read_csv(load_V1, index_col=0)

    V1 = ['V1'] * len(V1_df)
    gaussian = ['gaussian'] * len(gaussian_df)
    pixel = ['pixel'] * len(pixel_df)
    #    print(len(V1_df))
    pixel_df['type'] = pixel
    V1_df['type'] = V1
    gaussian_df['type'] = gaussian
    
    df = pd.concat([pixel_df, V1_df, gaussian_df])
    return df

def process_result_data(img_file, method, target_param, pixel_file=None,
                        gaussian_file=None, V1_file=None):
    ''' 
    Open 3 csv data files, make it as pandas dataframe, remove unnecessary 
    column, find the plotting data with minimum mean error 
    for each of target_param.
    
    Parameters
    ----------
    img_file : String
        The name of image file that will be worked on
        
    method : String
        Basis the data file was worked on. 
        Currently upports dct and dwt (discrete cosine/wavelet transform).
        
    target_param : String 
        Param for which to get min error
        [num_cell, alpha] for num_cell and alpha error plots respectively.
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.

    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.
        
    Returns
    ----------
    obs_dict : python dictionary
        Dictionary that contains ['V1', 'gaussian', 'pixel'] as a key
        [0]th value storing plotting data with [1]st data containing 
        minimum mean error parameter for each num_cell.
    '''
    img_nm = img_file.split('.')[0]
    
    # TODO: Currently all three files required,
    # thinking if plot can be generated with just one or two files as well
    if (V1_file==None or gaussian_file==None or pixel_file==None):
        print("All three files required to generate figure")
        sys.exit(0)
        
    
    V1_df, gaussian_df, pixel_df = load_dataframe(img_nm, method, pixel_file,
                        gaussian_file, V1_file)
    
    obs_dict= {'V1': V1_df,
               'gaussian': gaussian_df, 
               'pixel': pixel_df, 
              }
    # Remove unnecessary index column in pandas
    for obs, file in obs_dict.items():  
        obs_dict.update({obs: remove_unnamed_data(file)})
        
    for obs, file in obs_dict.items():
        obs_dict.update({obs: get_min_error_data(method, obs, file, target_param)})
    
    return obs_dict

def process_result_data_new(img_file, method, target_param, pixel_file=None,
                        gaussian_file=None, V1_file=None):
    ''' 
    Open 3 csv data files, make it as pandas dataframe, remove unnecessary 
    column, find the plotting data with minimum mean error 
    for each of target_param.
    
    Parameters
    ----------
    img_file : String
        The name of image file that will be worked on
        
    method : String
        Basis the data file was worked on. 
        Currently upports dct and dwt (discrete cosine/wavelet transform).
        
    target_param : String 
        Param for which to get min error
        [num_cell, alpha] for num_cell and alpha error plots respectively.
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.

    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.
        
    Returns
    ----------
    obs_dict : python dictionary
        Dictionary that contains ['V1', 'gaussian', 'pixel'] as a key
        [0]th value storing plotting data with [1]st data containing 
        minimum mean error parameter for each num_cell.
    '''
    img_nm = img_file.split('.')[0]
    
    # TODO: Currently all three files required,
    # thinking if plot can be generated with just one or two files as well
    if (V1_file==None or gaussian_file==None or pixel_file==None):
        print("All three files required to generate figure")
        sys.exit(0)
        
    data = load_dataframe_new(img_nm, method, pixel_file, gaussian_file, V1_file)
    data['blob_size'] = data['blob_size'].fillna(0)
    data['cell_size'] = data['cell_size'].fillna(0)
    return data


def save_reconstruction_error(img_name, method, observation):
    '''
    Saves the reconstruction error figure to a filepath built from params 

    Parameters
    ----------
    img_name : String
        Name of image file to reconstruct

    method : String
        Basis the data file was worked on. 
        Currently supports dct and dwt (discrete cosine/wavelet transform).

    observation : String
        Observation technique to be used for sampling image data.
    '''
    outfile = fig_save_path(img_name, method, observation, "colorbar")
    plt.savefig(outfile, dpi = 300, bbox_inches = "tight")
    print(f'saving reconstruction error figure to {outfile}')

def save_num_cell(img_name, pixel_file, gaussian_file, V1_file, method):
    '''
    Saves the num cell vs error figure to a filepath built from params 

    Parameters
    ----------
    img_name : String
        Name of image file to reconstruct

    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
    
    V1_file : String
        V1 observation data file from hyperparameter sweep.

    method : String
        Basis the data file was worked on. 
        Currently supports dct and dwt (discrete cosine/wavelet transform).
    '''
    # for its save name, the name of file order is pixel -> gaussian -> V1 
    save_name = pixel_file.split('.')[0] + '_' + \
        gaussian_file.split('.')[0] + '_' + V1_file.split('.')[0]
    save_path = fig_save_path(img_name, method, 'num_cell_error', save_name)
    plt.savefig(save_path, dpi = 200)
    print(f'saving error vs num_cell figure to {save_path}')

def save_alpha(img_name, pixel_file, gaussian_file, V1_file, method):
    '''
    Saves the alpha vs error figure to a filepath built from params 

    Parameters
    ----------
    img_name : String
        Name of image file to reconstruct

    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
    
    V1_file : String
        V1 observation data file from hyperparameter sweep.

    method : String
        Basis the data file was worked on. 
        Currently supports dct and dwt (discrete cosine/wavelet transform).
    '''
    # for its save name, the name of file order is pixel -> gaussian -> V1 
    save_name = pixel_file.split('.')[0] + '_' + \
        gaussian_file.split('.')[0] + '_' + V1_file.split('.')[0]
    save_path = fig_save_path(img_name, method, 'alpha_error', save_name)
    plt.savefig(save_path, dpi = 200)
    print(f'saving error vs alpha figure to {save_path}')

def save_filter_dim(img_name, pixel_file, gaussian_file, V1_file, method):
    '''
    Saves the alpha vs error figure to a filepath built from params 

    Parameters
    ----------
    img_name : String
        Name of image file to reconstruct

    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
    
    V1_file : String
        V1 observation data file from hyperparameter sweep.

    method : String
        Basis the data file was worked on. 
        Currently supports dct and dwt (discrete cosine/wavelet transform).
    '''
    # for its save name, the name of file order is pixel -> gaussian -> V1 
    save_name = pixel_file.split('.')[0] + '_' + \
        gaussian_file.split('.')[0] + '_' + V1_file.split('.')[0]
    save_path = fig_save_path(img_name, method, 'filter_dim', save_name)
    plt.savefig(save_path, dpi = 200)
    print(f'saving error vs filter figure to {save_path}')

    
def get_min_error_data(method, observation, data_df, target_param):
    ''' 
    Retrieve plotting data and minimum error parameter to be returned.
    
    Parameters
    ----------
    method : String
        Basis the data file was worked on. 
        Currently supports dct and dwt (discrete cosine/wavelet transform).
        
    observation : String
        Observation technique to be used for sampling image data.
    
    data_df : pandas dataframe
        Stores reconstruction hyperparameters and errors.
            
    target_param : String 
        Param for which to get min error
        [num_cell, alpha] for num_cell and alpha error plots respectively.
    
    Returns
    ----------
    data_plotting_data : pandas dataframe
        Dataframe storing all hyperparameters that have the minimum error.
    
    data_min_df : pandas dataframe
        Dataframe storing hyperparameters that have minimum mean errors 
        for each num_cell, but without unnecessary column.
    '''
    param_list = []
    
    # V1 observation takes two more parameter
    if (observation.upper() == 'V1'):
        # Check the method first to check what parameter it has to deal with
        # since dwt takes two more parameters
        if (method.lower() == 'dct'):
            param_list = ['num_cell', 'blob_size', 'cell_size', 'alp']
        elif (method.lower() == 'dwt') :
            param_list = ['num_cell', 'blob_size', 'cell_size', 'alp', 'lv']
            
    else :
        if (method == 'dct'):
            param_list = ['num_cell', 'alp']
        elif (method == 'dwt') :
            param_list = ['num_cell', 'alp', 'lv']
    
    # For each hyperparameter, gives mean of its own repetition
    data_mean_df = data_df.groupby(param_list,
                                   as_index=False).mean().drop('rep', axis=1)
    
    # Grab the lowest mean error from each number of cell
    data_min_df = data_mean_df.sort_values('error').drop_duplicates(target_param)
    data_min_df = data_min_df.rename(columns={'error': 'min_error'})
    
    # Mark the hyperparameter that gives lowest mean error to whole dataset
    data_merged_df = pd.merge(data_df, data_min_df, on=param_list, how='left')
    
    # Grab hyperparameters that was marked
    data_plotting_data = data_merged_df.loc[
        data_merged_df['min_error'].notnull()
    ]

    return (data_plotting_data, data_min_df)

if __name__ == "__main__":
    print(get_min_error_data("a", "b", "c"))