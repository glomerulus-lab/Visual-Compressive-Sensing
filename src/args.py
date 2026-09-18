# argparse stuff
import pywt
import argparse
import sys

def add_colorbar_args(parser):
    '''
    Add arguments to the parser that are required for 
    colorbar_live_reconst function in figure_library.py

    Parameters
    ----------
    parser : ArgumentParser
        parser object used to hold argument information
    '''

    # theres a lot of these -- use this function instead of manually typing all
    wavelist = pywt.wavelist()
    # get image infile
    parser.add_argument(
        '-observation', choices=['pixel', 'V1', 'gaussian'], action='store',
        help='[Colorbar Figure] : observation type to use when sampling',
        metavar = 'OBS', required=False, nargs=1)
    parser.add_argument(
        '-color', action='store_true',
        help='[Colorbar Figure] : color mode of reconstruction',
        required=False)
    # add hyperparams REQUIRED for dwt ONLY
    parser.add_argument(
        '-dwt_type', choices=wavelist, action='store',
        help='[Colorbar Figure] : dwt type',
        metavar='DWT_TYPE', required=False, nargs=1)
    parser.add_argument(
        '-level', choices=['1', '2', '3', '4'], action='store',
        help='[Colorbar Figure] : level',
        metavar = 'LEVEL', required=False, nargs=1)
    # add hyperparams REQUIRED for V1 ONLY
    parser.add_argument(
        '-cell_size', action='store', 
        help='[Colorbar Figure] : cell size', 
        metavar='CELL_SIZE', required=False, nargs=1)
    parser.add_argument(
        '-sparse_freq', action='store', 
        help='[Colorbar Figure] : sparse frequency',
        metavar='FREQ', required=False, nargs=1)
    # add hyperparams that are used for both dct and dwt
    parser.add_argument(
        '-alpha', action='store', 
        help='[Colorbar Figure] : alpha values to use',
        metavar="ALPHA", required=False, nargs=1)
    parser.add_argument(
        '-num_cells', action='store', 
        help='[Colorbar Figure] : Method you would like' +
        ' to use for reconstruction',
        metavar='NUM_CELLS', required=False, nargs=1)
    parser.add_argument(
        '-filter_dim', action='store', 
        help='[Colorbar Figure] : Dimensions of filter per batch' +
        ' to use for reconstruction',
        metavar='FILTER_DIM', required=False, nargs=1)
    parser.add_argument(
        '-fixed_weights', action ='store_true',
        help='[Colorbar Figure and Hyperparam Sweep] : fix weights from start or randomly generate',
        required=False)
    

def eval_colorbar_args(args, parser):
    '''
    Evaluate the colorbar arguments to 
    ensure that they are sufficient for plotting.
    
    Parameters
    ----------
    args : NameSpace
        Contains each arg and its assigned value.

    parser : ArgumentParser
        Parser object used to give error if needed.

    Returns
    -------
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).

    img_name : String
        The name of image file to reconstruct from.
            
    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']
        
    color : boolean
        Color format for how image should be reconstructed.
        True if reconstructing image in color, False if grayscaled.
    
    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    level : int
        Level of signal frequencies for dwt.
        Should be an integer in [1, 4].
        
    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cells : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training

    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training
    '''
    method = args.method[0] if args.method is not None else None
    img_name = args.img_name[0] if args.img_name is not None else None
    observation = args.observation[0] if args.observation is not None else None
    color = args.color
    fixed_weights = args.fixed_weights
    filter_dim = (eval(args.filter_dim[0]), eval(args.filter_dim[0]))
    print(filter_dim)
    num_cells = eval(args.num_cells[0]) if args.num_cells is not None else None
    if None in [method, img_name, observation, num_cells, filter_dim]:
        parser.error('[Colorbar Figure] : at least method, img_name, filter_dim'+
                     ', observation, num_cells required for colorbar figure')
    # deal with missing or unneccessary command line args
    if method == "dwt" and (args.dwt_type is None or args.level is None):
        parser.error(
            '[Colorbar Figure] : dwt method requires -dwt_type and -level.')
    elif method == "dct" and (args.dwt_type is not None
                              or args.level is not None):
        parser.error(
            '[Colorbar Figure] : dct method does not use -dwt_type and -level.')
    if observation.lower() == "v1" and (args.cell_size is None
                                        or args.sparse_freq is None):
        parser.error(
            '[Colorbar Figure] : V1 observation requires'+
            ' cell size and sparse freq.')
    elif observation.lower() != "v1" and (args.cell_size is not None
                                          or args.sparse_freq is not None):
        parser.error('[Colorbar Figure] : Cell size and sparse freq params'+
                     ' are only required for V1 observation.')
    dwt_type = args.dwt_type[0] if args.dwt_type is not None else None
    level = eval(args.level[0]) if args.level is not None else None
    alpha = eval(args.alpha[0]) if args.alpha is not None else None
    cell_size = eval(args.cell_size[0]) if args.cell_size is not None else None
    sparse_freq = eval(args.sparse_freq[0]) \
        if args.sparse_freq is not None else None
    return method, img_name, observation, color, dwt_type, level, alpha,\
        num_cells, cell_size, sparse_freq, fixed_weights, filter_dim


def add_plot_args(parser):
    '''
    Add arguments to the parser that are required for 
    the error_vs_alpha and error_vs_num_cell functions in figure_library.py

    Parameters
    ----------
    parser : ArgumentParser
        parser object from argparse used to add arguments
    '''
    
    parser.add_argument(
        '-pixel_file', action='store', 
        help='[Alpha and Num Cell Figure] : file to read pixel data from',
        metavar='FILE', required=False, nargs=1)
    parser.add_argument(
        '-gaussian_file', action='store', 
        help='[Alpha and Num Cell Figure] : file to read gaussian data from',
        metavar='FILE', required=False, nargs=1)
    parser.add_argument(
        '-v1_file', action='store', 
        help='[Alpha and Num Cell Figure] : file to read V1 data from',
        metavar='FILE', required=False, nargs=1)
    parser.add_argument(
        '-data_grab', action='store_true',
        help='[Alpha and Num Cell Figure] : auto grab data when '+
        'argument is present',
        required=False)


def eval_plot_args(args, parser):
    '''
    Evaluate the args to ensure that they are sufficient for 
    generating alpha error and num cell error plot.
    
    Parameters
    ----------
    args : NameSpace
        Contains each arg and its assigned value.

    parser : ArgumentParser
        Parser object used to give error if needed.

    Returns
    -------
    img_name : String
        The name of the image that was reconstructed

    method : String
        The basis being used for reconstruction [dct, dwt]

    pixel: String
       csv file containing pixel reconstruction data

    gaussian : String
       csv file containing gaussian reconstruction data

    v1 : String
       csv file containing V1 reconstruction data

    data_grab : boolean
       TODO: convert boolean to 'auto' or 'manual'
    '''
    
    img_name = args.img_name[0] if args.img_name is not None else None
    method = args.method[0] if args.method is not None else None
    pixel = args.pixel_file[0] if args.pixel_file is not None else None
    gaussian = args.gaussian_file[0] if args.gaussian_file is not None else None
    v1 = args.v1_file[0] if args.v1_file is not None else None
    data_grab = args.data_grab
    if None in [method, img_name, pixel, gaussian, v1]:
        parser.error(
            '[Alpha Figure] : at least method, img_name, pixel_file, '+
            'gaussian_file, V1_file required for alpha error figure')
    return img_name, method, pixel, gaussian, v1, data_grab
    
def add_generic_figure_args(parser):
    ''' 
    Add arguments that are used in every plotting function
    
    Parameters
    ----------
    parser : ArgumentParser
        Parser object used to hold argument information
    '''
    # add figtype -- this is the only required argparse arg, determine which others should be there based on figtype
    parser.add_argument(
        '-fig_type', choices=['colorbar', 'num_cell', 'alpha', 'filter_dim'],
        action='store',
        help='[Alpha, Colorbar and Num Cell Figure] : type of figure to generate',
        metavar='FIGTYPE', required=True, nargs=1)
    # add arguments used by both num cell and colorbar
    parser.add_argument(
        '-img_name', action='store', 
        help='[Alpha, Colorbar and Num Cell Figure] : filename of image'+
        ' to be reconstructed',
        metavar='IMG_NAME', required=False, nargs=1)
    parser.add_argument(
        '-method', choices=['dct', 'dwt'], action='store',
        help='[Alpha, Colorbar and Num Cell Figure] : Method to use for reconstruction',
        metavar='METHOD', required=False, nargs=1)
    parser.add_argument(
        '-save', action='store_true',
        help='[Alpha, Colorbar and Num Cell Figure] : save into specified path '+
        'when argument is present',
        required=False)
    

    
def parse_figure_args():
    '''
    Parse the command line args for the figure library
    
    Returns
    -------
    figure type : String
        Desired type of figure to plot 'colorbar' or 'num_cell'

    params : List
        Returns the appropriate parameters to run for a given figure type.
    '''
    parser = argparse.ArgumentParser(
        description='Generate a figure of your choosing.')
    add_generic_figure_args(parser)
    add_colorbar_args(parser)
    add_plot_args(parser) # args for num cell and alpha
    args = parser.parse_args()
    fig_type = args.fig_type[0]
    save = args.save
    if fig_type == 'colorbar':
        params = eval_colorbar_args(args, parser)
    elif fig_type in ['num_cell', 'alpha', 'filter_dim']:
        params = eval_plot_args(args, parser)
    return fig_type, params, save




def parse_sweep_args():
    '''
    Parse the command line args for the hyperparam sweep filter
    
    Returns
    ----------
    method : String
        Method of reconstruction ('dwt' or 'dct').
    
    img : String
        Name of image to reconstruct (e.g. 'tree_part1.jpg').

    observation : String
        Method of observation (e.g. pixel, gaussian, v1).
    
    color : boolean
        Color format for how image should be reconstructed.
        True if reconstructing image in color, False if grayscaled.

    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    lv : List of int
        List of one or more integers in [1, 4].
        
    alpha_list : List of float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cell : List of int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : List of int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : List of int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.

    num_reps : int
        Number of repetitions per hyperparameter combination. Defaults to 10,
        the value the sweep used to hardcode.

    algorithm : String
        Solver used to recover the sparse coefficients.
        One of ['lasso', 'ridge', 'omp', 'bp']. Defaults to 'lasso'.
        'bp' and 'omp' have no alpha penalty, so they are swept without
        -alpha_list and are supported for the dct method only.
    '''

    parser = argparse.ArgumentParser(description='Create a hyperparameter sweep')
    add_sweep_args(parser)
    args = parser.parse_args()
    
    method, img_name, observation, color, dwt_type, level, alpha_list, num_cells, cell_size, sparse_freq, fixed_weights, filter_dim, num_reps, algorithm = eval_sweep_args(args, parser)    
    return method, img_name, observation, color, dwt_type, level, alpha_list, num_cells, cell_size, sparse_freq, fixed_weights, filter_dim, num_reps, algorithm

def add_sweep_args(parser):
    '''
    Add arguments to the parser that are required for the hyperparameter sweep

    Parameters
    ----------
    parser : ArgumentParser
        parser object used to hold argument information
    '''
    # theres a lot of these -- use this function instead of manually typing all
    wavelist = pywt.wavelist()
    
    # get image infile
    parser.add_argument(
        '-img_name', action='store', 
        help='filename of image to be reconstructed',
        metavar='IMG_NAME', required=True, nargs=1)
    # add standard params
    parser.add_argument(
        '-method', choices=['dct', 'dwt'], action='store',
        help='Method you would like to use for reconstruction',
        metavar='METHOD', required=True, nargs=1)
    parser.add_argument(
        '-observation', choices=['pixel', 'V1', 'gaussian'], action='store',
        help='observation type to use when sampling',
        metavar='OBSERVATION', required=True, nargs=1)
    parser.add_argument(
        '-color', action='store_true', 
        help='Color mode of reconstruction',
        required=False)
    # add hyperparams REQUIRED for dwt ONLY
    parser.add_argument(
        '-dwt_type', choices=wavelist, action='store', 
        help='dwt type',
        metavar='DWT_TYPE', required=False, nargs=1)
    parser.add_argument(
        '-level', choices=['1', '2', '3', '4'], action='store', 
        help='level of signal frequency for dwt',
        metavar='LEVEL', required=False, nargs="+")
    # add hyperparams REQUIRED for v1 only
    parser.add_argument(
        '-cell_size', action='store', 
        help='cell size',
        metavar='CELL_SIZE', required=False, nargs="+")
    parser.add_argument(
        '-sparse_freq', action='store', 
        help='sparse frequency',
        metavar='SPARSE_FREQUENCY', required=False, nargs="+")
    # add hyperparams that are used for both dct and dwt
    parser.add_argument(
        '-algorithm', choices=['lasso', 'ridge', 'omp', 'bp'], action='store',
        help='solver used to recover the sparse coefficients (default lasso)',
        metavar='ALGORITHM', required=False, nargs=1)
    # Required for the alpha-penalised solvers only; 'bp'/'omp' have no alpha,
    # so they reject -alpha_list rather than sweeping a parameter they ignore.
    parser.add_argument(
        '-alpha_list', action='store', 
        help='alpha values to use (lasso/ridge only)',
        metavar="ALPHAS", required=False, nargs="+")
    parser.add_argument(
        '-num_reps', action='store',
        help='number of repetitions per hyperparameter combination'
             ' (default 10)',
        metavar='NUM_REPS', required=False, nargs=1)
    parser.add_argument(
        '-num_cells', action='store', 
        help='Method you would like to use for reconstruction',
        metavar='NUM_CELLS', required=True, nargs="+")
    parser.add_argument(
        '-filter_dim', action='store', 
        help='Filter size you would like to use for reconstruction',
        metavar='NUM_CELLS', required=True, nargs="+")
    parser.add_argument(
        '-fixed_weights', action ='store_true',
        help='[Colorbar Figure and Hyperparam Sweep] : fix weights from start or randomly generate',
        required=False)
    
def eval_sweep_args(args, parser):
    '''
    Evaluate the colorbar arguments to 
    ensure that they are sufficient for plotting.
    
    Parameters
    ----------
    args : NameSpace
        Contains each arg and its assigned value.

    parser : ArgumentParser
        Parser object used to give error if needed.

    Returns
    -------
    method : String
        Method of reconstruction ('dwt' or 'dct').
    
    img : String
        Name of image to reconstruct (e.g. 'tree_part1.jpg').

    observation : String
        Method of observation (e.g. pixel, gaussian, v1).
    
    color : boolean
        Color format for how image should be reconstructed.
        True if reconstructing image in color, False if grayscaled.

    dwt_type : String
        Type of dwt method to be used
        See pywt.wavelist() for all possible dwt types.
        
    lv : List of int
        List of one or more integers in [1, 4].
        
    alpha_list : List of float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cell : List of int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : List of int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : List of int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.

    num_reps : int
        Number of repetitions per hyperparameter combination. Defaults to 10,
        the value the sweep used to hardcode.

    algorithm : String
        Solver used to recover the sparse coefficients.
        One of ['lasso', 'ridge', 'omp', 'bp']. Defaults to 'lasso'.
        'bp' and 'omp' have no alpha penalty, so they are swept without
        -alpha_list and are supported for the dct method only.
    '''
    
    #args = parser.parse_args()
    method = args.method[0]
    img_name = args.img_name[0]
    observation = args.observation[0]
    color = args.color
    fixed_weights = args.fixed_weights
    algorithm = args.algorithm[0] if args.algorithm is not None else 'lasso'

    # deal with missing or unneccessary command line args
    if algorithm in ('lasso', 'ridge') and args.alpha_list is None:
        parser.error(f'{algorithm} requires -alpha_list.')
    elif algorithm not in ('lasso', 'ridge') and args.alpha_list is not None:
        parser.error(f'{algorithm} has no alpha penalty, so it does not'
                     ' use -alpha_list.')
    if algorithm != 'lasso' and method == 'dwt':
        parser.error(f'dwt method only supports -algorithm lasso,'
                     f' got {algorithm}.')
    if method == "dwt" and (args.dwt_type is None
                            or args.level is None):
        parser.error('dwt method requires -dwt_type and -level.')
    elif method == "dct" and (args.dwt_type is not None
                              or args.level is not None):
        parser.error('dct method does not use -dwt_type and -level.')
    if observation.lower() == "v1" and (args.cell_size is None
                                        or args.sparse_freq is None):
        parser.error('v1 observation requires cell size and sparse freq.')
    elif observation.lower() != "v1" and (args.cell_size is not None
                                          or args.sparse_freq is not None):
        parser.error('Cell size and sparse freq params are'+
                     ' only required for V1 observation.')
    dwt_type = args.dwt_type
    level = [eval(i) for i in args.level] if args.level is not None else None
    alpha_list = [eval(i) for i in args.alpha_list] \
        if args.alpha_list is not None else None
    
    num_reps = eval(args.num_reps[0]) if args.num_reps is not None else 10
    if not isinstance(num_reps, int) or num_reps < 1:
        parser.error('-num_reps must be a positive integer.')

    num_cells = [eval(i) for i in args.num_cells]
    filter_dim = [(eval(i), eval(i)) for i in args.filter_dim]
    
    cell_size = [eval(i) for i in args.cell_size] \
        if args.cell_size is not None else None
    sparse_freq = [eval(i) for i in args.sparse_freq] \
        if args.sparse_freq is not None else None
    return method, img_name, observation, color, dwt_type, level, alpha_list, \
        num_cells, cell_size, sparse_freq, fixed_weights, filter_dim, \
        num_reps, algorithm