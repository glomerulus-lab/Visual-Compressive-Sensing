# Signal Reconstruction Research
Description: Compressed sensing [(CS)](https://github.com/qkstngus000/Compress-Sensing) is a widely accepted theory for signal-based data recovery using trained samples, which enables the reconstruction of a full signal at a low cost. Typically, pixel-based sampling techniques are utilized, but in this study, we propose a new observation technique using a biological model, the Visual Cortex (V1), to acquire more robust and improved signal reconstruction. The V1 is a biological model that has been shown to be effective in processing visual information in the brain. We hypothesize that this biological approach to observation combined with LASSO sparse coding prediction could lead to a better minimization of reconstruction errors compared to the traditional mathematical approach. Thus, we compare the effectiveness of the signal reconstruction between the V1 model and two classical models, namely pixel selection and Gaussian, to determine which method performs best. This study aims to highlight the potential of utilizing biological models in CS and to provide a better understanding of the performance of different observation techniques.

## Table of Contents
1. [Installation](#installation)
2. [Main Functions](#main-functions)
3. [Function Usage](#function-usage)

## Installation
Assuming you have python environment, fork this project into desired directory.

To use the libraries in the src directory as a package, run command below in the root directory of the project.

    pip install -e .

## Organization

### src
This is where all source codes are stored. Source code mainly separated into 5 different files. 
    
- Compress Sensing Library (1): Contains all codes that deals with computations of signal data with its observation method. 
- (hyperparam sweep file) (2): Calls Signal reconstruction method defined in Compress Sensing Library and call dask (parallel computation library) to compute hyperparameters that user wants to test. There are total two hyperparameter sweep files, which one deals with descrete cosine transform ([DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform)) and the other deals with descrete wavelet transform ([DWT](https://en.wikipedia.org/wiki/Discrete_wavelet_transform)). 
- Utility Library (1): Contains all methods that are not directly related to signal computation nor observation method, but used to approach desired goal such as saving function, connecting functions and others.
- Figure Library (1): Once there is data, this figure library would grab the resultant data file and change data to figures for visualization



### result
This is where all hyperparameter sweeped data is getting stored. Once hyperparameter sweep function is run, the source code will find the result directory with its save path and save it into correspondent path.
File saving path is determined by root -> result -> method -> image_name -> observation -> color/black_and_white.csv

### figures
This is where figures generated from result data is getting stored
File saving path is determined by root -> figure -> method -> image_name -> observation -> color/black_and_white.csv

### structured_random_features
This is the base of our project where folder contains neural-network V1 model that allows to generate reconstruction using V1 observation.

## Function Usage
### compress_sensing_library
For examples, please look at this [example](./compress_sensing_example.md) to see how to apply functions listed in compressed_sensing_library.
    
### figure library
This library is still in development phase.
