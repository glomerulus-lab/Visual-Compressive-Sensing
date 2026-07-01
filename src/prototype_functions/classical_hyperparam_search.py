import numpy as np
import sys
# sys.path.append("../")
from pathlib import Path

# Package for importing image representation
from PIL import Image, ImageOps

from compress_sensing_library import reconstruct

import pandas as pd
import itertools
import dask
from dask.distributed import Client, progress
import time
import os.path


def search(img_arr, repetition = 20, alpha = np.logspace(-3, 3, 7), classical_samp_list = [50, 100, 200, 500]):
    cn, cm = img_arr.shape
    params = []
    results = []
    
    # List to make combination of
    search_list = {'repetition': repetition,
                   'alpha': alpha,
                   'classical_samp_list': classical_samp_list
                  }



    # Get repetition * alpha combination amount of result
    for row in itertools.product(*search_list.values()):
        product = dict(zip(search_list.keys(), row))
        rep, alp, classical_samp = product.values()
        
        rand_index = np.random.randint(0, cn * cm, classical_samp)
        classical_Y = classical_arr.flatten()[rand_index]
        classical_Y = classical_Y.reshape(classical_samp, 1)

        # Generate C matrix
        C = np.eye(cn * cm)[rand_index, :] * np.sqrt(cn * cm)
        C3D = C.reshape(classical_samp, cn, cm)
        classical_Y = classical_Y * np.sqrt(cn * cm)
        theta, classical_reform, s = reconstruct(C3D, classical_Y, alp)
        
        error = np.linalg.norm(classical_arr - classical_reform, 'fro') / np.sqrt(cm*cn)
        
        params.append({'classical_samp' : classical_samp,
                       'alpha' : alp,
                       'repetition' : rep,
                       'error' : error
                      })
        results.append({'s' : s,
                        'theta' : theta,
                        'reform' : classical_reform,
                       })
        
    return params, results

def main():
    file = sys.argv[1]
    image_path ='../images/{img}'.format(img = file)
    image_nm = image_path.split('/')[2].split('.')[0]
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img_arr = np.asarray(img)
    save_path = os.path.join("../result/{img_nm}/V1".format(img_nm = image_nm))
    params, results = search(img_arr = img_arr)
    print('Process Completed')
    
    save_path = os.path.join('./result/{img}/Classical/'.format(img = image_nm))

    classical_param_df = pd.DataFrame(params)
    classical_param_df.to_csv(os.path.join(save_path, "Classical_Param_" + "_".join(str.split(time.ctime().replace(":", "_"))) + ".csv"))
    print('Save Completed. Terminate program')

if __name__ == "__main__":
    main()
    
    
    
