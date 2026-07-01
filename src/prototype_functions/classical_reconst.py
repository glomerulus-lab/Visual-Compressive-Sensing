import numpy as np

# Package for importing image representation
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from V1_reconst import reconstruct

def reconst(img_arr, alpha, sample_sz):
    ''' TODO Function Description
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of V1 weights
        
    img : array_like
          (n, m) shape image array
    
    Returns
    ----------
    y : vector
        Dot product of W and image
    
    '''
        
    cn, cm = img_arr.shape
    classical_samp = np.floor(cn * cm * 0.5).astype(int)
    rand_index = np.random.randint(0, cn * cm, classical_samp)
    classical_y = img_arr.flatten()[rand_index].reshape(classical_samp, 1)
    
    
    classical_y = classical_y * np.sqrt(cn * cm)
    C = np.eye(cn * cm)[rand_index, :] * np.sqrt(cn * cm)
    C3D = C.reshape(classical_samp, cn, cm)
    theta, classical_reconst, s = reconstruct(C3D, classical_y, alpha)
    error = np.linalg.norm(img_arr - classical_reconst, 'fro') / np.sqrt(cm*cn)
    print("$\\alpha = {alpha}, sample_size = {sample_sz}%$\nError: {err}\n".format(err = error, alpha = alpha, sample_sz = sample_sz * 100))
    
    return theta, classical_reconst, s, error

def main():
    file = input("Enter image File: ")
    image_path = '../images/{image}'.format(image = file)
    
    image_nm = file.split('.')[0]
    
    # Read Image in GrayScale and show grayscale image
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img_arr = np.asarray(img)
    sample_sz = float(input("Enter size of sample for measure(ex: 50% of sample as 0.5): "))
    
    if (input("Do you want to save original grayscale figure? please answer (y/n): ") == 'y'):
        plt.imshow(img_arr, 'gray')
        plt.title("Original {img_nm} image".format(img_nm = image_nm))
        plt.axis("off")
        plt.clim(0, 255)
        plt.savefig("../result/{img_nm}/Classical/alpha_reconst/Original image.png".format(img_nm = image_nm).replace(" ", "_"), dpi = 1000, transparent = True)
        plt.show()
        
    flag = False
    save = False
    
    alpha = input("Give alpha value. \nIf you want lists of alpha values by default, just press enter: ")
    if (alpha == ""):
        alpha = np.logspace(-4, 3, 8)
        flag = True
    else:
        alpha = float(alpha)
    
    if (input("Do you want to save reconst picture(y/n)?: ") == 'y'):
        save = True
    if (flag):
        for alp in alpha:
            theta, classical_reconst, s, error = reconst(img_arr, alp, sample_sz)
            if (save):
                title = "$\\alpha = {alpha}, sample\_size = {sample_sz}%, error = {err}$".format(alpha = alp, err = error, sample_sz = sample_sz * 100)
                plt.figure(figsize=(6,6))
                plt.imshow(classical_reconst, 'gray')
                plt.clim(0, 255)
                plt.title(title)
                plt.axis("off")
                plt.savefig("../result/{img_nm}/Classical/alpha_reconst/alpha_{alpha}_sample_{samp}_error_{err: .3f}.png".
                            format(img_nm = image_nm, alpha = alp, samp = sample_sz, err = error), dpi = 300, transparent = True)
                plt.show()

    else :
        theta, classical_reconst, s, error = reconst(img_arr, alpha, sample_sz)
        if (save):
            title = "$\\alpha = {alpha}, sample\_size = {sample_sz}%, error = {err}$".format(alpha = alpha, sample_sz = sample_sz * 100, err = error)
            plt.imshow(classical_reconst, 'gray')
            plt.clim(0, 255)
            plt.title(title)
            plt.axis("off")
            plt.savefig("../result/{img_nm}/Classical/alpha_reconst/alpha_{alp}_sample_{samp}_error_{err}.png".
                        format(img_nm = image_nm, alp = alpha, samp = sample_sz, err = error), dpi = 300, transparent = True)
            plt.show()

main()
