import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import cv2
from scipy.stats import pearsonr
import math
import matplotlib.pyplot as plt


fontdict = {'fontsize' : 170}


def EME(img, L=8):
    m, n = img.shape
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    m1 = 0
    E = 0
    for _ in range(number_m):
        n1 = 0
        for __ in range(number_n):
            A1 = img[m1:m1+L, n1:n1+L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))
 
            if rbg_min > 0 :
                rbg_ratio = rbg_max/rbg_min
            else :
                rbg_ratio = rbg_max  ###
            E = E + np.log(rbg_ratio + 1e-5)
 
            n1 = n1 + L
        m1 = m1 + L
    E_sum = 20*E/(number_m*number_n)
    return E_sum


def normalize(img):
    return cv2.normalize(np.abs(img).astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def showImg(img,axis="off"):
    plt.imshow(normalize(img),cmap="gray",norm=plt.Normalize(0,1))
    plt.axis(axis)
        
def compareRes(img,gen):
    img=normalize(img)
    gen=normalize(gen)
    return {
        "mse":compare_mse(img,gen),
        "psnr":compare_psnr(img,gen,data_range=1), 
        "ssim":compare_ssim(img,gen,data_range=1),
        "pncc":pearsonr(img.flatten(), gen.flatten()),
        "emes":(EME(img),EME(gen))    
    }  