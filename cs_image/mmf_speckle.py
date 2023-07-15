import scipy.io as scio
import numpy as np
import torch
import cv2

def uniform_picker_column(matrix, N: int):
    # M = matrix.shape[1]
    # picked_cols = np.arange(N)
    # picked_cols = np.random.choice(range(M), N)
    # return matrix[:, picked_cols], picked_cols
    
    L = matrix.shape[0]
    return (( matrix.reshape((L,150,150)) )[:,11:150-11,11:150-11]).reshape((L,-1)), None
    
"""
0 10 25
"""
def get_mmf_speckle_measure_matrix(dis:int, C: int, M: int, N:int, device, sigma=None, channel_common=True):
    path = r'./cs_image/mmf_displacement/{}/A_500_256_1.mat'.format(dis)
    matdata = scio.loadmat(path)
    mat, picker = uniform_picker_column(matdata["A1"], N)
    matrix = torch.tensor(mat, device=device, dtype=torch.float32)
    return matrix, picker


def get_mmf_ori_image(dis:int ,device, W, H, picker=None):
    path = r'./cs_image/mmf_displacement/{}/GI_x0y{}.mat'.format(dis,dis)
    matdata = scio.loadmat(path)
    # img = (matdata["File_image_temp"].flatten())[picker].reshape((W,H))
    img = matdata["File_image_temp"][11:150-11,11:150-11]
    return torch.tensor(img, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
def get_mmf_measure(dis:int, device):
    path = r'./cs_image/mmf_displacement/{}/y_original_500_256_1.mat'.format(dis)
    matdata = scio.loadmat(path)
    y = matdata["Data_after"][11:150-11,11:150-11,:].reshape((-1,2500)).sum(0)
    return torch.tensor(y, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 1 1 M

def get_t_ori_image(name, device, W, H):
    path = r'./cs_image/timg/{}.bmp'.format(name)
    img = cv2.resize(cv2.imread(path, 0), (W, H))
    return torch.tensor(img / 255.0, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def get_pre_measure_img(W,H):
    return 0.5 * torch.ones(W,H)

# just test uniform_picker_column is useful
def get_mmf_gussain_speckle_measure_matrix(C: int, M: int, N:int, device, sigma: float, channel_common=True):
    matrix = sigma * torch.randn(M, 150*150, device=device,dtype=torch.float32)
    return uniform_picker_column(matrix, N)