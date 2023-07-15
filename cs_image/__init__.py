from cs_image.measure_matrix import *
from cs_image.invert_sample import *
from cs_image.svd_sample import *
from cs_image.mmf_speckle import *
from torch import inverse, det, logdet

def predict_unknown_measure_matrix_float64(A_1, y_1, y_2):
    A_1 = A_1.to(torch.float64)
    y_1 = y_1.to(torch.float64)
    y_2 = y_2.to(torch.float64)
    L = A_1 @ A_1.T
    print(det(L), logdet(L), A_1.dtype, y_1.dtype, y_2.dtype, y_1.shape)
    y_1 = y_1.squeeze(0).T
    y_2 = y_2.squeeze(0).T
    ND = y_1.T @ inverse(L) @ y_1
    res = 1 / ND * (y_2 @ y_1.T @ inverse(L) @ A_1)
    return res.to(torch.float32)

def predict_unknown_measure_matrix_float32(A_1, y_1, y_2):
    L = A_1 @ A_1.T
    print(det(L), logdet(L), A_1.dtype, y_1.dtype, y_2.dtype, y_1.shape)
    y_1 = y_1.squeeze(0).T
    y_2 = y_2.squeeze(0).T
    ND = y_1.T @ inverse(L) @ y_1
    res = 1 / ND * (y_2 @ y_1.T @ inverse(L) @ A_1)
    return res

def predict_measure_matrix_from_recv(y_2, A_1, A_recv):
    A_1 = A_1.to(torch.float64)
    y_2 = y_2.to(torch.float64)
    A_recv = A_recv.to(torch.float64)
    y_2 = y_2.squeeze(0).T
    Y = y_2 @ y_2.T
    L = A_1.T @ inverse(A_1 @ A_1.T) @ A_1
    P_1 = inverse(Y) @ A_recv @ inverse(L)
    E = torch.randn(P_1.shape[1], P_1.shape[0], device = P_1.device)
    A_2 = inverse(E @ P_1) @ E
    return A_2.T

def judge_two_measure_error(A_1,A_2,y_1,y_2):
    A_1 = A_1.to(torch.float64)
    A_2 = A_2.to(torch.float64)
    y_1 = y_1.to(torch.float64)
    y_2 = y_2.to(torch.float64)
    y_1 = y_1.squeeze(0).T
    y_2 = y_2.squeeze(0).T
    y_2_recv = A_1 @ A_2.T @ inverse(A_2 @ A_2.T) @ y_2
    print(y_2_recv - y_2)

def decoder_measure_and_matrix(y, measure_matrix):
    M = measure_matrix.shape[0]    
    device = measure_matrix.device
    Q = get_gussian_measure_matrix(0,M,M,device,0.1,True)
    K = get_random_svd_compose_measure_matrix(0,M,M,device,0.1,True)
    V = get_bernoulli_measure_matrix(0,M,M,device,0.1,True)
    C = K
    measure_matrix = C @ measure_matrix
    y = C @ y.transpose(-1,-2)
    return y.transpose(-1,-2), measure_matrix