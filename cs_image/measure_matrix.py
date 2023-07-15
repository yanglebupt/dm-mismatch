import torch
import numpy as np
from scipy.linalg import hadamard,toeplitz
from utils import normalize

def delta(m,n,u,v,fi=0):
    res=np.zeros((m,n))
    res[u,v]=np.exp(complex('j')*fi)
    return res 


"""
.unsqueeze(0).unsqueeze(0)
"""
def unsqueeze(matrix):
    lens = len(matrix.shape)
    x_matrix = matrix
    for _ in range(4-lens):
        x_matrix = x_matrix.unsqueeze(0)
        
    return x_matrix

"""
随机 svd 组合而成的观测矩阵
B,C 共用同一个观测矩阵 
"""
def get_random_svd_compose_measure_matrix(C: int, M: int, N:int, device, sigma=None, channel_common=True):
    some=False
    Vt = torch.rand(N, N).to(device)  
    Vt, _ = torch.linalg.qr(Vt, 'reduced' if some else 'complete')   # N*N  正交矩阵
    U = torch.rand(M, M).to(device)
    U, _ = torch.linalg.qr(U, 'reduced' if some else 'complete')     # M*M  正交矩阵
    S = torch.hstack(
      (
        torch.eye(M),   # M*M 对角矩阵
        torch.zeros(M, N-M)  # M*(N-M) 全0 
      )
    ).to(device)  # M*N
    H = torch.matmul(U, torch.matmul(S, Vt))  # H = USV
    return H
            
"""
随机高斯观测矩阵
"""
def get_gussian_measure_matrix(C: int, M: int, N:int, device, sigma: float, channel_common=True):
    return sigma * torch.randn(M,N, device=device) if channel_common else sigma * torch.randn(C,M,N, device=device)


"""
随机伯努利观测矩阵
"""
def get_bernoulli_measure_matrix(C: int, M: int, N:int, device, sigma=None, channel_common=True, three=False):
    if three:
        H = torch.randint(-1,5, (M,N), device=device, dtype=torch.float) if channel_common else torch.randint(-1,5, (C,M,N), device=device, dtype=torch.float)
        H[H==2] = 0
        H[H==3] = 0
        H[H==4] = 0
    else:    
        H = torch.randint(0,2, (M,N), device=device, dtype=torch.float) if channel_common else torch.randint(0,2, (C,M,N), device=device, dtype=torch.float)
        H[H==0] = -1
        
    return H


"""
随机稀疏矩阵
"""
def get_sparse_random_measure_matrix(C: int, M: int, N:int, device, sigma=None, channel_common=True, d=0.5):
    H = torch.zeros((M, N),device=device,dtype=torch.float)
    d = int(M*d) if d<1 else d
    for i in range(N):
        H[np.random.choice(np.arange(M), d),i]=1
    return H


"""
托普利兹矩阵和循环矩阵
"""
def get_toeplitz_loop_measure_matrix(C: int, M: int, N:int, device, sigma=None, channel_common=True):
    u = np.random.randint(0,2, (N,))
    u[u==0] = -1
    H=np.empty((M,N))
    for i in range(M):
        u=np.roll(u,1)
        H[i,:]=u
    return torch.tensor(H, device=device, dtype=torch.float)
        


"""
部分哈达玛观测矩阵 N 无法大太
"""
def get_part_hada_measure_matrix(C: int, M: int, N:int, device, sigma=None, channel_common=True):
    hada = hadamard(N)
    random_rows = np.random.choice(np.arange(N), M)
    return torch.tensor(hada[random_rows, :], device=device, dtype=torch.float)
    

"""
上一个方法的替代 部分哈达玛观测矩阵
"""
def get_hada_measure_matrix(C: int, M: int, N:int, device, sigma=None, channel_common=True):
    size=int(np.sqrt(N))
    measure_matrix = np.empty((M,N))
    k=0
    Hada = hadamard(size)
    x,y = np.arange(size),np.arange(size)
    np.random.shuffle(x)
    np.random.shuffle(y)
    for i in x:
        for j in y:
            Delta=delta(size,size,i,j)
            W=np.dot(np.dot(Hada,Delta),Hada)
            measure_matrix[k,:] = W.reshape((N,))
            k+=1
            if k>=M:
                return torch.tensor(measure_matrix, device=device,dtype=torch.float)
            
    return torch.tensor(measure_matrix, device=device,dtype=torch.float)

"""
正弦余弦观测
"""
def get_fft_measure_matrix(C: int, M: int, N:int, device, sigma=None, fi=2*np.pi/3, channel_common=True):
    size=int(np.sqrt(N))
    measure_matrix = np.empty((M,N))
    k=0
    x,y = np.arange(size),np.arange(size)
    np.random.shuffle(x)
    np.random.shuffle(y)
    for i in x:
        for j in y:
            Delta=delta(size,size,i,j,fi)
            W=np.fft.ifft2(Delta).real
            measure_matrix[k,:] = W.reshape((N,))
            k+=1
            if k>=M:
                return torch.tensor(measure_matrix, device=device,dtype=torch.float)
            
    return torch.tensor(measure_matrix, device=device,dtype=torch.float)


def speckle_measure(img: torch.Tensor, measure_matrix: torch.Tensor, nosie_sigma: float, nosie=False, channel_common=True):
    B,C,H,W = img.shape
    N = H*W
    M = measure_matrix.shape[0]
    x = img.reshape((B,C,N,1))   
    """
    [M N] * [B*C N 1] = [B C M 1]  所有 channel 共用一个观测矩阵
    [1 C M N] * [B C N 1] = [B C M 1]
    """
    y = torch.matmul(measure_matrix, x.view(B * C, N, 1)).view(B, C, M) if channel_common else torch.matmul(unsqueeze(measure_matrix), x).squeeze(dim=-1)
    
    if nosie:
        y += (nosie_sigma * torch.randn(B,C,M,device = y.device))
    return y   # B C M

