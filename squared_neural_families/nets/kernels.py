import torch
import numpy as np

def cos_kernel(W1, W2, B1, B2):
    dist = torch.cdist(torch.unsqueeze(W1, 0), torch.unsqueeze(W2, 0)) 
    sum_ = torch.cdist(torch.unsqueeze(W1, 0), -torch.unsqueeze(W2, 0)) 
    
    b1 = torch.unsqueeze(B1.T, 2)
    b2 = torch.unsqueeze(B2.T, 2)

    distb = torch.cdist(b1, b2, p=1)
    sumb = torch.cdist(b1, -b2, p=1)

    ret = 0.5*(torch.exp(-0.5*dist**2)*torch.cos(distb) + \
            torch.exp(-0.5*sum_**2)*torch.cos(sumb))
    return ret

def linear_kernel(W1, W2, B1, B2):
    B1 = B1.T # (M, n) where M is batch size and n is num neurons
    B2 = B2.T 
    B1 = B1.reshape((B1.shape[0], B1.shape[1], 1)) # (M, n, 1)
    B2 = B2.reshape((B2.shape[0], 1, B2.shape[1])) # (M, 1, n)

    B12 = B1 @ B2 # Pairwise product of elements of B1 and B2
    W12 = W1 @ W2.T # Same for W
    return W12 + B12

def cos_minus_ident_kernel(W1, W2, B1, B2, a=0.5):
    # Same as ident minus cos
    cos_ker = 1/(4*a**2)*cos_kernel(2*a*W1, 2*a*W2, 2*a*B1, 2*a*B2)
    lin_ker = linear_kernel(W1, W2, B1, B2)

    sq_norm1 = torch.sum(W1**2, dim=1).reshape((-1, 1))
    sq_norm2 = torch.sum(W2**2, dim=1).reshape((-1, 1))

    exp1 = torch.exp(-2*a**2*sq_norm1).reshape((1, -1,1))
    exp2 = torch.exp(-2*a**2*sq_norm2).reshape((1, 1,-1))

    B1 = B1.T # (M, n) where M is batch size and n is num neurons
    B2 = B2.T 
    B1 = B1.reshape((B1.shape[0], B1.shape[1], 1)) # (M, n, 1)
    B2 = B2.reshape((B2.shape[0], 1, B2.shape[1])) # (M, 1, n)
    
    cross1 = (W1 @ W2.T) *(exp1*torch.sin(2*a*B1) + torch.sin(2*a*B2) * exp2)
    cross2 = B1 @ (torch.cos(2*a*B2) * exp2)/(2*a)
    cross3 = (torch.cos(2*a*B1)*exp1) @ B2/(2*a)

    return cos_ker + cross1 - cross2 - cross3 + lin_ker

def snake_kernel(W1, W2, B1, B2, a=0.5):
    cosmident_ker = cos_minus_ident_kernel(W1, W2, B1, B2, a)

    sq_norm1 = torch.sum(W1**2, dim=1).reshape((-1, 1))
    sq_norm2 = torch.sum(W2**2, dim=1).reshape((-1, 1))

    exp1 = torch.exp(-2*a**2*sq_norm1).reshape((1, -1,1))
    exp2 = torch.exp(-2*a**2*sq_norm2).reshape((1, 1,-1))

    B1 = B1.T # (M, n) where M is batch size and n is num neurons
    B2 = B2.T 
    B1 = B1.reshape((B1.shape[0], B1.shape[1], 1)) # (M, n, 1)
    B2 = B2.reshape((B2.shape[0], 1, B2.shape[1])) # (M, 1, n)

    cross1 = B1 - 1/(2*a)*torch.cos(2*a*B1)*exp1
    cross2 = B2 - 1/(2*a)*torch.cos(2*a*B2)*exp2
    return cosmident_ker + 1/(4*a**2) + (cross1+cross2)/(2*a)


def sin_kernel(W1, W2, B1, B2):
    dist = torch.cdist(torch.unsqueeze(W1, 0), torch.unsqueeze(W2, 0)) 
    sum_ = torch.cdist(torch.unsqueeze(W1, 0), -torch.unsqueeze(W2, 0)) 
    
    b1 = torch.unsqueeze(B1.T, 2)
    b2 = torch.unsqueeze(B2.T, 2)

    distb = torch.cdist(b1, b2, p=1)
    sumb = torch.cdist(b1, -b2, p=1)

    ret = 0.5*(torch.exp(-0.5*dist**2)*torch.cos(distb) - \
            torch.exp(-0.5*sum_**2)*torch.cos(sumb))
    return ret


def arc_cosine_kernel(W1, W2, B1, B2):
    prod12 = W1 @ W2.T
    prod11 = torch.sum(W1**2, dim=1).reshape((-1,1))
    prod22 = torch.sum(W2**2, dim=1).reshape((1,-1))

    eps = 1e-7
    cos = torch.clip(((prod12 / torch.sqrt(prod11)) / torch.sqrt(prod22)),
            -1+eps, 1-eps)
    cos = torch.nan_to_num(cos, nan=1, posinf=1, neginf=1)
    sin = torch.clip(torch.clip(torch.sqrt(1 - cos**2), 0, 1), 0, 1)
    theta = torch.arccos(cos)
    ret = torch.sqrt(prod11 @ prod22)\
            /(2*np.pi) * (sin + (np.pi - theta)*cos)
    return ret 

def arc_sine_kernel(W1, W2, B1, B2):
    prod12 = W1 @ W2.T
    prod11 = torch.sum(W1**2, dim=1).reshape((-1,1))
    prod22 = torch.sum(W2**2, dim=1).reshape((-1,1))

    arg = (2*prod12 / torch.sqrt(1 + 2*prod11))/torch.sqrt(1+2*prod22)

    return 2/np.pi * torch.arcsin(arg)
