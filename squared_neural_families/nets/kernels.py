import torch
import numpy as np

def cos_kernel(W1, W2, B1, B2):
    dist = torch.cdist(torch.unsqueeze(W1, 0).contiguous(), 
        torch.unsqueeze(W2, 0).contiguous()) 
    sum_ = torch.cdist(torch.unsqueeze(W1, 0).contiguous(), 
        -torch.unsqueeze(W2, 0).contiguous()) 
    
    b1 = torch.unsqueeze(B1.T, 2)
    b2 = torch.unsqueeze(B2.T, 2)

    distb = torch.cdist(b1.contiguous(), b2.contiguous(), p=1)
    sumb = torch.cdist(b1.contiguous(), -b2.contiguous(), p=1)

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

def _snake_kernel(W1, W2, B1, B2, a=0.5):
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

def snake_kernel(W1, W2, B1, B2, a=0.5):
    """
    Implemented without calling other methods. Uglier code but faster.
    """
    #TODO: This could surely be refactored and still made fast
    cos_ker = 1/(4*a**2)*cos_kernel(2*a*W1, 2*a*W2, 2*a*B1, 2*a*B2)

    sq_norm1 = torch.sum(W1**2, dim=1).view(-1, 1)
    sq_norm2 = torch.sum(W2**2, dim=1).view(-1, 1)

    exp1 = torch.exp(-2*a**2*sq_norm1).view(1, -1,1)
    exp2 = torch.exp(-2*a**2*sq_norm2).view(1, 1,-1)

    B1 = B1.T # (M, n) where M is batch size and n is num neurons
    B2 = B2.T 
    B1 = B1.view(B1.shape[0], B1.shape[1], 1) # (M, n, 1)
    B2 = B2.view(B2.shape[0], 1, B2.shape[1]) # (M, 1, n)
    
    B12 = B1 @ B2 # Pairwise product of elements of B1 and B2
    W12 = W1 @ W2.T # Same for W
    lin_ker = W12 + B12

    cross1 = W12 *(exp1*torch.sin(2*a*B1) + torch.sin(2*a*B2) * exp2)
    cross2 = B1 @ (torch.cos(2*a*B2) * exp2)/(2*a)
    cross3 = (torch.cos(2*a*B1)*exp1) @ B2/(2*a)

    cosmident_ker = cos_ker + cross1 - cross2 - cross3 + lin_ker

    cross1 = B1 - 1/(2*a)*torch.cos(2*a*B1)*exp1
    cross2 = B2 - 1/(2*a)*torch.cos(2*a*B2)*exp2
    return cosmident_ker + 1/(4*a**2) + (cross1+cross2)/(2*a)


def sin_kernel(W1, W2, B1, B2):
    dist = torch.cdist(torch.unsqueeze(W1, 0).contiguous(), 
        torch.unsqueeze(W2, 0).contiguous()) 
    sum_ = torch.cdist(torch.unsqueeze(W1, 0).contiguous(), 
        -torch.unsqueeze(W2, 0).contiguous()) 
    
    b1 = torch.unsqueeze(B1.T, 2)
    b2 = torch.unsqueeze(B2.T, 2)

    distb = torch.cdist(b1.contiguous(), b2.contiguous(), p=1)
    sumb = torch.cdist(b1.contiguous(), -b2.contiguous(), p=1)

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

def arc_cosine_kernel_sphere(W1, W2, B1, B2):
    # Integrate over uniform distribution on sphere instead of Gaussian
    assert W1.shape[1] == 3
    moment = 3
    ret = arc_cosine_kernel(W1, W2, B1, B2)/moment
    return ret

def arc_sine_kernel(W1, W2, B1, B2):
    prod12 = W1 @ W2.T
    prod11 = torch.sum(W1**2, dim=1).reshape((-1,1))
    prod22 = torch.sum(W2**2, dim=1).reshape((-1,1))

    arg = (2*prod12 / torch.sqrt(1 + 2*prod11))/torch.sqrt(1+2*prod22)

    return (2/np.pi * torch.arcsin(arg)).view((-1, W1.shape[0]. W2.shape[0]))

def vmf_kernel(W1, W2, B1, B2):
    assert W1.shape[1] == 3 # Currently only support data on sphere

    sum_ = torch.cdist(torch.unsqueeze(W1, 0).contiguous(), 
        -torch.unsqueeze(W2, 0).contiguous()) 
    
    b1 = torch.unsqueeze(B1.T, 2)
    b2 = torch.unsqueeze(B2.T, 2)

    b_sum = torch.cdist(b1.contiguous(), -b2.contiguous(), p=1)

    exp_fac = torch.squeeze(torch.exp(b_sum), dim=len(b_sum.shape)-1)
    fac = (torch.exp(sum_) - torch.exp(-sum_))/sum_
    fac[fac!=fac] = 2

    return fac*exp_fac/2

def log_linear_kernel(W1, W2, B1, B2, a=0, b=1):
    """
    The kernel corresponding with a log-linear model for the intensity
    /density over a rectangular domain. I.e.

    sigma = exp, t = ident, base measure = Lebesgue, domain = [a,b]**d
    
    B1 is (n, M), where M is batch size and n is number of neurons
    """
    B1 = B1.T # (M, n) where M is batch size and n is num neurons
    B2 = B2.T 
    B1 = B1.view(B1.shape[0], B1.shape[1], 1) # (M, n, 1)
    B2 = B2.view(B2.shape[0], 1, B2.shape[1]) # (M, 1, n)

    sum_b = B1 + B2 # M x n x n
    sum_w = torch.unsqueeze(W1.T, 2) + torch.unsqueeze(W2.T, 1) # d x n x n
    
    a = a*torch.ones((sum_w.shape[0], 1, 1),  device=B1.device)
    b = b*torch.ones((sum_w.shape[0], 1, 1),  device=B1.device)
    
    all_exp = (torch.exp(b*sum_w) - torch.exp(a*sum_w))/sum_w

    ret = torch.exp(sum_b) * torch.unsqueeze(torch.prod(all_exp, dim=0),0)
    return ret

def relu1d_kernel_(W1, W2, B1, B2, a=0, b=1):
    """
    Lebesgue base measure on an interval (a, b), identity sufficient stat,
    ReLU activations. In this 1D example, can use nonzero bias.
    """
    assert W1.shape[1] == 1

    W1 = torch.tile(W1, [1, W2.shape[0]])
    W2 = torch.tile(W2.T, [W1.shape[0], 1])
    B1 = torch.tile(B1, [1, B2.shape[0]])
    B2 = torch.tile(B2.T, [B1.shape[0], 1])

    antideriv = lambda x: W1 * W2/3 * x**3 + (W1*B2 + W2*B1)/2*x**2 + B1*B2*x

    min2 = torch.minimum(b*torch.ones_like(W1, device=W1.device), -B1/W1)
    min3 = torch.minimum(b*torch.ones_like(W2, device=W2.device), -B2/W2)
    min4 = torch.minimum(min3, -B1/W1)

    upper = (W1 > 0) * (W2 > 0) * b + \
            (W1 < 0) * (W2 > 0) * min2 + \
            (W1 > 0) * (W2 < 0) * min3 + \
            (W1 < 0) * (W2 < 0) * min4

    max3 = torch.maximum(a*torch.ones_like(W1, device=W1.device), -B1/W1)
    max2 = torch.maximum(a*torch.ones_like(W2, device=W2.device), -B2/W2)
    max1 = torch.maximum(max2, -B1/W1)

    lower = (W1 > 0) * (W2 > 0) * max1 + \
            (W1 < 0) * (W2 > 0) * max2 + \
            (W1 > 0) * (W2 < 0) * max3 + \
            (W1 < 0) * (W2 < 0) * a

    ret = (antideriv(upper) - antideriv(lower))*(upper > lower)
    return ret

def relu1d_kernel(W1, W2, B1, B2, a):
    s = 0
    for lower_upper in a:
        s = s + relu1d_kernel_(W1, W2, B1, B2, lower_upper[0], lower_upper[1])
    return s

