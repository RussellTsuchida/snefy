import numpy as np
import torch
from torch import linalg as LA
from . import kernels
from normflows.distributions import GaussianMixture

class SquaredNN(torch.nn.Module):
    """
    Defines a squared neural network multiplied by an appropriate measure.

    Args:
        domain (str): 'Rd' for \mathbb{R}^d.
        measure (str): 'gauss' for standard Gaussian measure
        activation (str): 'cos' for cosine activations
        preprocessing (str): 'ident' for no preprocessing
        d (int): dimension of the variable to be modelled
        n (int): Number of parameters in V
        dim (None or int): If not None, only model this index of the input.
            If not None, then d must be 1
        num_mix_components (int): Number of mixture components if using a
            Gaussian base measure.
        m (int) The number of rows in V, i.e. the width of the readout
            layer.
    Methods:
        integrate - Integrate the squared neural network against the measure.
            Optionally takes an extra_input, which could be the output of
            another neural network for conditional density estimation.
        forward - Forward pass through the squared network multiplied
            by the measure.
    """
    def __init__(self, domain, measure, activation, preprocessing, d=2, n=100,
        dim = None, num_mix_components=8, m=1):
        super().__init__()
        self.d = d
        self.n = n
        self.a = 1 #TODO: this is the parameter for snake activaitons.
        self.dim = dim
        self.measure = measure
        self.preprocessing = preprocessing

        self._initialise_params(d, n, m)
        self._initialise_measure(measure,num_mix_components)
        self._initialise_activation(activation)
        self._initialise_kernel(domain, measure, activation, preprocessing)

    def _initialise_activation(self, activation):
        if (activation == 'cos'):
            self.act = torch.cos
        elif (activation == 'cosmident'):
            self.act = lambda x: x - torch.cos(x)
        elif (activation == 'snake'):
            #self.act = lambda x: x + torch.sin(self.a*x)**2/self.a
            self.act = lambda x: x + (1-torch.cos(2*self.a*x))/(2*self.a)
        elif (activation == 'sin'):
            self.act = torch.sin
        elif (activation == 'relu'):
            self.act = torch.nn.ReLU()
            self.B.requires_grad = False
        elif (activation == 'erf'):
            self.act = torch.erf
            self.B.requires_grad = False
        elif (activation == 'exp'):
            self.act = torch.exp
        else:
            raise Exception("Unexpected activation.")

    def _initialise_measure(self, measure, num_mix_components):
        if (measure == 'gauss'):
            #self.pdf = lambda x, log_scale: \
            #        normpdf(x, std=self.s, log_scale=log_scale)
            self.base_measure = GaussianMixture(num_mix_components, self.d).float()
            self.pdf = lambda x, log_scale: \
                self.base_measure.log_prob(x) if log_scale\
                else torch.exp(self.base_measure.log_prob(x))
        elif (measure == 'uniformsphere'):
            # Reciprocal of surface area of sphere
            self.base_measure = None
            self.pdf_ = lambda x, log_scale:\
                -self.d/2 * np.log(2*np.pi) + \
                torch.lgamma(torch.tensor(self.d/2)) if log_scale \
                else torch.exp(torch.lgamma(torch.tensor(self.d/2)))/\
                    (2*np.pi**(self.d/2))
            self.pdf = lambda x, log_scale: self.pdf_(x, log_scale).to(x.device)
        else:
            raise Exception("Unexpected measure.")

    def _initialise_params(self, d, n, m):
        W = np.random.normal(0, 1, (n, d))*2
        V = np.random.normal(0, 1, (m, n))*np.sqrt(1/(n*m))
        B = np.zeros((n, 1))

        self.W = torch.nn.Parameter(torch.from_numpy(W).float())
        self.V = torch.nn.Parameter(torch.from_numpy(V).float())
        self.B = torch.nn.Parameter(torch.from_numpy(B).float())
        self.v0 = torch.nn.Parameter(torch.from_numpy(np.asarray([1.])).float())
        # TODO: STD of Gaussian base measure
        #self.s = torch.nn.Parameter(torch.from_numpy(np.asarray([1.])).float())
        #if not ((self.measure == 'gauss') and (self.preprocessing == 'ident')):
        #    self.s.requires_grad = False

    def _initialise_kernel(self, domain, measure, activation, preprocessing):
        if (domain == 'Rd') and (measure == 'gauss') and (activation == 'cos')\
                and (preprocessing == 'ident'):
            name = 'cos'
        elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'snake')\
                and (preprocessing == 'ident'):
            name = 'snake'
        elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'cosmident')\
                and (preprocessing == 'ident'):
            name = 'cosmident'
        elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'sin')\
                and (preprocessing == 'ident'):
            name = 'sin'
        elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'relu')\
                and (preprocessing == 'ident'):
            name = 'arccos'
            self.B.requires_grad = False
        elif (domain == 'Rd') and (measure == 'gauss') and (activation == 'erf')\
                and (preprocessing == 'ident'):
            name = 'arcsin'
            self.B.requires_grad = False
        elif ((domain == 'Rd') and (measure == 'gauss') and (activation == 'exp')\
            and (preprocessing == 'sphere')) or \
            ((domain == 'sphere') and (measure == 'uniformsphere') and \
            (activation == 'exp') and (preprocessing == 'ident')):
            name = 'vmf'
        else:
            raise Exception("Unexpected integration parameters.")
        
        self.kernel = Kernel(name, self.a)

    def _mathcal_T(self, A, m):
        """
        Return the result \mathcal{T} \Theta
        """
        #(num_mix_components x d x d)
        Amat = torch.diag_embed(torch.squeeze(A, dim=0))\
            .view(A.shape[1],self.d,self.d)
        
        # (num_mix_components x n x 1)
        m_ = torch.swapaxes(m, 0, 2)
        m_ = torch.swapaxes(m_, 0, 1)

        # (num_mix_components, n, d)
        # (num_mix_components, n, 1)
        return (self.W @ Amat, self.B + self.W @ m_)

    def integrate(self, extra_input=0, log_scale=False):
        if self.measure == 'gauss':
            t_param1, t_param2 = \
                self._mathcal_T(torch.exp(self.base_measure.log_scale), 
                self.base_measure.loc)
        else:
            t_param1 = self.W.view((1, self.W.shape[0], self.W.shape[1]))
            t_param2 = self.B.view((1, self.B.shape[0], 1))

        # Iterate over the number of mixture components. 
        # Probably a better way to do this. #TODO
        self.K = 0 
        if self.measure == 'gauss':
            weights = torch.softmax(self.base_measure.weight_scores, 1)
        else:
            weights = np.asarray([[1]])
        for i in range(t_param1.shape[0]):
            self.K = self.K + weights[0,i]*\
                self.kernel(t_param1[i,:,:], t_param2[i,:,:], extra_input)
        #VKV = self.V.T @ self.K @ self.V+ self.v0**2 ## <- m=1 case transpose
        #torch.vmap vectorises the operation. So we can do a batch trace on
        # (B, m, m) for B traces of mxm matrices
        #VKV = torch.vmap(torch.trace)(self.V @ self.K @ self.V.T)
        # Not available until very recent so we do something else instead
        VKV = (self.V @ self.K @ self.V.T).diagonal(offset=0, dim1=-1, 
            dim2=-2).sum(-1).view((-1, 1, 1)) + self.v0**2
        if log_scale:
            ret = torch.log(VKV)
        else:
            ret = VKV

        return ret

    def forward(self, y, extra_input=0, log_scale=False):
        y = self._mask(y)
        # m=1 case transpose
        #squared_net = (self.V.T @ self.act(self.W @ y.T + self.B\
        #    + extra_input))**2+ self.v0**2
        net_out = (self.V @ self.act(self.W @ y.T + self.B\
            + extra_input)).T
        squared_net = torch.norm(net_out, dim=1)**2 + self.v0**2
        squared_net = squared_net.view((1, -1))
        if log_scale:
            logpdf = self.pdf(y, log_scale)
            return torch.log(squared_net) + logpdf

        pdf = self.pdf(y, log_scale)
        return squared_net*pdf

    def _mask(self, y):
        if not (self.dim is None):
            if not (len(y.shape) == 1):
                if not (y.shape[1] == 1):
                    #return y[:,self.dim].reshape((-1,1))
                    return y[:,self.dim].view(-1,1)
        return y

class Kernel(torch.nn.Module):
    """
    These are kernels used by the SquaredNN for closed-form integration.
    
    Args:
        name (str): Name of the kernel. 'cos'
    """
    def __init__(self, name, a):
        super().__init__()
        self.a = a
        self._init_kernel(name)

    def _init_kernel(self, name):
        if name == 'cos':
            self.kernel = lambda W, B, extra_input: \
                kernels.cos_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'snake':
            self.kernel = lambda W, B, extra_input: \
                kernels.snake_kernel(W, W, B+extra_input, B+extra_input,
                        a=self.a)
        elif name == 'cosmident':
            self.kernel = lambda W, B, extra_input: \
                kernels.cos_minus_ident_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'sin':
            self.kernel = lambda W, B, extra_input: \
                kernels.sin_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'arccos':
            self.kernel = lambda W, B, extra_input: \
                kernels.arc_cosine_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'arcsin':
            self.kernel = lambda W, B, extra_input: \
                kernels.arc_sine_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'vmf':
            self.kernel = lambda W, B, extra_input: \
                kernels.vmf_kernel(W, W, B+extra_input, B+extra_input)
        else:
            raise Exception("Unexpected kernel name.")

    def forward(self, W,  B, extra_input=0):
        return self.kernel(W, B, extra_input)


"""
Miscellaneous functions used in the above implementations
"""
def normpdf(val, std=1, log_scale = False):
    if log_scale == True:
        #ret = torch.sum(\
        #    -0.5*torch.log(2*np.pi)-0.5*val**2/std**2\
        #    -torch.log(std),dim=1)
        ret = torch.sum(-0.5*val**2/std**2, dim=1)\
            -0.5*np.log(2*np.pi)*val.shape[1]\
            -torch.log(std)*val.shape[1]
    else:
        ret = torch.prod((1/torch.sqrt(2*np.pi*std**2))*\
            torch.exp(-0.5*val**2/std**2), dim=1)
    return ret



"""
Begin test code
"""
if __name__== '__main__':
    # Test the integration and forward methods
    squared_nn = SquaredNN(domain='Rd', measure='gauss', activation='cos', 
            preprocessing='ident', d=2, n=100)
    print(squared_nn.integrate())
    inp = torch.from_numpy(np.random.normal(0, 1, (1000, 2))).float()
    print(squared_nn(inp))

