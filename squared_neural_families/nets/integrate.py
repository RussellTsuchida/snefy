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
        m (int): The number of rows in V, i.e. the width of the readout
            layer. If m is -1, parameterise by PD matrix V.T V
        temporal (None or list of list of floats): 
            In addition to the d dimensions for the variable
            to be modelled, include an extra dimension for time.
            Always use ReLU activations, Lebesgue base measure on
            interval for this extra dimension. If None, don't include temporal
            dimension. If a list of list of two floats, use a temporal dimension
            and compute the integral over the union of the integrals.
    Methods:
        integrate - Integrate the squared neural network against the measure.
            Optionally takes an extra_input, which could be the output of
            another neural network for conditional density estimation.
        forward - Forward pass through the squared network multiplied
            by the measure.
    """
    def __init__(self, domain, measure, activation, preprocessing, d=2, n=100,
        dim = None, num_mix_components=8, m=1, diagonal_V = False, 
        temporal = None):
        super().__init__()
        self.finetuning = False
        self.d = d
        self.n = n
        self.a = 1 #TODO: this is the parameter for snake activaitons.
        self.bound0 = None
        self.bound1 = None
        self.dim = dim
        self.measure = measure
        self.preprocessing = preprocessing
        self.num_mix_components = num_mix_components

        self._initialise_params(d, n, m, diagonal_V, temporal)
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
            self.B.requires_grad = False # B may be folded into V for exp fam
        else:
            raise Exception("Unexpected activation.")

    def _initialise_measure(self, measure, num_mix_components):
        if (measure == 'leb'):
            self.base_measure = None
            self.pdf = lambda x, log_scale: 0 if log_scale else 1
        elif (measure == 'gauss'):
            self.base_measure = GaussianMixture(num_mix_components, self.d).float()
            self.pdf = lambda x, log_scale: \
                self.base_measure.log_prob(x) if log_scale\
                else torch.exp(self.base_measure.log_prob(x))
        elif (measure == 'uniformsphere'):
            # Reciprocal of surface area of sphere
            self.base_measure = None
            # This doesn't affect the fit anyway. Affects units of log likelihood
            self.pdf_ = lambda x, log_scale:\
                -self.d/2 * np.log(np.pi) - np.log(2) + \
                torch.lgamma(torch.tensor(self.d/2)) if log_scale \
                else torch.exp(torch.lgamma(torch.tensor(self.d/2)))/\
                    (2*np.pi**(self.d/2))
            self.pdf = lambda x, log_scale: self.pdf_(x, log_scale).to(x.device)
        else:
            raise Exception("Unexpected measure.")

    def marginal_base_pdf(self, keep_dims, x, log_scale=True):
        """
        Evaluate the marginal base density pdf over keep_dims at x.
        Assume that the base pdf factors as a diagonal Gaussian.
        """
        if keep_dims is None:
            return self.pdf(x, log_scale)

        assert self.measure=='gauss'
        weights = torch.softmax(self.base_measure.weight_scores, 1)
        assert weights.shape[0] == weights.shape[1]
        assert weights.shape[0] == 1

        log_std = self.base_measure.log_scale[:,:,keep_dims]
        std = torch.exp(log_std)
        means = self.base_measure.loc[:,:,keep_dims]

        ret = 0
        k = len(keep_dims)
        for i in range(weights.shape[1]):
            # Note: Weights already included in kernel calc
            ret = ret  - k/2*np.log(2*np.pi) - \
                0.5*torch.log(torch.prod(std[0,i]**2)) - \
                0.5*torch.sum((x - means[0,i])**2/std[0,i]**2, dim=1)
        
        if not log_scale:
            ret = torch.exp(ret)

        return ret

    def _initialise_params(self, d, n, m, diagonal_V, temporal):
        self.temporal = temporal 
        self.m = m
        W = np.random.normal(0, 1, (n, d))*2
        B = np.zeros((n, 1))

        self.W = torch.nn.Parameter(torch.from_numpy(W).float())

        if diagonal_V:
            assert (n == m)
            self.V = torch.nn.Parameter(torch.diag(torch.ones((n)).float()))
        else:
            if m == -1:
                size = n
            else:
                size = m
            V = np.random.normal(0, 1, (size, n))*np.sqrt(1/(n*size))
            self.V = torch.nn.Parameter(torch.from_numpy(V).float())
        if m == -1:
            self.initialise_vtv()

        self.B = torch.nn.Parameter(torch.from_numpy(B).float())
        self.v0 = torch.nn.Parameter(torch.from_numpy(np.asarray([1.])).float())
        # TODO: STD of Gaussian base measure
        #self.s = torch.nn.Parameter(torch.from_numpy(np.asarray([1.])).float())
        #if not ((self.measure == 'gauss') and (self.preprocessing == 'ident')):
        #    self.s.requires_grad = False

        if not (self.temporal is None):
            Wt = np.random.normal(0, 1, (n, 1))/100
            #Wt = np.ones((n,1))*0
            self.Wt = torch.nn.Parameter(torch.from_numpy(Wt).float())
            #Bt = np.random.normal(0, 1, (n, 1))
            Bt = np.ones((n, 1))
            self.Bt = torch.nn.Parameter(torch.from_numpy(Bt).float())

    def initialise_vtv(self):
        self.m = -1
        self.VTV = torch.nn.Parameter((self.V.T @ self.V).data)

    def finetune_model(self):
        if not self.finetuning:
            self.initialise_vtv()
            self.V.requires_grad = False
            self.W.requires_grad = False
            self.B.requires_grad = False
            self.v0.requires_grad = False
            self.Wt.requires_grad = False
            self.Bt.requires_grad = False
        self.finetuning = True

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
        elif (domain == 'sphere') and (measure == 'uniformsphere') and \
            (activation == 'relu') and (preprocessing == 'ident'):
            name = 'arccossphere'
            self.B.requires_grad = False
        elif (type(domain) is list) and (measure == 'leb') and (activation == 'exp')\
            and (preprocessing == 'ident'):
            name = 'loglinear'
            self.bound0 = domain[0]; self.bound1 = domain[1]
            self.B.requires_grad = False
        else:
            raise Exception("Unexpected integration parameters.")
        
        self.kernel = Kernel(name, self.a, self.bound0, self.bound1)
        if not (self.temporal is None):
            self.kernelt = Kernel('relu1d', self.temporal)

    def _mathcal_T(self, A, m, keep_dims = []):
        """
        Return the result \mathcal{T} \Theta
        """
        assert not (keep_dims is None)
        idx = list(range(0, self.d))
        [idx.remove(i) for i in keep_dims]
        A_ = A[:,:,idx]
        m_ = m[:,:,idx]
        d = len(idx)

        #(num_mix_components x d x d)
        Amat = torch.diag_embed(torch.squeeze(A_, dim=0))\
            .view(A_.shape[1],d, d)
        
        # (num_mix_components x n x 1)
        m_ = torch.swapaxes(m_, 0, 2)
        m_ = torch.swapaxes(m_, 0, 1)

        # (num_mix_components, n, d)
        # (num_mix_components, n, 1)
        ret1 = self.W[:,idx] @ Amat
        ret2 = self.B + self.W[:,idx] @ m_

        return (ret1, ret2)

    def integrate(self, extra_input=0, log_scale=False):
        self.K = self._evaluate_kernel(extra_input, keep_dims=[])
        # Multiply by temporal kernel if required [ALREADY DONE IN EVALUATE]
        #if not (self.temporal is None):
        #    self.K = self.K * self.kernelt(self.Wt, self.Bt, extra_input)
        #VKV = self.V.T @ self.K @ self.V+ self.v0**2 ## <- m=1 case transpose
        #torch.vmap vectorises the operation. So we can do a batch trace on
        # (B, m, m) for B traces of mxm matrices
        #VKV = torch.vmap(torch.trace)(self.V @ self.K @ self.V.T)
        # Not available until very recent so we do something else instead
        #VKV = (self.V @ self.K @ self.V.T).diagonal(offset=0, dim1=-1, 
        #    dim2=-2).sum(-1).view((-1, 1, 1)) + self.v0**2
        VTV = self.VTV + 1e-3*torch.eye(self.VTV.shape[0],
            device = self.VTV.device)\
            if self.m == -1 else self.V.T @ self.V

        VKV = torch.sum(self.K * VTV, dim=[1,2]) + self.v0**2
        if log_scale:
            ret = torch.log(VKV)
        else:
            ret = VKV

        return ret

    def _evaluate_kernel(self, extra_input=0, keep_dims = []):
        # Preprocess Gaussian mixture model parameters via affine transform
        if self.measure == 'gauss':
            t_param1, t_param2 = \
                self._mathcal_T(torch.exp(self.base_measure.log_scale), 
                self.base_measure.loc, keep_dims)
            t_param1 = t_param1.view((self.num_mix_components, self.n, -1))
            t_param2 = t_param2.view(\
                (self.num_mix_components, self.B.shape[0], 1))
        else:
            t_param1 = self.W.view((1, self.W.shape[0], self.W.shape[1]))
            t_param2 = self.B.view((1, self.B.shape[0], 1))

        # Only integrate over the dimensions we don't want to keep
        if not (keep_dims is None):
            idx = list(range(0, self.d))
            [idx.remove(i) for i in keep_dims]
            if not (extra_input == 0):
                if not (extra_input.nelement() == 1):
                    extra_input = \
                        self.W.view((1, self.W.shape[0], self.W.shape[1]))\
                        [:,:,keep_dims].contiguous() @\
                        extra_input.T.contiguous()
                    extra_input = torch.squeeze(extra_input).contiguous()

            """
            extra_input = t_param1[:,:,keep_dims].contiguous() @\
                extra_input.T.contiguous()
            extra_input = torch.squeeze(extra_input).contiguous()
            t_param1 = t_param1[:,:,idx].contiguous()
            """
        # Iterate over the number of mixture components. 
        K = 0 
        if self.measure == 'gauss':
            weights = torch.softmax(self.base_measure.weight_scores, 1)
        else:
            weights = np.asarray([[1]])

        # Iterate over mixture components
        for i in range(t_param1.shape[0]):
            K = K + weights[0,i]*\
                self.kernel(t_param1[i,:,:].contiguous(), 
                t_param2[i,:,:].contiguous(), extra_input)
        K = K.view((-1, self.n, self.n))

        # Multiply by temporal kernel if required
        if not (self.temporal is None):
            K = K * self.kernelt(self.Wt.contiguous(), self.Bt.contiguous(), 
                extra_input)
        return K

    def forward(self, y, extra_input=0, log_scale=False):
        y = self._mask(y)
        if not (self.temporal is None):
            t = y[:,-1].view((-1,1))
            y = y[:,:-1]
        
        # If M is batch size, below is shape M x n
        feat = self.act(self.W @ y.T + self.B + extra_input).T 

        #Multiply by temporal features if required
        if not (self.temporal is None):
            feat = feat * torch.nn.functional.relu(self.Wt @ t.T + self.Bt\
                + extra_input).T

        if self.m == -1:
            # Batch matrix multiply of features gives shape M x n x n
            """
            feat = feat.unsqueeze(2)
            Ktilde = torch.bmm(feat, torch.swapaxes(feat, 1, 2))
            VTV = self.VTV if self.m == -1 else self.V.T @ self.V
            squared_net = torch.sum(Ktilde*VTV, dim=[1,2]) + \
                self.v0**2
            """
            VTV = self.VTV + 1e-3*torch.eye(self.VTV.shape[0],
                device = self.VTV.device)\
                if self.m == -1 else self.V.T @ self.V
            psiT_VTV = feat @ VTV
            squared_net = torch.bmm(psiT_VTV.view(-1, 1, self.n),
                feat.view(-1, self.n, 1)) + self.v0**2
        else:
            net_out = (self.V @ feat.T).T
            squared_net = torch.norm(net_out, dim=1)**2 + self.v0**2
            squared_net = squared_net.view((1, -1))
        if log_scale:
            logpdf = self.pdf(y, log_scale)
            return torch.log(squared_net) + logpdf

        pdf = self.pdf(y, log_scale)
        return squared_net*pdf

    def l2_lastlayer(self):
        VTV = self.VTV if self.m == -1 else self.V.T @ self.V
        return torch.norm(VTV)

    def _mask(self, y, extra_input = 0):
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
    def __init__(self, name, a, bound0=None, bound1=None):
        super().__init__()
        self.a = a
        self.bound0 = bound0
        self.bound1 = bound1
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
        elif name == 'arccossphere':
            self.kernel = lambda W, B, extra_input: \
                kernels.arc_cosine_kernel_sphere\
                (W, W, B+extra_input, B+extra_input)
        elif name == 'arcsin':
            self.kernel = lambda W, B, extra_input: \
                kernels.arc_sine_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'vmf':
            self.kernel = lambda W, B, extra_input: \
                kernels.vmf_kernel(W, W, B+extra_input, B+extra_input)
        elif name == 'loglinear':
            self.kernel = lambda W, B, extra_input: \
                kernels.log_linear_kernel(W, W, B+extra_input, B+extra_input,
                        a=self.bound0, b=self.bound1)
        elif name == 'relu1d':
            self.kernel = lambda W, B, extra_input:\
                kernels.relu1d_kernel(W, W, B+extra_input, B+extra_input,
                    a=self.a)
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

