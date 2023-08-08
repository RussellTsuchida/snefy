import torch
import torch.nn as nn
import numpy as np

from normflows.distributions import BaseDistribution

class ConditionalDensity(BaseDistribution):
    """
    Squared neural family conditional density. The feature_net operates on the
    variable we are conditioning on
    """
    def __init__(self, squared_nn, feature_net, proposal_dist=None):
        super().__init__()
        self.squared_nn = squared_nn
        self.dim = squared_nn.d
        self.feature_net = feature_net
        self._init_proposal_dist(proposal_dist)

    def _init_proposal_dist(self, proposal_dist):
        """
        Default proposal distribution is Gaussian with scale 2.
        Scale 2 so that we are slightly wider than the default base 
        distribution of the squared neural family, which has identity scale

        """
        #TODO: Currently all on CPU
        device = self.squared_nn.B.device
        if proposal_dist is None:
            proposal_dist = torch.distributions.multivariate_normal.\
            MultivariateNormal(torch.zeros(self.dim), 
            2*torch.eye(self.dim))
        self.proposal_dist = proposal_dist

    def sample(self, num_samples=1, M=1000, x=None, num_batch_samples=None,
        keep_leftovers = False):
        """
        Rejection sampling.
        """
        #TODO: Currently assumes proposal dist is on CPU
        device = self.squared_nn.B.device
        values = torch.zeros(0, self.dim).to(device)

        # Sample twice as many as expected from the proposal
        if num_batch_samples is None:
            num_batch_samples = min(x.shape[0], 2*M*num_samples) if \
                not (x is None) else 2*M*num_samples

        while values.shape[0] < num_samples:
            y = self.proposal_dist.sample([num_batch_samples])
            # (-1, self.dim)

            #  Check whether to accept or reject
            log_prob = -self.proposal_dist.log_prob(y).to(device) + \
                self.log_prob(y.to(device), x) - \
                np.log(M)
            u = torch.rand(y.shape[0]).to(device)
            # Keep relevant samples 
            idx = (torch.log(u) < log_prob).nonzero()
            values_ = y[idx[:,0], :].to(device)
            values = torch.vstack((values, values_))
            if torch.any(log_prob > 0):
                print('Warning: M may not be high enough')

        if keep_leftovers:
            return values.to(device)
        return values[:num_samples, :].to(device)
        
    def forward(self, num_samples=1, M=1000, x=None):
        """
        Conforming to normalising flows API. Generate samples and compute
        their log probability.
        """
        logp = self.log_prob(samples, x)
        samples = self.sample(num_samples, M, x)
        return self.samples, logp

    def log_prob(self, y, x, keep_dims=None):
        """
        Log probability of y given x (as a batch)
        """
        feat = self.feature_net(x).T

        # Joint distribution (keep_dims=None) or marginal distribution
        if keep_dims is None:
            log_num = torch.squeeze(self.squared_nn(y, extra_input=feat,
                log_scale=True))
        else:
            VTV = self.squared_nn.VTV + 1e-3*torch.eye(\
                self.squared_nn.VTV.shape[0], 
                device = self.squared_nn.VTV.device)\
                if self.squared_nn.m == -1 else \
                self.squared_nn.V.T @ self.squared_nn.V

            C = self.squared_nn._evaluate_kernel(extra_input=y,
                keep_dims=keep_dims)

            # NOTE: HERE WE ASSUME THE BASE PDF IS FACTORISED INDEPENDENT
            # And also v0 == 0
            logpdf = self.squared_nn.marginal_base_pdf\
                (keep_dims, y, log_scale=True)
            log_num = torch.log(torch.sum(VTV * C, dim=[1,2]) + \
                self.squared_nn.v0**2) + logpdf
        # Only update the partition function during training. 
        # First evaluation after model.eval() requires manual call to update_partfun
        if self.training:
            self.update_partfun(x)
        return log_num - self.log_part

    def update_partfun(self, x):
        # First evaluation after model.eval() requires manual call to update_partfun
        feat = self.feature_net(x).T
        self.log_part = torch.squeeze(\
                self.squared_nn.integrate(extra_input=feat, log_scale=True))

class Density(ConditionalDensity):
    """
    Same as conditional density but without conditioning on any input.
    This is implemented as a conditional density but with a feature network
    which always outputs zero
    """
    def __init__(self, squared_nn):
        self.param = 0
        super().__init__(squared_nn, lambda x: self.param)
        self.param = torch.nn.Parameter(\
            torch.from_numpy(np.asarray([[0.]])).float())
        self.param.requires_grad = False
    
    def sample(self, num_samples=1, M=1000, x=None, num_batch_samples=None,
        keep_leftovers = False):
        return super().sample(num_samples, M, x, num_batch_samples,
            keep_leftovers)

    def forward(self, num_samples=1, M=1000, x=None):
        return super().forward(num_samples, M, x=x)

    def log_prob(self, y, x=None, keep_dims=None):
        return super().log_prob(y, x=x, keep_dims=keep_dims)

    def update_partfun(self, x=None):
        return super().update_partfun(x=x)


class AutoregressiveDensity(BaseDistribution):
    """
    We assume that the masking is performed by the feature net
    """
    def __init__(self, density_list, proposal_dist=None):
        super().__init__()
        self.density_list = torch.nn.ModuleList(density_list)
        self.dim = len(density_list)
        self._init_proposal_dist(proposal_dist)

    def _init_proposal_dist(self, proposal_dist):
        """
        Default proposal distribution is Gaussian with scale 2.
        Scale 2 so that we are slightly wider than the default base 
        distribution of the squared neural family, which has identity scale

        """
        #TODO: Currently all on CPU
        device = self.density_list[0].squared_nn.B.device
        if proposal_dist is None:
            proposal_dist = torch.distributions.multivariate_normal.\
            MultivariateNormal(torch.zeros(self.dim), 
            2*torch.eye(self.dim))
        self.proposal_dist = proposal_dist

    def log_prob(self, y):
        """

        """
        ret = 0. 
        for l in range(len(self.density_list)):
            p = self.density_list[l]
            ret = ret + p.log_prob(y, y)
        return ret

    def sample(self, num_samples=1, M=1000, method='multi',
        num_batch_samples = None):
        if method == 'autoregressive':
            return self._sample_autoregressive(num_samples, M,
            num_batch_samples = num_batch_samples)
        elif method == 'multi':
            return self._sample_multi(num_samples, M, 
            num_batch_samples = num_batch_samples)

    def _sample_autoregressive(self, num_samples=1, M=1000, 
        num_batch_samples=None):
        device = self.density_list[0].squared_nn.B.device
        samples = None
        x = None
        if num_batch_samples is None:
            num_batch_samples = num_samples*M*2

        for l in range(len(self.density_list)):
            p = self.density_list[l]
            if not (samples is None):
                x = torch.hstack((\
                    torch.zeros(samples.shape[0],
                    len(self.density_list)-samples.shape[1]).to(device),
                    samples))
                num_batch_samples = min(num_batch_samples, x.shape[0])
            new = p.sample(num_samples, M, x, 
                num_batch_samples=num_batch_samples, keep_leftovers=True)
            samples = new if (samples is None) else torch.hstack((new,
                samples[:new.shape[0], :]))

        return samples[:num_samples,:]


    def _sample_multi(self, num_samples=1, M=1000, num_batch_samples=None):
        """
        Rejection sampling.
        """
        #TODO: Currently assumes proposal dist is on CPU
        device = self.density_list[0].squared_nn.B.device
        values = torch.zeros(0, self.dim).to(device)

        # Sample twice as many as expected from the proposal
        if num_batch_samples is None:
            num_batch_samples = 2*M*num_samples

        while values.shape[0] < num_samples:
            y = self.proposal_dist.sample([num_batch_samples])
            # (-1, self.dim)

            #  Check whether to accept or reject
            log_prob = -self.proposal_dist.log_prob(y).to(device) + \
                self.log_prob(y.to(device)) - np.log(M)
            u = torch.rand(y.shape[0]).to(device)
            # Keep relevant samples 
            idx = (torch.log(u) < log_prob).nonzero()
            values_ = y[idx[:,0], :].to(device)
            values = torch.vstack((values, values_))
            if torch.any(log_prob > 0):
                print('Warning: M may not be high enough')

        return values[:num_samples, :].to(device)

    def forward(self, num_samples=1, M=1000):
        """
        Conforming to normalising flows API. Generate samples and compute
        their log probability.
        """
        samples = self.sample(num_samples, M)
        return self.samples, self.log_prob(samples)

