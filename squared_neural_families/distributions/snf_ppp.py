import torch
import torch.nn as nn
import math
import numpy as np

class PoissonPointProcess(torch.nn.Module):
    def __init__(self, squared_nn, alpha=1):
        super().__init__()
        self.squared_nn = squared_nn # This is the intensity function of the PPP
        self.alpha = alpha

    def log_prob(self, y, N):
        """
        log probability of observations y

        y is (M, d) where M is the number of BATCH points and d is the dimension

        Note that M <= N, where N is the number of events in the realisation.

        NOTE: RETURN SIZE IS SCALAR, NOT (M, 1) AS IT WOULD BE FOR DENSITY
        ESTIMATION! 
        """
        if self.training:
            self.update_iif()
        
        # Careful to appropriately handle ratio of batch size to realisation size
        M = y.shape[0]
        return torch.sum(self.squared_nn(y, log_scale=True))*N/M - \
            self.alpha*self.iif - math.lgamma(N+1) + N*np.log(self.alpha)

    def update_iif(self):
        """
        Update calculation of the integrated intensity function
        """
        self.iif = self.squared_nn.integrate(log_scale=False)

    def integrate_over_A(self, samples):
        """
        We assume that samples is (N, d), where N is the number of samples and
        d is the dimension. Each row is a sample from the base measure \mu, 
        followed by a filtering process which removes samples that do not
        belong to A.
        The integrated intensity function is just the expected value of the
        intensity function with respect to \mu.
        """
        lambda_eval = self.alpha*self.squared_nn(samples, log_scale=False)
        return torch.mean(lambda_eval)
