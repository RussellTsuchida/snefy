import torch
import torch.nn as nn

class PoissonPointProcess(torch.nn.Module):
    def __init__(self, squared_nn):
        super().__init__()
        self.squared_nn = squared_nn # This is the intensity function of the PPP


    def log_prob(self, y):
        """
        log probability of observations y

        y is (n, d) where n is the number of points and d is the dimension

        return size (n, 1)
        """
        if self.training:
            self.update_iif()

        return (self.squared_nn(y, log_scale=True) - self.log_iif).reshape((-1,1))


    def update_iif(self):
        """
        Update calculation of the integrated intensity function
        """
        self.log_iif = self.squared_nn.integrate(log_scale=True)


