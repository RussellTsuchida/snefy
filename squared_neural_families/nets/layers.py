import torch

class Mask(torch.nn.Module):
    """
    A torch layer which masks indices of a 1D input.

    Args:
        idx (list of ints or int): indices to mask (i.e. set to zero)
    """
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        x_ = torch.clone(x)
        x_[:, self.idx] = 0
        return x_


class Mlp(torch.nn.Module):
    """
    Basic MLP with ReLU activations, batch norm, and input masking.

    Args:
        layer_sizes (list of ints): Size of each layer, including inputs and
            outputs
        mask_idx (None or list of ints): Input indices to mask. If None, don't
            mask any inputs.
    """
    def __init__(self, layer_sizes, mask_idx = None):
        super().__init__()

        modules = []
        if not (mask_idx is None):
            modules.append(Mask(mask_idx))
        for i in range(1, len(layer_sizes)):
            in_size = layer_sizes[i-1]
            out_size = layer_sizes[i]
            modules.append(torch.nn.Linear(in_size, out_size))
            if (i == (len(layer_sizes) - 1)):
                pass
            else:
                modules.append(torch.nn.BatchNorm1d(out_size))
                modules.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*modules)

        
    def forward(self, x):
        if (x is None):
            return 0
        return self.net(x)


