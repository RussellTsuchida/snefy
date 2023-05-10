# Direct edit of https://github.com/VincentStimper/normalizing-flows/blob/master/examples/real_nvp_colab.ipynb
import torch
import numpy as np
import normflows as nf
import math
from matplotlib import pyplot as plt
import time

from tqdm import tqdm

from squared_neural_families.nets.integrate import SquaredNN
from squared_neural_families.nets.layers import Mlp 
from squared_neural_families.utils.plotters import matplotlib_config
from squared_neural_families.distributions.snf import Density, ConditionalDensity, AutoregressiveDensity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib_config()

##################################################
# Set up model
# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(2)

# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
"""
# This models p(x1)
squared_nn1 = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
    n = 100, dim=1)
p1 = Density(squared_nn1)

# This models p(x0 | x1)
squared_nn2 = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
    n = 50, dim=0)
# Feature network masks out x0 (so only operates on x1)
feat_net = Mlp([2, 200, 200, 50], mask_idx=0)
p0given1 = ConditionalDensity(squared_nn2, feat_net)

# Create the joint density
snf_base = AutoregressiveDensity([p1, p0given1]).to(device)
"""

squared_nn = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=2,
    n=100)
snf_base = Density(squared_nn)

# Construct NVP and snf models
nvp = nf.NormalizingFlow(base, flows)
nvp = nvp.to(device)

snf = nf.NormalizingFlow(snf_base, [])
snf = snf.to(device)



def train_and_visualise(model, name):
    # Define target distribution
    target = nf.distributions.TwoMoons()

    # Plot target distribution
    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig('examples/outputs/moon/' + name + \
        '_target_moon.png', bbox_inches='tight')
    plt.close()

    def log_prob_batch(density, zz, batch_size=512):
        log_prob = torch.zeros((0,)).to('cpu')
        for i in range( int(math.ceil(zz.shape[0]/batch_size))):
            zzi = zz[batch_size*i:min(batch_size*(i+1),zz.shape[0]),:]
            log_probi = model.log_prob(zzi).to('cpu').data
            log_prob = torch.hstack((log_prob, log_probi))
        return log_prob

    # Train model
    max_iter = 2001
    num_samples = 2 ** 9
    show_iter = 500

    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters())
    t0 = time.time()
    for it in tqdm(range(max_iter)):
        # Plot learned distribution
        if (it) % show_iter == 0:
            model.eval()
            #log_prob = model.log_prob(zz)
            log_prob = log_prob_batch(model, zz)

            prob = torch.exp(log_prob.to('cpu').view(*xx.shape))

            plt.figure(figsize=(15, 15))
            plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')
            plt.gca().set_aspect('equal', 'box')
            plt.savefig('examples/outputs/moon/' + name + str(it)  + '_moon.png', 
                bbox_inches='tight')
            plt.close()
            model.train()

        optimizer.zero_grad()
        
        # Get training samples
        x = target.sample(num_samples).to(device)
        
        # Compute loss
        loss = model.forward_kld(x)
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    t1 = time.time()

    model.eval()
    x = target.sample(1000).to(device)
    final_log_prob = torch.mean(log_prob_batch(model, x))
    print('Final log probability for model ' + str(name) + ':' + str(final_log_prob))
    print('Training took ' + str(t1 - t0) + ' seconds')
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Parameter count: ' + str(param_count))

    # Plot loss
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig('examples/outputs/moon/' + name + 'loss_moon.png', bbox_inches='tight')
    plt.close()

    # Plot target distribution
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)

    ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')

    ax[0].set_aspect('equal', 'box')
    ax[0].set_axis_off()
    ax[0].set_title('Target', fontsize=24)

    # Plot learned distribution
    log_prob = log_prob_batch(model, zz)
    prob = torch.exp(log_prob.to('cpu').view(*xx.shape))

    ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')

    ax[1].set_aspect('equal', 'box')
    ax[1].set_axis_off()
    ax[1].set_title('Real NVP', fontsize=24)

    plt.subplots_adjust(wspace=0.1)
    plt.savefig('examples/outputs/moon/' + name + 'all_moon.png', bbox_inches='tight')
    plt.close()


train_and_visualise(snf, 'snf')
train_and_visualise(nvp, 'nvp')

