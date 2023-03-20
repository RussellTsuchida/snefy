from squared_neural_families.nets.integrate import SquaredNN
from squared_neural_families.nets.layers import Mlp
from squared_neural_families.utils.plotters import matplotlib_config
from squared_neural_families.distributions.snf import Density, ConditionalDensity, AutoregressiveDensity

import matplotlib.pyplot as plt 
from scipy.integrate import dblquad
import torch
import numpy as np
import math
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib_config()
			
############################## UNCONDITIONAL DENSITY
# Initialise a density with a squared neural network
squared_nn = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=2, n=100, 
	dim=None).to(device)
unconditional_density = Density(squared_nn).to(device)
unconditional_density.eval()

# Generate 5 samples from said density 
t0 = time.time()
samples = unconditional_density.sample(50,M=10).to('cpu')
print('Took ' + str(time.time() - t0) + ' seconds to sample 50 points from unconditional density')

# Plot the density on a fine grid with the samples
grid_size = 200 
grid_extent = torch.max(torch.abs(samples)).item()
xx, yy = torch.meshgrid(\
	torch.linspace(-grid_extent, grid_extent, grid_size), 
	torch.linspace(-grid_extent, grid_extent, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

log_prob = unconditional_density.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')

plt.scatter(samples[:,0], samples[:,1], c='r')
plt.gca().set_aspect('equal', 'box')
plt.savefig('examples/outputs/density.png', bbox_inches='tight')
plt.close()
# Verify using SciPy that the density integrates to one
np_prob = lambda x, y: torch.exp(unconditional_density.log_prob(\
	torch.from_numpy(np.asarray([x, y]).reshape((1,-1))).float()\
	.to(device)).to('cpu')).data.numpy()

#print(dblquad(np_prob, -np.inf, np.inf, -np.inf, np.inf, 
#	epsabs=0.1, epsrel=0.1))
############################## AUTOREGRESSIVE DENSITY
# This models p(x1)
squared_nn1 = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
    n = 100, dim=1)
p1 = Density(squared_nn1)

# This models p(x0 | x1)
squared_nn2 = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
    n = 100, dim=0)
# Feature network masks out x0 (so only operates on x1)
feat_net = Mlp([2, 100, 100, 100], mask_idx=0)
p0given1 = ConditionalDensity(squared_nn2, feat_net)

# Create the joint density
p01 = AutoregressiveDensity([p1, p0given1]).to(device)
p01.eval()

# Generate 50 samples from said density 
t0 = time.time()
samples = p01.sample(50, M=10, method='multi').to('cpu')
print('Took ' + str(time.time() - t0) + ' seconds to sample 50 points from autoregressive density')
# Plot the density on a fine grid with the samples
grid_size = 200 
grid_extent = torch.max(torch.abs(samples)).item()
xx, yy = torch.meshgrid(\
	torch.linspace(-grid_extent, grid_extent, grid_size), 
	torch.linspace(-grid_extent, grid_extent, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

batch_size=400
log_prob = torch.zeros((0,)).to(device)
for i in range( int(math.ceil(zz.shape[0]/batch_size))):
    zzi = zz[batch_size*i:min(batch_size*(i+1),zz.shape[0]),:]
    log_probi = p01.log_prob(zzi)
    log_prob = torch.hstack((log_prob, log_probi))
prob = torch.exp(log_prob.to('cpu').view(*xx.shape))

#log_prob = p01.log_prob(zz, zz).to('cpu').view(*xx.shape)
#prob = torch.exp(log_prob)

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')

plt.scatter(samples[:,0], samples[:,1], c='r')
plt.gca().set_aspect('equal', 'box')
plt.savefig('examples/outputs/autoregressive_density.png', bbox_inches='tight')
plt.close()
# Verify using SciPy that the density integrates to one
np_prob = lambda x, y: torch.exp(p01.log_prob(\
	torch.from_numpy(np.asarray([x, y]).reshape((1,-1))).float()\
	.to(device)).to('cpu')).data.numpy()

#print(dblquad(np_prob, -np.inf, np.inf, -np.inf, np.inf, 
#	epsabs=0.1, epsrel=0.1))


