import torch
from squared_neural_families.nets.integrate import SquaredNN
from squared_neural_families.distributions.snf_ppp import PoissonPointProcess
from squared_neural_families.utils.plotters import matplotlib_config
import normflows as nf
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import numpy as np
import pickle

matplotlib_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Path('examples/outputs/ppp/').mkdir(parents = True, exist_ok = True)
NUM_EPOCHS = 1000

################################################## Example in \mathbb{R}^2 ##############
# The SNF PPP model
squared_nn = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=2, n=100)
ppp = PoissonPointProcess(squared_nn).to(device)

# The data
target = nf.distributions.TwoMoons()
x = target.sample(1000).to(device)

# A grid for plotting stuff
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)


def plot_intensity(zz, prefix=0):
    #log_prob = ppp.log_prob(zz).to('cpu').view(*xx.shape)
    #prob = torch.exp(log_prob)

    intensity = ppp.squared_nn(zz).to('cpu').view(*xx.shape)

    plt.figure(figsize=(15, 15))
    #plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='viridis')
    plt.pcolormesh(xx, yy, intensity.data.numpy(), cmap='viridis')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig('examples/outputs/ppp/' + str(prefix) + \
        'ppp_plane.png', bbox_inches='tight')
    plt.close()

# The training loop
ppp.train()
optimizer = torch.optim.Adam(ppp.parameters(), lr=0.01)
for it in range(NUM_EPOCHS):
    if (it % 100) == 0:
        plot_intensity(zz, it)

    optimizer.zero_grad()

    loss = -1*torch.mean(ppp.log_prob(x))

    loss.backward()
    optimizer.step()

    print(loss)





################################################## Example on \mathbb{S}^2 ##############
NUM_EPOCHS = 10000
# The SNF PPP model
squared_nn = SquaredNN('sphere', 'uniformsphere', 'exp', 'ident', d=3, n=100)
ppp = PoissonPointProcess(squared_nn).to(device)

# The data
x = torch.normal(0, 1, (1000, 3))
x = (x / torch.norm(x, dim=0)).to(device)

# A grid for plotting stuff
grid_size = 200
theta1, theta2 = torch.meshgrid(torch.linspace(0, np.pi, grid_size), torch.linspace(0, 2*np.pi, grid_size))

xx = torch.sin(theta1) * torch.cos(theta2)
yy = torch.sin(theta1) * torch.sin(theta2)
zz = torch.cos(theta1)

xyz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2), zz.unsqueeze(2)], 2).view(-1, 3)
xyz = xyz.to(device)

def plot_intensity(xyz, prefix=0):
    intensity = ppp.squared_nn(xyz).to('cpu').view(*xx.shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    heatmap = intensity.to('cpu').data.numpy()
    heatmap = heatmap / np.amax(heatmap)
    
    ax.plot_surface(xx.to('cpu').data.numpy(), 
        yy.to('cpu').data.numpy(), 
        zz.to('cpu').data.numpy(), cstride=1, rstride=1, facecolors=cm.jet(heatmap))
    #plt.gca().set_aspect('equal', 'box')
    plt.savefig('examples/outputs/ppp/' + str(prefix) + \
        'ppp_sphere.png', bbox_inches='tight')
    plt.close()

# The training loop
ppp.train()
optimizer = torch.optim.Adam(ppp.parameters(), lr=0.001)
for it in range(NUM_EPOCHS):
    if (it % 100) == 0:
        plot_intensity(xyz, it)

    optimizer.zero_grad()

    loss = -1*torch.mean(ppp.log_prob(x))

    loss.backward()
    optimizer.step()

    print(loss)


