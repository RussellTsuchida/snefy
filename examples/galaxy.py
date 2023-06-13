import matplotlib.pyplot as plt
import numpy as np
from squared_neural_families.nets.integrate import SquaredNN
from squared_neural_families.distributions.snf import Density
import torch
from matplotlib import cm
import time

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

########################################## Load data
TRAIN_SPLIT = 0.8

data = np.genfromtxt('galaxies.tab', skip_header=1, encoding='utf-8')
data = data * np.pi/180
np.random.shuffle(data)

x = np.sin(data[:,0])*np.cos(data[:,1])
y = np.sin(data[:,0])*np.sin(data[:,1])
z = np.cos(data[:,0])

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
def plot_data(ax, fname):
    ax.view_init(90, 90)
    ax.scatter(x[:]*1.02, y[:]*1.02, z[:]*1.02, 
            s=1, c='k')
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('not_rotated/' + fname, bbox_inches='tight', transparent=True,
    pad_inches = 0)

    ax.view_init(270, 0)
    plt.tight_layout()
    plt.savefig('rotated/' + fname, bbox_inches='tight', transparent=True,
    pad_inches = 0)
    plt.close()

data = np.hstack((x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1))))
data = torch.from_numpy(data.astype(np.float32)).to(device)
data_train = data[:int(TRAIN_SPLIT*data.shape[0]),:]
data_test = data[data_train.shape[0]:,:]

############################################ Define model
n = 30
d = 3
m = n

t0 = time.time()
squared_nn = SquaredNN('sphere', 'uniformsphere', 'exp', 'ident', d=d, n=n,
m=n, diagonal_V=False).to(device)
snf = Density(squared_nn).to(device)

####################################### A grid for plotting stuff
grid_size = 200
theta1, theta2 = torch.meshgrid(torch.linspace(0, np.pi, grid_size), torch.linspace(0, 2*np.pi, grid_size))

xx = torch.sin(theta1) * torch.cos(theta2)
yy = torch.sin(theta1) * torch.sin(theta2)
zz = torch.cos(theta1)

xyz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2), zz.unsqueeze(2)], 2).view(-1, 3)
xyz = xyz.to(device)

# Define a plotting function
def plot_density(xyz, prefix=0):
    logp = snf.log_prob(xyz).to('cpu').view(*xx.shape)
    p = torch.exp(logp)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    heatmap = logp.to('cpu').data.numpy()
    heatmap = heatmap - np.amin(heatmap)
    if not (np.amax(heatmap) == 0):
        heatmap = heatmap / np.amax(heatmap)

    ax.plot_surface(xx.to('cpu').data.numpy(), 
        yy.to('cpu').data.numpy(), 
        zz.to('cpu').data.numpy(), cstride=1, rstride=1,
        facecolors=cm.jet(heatmap))
    plot_data(ax, str(prefix)+'galaxy.png')


##################################################### Train model
snf.train()
optimizer = torch.optim.Adam(snf.parameters())
#for epoch in range(20000):
for epoch in range(2000):
    #if epoch > 10:
    #    squared_nn.W.requires_grad=False
    optimizer.zero_grad()
    loss = -1*torch.mean(snf.log_prob(data_train))
    loss.backward()
    optimizer.step()

    #if (epoch % 1000) == 0:
    if (epoch % 5) == 0:
        #loss = -1*torch.mean(snf.log_prob(data_test))
        plot_density(xyz, epoch)
        #print(epoch)
        #print(loss.item())

print(str(loss.item()) + ',' + str(time.time() - t0))
