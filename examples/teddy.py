import numpy as np
from cdetools.cde_loss import cde_loss
from matplotlib import pyplot as plt
from squared_neural_families.distributions.snf import ConditionalDensity
from squared_neural_families.nets.integrate import SquaredNN
from squared_neural_families.nets.layers import Mlp
import torch
import os
import wget
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

np_data = 'data/teddy_x_train.npy'
data_dir = 'data/'
if not os.path.exists(np_data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print('"data" subfolder created')

    _ = wget.download('https://github.com/COINtoolbox/photoz_catalogues/raw/master/Teddy/teddy_A', 
                  out='data/teddy_A.txt')

    _ = wget.download('https://github.com/COINtoolbox/photoz_catalogues/raw/master/Teddy/teddy_B', 
                  out='data/teddy_B.txt')

    def extract_teddy_data(filename, train_data, directory='data/'):
        
        full_data = []
        outfiles = ('teddy_x_train.npy', 'teddy_z_train.npy') if train_data else ('teddy_x_test.npy', 'teddy_z_test.npy')
        with open(filename) as fp: 
            full_lines = fp.readlines()
            for line in full_lines:
                if '#' in line:
                    continue
                full_data.append([float(el) for el in line.strip().split(' ') if el])
            fp.close()
        
        # Saving the formatted Teddy data
        np.save(arr=np.array(full_data)[:, 7:12], file=directory + outfiles[0])
        np.save(arr=np.array(full_data)[:, 6], file=directory + outfiles[1])
        #print('Extraction and Saving Done!')

    extract_teddy_data(filename='data/teddy_A.txt', train_data=True, directory='data/')

    extract_teddy_data(filename='data/teddy_B.txt', train_data=False, directory='data/')


x_train = np.load(file='data/teddy_x_train.npy')#[:1000,:]
x_test = np.load(file='data/teddy_x_test.npy')
z_train = np.load(file='data/teddy_z_train.npy')#[:1000]
z_test = np.load(file='data/teddy_z_test.npy')


#x_train = np.load(file='data/teddy_x_train.npy')[:train_limit_points, :]
#x_test = np.load(file='data/teddy_x_test.npy')[:test_limit_points, :]
#z_train = np.load(file='data/teddy_z_train.npy')[:train_limit_points]
#z_test = np.load(file='data/teddy_z_test.npy')[:test_limit_points]

mu_train = np.tile(np.average(x_train, axis=0), (x_train.shape[0], 1))
std_train = np.tile(np.std(x_train, axis=0), (x_train.shape[0], 1))
x_train = np.divide(x_train - mu_train, std_train)


x=torch.from_numpy(x_train.astype(np.float64)).type(torch.Tensor).to(device)
z=torch.from_numpy(z_train.astype(np.float64)).type(torch.Tensor).to(device)

# Here we need to normalize the input (using the same mu and std for training, 
# so we need to be careful with dimensions)
mu_train = np.tile(np.average(x_train, axis=0), (x_test.shape[0], 1))
std_train = np.tile(np.std(x_train, axis=0), (x_test.shape[0], 1))

x_test = np.divide(x_test - mu_train, std_train)
x_test = torch.from_numpy(x_test.astype(np.float64)).type(torch.Tensor).to(device)
z_test_ = \
torch.from_numpy(z_test.astype(np.float64)).type(torch.Tensor).reshape((-1,1)).to(device)


class Dataset(torch.utils.data.Dataset):

    def __init__(self,x,z):
        self.x = x
        self.z = z

    def __getitem__(self,index):
        return self.x[index], self.z[index]

    def __len__(self):
        return len(self.x)


batch_size = 256
dataset = Dataset(x,z)
train_load = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

### The model
t0 = time.time()
squared_nn = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
    n = 32, dim=0, num_mix_components=1, m=4)
squared_nn.to(device)
squared_nn.a = 10
squared_nn._initialise_kernel('Rd', 'gauss', 'snake', 'ident')
feat_net = Mlp([5, 512, 256, 128, 64, 32])
#feat_net = Mlp([5, 256, 128, 64, 32, 16])
#feat_net = Mlp([5, 128, 64, 32, 16, 8])
model = ConditionalDensity(squared_nn, feat_net).to(device)
#pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print('Num params: ' + str(pytorch_total_params))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
n_epochs = 100

loss_list = []

model.train()
for epoch in range(n_epochs):
    #rint(epoch)
    epoch_loss = 0
    for batch_idx, (x_batch, z_train_batch) in enumerate(train_load):
        x_batch, z_train_batch = x_batch.to(device), z_train_batch.to(device)
        optimizer.zero_grad()
        bs = min(batch_size, x_batch.shape[0])
        loss = -torch.mean(\
                model.log_prob(z_train_batch.reshape((-1,1)), x_batch))
        loss.backward()
        #print(loss.item())
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()*bs
        loss_list.append(loss.item()*bs)
        
    #print(epoch_loss/x_train.shape[0])
    #z_data = z_test_[:batch_size, :]
    #x_data = x_test[:batch_size, :]
    #cde_test = -torch.mean(model.log_prob(z_data, x_data)).item()
    #print(cde_test)
    #scheduler.step()

########################## Evaluate test NLL. Need to do it in batches
idx = 0
nll = 0
while idx < z_test.shape[0]:
    end_idx = min(min(idx + batch_size, z_test.shape[0]), x_test.shape[0])
    z_data = z_test_[idx:end_idx, :]
    x_data = x_test[idx:end_idx, :]
    cde_test = torch.sum(model.log_prob(z_data, x_data)).item()
    idx = idx + batch_size
    nll = nll - cde_test
nll = nll/z_test.shape[0]
#nll = -torch.mean(model.log_prob(z_test_, x_test))
print(str(nll) + ',' + str(time.time() - t0))


####################### Evaluate PDF over fine grid for first 12 test instances
n_grid = 100
z_grid = np.linspace(np.amin(z_train), np.amax(z_train), 
        n_grid).reshape((-1,1))  # Creating a grid over the density range
z_grid_= torch.from_numpy(z_grid.astype(np.float64)).type(torch.Tensor).to(device)
x_test_ = torch.repeat_interleave(x_test[:12,:], n_grid, 0).to(device)
z_grid_ = torch.tile(z_grid_, (12, 1)).to(device)
idx = 0
cde_test = torch.zeros((0,)).to(device)
while idx < z_grid_.shape[0]:
    end_idx = min(min(idx + batch_size, z_grid_.shape[0]), x_test_.shape[0])
    z_data = z_grid_[idx:end_idx, :]
    x_data = x_test_[idx:end_idx, :]
    cde_test_ = model.log_prob(z_data, x_data)
    cde_test = torch.hstack((cde_test, cde_test_))
    idx = idx + batch_size
#cde_test = model.log_prob(z_grid_, x_test_)

cde_test = cde_test.reshape((12, -1)).cpu().detach().numpy()


########################### CHeck that first pdf integrates to 1
from scipy.integrate import simps
cde_test = np.exp(cde_test)
#print(type(cde_test), cde_test.shape)
den_integral = simps(cde_test[0, :], 
        x=z_grid.reshape((-1,)))
#print('Integral of the first density integrates to: %.2f' % den_integral)

#from cdetools.cde_loss import cde_loss as cde_loss_np
#cde_loss_val, std_cde_loss = cde_loss_np(cde_test, 
#        z_grid, z_test_[:12,:])
#print('CDE Loss: %4.2f \pm %.2f' % (cde_loss_val, std_cde_loss))

################################### Plot first 12 conditional PDFs
fig = plt.figure(figsize=(30, 20))
for jj, cde_predicted in enumerate(cde_test[:12,:]):
    ax = fig.add_subplot(3, 4, jj + 1)
    plt.plot(z_grid, 
            np.log(cde_predicted), label=r'$\log \hat{p}(z| x_{\rm obs})$')
    plt.axvline(z_test[jj], color='red', label=r'$z_{\rm obs}$')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel(r'Redshift $z$', size=20)
    plt.ylabel('CDE', size=20)
    plt.legend(loc='upper right', prop={'size': 20})
plt.savefig('snefy.png')



