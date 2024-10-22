import numpy as np
#from cdetools.cde_loss import cde_loss
from matplotlib import pyplot as plt
from squared_neural_families.distributions.snf import Density
from squared_neural_families.nets.integrate import SquaredNN
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

x=torch.from_numpy(x_train.astype(np.float64)).type(torch.Tensor).to(device)
z=torch.from_numpy(z_train.astype(np.float64)).type(torch.Tensor).to(device)

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
squared_nn = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=6,
    n = 10, num_mix_components=1, m=10)
squared_nn.to(device)
squared_nn.a = 1
squared_nn._initialise_kernel('Rd', 'gauss', 'snake', 'ident')
squared_nn.v0 = torch.nn.Parameter(torch.tensor([0.]))
squared_nn.v0.requires_grad=False

model = Density(squared_nn).to(device)
#pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print('Num params: ' + str(pytorch_total_params))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
n_epochs = 10

loss_list = []

model.train()
q = 0.5 # Probability of deleting a whole coordinate from a batch
for epoch in range(n_epochs):
    epoch_loss = 0
    print(epoch)
    for batch_idx, (x_batch, z_train_batch) in enumerate(train_load):
        x_batch, z_train_batch = x_batch.to(device), z_train_batch.to(device)
        data_batch = torch.hstack((x_batch, z_train_batch.reshape((-1,1))))

        optimizer.zero_grad()
        bs = min(batch_size, x_batch.shape[0])

        ### TRAINING WITH RANDOMLY PARTIAL OBSERVATIONS
        keep_dims = np.arange(x_batch.shape[1]+1)
        keep_dims_idx = tuple(np.nonzero(\
                np.random.choice([0, 1], size=(x_batch.shape[1]+1,), 
                p=[q, 1-q]).astype(int)))
        keep_dims = keep_dims[keep_dims_idx]
        data_batch = data_batch[:,keep_dims]

        loss = -torch.mean(\
                model.log_prob(data_batch, keep_dims=keep_dims))

        """
        ### TRAINING WITH FULL JOINT OBSERVATIONS
        loss = -torch.mean(\
                model.log_prob(data_batch, x=0))
        """
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()*bs
        loss_list.append(loss.item()*bs)
        
########################## Evaluate test NLL. Need to do it in batches
### EVALUATE ON FULL JOINT OBSERVATIONS
idx = 0
nll = 0
while idx < z_test.shape[0]:
    end_idx = min(min(idx + batch_size, z_test.shape[0]), x_test.shape[0])
    z_data = z_test_[idx:end_idx, :]
    x_data = x_test[idx:end_idx, :]
    batch_data = torch.hstack((x_data, z_data))

    cde_test = torch.sum(model.log_prob(batch_data)).item()
    idx = idx + batch_size
    nll = nll - cde_test
nll = nll/z_test.shape[0]
print(str(nll) + ',' + str(time.time() - t0))



### EVALUATE ON PARTIAL OBSERVATIONS FIRST 2 COORDINATES
idx=0
nll = 0
keep_dims = [0,1,2,3,4]
while idx < z_test.shape[0]:
    end_idx = min(min(idx + batch_size, z_test.shape[0]), x_test.shape[0])
    z_data = z_test_[idx:end_idx, :]
    x_data = x_test[idx:end_idx, :]
    batch_data = torch.hstack((x_data, z_data))
    batch_data = batch_data[:,keep_dims]

    cde_test = torch.sum(model.log_prob(batch_data, keep_dims=keep_dims)).item()
    idx = idx + batch_size
    nll = nll - cde_test
nll = nll/z_test.shape[0]
print(str(nll) + ',' + str(time.time() - t0))

