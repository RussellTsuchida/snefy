# Modified version of https://github.com/VincentStimper/resampled-base-flows/blob/master/experiments/2d_toy_problems/rnvp_fkld.ipynb
# Minor adjustments to original: 
# (1) Do not provide the target distribution to the model (this
# is only a software change --- the mathematics is still the same).
# (2) Include the new SNEFY model
# (3) Remove manual seed setting
# (4) The number of flow layers K is now an argument (which is sometimes set to 0)
# (5) Remove the part which ignores nans and sets them to zero probability (!) for
# plotting. We still keep this part when evaluating the KL divergence.



# Import packages
import numpy as np
import torch
import math
import normflows as nf
import larsflow as lf

from matplotlib import pyplot as plt
from tqdm import tqdm
import time

from squared_neural_families.nets.integrate import SquaredNN
from squared_neural_families.nets.layers import Mlp 
from squared_neural_families.utils.plotters import matplotlib_config
from squared_neural_families.distributions.snf import Density, ConditionalDensity, AutoregressiveDensity

PLOT_BOOL = False
matplotlib_config()
# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function for model creation
def create_model(base='gauss', K=8):
    # Set up model

    # Define flows
    #torch.manual_seed(10)
    latent_size = 2
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        param_map = nf.nets.MLP([latent_size // 2, 32, 32, latent_size], init_zeros=True)
        flows += [nf.flows.AffineCouplingBlock(param_map)]
        flows += [nf.flows.Permute(latent_size, mode='swap')]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set prior and q0
    if base == 'resampled':
        a = nf.nets.MLP([latent_size, 256, 256, 1], output_fn="sigmoid")
        q0 = lf.distributions.ResampledGaussian(latent_size, a, 100, 0.1, trainable=False)
    elif base == 'gaussian_mixture':
        n_modes = 10
        q0 = nf.distributions.GaussianMixture(n_modes, latent_size, trainable=True,
                                              loc=(np.random.rand(n_modes, latent_size) - 0.5) * 5,
                                              scale=0.5 * np.ones((n_modes, latent_size)))
    elif base == 'gauss':
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=True)
    elif base == 'snefy':
        """
        squared_nn1 = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
            n = 50, dim=1)
        p1 = Density(squared_nn1)
        squared_nn2 = SquaredNN('Rd', 'gauss', 'snake', 'ident', d=1,
            n = 30, dim=0)
        feat_net = Mlp([2, 200, 200, 200, 30], mask_idx=0)
        p0given1 = ConditionalDensity(squared_nn2, feat_net)
        q0 = AutoregressiveDensity([p1, p0given1]).to(device)
        """
        squared_nn = SquaredNN('Rd', 'gauss', 'cos', 'ident', d=2, n=50) #50
        q0 = Density(squared_nn)
    else:
        raise NotImplementedError('This base distribution is not implemented.')

    # Construct flow model
    model = lf.NormalizingFlow(q0=q0, flows=flows)

    # Move model on GPU if available
    return model.to(device)


# Function to train model
def train(model, p, max_iter=20000, num_samples=2 ** 10, lr=1e-3, weight_decay=1e-3):
    # Do mixed precision training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    losses_train = -np.inf*np.ones(max_iter)
    losses_valid = -np.inf*np.ones(max_iter)
    losses_test = -np.inf*np.ones(max_iter)
    
    t0 = time.time()
    for it in tqdm(range(max_iter)):
        model.train()
        x = p.sample(num_samples).to(device)
        loss = model.forward_kld(x)

        loss.backward()
        optimizer.step()

        # Clear gradients
        nf.utils.clear_grad(model)
        model.eval()
        if isinstance(model.q0, Density):
            model.q0.update_partfun()

        # Compute training, validation and test statistics.
        # Training and testing are easy
        # Note that for resampled bases, the loss should be understood
        # as a thing to minimise/maximise but not an absolute measure
        # because there is error in estimating the normalising constant
        # This error can be quite significant when comparison with
        # other methods is important
        losses_train[it] = torch.mean(model.log_prob(x)).item()
        x = p.sample(num_samples).to(device)
        loss = torch.mean(model.log_prob(x))
        losses_valid[it] = loss.item()

        # For testing, resampled bases are more complicated.
        # Estimate Z if validation loss is small and using resampled base
        # Only bother estimating Z if we think the validation loss is good.
        # Otherwise, just skip this step to save time
        # Check the rough estimate of validation error every 100 epochs
        # Also check test error at last epoch
        low_valid = (losses_valid[it] >= np.nanmax(losses_valid[::100]))
        big_iter = (it > 0) and ((it % 100) == 0)
        if isinstance(model.q0, lf.distributions.ResampledGaussian):
            losses_test[it] = np.nan
            if (low_valid and big_iter) or (it+1 == max_iter):
                model.q0.estimate_Z(2**17, 2**10) # Taken from uci example
                x = p.sample(num_samples).to(device)
                loss = torch.mean(model.log_prob(x))
                losses_test[it] = loss.item()
                if losses_valid[it] >= np.nanmax(losses_valid[::100]):
                    best_idx = it
        else:
            x = p.sample(num_samples).to(device)
            loss = torch.mean(model.log_prob(x))
            losses_test[it] = loss.item()

        # Freeze hidden params if using SNEFY
        #if (it == 3000) and isinstance(model.q0, Density):
        #    model.q0.freeze_features()

    time_taken = time.time() - t0
    print('Model took ' + str(time_taken) + ' seconds to train.')
    if not isinstance(model.q0, lf.distributions.ResampledGaussian):
        best_idx = np.nanargmax(losses_valid)
    best_test = losses_test[best_idx]
    print('Log likelihood at best validation epoch: ' + str(best_test) + \
    ' at ' + str(best_idx))
    return losses_train, losses_valid, losses_test, best_test, time_taken



# Plot function
def plot_results(model, p, target=True, a=False, save=False, prefix=''):
    # Prepare z grid for evaluation
    grid_size = 300
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)
    
    log_prob = p.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    #prob[torch.isnan(prob)] = 0
    prob_target = prob.data.numpy()
    
    if target:
        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob_target)
        #cs = plt.contour(xx, yy, prob_target, [.025, .15, .7], colors='w', linewidths=3)#, linestyles='dashed')
        #for c in cs.collections:
        #    c.set_dashes([(0, (10.0, 10.0))])
        plt.gca().set_aspect('equal', 'box')
        plt.axis('off')
        if save:
            plt.savefig(prefix + 'target.png', dpi=300, bbox_inches='tight')
            plt.close()

    model.eval()
    log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)

    prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
    #prob[torch.isnan(prob)] = 0
    prob_model = prob.data.numpy()

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob_model)#, cmap=plt.get_cmap('coolwarm'))
    #cs = plt.contour(xx, yy, prob_model, [.025, .2, .35], colors='w', linewidths=3)#, linestyles='dashed')
    #for c in cs.collections:
    #    c.set_dashes([(0, (10.0, 10.0))])
    plt.gca().set_aspect('equal')
    plt.axis('off')
    if save:
        plt.savefig(prefix + 'model.png', dpi=300, bbox_inches='tight')
        plt.close()

    def log_prob_batch(density, zz, batch_size=512):
        log_prob = torch.zeros((0,)).to('cpu')
        for i in range( int(math.ceil(zz.shape[0]/batch_size))):
            zzi = zz[batch_size*i:min(batch_size*(i+1),zz.shape[0]),:]
            log_probi = model.log_prob(zzi).to('cpu').data
            log_prob = torch.hstack((log_prob, log_probi))
        return log_prob
    # Skip visualising the base distribution
    """
    #log_prob = model.q0.log_prob(zz).to('cpu').view(*xx.shape)
    log_prob = log_prob_batch(model.q0, zz, batch_size=512).view(*xx.shape)
    prob = torch.exp(log_prob)
    #prob[torch.isnan(prob)] = 0
    prob_base = prob.data.numpy()

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob_base)
    #cs = plt.contour(xx, yy, prob_base, [.025, .075, .135], colors='w', linewidths=3)#, linestyles='dashed')
    #for c in cs.collections:
    #    c.set_dashes([(0, (10.0, 10.0))])
    plt.gca().set_aspect('equal')
    plt.axis('off')
    if save:
        plt.tight_layout()
        plt.savefig(prefix + 'base.png', dpi=300, bbox_inches='tight')
        plt.close()
    if a:
        prob = model.q0.a(zz).to('cpu').view(*xx.shape)
        #prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(prefix + 'a.png', dpi=300, bbox_inches='tight')
            plt.close()
    """
    # Compute KLD. Unfortunately NVPs output nans, so need to hardcode them to
    # zero. Together with the epsilon on the denominator, this is kind of 
    # unprincipled, so we prefer to report test log likelihood
    #prob_model[torch.isnan(prob_model)] = 0
    """
    prob_model = np.nan_to_num(prob_model)
    eps = 1e-10
    kld = np.sum(prob_target * np.log((prob_target + eps) / (prob_model + eps)) * 6 ** 2 / grid_size ** 2)
    print('KL divergence: %f' % kld)

    # Compute log likelihood
    x = p.sample(10000).to(device)
    log_prob = torch.mean(log_prob_batch(model, x, batch_size=512))
    print('Average log likelihood: %f' % log_prob)
    """

# Train models
p = [nf.distributions.TwoMoons(), nf.distributions.CircularGaussianMixture(11),
nf.distributions.RingMixture()]
name = ['moons', 'circle', 'rings']
NUM_LAYERS = 3

for i in range(len(p)):
    """
    print('SNEFY base only:')
    model = create_model('snefy', K=0)
    losstr,lossv,losst,bt1 = train(model, p[i])
    if PLOT_BOOL:
        # Plot and save results
        plt.plot(losstr)
        plt.plot(lossv)
        plt.plot(losst)
        prefix = 'examples/outputs/2d_distributions/'+ name[i] + '_snefy_'
        plt.savefig(prefix + 'loss.png')
        plt.close()
        plot_results(model, p[i], save=True, a=False,
                     prefix=prefix)
    """

    print('SNEFY base:')
    model = create_model('snefy', K=NUM_LAYERS)
    losstr,lossv,losst,bt2,time_taken2= train(model, p[i])
    num_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if PLOT_BOOL: 
        # Plot and save results
        plt.plot(losstr)
        plt.plot(lossv)
        plt.plot(losst)
        prefix = 'examples/outputs/2d_distributions/'+ name[i] + '_snefynvp_'
        plt.savefig(prefix + 'loss.png')
        plt.close()
        plot_results(model, p[i], save=True, a=False,
                     prefix=prefix)

    """
    # Train model with (only) resampled base distribution
    print('Resampled base only:')
    model = create_model('resampled', K=0)
    losstr,lossv,losst,bt3 = train(model, p[i])
    if PLOT_BOOL: 
        plt.plot(losstr)
        plt.plot(lossv)
        plt.scatter(np.arange(len(losst)), losst, s=100, c='g', zorder=10)
        prefix = 'examples/outputs/2d_distributions/'+ name[i] + '_resampled_'
        plt.savefig(prefix + 'loss.png')
        plt.close()
        # Plot and save results
        plot_results(model, p[i], save=True, a=False,
                     prefix=prefix)
    """

    # Train model with resampled base distribution
    print('Resampled base:')
    model = create_model('resampled', K=NUM_LAYERS)
    losstr,lossv,losst,bt4,time_taken4 = train(model, p[i])
    num_params4 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if PLOT_BOOL: 
        plt.plot(losstr)
        plt.plot(lossv)
        plt.scatter(np.arange(len(losst)), losst)
        prefix = 'examples/outputs/2d_distributions/'+ name[i] + '_rnvp_'
        plt.savefig(prefix + 'loss.png')
        plt.close()
        # Plot and save results
        plot_results(model, p[i], save=True, a=False,
                     prefix=prefix)



    # Train model with Gaussain base distribution
    print('Gauss base:')
    model = create_model('gauss', K=NUM_LAYERS)
    losstr,lossv,losst,bt5,time_taken5 = train(model, p[i])
    num_params5 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if PLOT_BOOL: 
        plt.plot(losstr)
        plt.plot(lossv)
        plt.plot(losst)
        prefix = 'examples/outputs/2d_distributions/'+ name[i] + '_gauss_'
        plt.savefig(prefix + 'loss.png')
        plt.close()
        # Plot and save results
        plot_results(model, p[i], save=True,
                     prefix=prefix)
    
    # Train model with mixture of Gaussians base distribution
    print('GMM base:')
    model = create_model('gaussian_mixture', K=NUM_LAYERS)
    losstr,lossv,losst,bt6,time_taken6 = train(model, p[i])
    num_params6 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if PLOT_BOOL: 
        plt.plot(losstr)
        plt.plot(lossv)
        plt.plot(losst)
        prefix = 'examples/outputs/2d_distributions/'+ name[i] + '_gmm_'
        plt.savefig(prefix + 'loss.png')
        plt.close()
        # Plot and save results
        plot_results(model, p[i], save=True,
                     prefix=prefix)

    result = ''
    times = ''
    param_counts = ''
    for bt in [bt2, bt4, bt5, bt6]:
        result = result + str(bt) + ','
    print('RESULTS:')
    print(result)

    for bt in [num_params2, num_params4, num_params5, num_params6]:
        param_counts = param_counts + str(bt) + ','
    print('PARAMETER COUNTS:')
    print(param_counts)

    for bt in [time_taken2, time_taken4, time_taken5, time_taken6]:
        times = times + str(bt) + ','
    print('TIMES:')
    print(times)


