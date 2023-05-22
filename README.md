# Squared Neural Families (SNEFY)

## Installation
Clone the repo, CD into the downloaded directory, and then install the package:
`python -m pip install --upgrade pip`
`python -m  pip install -r requirements.txt`
`python -m pip install .`

## Usage
The model is easily instantiated in two steps:
1. Define a squaredNN using the `squared_neural_families.nets.integrate.SquaredNN` class
2. Define a density model using `squared_neural_families.distributions.snf.Density` or `squared_neural_families.distributions.snf.ConditionalDensity` 

For example, the following two lines initialise an (unconditional) density model on the d-dimensional hypersphere with a uniform base measure, exponential activations, identity sufficient statistic, n hidden units, m output units and an unconstrained $\mathbf{V}$ matrix:
```
squared_nn = SquaredNN('sphere', 'uniformsphere', 'exp', 'ident', d=d, n=n,
    m=m, diagonal_V=False)
snf = Density(squared_nn).to(device)
```
The model is compatible with regular PyTorch code (e.g, it can be optimised using the usual kind of training loop for a neural net). E.g.
```
for epoch in range(20000):
    optimizer.zero_grad()
    loss = -1*torch.mean(snf.log_prob(data_train))
    loss.backward()
    optimizer.step()
```
## Examples
We provide three examples:
1. `2d_toy.py` shows the classic moons, circles and rings datasets using a distribution defined on $\mathbb{R}^2$. This example also shows how SNEFY can be used as base distributions in normalising flows.
2. `teddy.py` shows the z-photometric redshift example for conditional density estimation on $\mathbb{R}^2$. This example shows how deep neural network feature extractors can be used as conditioning variables inside SNEFY.
3. `galaxy.py` shows density estimation on the sphere $\mathbb{S}^2$. It uses a dataset of galaxies, as referenced in the paper.
