# Squared Neural Families (SNEFY)

PyTorch implementation of [squared neural families](https://arxiv.org/abs/2305.13552).

## Installation
Clone the repo, CD into the downloaded directory, and then install the package:
```
python -m pip install --upgrade pip
python -m  pip install -r requirements.txt
python -m pip install .
```

<table>
<tr>
<td>

A $\texttt{SNEFY}_{\mathbb{S}^2, \text{Id}, \exp, U(\mathbb{S}^2)}$ density model training for the first $1000$ epochs  (log scale). Here the subscript $\mathbb{S}^2$ (the sphere) denotes the support, $\text{Id}$ denotes the identity sufficient statistic, $\exp$ denotes the exponential activation function, and $U(\mathbb{S}^2)$ means the uniform distribution on the sphere. This example comes from `examples/galaxy.py`.

</td>
<td>

![animation](https://github.com/RussellTsuchida/snefy/assets/28694114/907cba96-4809-4bc8-9aa3-c493c9bea996)

</td>
</tr>
</table>



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
