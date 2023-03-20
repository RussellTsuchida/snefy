#!/bin/bash -l

#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --account=OD-217715

module load python
module load texlive/2021
module load pytorch
module load torchvision
source env/bin/activate

python -u crimes.py
#python -u ppp_torch.py

