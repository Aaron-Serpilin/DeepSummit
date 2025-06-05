#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C RTX2080Ti
#SBATCH -p proq
#SBATCH --gres=gpu:1

module load cuda10.2/toolkit

# (1) Ensure conda is in PATH (adjust if your system uses a different location):
source /var/scratch/ase347/anaconda3/etc/profile.d/conda.sh

# (2) Activate the `deepsummit` environment:
conda activate deepsummit

cd /var/scratch/ase347/DeepSummit
python test.py
