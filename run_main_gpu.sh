#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C RTX2080Ti
#SBATCH -p proq
#SBATCH --gres=gpu:1

module load cuda10.1/toolkit/10.1.243

cd /var/scratch/ase347/DeepSummit/scripts
/var/scratch/ase347/anaconda3/envs/deepsummit/bin/python train_stormer.py