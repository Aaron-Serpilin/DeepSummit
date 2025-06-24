#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C RTX2080Ti
#SBATCH -p proq
#SBATCH --gres=gpu:1

#SBATCH --output=/var/scratch/ase347/DeepSummit/runs/slurm_logs/%x_%j.out
#SBATCH --error=/var/scratch/ase347/DeepSummit/runs/slurm_logs/%x_%j.err

module load cuda10.1/toolkit/10.1.243

cd /var/scratch/ase347/DeepSummit
/var/scratch/ase347/anaconda3/envs/deepsummit/bin/python src/scripts/train_saint.py

