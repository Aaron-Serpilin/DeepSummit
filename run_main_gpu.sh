#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -C RTX2080Ti
#SBATCH -p proq
#SBATCH --gres=gpu:1

module load cuda10.1/toolkit/10.1.243

cd /var/scratch/ase347/DeepSummit

srun --exclusive -N1 -n1 /var/scratch/ase347/anaconda3/envs/deepsummit/bin/python src/scripts/train_saint.py &

srun --exclusive -N1 -n1 /var/scratch/ase347/anaconda3/envs/deepsummit/bin/python src/scripts/train_stormer.py &

wait