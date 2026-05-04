#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# module load Python/3.11.3

source ../.venv/bin/activate

srun python train_MPI.py -c ../configs/RF_SKL.txt