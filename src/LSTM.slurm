#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=0:15:00

source ../.venv/bin/activate

python train_torch.py -e 10 -c ../configs/LSTM_SKL.txt