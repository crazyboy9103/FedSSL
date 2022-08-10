#!/bin/bash

#SBATCH --job-name=fl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=32000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_clean/scripts/slurm/R-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL
cd /home/kwangyeongill/FedSSL_clean/ && python main.py --parallel True --group_norm True