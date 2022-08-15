#!/bin/bash

#SBATCH --job-name=simsiam_iid_bn
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_clean/scripts/slurm/name_%x_id_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL
cd /home/kwangyeongill/FedSSL_clean/ && python main.py --parallel True --group_norm False --exp simsiam --iid True --wandb_tag simsiam_iid_bn
