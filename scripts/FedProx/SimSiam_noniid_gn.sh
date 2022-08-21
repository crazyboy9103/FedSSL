#!/bin/bash

#SBATCH --job-name=simsiam_noniid_gn_fedprox
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_clean/scripts/slurm/simsiam_noniid_gn_fedprox_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="simsiam"
dist="noniid"
iid="False"
norm="gn"
gn="True"
agg="fedprox"

wandb_tag="$exp"_"$dist"_"$norm"_"$agg"
ckpt_path=./checkpoints/"$exp"_"$dist"_"$norm"_"$agg".pth.tar
cd /home/kwangyeongill/FedSSL_clean/ && python main.py \
                                        --parallel True \
                                        --group_norm $gn \
                                        --exp $exp \
                                        --iid $iid \
                                        --agg $agg \
                                        --wandb_tag $wandb_tag \
                                        --ckpt_path $ckpt_path
