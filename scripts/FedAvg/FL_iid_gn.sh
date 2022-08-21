#!/bin/bash

#SBATCH --job-name=FL_iid_gn_fedavg
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_clean/scripts/slurm/FL_iid_gn_fedavg_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="FL"
dist="iid"
iid="True"
norm="gn"
gn="True"
agg="fedavg"

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
