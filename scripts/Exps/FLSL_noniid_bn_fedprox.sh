#!/bin/bash

#SBATCH --job-name=prox_ni_0.01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_clean/scripts/slurm/prox_noniid_0.01_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="FLSL"
dist="noniid"
iid="False"
norm="bn"
gn="False"
agg="fedprox"
ema="0.01"
wandb_tag=prox_"$agg"_"$dist"_"ema"_"$ema"
ckpt_path=./checkpoints/"$exp"_"$dist"_"$norm"_"$agg".pth.tar
cd /home/kwangyeongill/FedSSL_clean/ && python main.py \
                                        --parallel True \
                                        --group_norm $gn \
                                        --exp $exp \
                                        --iid $iid \
                                        --agg $agg \
                                        --wandb_tag $wandb_tag \
                                        --ckpt_path $ckpt_path \
                                        --bn_stat_momentum $ema