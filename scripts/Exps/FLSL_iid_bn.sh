#!/bin/bash

#SBATCH --job-name=avg_i_0.01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_clean/scripts/slurm/fedavg_iid_0.01_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="FLSL"
dist="iid"
iid="True"
norm="bn"
gn="False"
agg="fedavg"
ema="0.01"
wandb_tag="$agg"_"$dist"_"ema"_"$ema"
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