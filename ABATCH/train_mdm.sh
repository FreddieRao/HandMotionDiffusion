#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -N 1
#SBATCH -n 8 --mem=64g -p "gpu" --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o /users/rfu7/data/code/24Text2Action/MotionDiffusionModel/ABATCH/out/v5_SP_sall_step50_bs32_l200_split.out

CUDA_VISIBLE_DEVICES=0 source ~/.bashrc && conda init && conda activate /users/rfu7/data/anaconda/mdm_env && cd MotionDiffusionModel && \
    bash scripts/train_unconstrained_vanilla.sh

# 09 37
# 12 119
# 17 278
# 18 376
# 20 118
# 21 350?