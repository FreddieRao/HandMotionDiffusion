#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -N 1
#SBATCH -n 8 --mem=64g -p "3090-gcondo" --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o /users/rfu7/data/code/24Text2Action/MotionDiffusionModel/ABATCH/out/unconstrained_vanilla_v9_SP_ANNO_l1000.out

CUDA_VISIBLE_DEVICES=0 source ~/.bashrc && conda init && conda activate /users/rfu7/data/anaconda/mdm_env && cd MotionDiffusionModel && \
    bash scripts/train_unconstrained_vanilla.sh