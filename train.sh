#!/bin/bash

#SBATCH -p vip_gpu_01
#SBATCH --gpus=8

module load anaconda/2024.10 cuda/12.1
source activate LD

set -x

python -V
nvcc -V

export PYTHONPATH=$(pwd):${PYTHONPATH}
export WANDB_API_KEY=05ac0c7fac19bec004160369c32723326fa8a618

PROJECT=LD-llama2-7b
NAME=Eagle2-800k

LARGE_PATH=/home/dalhxwlyjsuo_20T
BASE_PATH=/home/dalhxwlyjsuo/criait_liuf/wxl_model/Llama-2-7b-chat-hf

echo "start time: $(date)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision=bf16 train/main_LD.py \
    --project ${PROJECT} \
    --name ${NAME} \
    --basepath ${BASE_PATH} \
    --tmpdir ${LARGE_PATH}/LD_train_data/llama2-7b \
    --cpdir ${LARGE_PATH}/LD_checkpoints/llama2-7b/${NAME} \
    --configpath train/LD_llama_2_7B_config.json \
    --epoch 15 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 1 \
    --data_num 800000

echo "end time: $(date)"
