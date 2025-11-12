#!/bin/bash
#SBATCH -J LD_train_model
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8

set -x

nvcc -V
python -V

export PYTHONPATH=$(pwd):${PYTHONPATH}
export WANDB_API_KEY=05ac0c7fac19bec004160369c32723326fa8a618

PROJECT=LD-llama3-8b
NAME=LD-3-100k

BASE_PATH=/mnt/inaisfs/data/home/liuf_criait/data/model/Llama-3-8B-Instruct
CONFIG_PATH=train/llama3-8b/LD-3_config.json

echo "start time: $(date)"

accelerate launch train/main_LD.py \
    --project ${PROJECT} \
    --name ${NAME} \
    --basepath ${BASE_PATH} \
    --tmpdir ge_data/llama3-8b \
    --cpdir checkpoints/llama3-8b/${NAME} \
    --configpath ${CONFIG_PATH} \
    --epoch 8 \
    --bs 1 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 100000 \
    --lr 1e-5 \
    --train_LD \
    --hass_path checkpoints/llama3-8b/HASS-1-100k/state_39/pytorch_model.bin \
    --v_w 0

echo "end time: $(date)"
