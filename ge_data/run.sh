#!/bin/bash

#SBATCH -p vip_gpu_01
#SBATCH --gpus=4

module load cuda/12.1
nvcc --version

module load anaconda/2024.10
source activate LD
python --version

#export TOKENIZERS_PARALLELISM=false

SCRIPT=ge_data_all_llama2chat.py

OUTPUT_DIR=/home/dalhxwlyjsuo_20T/LD_train_data/llama2-7b
DATA_DIR=/home/dalhxwlyjsuo/criait_liuf/zmc_data
MODEL_DIR=/home/dalhxwlyjsuo/criait_liuf/wxl_model/Llama-2-7b-chat-hf

echo "start time: $(date)"

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATA_DIR}/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --model_path ${MODEL_DIR} \
    --dataset_name ShareGPT \
    --num_rows 68000 \
    --num_gpus 4

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATA_DIR}/ultrachat_200k \
    --model_path ${MODEL_DIR} \
    --dataset_name UltraChat \
    --num_rows 463000 \
    --num_gpus 4

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATA_DIR}/OpenThoughts2-1M \
    --model_path ${MODEL_DIR} \
    --dataset_name OpenThoughts2 \
    --num_rows 1143000 \
    --num_gpus 4

echo "end time: $(date)"
