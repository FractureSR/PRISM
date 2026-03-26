#!/bin/bash
#SBATCH -J DATA
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8

SCRIPT=ge_data_all_llama2chat.py

DATA_DIR=/mnt/inaisfs/data/home/liuf_criait/data
DATASET_DIR=${DATA_DIR}/dataset
MODEL_DIR=${DATA_DIR}/model/Llama-2-70b-chat-hf

# Output directories (each dataset has its own)
SHAREGPT_OUTDIR=llama2-70b/ShareGPT
ULTRACHAT_OUTDIR_PART1=llama2-70b/UltraChat_part1
ULTRACHAT_OUTDIR_PART2=llama2-70b/UltraChat_part2
OPENTHOUGHTS_OUTDIR=llama2-70b/OpenThoughts2

# UltraChat split configuration (adjust ULTRACHAT_SPLIT to change the ratio)
ULTRACHAT_TOTAL=463000
ULTRACHAT_SPLIT=231500

echo "start time: $(date)"

# ShareGPT
python allocation.py \
    --script ${SCRIPT} \
    --outdir ${SHAREGPT_OUTDIR} \
    --data_path ${DATASET_DIR}/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --model_path ${MODEL_DIR} \
    --dataset_name ShareGPT \
    --start_row 0 \
    --end_row 68000

# UltraChat - part 1
python allocation.py \
    --script ${SCRIPT} \
    --outdir ${ULTRACHAT_OUTDIR_PART1} \
    --data_path ${DATASET_DIR}/ultrachat_200k \
    --model_path ${MODEL_DIR} \
    --dataset_name UltraChat \
    --start_row 0 \
    --end_row ${ULTRACHAT_SPLIT}

# UltraChat - part 2
python allocation.py \
    --script ${SCRIPT} \
    --outdir ${ULTRACHAT_OUTDIR_PART2} \
    --data_path ${DATASET_DIR}/ultrachat_200k \
    --model_path ${MODEL_DIR} \
    --dataset_name UltraChat \
    --start_row ${ULTRACHAT_SPLIT} \
    --end_row ${ULTRACHAT_TOTAL}

# OpenThoughts2
python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OPENTHOUGHTS_OUTDIR} \
    --data_path ${DATASET_DIR}/OpenThoughts2-1M \
    --model_path ${MODEL_DIR} \
    --dataset_name OpenThoughts2 \
    --start_row 0 \
    --end_row 269000

echo "end time: $(date)"
