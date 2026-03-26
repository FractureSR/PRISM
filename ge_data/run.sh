#!/bin/bash
#SBATCH -J DATA
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PART=${PART:-llama2-70b}
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-/mnt/data2/ytchen/cache/${PART}}
HDFS_OUTPUT_ROOT=${HDFS_OUTPUT_ROOT:-/mnt/data2/ytchen/cache/hdfs}
SAMPLES_PER_FILE=${SAMPLES_PER_FILE:-2}
MOVE_EVERY_FILES=${MOVE_EVERY_FILES:-10}

# DATA_DIR=/mnt/inaisfs/data/home/liuf_criait/data
DATASET_DIR=/mnt/data1/ytchen/cache/prism-datasets/dataset/

if [ -z "${SCRIPT_NAME:-}" ]; then
    if [ "${PART}" = "llama3-8b" ]; then
        SCRIPT_NAME=ge_data_all_llama3.py
    else
        SCRIPT_NAME=ge_data_all_llama2chat.py
    fi
fi
SCRIPT=${SCRIPT_DIR}/${SCRIPT_NAME}

MODEL_DIR=/mnt/data1/ytchen/cache/Llama-2-70b-chat-hf/

echo "start time: $(date)"

python "${SCRIPT_DIR}/allocation.py" \
    --script "${SCRIPT}" \
    --local_outdir "${LOCAL_OUTPUT_ROOT}" \
    --hdfs_outdir "${HDFS_OUTPUT_ROOT}" \
    --data_path "${DATASET_DIR}/ShareGPT_V4.3_unfiltered_cleaned_split.json" \
    --model_path "${MODEL_DIR}" \
    --dataset_name ShareGPT \
    --num_rows 68000 \
    --samples_per_file "${SAMPLES_PER_FILE}" \
    --move_every_files "${MOVE_EVERY_FILES}"

python "${SCRIPT_DIR}/allocation.py" \
    --script "${SCRIPT}" \
    --local_outdir "${LOCAL_OUTPUT_ROOT}" \
    --hdfs_outdir "${HDFS_OUTPUT_ROOT}" \
    --data_path "${DATASET_DIR}/ultrachat_200k" \
    --model_path "${MODEL_DIR}" \
    --dataset_name UltraChat \
    --num_rows 463000 \
    --samples_per_file "${SAMPLES_PER_FILE}" \
    --move_every_files "${MOVE_EVERY_FILES}"

python "${SCRIPT_DIR}/allocation.py" \
    --script "${SCRIPT}" \
    --local_outdir "${LOCAL_OUTPUT_ROOT}" \
    --hdfs_outdir "${HDFS_OUTPUT_ROOT}" \
    --data_path "${DATASET_DIR}/OpenThoughts2-1M" \
    --model_path "${MODEL_DIR}" \
    --dataset_name OpenThoughts2 \
    --num_rows 269000 \
    --samples_per_file "${SAMPLES_PER_FILE}" \
    --move_every_files "${MOVE_EVERY_FILES}"

echo "end time: $(date)"
