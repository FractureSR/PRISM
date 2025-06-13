#!/bin/bash

#SBATCH -p vip_gpu_01
#SBATCH --gpus=1

module load anaconda/2024.10 cuda/12.1
source activate LD

python -V
nvcc -V

export PYTHONPATH=$(pwd):${PYTHONPATH}

PROJECT=llama2-7b
NAME=LD-800k

EA_MODEL_DIR=/home/dalhxwlyjsuo_20T/LD_checkpoints/${PROJECT}/${NAME}
EA_CONFIG_PATH=train/llama2-7b/LD_config.json
BASE_MODEL_PATH=/home/dalhxwlyjsuo/criait_liuf/wxl_model/Llama-2-7b-chat-hf

echo "start time: $(date)"

for iter in {0..15}
do
  echo "iter: ${iter}"

  EA_MODEL_PATH=${EA_MODEL_DIR}/state_${iter}
  cp ${EA_CONFIG_PATH} ${EA_MODEL_PATH}/config.json

  for bench_name in "mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa"
  do
    echo "bench_name: ${bench_name}"

    for temperature in "0.0" "1.0"
    do
      echo "temperature: ${temperature}"

      CUDA_VISIBLE_DEVICES=0 python evaluation/gen_ea_answer_llama2chat.py \
        --ea-model-path ${EA_MODEL_PATH} \
        --base-model-path ${BASE_MODEL_PATH} \
        --model-id ${PROJECT}/${NAME} \
        --bench-name ${bench_name} \
        --total-token 60 \
        --depth 5 \
        --top-k 10 \
        --temperature ${temperature}

      CUDA_VISIBLE_DEVICES=0 python evaluation/gen_baseline_answer_llama2chat.py \
        --ea-model-path ${EA_MODEL_PATH} \
        --base-model-path ${BASE_MODEL_PATH} \
        --model-id ${PROJECT}/Naive \
        --bench-name ${bench_name} \
        --temperature ${temperature}

      python evaluation/summary.py \
        --model_path ${BASE_MODEL_PATH} \
        --baseline_json ${bench_name}/${PROJECT}/Naive-temperature-${temperature}.jsonl \
        --LD_json ${bench_name}/${PROJECT}/${NAME}-temperature-${temperature}.jsonl
    done
  done
done

echo "end time: $(date)"
