#!/bin/bash

declare -a all_num_topics=(2 4 8 16 32 64 128 256)

for num_topics in "${all_num_topics[@]}"; do
  sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=32G -t 24:00:00 ./fedavg_etm_script.sh ${num_topics}
done