#!/bin/bash

set -e

gpu=$1
seed=$2
tag=$3
dataset=$4
num_tasks=$5
nc_first_task=$6
network=$7
num_epochs=$8
lamb=$9

if [ "${dataset}" = "imagenet_subset_kaggle" ]; then
  clip=1.0
else
  clip=100.0
fi

exp_name="${tag}:lamb_${lamb}"
result_path="results/${tag}/lwf_${lamb}_${seed}"
python3 src/main_incremental.py \
  --datasets ${dataset} \
  --num-tasks ${num_tasks} \
  --nc-first-task ${nc_first_task} \
  --use-test-as-val \
  --seed ${seed} \
  --lr 0.1 \
  --clipping ${clip} \
  --nepochs ${num_epochs} \
  --batch-size 128 \
  --gpu ${gpu} \
  --log disk wandb \
  --exp-name ${exp_name} \
  --results-path ${result_path} \
  --tags ${tag} \
  --network ${network} \
  --approach lwf \
  --lamb ${lamb}
