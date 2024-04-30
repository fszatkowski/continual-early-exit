#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
n_epochs=200
tag="tiny200x10"
approach='gdumb'
num_exemplars=2000

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --datasets tiny_imnet \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --log disk wandb \
    --results-path ./results/Tiny200x${num_tasks}/${approach}_ex${num_exemplars}/seed${seed} \
    --exp-name ${tag} \
    --save-models \
    --tags ${tag}
done
