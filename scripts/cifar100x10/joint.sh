#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
n_epochs=200
tag="cifar100x10"
approach='joint'

for seed in 0 1 2; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}/seed${seed} \
    --exp-name ${tag} \
    --save-models \
    --tags ${tag}
done
