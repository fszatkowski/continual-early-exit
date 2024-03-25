#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
num_epochs=200
dataset=imagenet_subset_kaggle
network=resnet18
tag=imagenet_subset_t${num_tasks}s${nc_first_task}

lamb=10

for seed in 0 1 2; do
  ./experiments/lwf.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} &
done
