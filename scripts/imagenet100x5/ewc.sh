#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=5
n_epochs=200
tag="imagenet100x5"
approach='ewc'
lamb=10000
alpha=0.5

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --lamb ${lamb} \
    --alpha ${alpha} \
    --log disk wandb \
    --results-path ./results/ImageNet100x${num_tasks}/${approach}_lamb_${lamb}_alpha_${alpha}/seed${seed} \
    --exp-name ${tag} \
    --save-models \
    --tags ${tag}
done
