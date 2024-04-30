#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=5
n_epochs=200
tag="tiny200x5"
approach='ssil'
lamb=0.25
num_exemplars=2000

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --num-workers 0 \
    --seed ${seed} \
    --network resnet18 \
    --ic-layers layer1.0 layer1.1 layer2.0 layer2.1 layer3.0 layer3.1 layer4.0 \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv \
    --ic-weighting sdn \
    --input-size 3 64 64 \
    --datasets tiny_imnet \
    --num-workers 0 \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --lamb ${lamb} \
    --log disk wandb \
    --results-path ./results/Tiny200x${num_tasks}/${approach}_ex_${num_exemplars}_lamb_${lamb}_ee/seed${seed} \
    --exp-name ee_${tag} \
    --save-models \
    --tags ${tag}
done
