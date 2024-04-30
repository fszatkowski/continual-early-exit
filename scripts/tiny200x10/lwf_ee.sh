#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
n_epochs=200
tag="cifar100x5"
approach='lwf'
lamb=0.5

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --ic-layers layer1.0 layer1.1 layer2.0 layer2.1 layer3.0 layer3.1 layer4.0 \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv \
    --ic-weighting sdn \
    --input-size 3 64 64 \
    --datasets tiny_imnet \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb ${lamb} \
    --log disk wandb \
    --results-path ./results/Tiny200x${num_tasks}/${approach}_tw_lamb_${lamb}_ee/seed${seed} \
    --exp-name ee_${tag} \
    --save-models \
    --tags ${tag}
done
