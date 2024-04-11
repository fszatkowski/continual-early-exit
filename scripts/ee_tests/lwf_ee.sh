#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
n_epochs=200
tag="test"
approach='lwf'

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-layers layer1.2 layer1.4 layer2.1 layer2.3 layer3.0 layer3.2 \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv \
    --ic-weighting sdn \
    --input-size 3 32 32 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb 1 \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ee/seed${seed} \
    --exp-name ee_${tag} \
    --save-models \
    --tags ${tag}
done

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-layers layer1.2 layer1.4 layer2.1 layer2.3 layer3.0 layer3.2 \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv \
    --ic-weighting uniform \
    --input-size 3 32 32 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb 1 \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ee_uniform/seed${seed} \
    --exp-name ee_${tag} \
    --save-models \
    --tags ${tag}
done

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-layers layer1.1 layer1.4 layer2.1 layer2.3 layer3.0 layer3.2 \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv \
    --ic-weighting sdn \
    --input-size 3 32 32 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb 1 \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ee_alt_placement/seed${seed} \
    --exp-name ee_${tag} \
    --save-models \
    --tags ${tag}
done

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-layers layer1.1 layer1.4 layer2.1 layer2.3 layer3.0 layer3.2 \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv \
    --ic-weighting uniform \
    --input-size 3 32 32 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb 1 \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ee_alt_placement_uniform/seed${seed} \
    --exp-name ee_${tag} \
    --save-models \
    --tags ${tag}
done

