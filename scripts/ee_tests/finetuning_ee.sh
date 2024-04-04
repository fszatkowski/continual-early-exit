#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

seed=1221
num_tasks=10
n_epochs=200
tag="test"
approach='finetuning'

python src/main_incremental.py \
  --gpu 0 \
  --seed ${seed} \
  --network resnet32 \
  --ic-layers layer1.1 layer1.3 layer2.1 layer2.3 layer3.0 layer3.2 \
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
  --log disk wandb \
  --results-path ./results/CIFAR100/${approach}_ee/${num_tasks}splits/seed${seed} \
  --exp-name ee_${tag} \
  --save-models \
  --tags ${tag}
