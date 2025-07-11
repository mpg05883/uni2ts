#!/bin/bash

# Delta partition information:
# https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html#delta-partitions-queues

#SBATCH --job-name=pretrain_moirai_small
#SBATCH --partition=gpuA40x4  
#SBATCH --mem=120GB  # Total memory per node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1  # * Ensure --gpus-per-node equals --ntasks-per-node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x/out/%A.out
#SBATCH --error=logs/%x/err/%A.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

python -m cli.eval \
  run_name=example_eval_2 \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96