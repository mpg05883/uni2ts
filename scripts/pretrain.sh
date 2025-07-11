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

mkdir -p logs

# Load helper functions
source ./scripts/utils.sh

delete_lightning_logs
delete_wandb_logs

# Load conda environment
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate uni2ts

# Redirect wandb cache
export WANDB_CACHE_DIR=/tmp

log_job_info

python -m cli.train \
  -cp conf/pretrain \
  run_name=first_run \
  model=moirai_small \
  data=lotsa_v1_weighted