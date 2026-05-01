#!/bin/bash
#SBATCH --job-name=glm-prior-train
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --export=NONE
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=h200
#SBATCH --mem=1500GB
#SBATCH --partition=gpuxl
#SBATCH --reservation=rocky9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@nyu.edu

set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pmf-prior-network-rusty9

cd "$ROOT"

export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node 8 -m train_prior_network.finetune_nt "$@"
