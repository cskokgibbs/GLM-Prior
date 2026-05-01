#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --output=./train_prior_network/slurm_output/%x_%j.out
#SBATCH --error=./train_prior_network/slurm_output/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=47:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|rtx8000
#SBATCH --mem=500GB
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@nyu.edu
#SBATCH --job-name=finetune_nt

CONDA_ENV=pmf-prior-network

RUN_NAME="tokenize_and_train"
OUTPUT_DIR="./train_prior_network/experiment_results/finetune_nt_${RUN_NAME}"

export WANDB_LOG_MODEL="false"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

if [ -z "$1" ]; then
    echo "Running regular training"
    python -m train_prior_network.finetune_nt \
        training_args.output_dir=${OUTPUT_DIR} \
        script_args.wandb_run_name=${RUN_NAME}
else
    echo "Running with additional arguments: $@"
    python -m train_prior_network.finetune_nt "$@"
fi
