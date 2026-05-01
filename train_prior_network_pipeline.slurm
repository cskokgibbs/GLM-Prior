#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=GLM_prior_pipeline
#SBATCH --mail-type=END
#SBATCH --export=NONE
#SBATCH --mem=50GB
#SBATCH --time=147:00:00
#SBATCH --output=./TPN/slurm_output/%x_%j.out
#SBATCH --error=./TPN/slurm_output/%x_%j.err
#SBATCH --mail-user=%u@nyu.edu

# args 1: name of working dir

CONDA_ENV=pmf-prior-network
WORKING_DIR="${1}/experiment_results/"

source ~/miniforge3/bin/activate ${CONDA_ENV}

python -m train_prior_network_pipeline \
  working_dir=${WORKING_DIR} \
  email=${USER}@nyu.edu
