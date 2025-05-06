#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=GLM_prior_pipeline
#SBATCH --mail-type=END
#SBATCH --export=NONE
#SBATCH --mem=50GB
#SBATCH --time=147:00:00
#SBATCH --output=./${1}/slurm_output/%x_%j.out
#SBATCH --error=./${1}/slurm_output/%x_%j.err
#SBATCH --mail-user=%u@nyu.edu

# args 1: name of working dir

source /scratch/work/public/singularity/greene-ib-slurm-bind.sh

SINGULARITY_OVERLAY="/scratch/${USER}/envs/overlay-25GB-500K.ext3"
SINGULARITY_IMG="/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
WORKING_DIR="${1}/experiment_results/"
SLURM_OUTPUT_DIR="$(dirname "$(pwd)")/slurm_output"
singularity exec --overlay ${SINGULARITY_OVERLAY}:ro ${SINGULARITY_IMG} /bin/bash -c "
source /ext3/env.sh
conda deactivate && conda activate /ext3/pmf-grn
python -m train_prior_network_pipeline working_dir=${WORKING_DIR} slurm_output_dir=${SLURM_OUTPUT_DIR} email=${USER}@nyu.edu || exit
"
