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
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@nyu.edu
#SBATCH --job-name=finetune_nt

# Check if USER, Singularity overlay, and Singularity image are provided as arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: USER, Singularity overlay, or Singularity image not provided."
    echo "Usage: sbatch unified_finetune_nt.slurm <USER> <singularity_overlay> <singularity_img> [additional_hyperparameters] [--eval]"
    exit 1
fi

USER=$1
SINGULARITY_OVERLAY=$2
SINGULARITY_IMG=$3
CONDA_ENV=pmf-prior-network

# Default parameters for a regular run
RUN_NAME="tokenize_and_train"
OUTPUT_DIR="./train_prior_network/experiment_results/finetune_nt_${RUN_NAME}"
HUGGINGFACE_REPO="cskokgibbs/pretokenized_data_yeast"

export WANDB_LOG_MODEL="false"
# Shift arguments to remove the first three arguments (USER, Singularity overlay, and Singularity IMG)
shift 3

# Check if there are any additional arguments passed (for hyperparameter sweep)
if [ -z "$1" ]; then
    # No additional arguments, run the regular training script using config file
    echo "Running regular training with USER: ${USER} and Singularity overlay: ${SINGULARITY_OVERLAY}"

    singularity exec --nv --overlay ${SINGULARITY_OVERLAY}:ro ${SINGULARITY_IMG} /bin/bash -c "
    source /ext3/env.sh
    conda deactivate && conda activate ${CONDA_ENV}
    python -m train_prior_network.finetune_nt training_args.output_dir=${OUTPUT_DIR} script_args.wandb_run_name=${RUN_NAME}
    "

else
     # Additional arguments provided, assume this is for hyperparameter sweep
    echo "Running hyperparameter sweep with USER: ${USER}, Singularity overlay: ${SINGULARITY_OVERLAY}, and arguments: $@"

    singularity exec --nv --overlay ${SINGULARITY_OVERLAY}:ro ${SINGULARITY_IMG} /bin/bash -c "
    source /ext3/env.sh
    conda deactivate && conda activate ${CONDA_ENV}
    python -m train_prior_network.finetune_nt $@
    "
fi

