# GLM-Prior: A Transformer-Based Approach for Biologically Informed Prior-Knowledge
-----------

This repository contains the models, data, and scripts used in the GLM-Prior paper, which introduces a transformer-based nucleotide sequence classification approach for inferring transcription factor–target gene interaction priors.

GLM-Prior is the first stage in a **dual-stage training pipeline** that includes:

1. **GLM-Prior** – a fine-tuned genomic language model that predicts TF–gene interactions from nucleotide sequences.
2. **PMF-GRN** – a probabilistic matrix factorization model that performs GRN inference using prior knowledge from GLM-Prior.


![GLM-Prior](dual-stage-schematic.png)

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [GLM-Prior Pipeline (Stage 1)](#glm-prior-pipeline-stage-1)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [GRN Inference with PMF-GRN (Stage 2)](#grn-inference-with-pmf-grn-stage-2)

---

## 🌳 Environment Setup
GLM-Prior is designed to run within a **Singularity container** using a Conda environment. Follow the steps below to create and activate the environment:

### 1. Create Conda Environment
```
conda create -p /ext3/pmf-prior-network python=3.10 -y
```

### 2. Install Required Packages
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install hydra-core pandas datasets scikit-learn transformers wandb
```

### 3. Launch Singularity Container
```
singularity exec --nv --overlay overlay-15GB-500K.ext3:ro --bind local:$HOME/.local cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
```

### 4. Final Environment Installations
```
conda env update --prefix /ext3/pmf-prior-network --file environment_prior_network.yaml --prune
```

### 5. Activate the Environment
```
source /ext3/env.sh
conda activate /ext3/pmf-prior-network
```

## 🧬 Dataset Preparation
To prepare training data, use the notebooks provided in the `create_sequence_datasets/` directory. There are separate notebooks for yeast, mouse, and human.
Each notebook:
- Lists required downloads (FASTA files, GTF annotations, TF motifs)
- Walks through generating gene/TF sequences
- Saves output sequences and prior matrices for model input

Dataset Locations:
- Yeast: `data/yeast/`
- Mouse: `data/mouse/`
- Human: `data/human/`

For mouse and human reference datasets, download from [BEELINE](https://zenodo.org/records/3701939).

## GLM-Prior Training Pipeline (Stage 1)
To train the GLM-Prior model on DNA sequence input:
### 1. Edit configuration files with appropriate file paths and optimal hyperparameters for full training:
- `config/train_prior_network_pipeline.yaml`
- `config/prior_network/finetune_nt.yaml`
### 2. Set Singularity paths in `config/train_prior_network_pipeline.yaml`
- `prior_network_singularity_overlay` - overlay path for environment
- `prior_network_singularity_img` - path to Singularity image
### 3. Launch training pipeline:
`sbatch train_prior_network_pipeline.slurm TPN_pipeline`
This will launch dynamic slurm scripts to:
- Tokenize sequences
- Train GLM-Prior
- Perform inference on gene-TF pairs
- Evaluate predictions vs. a gold standard
- Save outputs to:
`output/<experiment_name>/prior_network_predictions.tsv`
`output/<experiment_name>/auprc_vs_gold.json`

## 🔍 Hyperparameter Sweep
To optimize GLM-Prior for a new dataset, run a hyperparameter sweep:
### 1. Modify sweep parameters:
Edit `./train_prior_network/finetune_nt_hp_sweep.sh` for:
- class weights
- learning rates
- downsampling rates
- gradient accumulation steps
Confirm paths and set `num_train_epochs: 1` in `config/prior_network/finetune_nt.yaml`

### 2. Submit sweep jobs
```
./train_prior_network/finetune_nt_hp_sweep.sh $USER /scratch/$USER/GLM-Prior/envs/overlay-15GB-500K.ext3 /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif ./train_prior_network/hp-sweep/
```
Each configuration will be submitted as an individual SLURM job. Weights & Biases will automatically track all sweeps. Select the configuration with the best F1 score and update your training config accordingly.

## 🧠 GRN Inference with PMF-GRN (Stage 2)
Binarized prior-knowledge can be used as input for the [PMF-GRN](https://github.com/nyu-dl/pmf-grn) model to perform full GRN inference.
- PMF-GRN takes this prior-knowledge matrix and single cell gene expression data to infer directed regulatory edges between TFs and their target genes 
