# GLM-Prior: a transformer-based nucleotide sequence classification model for inference of prior-knowledge GRNs. 
-----------

This repository and its references contain the models, data, and scripts used to perform the experiments in the 
GLM-Prior paper.

![GLM-Prior](dual-stage-schematic.png)

------------
## Installation Steps


## Train Prior Network Pipeline
To train a prior network from DNA sequences corresponding to transcription factors and target genes in a species or cell line of interest:
Step 1: Specify correct file paths in `config/train_prior_network_pipeline.yaml` and `config/prior_network/finetune_nt.yaml`
Step 2: Change the values for the singularity options in `config/train_prior_network_pipeline.yaml`. The options `prior_network_singularity_overlay` and `prior_network_singularity_img` should correspond to the Conda environment `pmf-prior-network`.
Step 3: Run the pipeline using the following command: `sbatch train_prior_network_pipeline.slurm TPN_pipeline`, where the second argument will specify which directory the experiment should be saved into.

