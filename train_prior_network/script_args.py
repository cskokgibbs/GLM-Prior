from dataclasses import dataclass, field

from typing import Optional


@dataclass
class ScriptArguments:
    """
    Extra arguments not already in HF TrainingArguments.
    """

    # pipeline orchestration
    overwrite: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, re-run steps even if their outputs already exist."},
    )
    conda_env: Optional[str] = field(
        default="pmf-prior-network",
        metadata={"help": "Conda environment name to activate in SLURM jobs."},
    )
    inference_chunk_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of genes per inference SLURM array task."},
    )
    num_gpus: Optional[int] = field(
        default=1,
        metadata={"help": "Number of GPUs for training. >1 launches with torchrun (DDP)."},
    )
    email: Optional[str] = field(
        default=None,
        metadata={"help": "Email address for SLURM job notifications."},
    )
    slurm_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for SLURM stdout/stderr logs. Defaults to <cwd>/slurm_output."},
    )
    log_level: Optional[str] = field(
        default="info",
        metadata={"help": "Logging level (e.g. 'info', 'debug')."},
    )

    # data
    gene_tf_prior_data: Optional[str] = field(
        default="data/yeast/YEASTRACT_20190713_BOTH.tsv",
        metadata={"help": "TSV containing prior knowledge of gene-TF interactions."},
    )
    gene_dna_sequences: Optional[str] = field(
        default="preprocessing/yeast_genome/gene_DNA_sequences_ordered.tsv",
        metadata={"help": "File containing DNA sequences for all genes."},
    )
    tf_dna_sequences: Optional[str] = field(
        default="preprocessing/yeast_genome/TF_info_scores_with_DBID.tsv",
        metadata={"help": "File containing DNA sequences for all TFs."},
    )
    train_prop: Optional[float] = field(
        default=0.9,
        metadata={
            "help": "What proportion to use as training data. If 1.0, then no validation dataset will be created."
        },
    )
    downsample_rate: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "What proportion to use to downsample nonreg interactions. If 1.0, then no downsampling will be performed."
        },
    )
    class_weights: Optional[list] = field(
        default=lambda: [1.0, 1.0],
        metadata={"help": "Class weights for the loss function."},
    )
    classification_threshold: Optional[float] = field(
        default=0.5,
        metadata={"help": "Probability threshold for classifying examples."},
    )
    max_eval_examples: Optional[int] = field(
        default=10000,
        metadata={"help": "Maximum limit on the number of eval examples."},
    )

    # other training settings
    model_name_or_path: Optional[str] = field(
        default="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        metadata={"help": "the model name"},
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"}
    )
    use_even_class_sampler: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use a custom sampler to sample an equal amount of positives and negatives in each batch."
        },
    )

    # instrumentation
    wandb_project: Optional[str] = field(
        default="prior_network", metadata={"help": "wandb project name"}
    )
    wandb_run_name: Optional[str] = field(
        default="gene_tf_interactions", metadata={"help": "wandb run name"}
    )
    wandb_entity: Optional[str] = field(
        default="pmf-grn-3dc", metadata={"help": "wandb username or team name"}
    )
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Torch_dtype to use when loading and training the model (e.g. 'float16,' 'bfloat16', etc."
        },
    )

