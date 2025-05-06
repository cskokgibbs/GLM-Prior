from dataclasses import dataclass, field

from typing import Optional


@dataclass
class ScriptArguments:
    """
    Extra arguments not already in HF TrainingArguments.
    """

    # data
    gene_tf_prior_data: Optional[str] = field(
        default="data/yeast/YEASTRACT_20190713_BOTH.tsv",
        metadata={"help": "TSV containing prior knowledge of gene-TF interactions."},
    )
    gold_standard_data: Optional[str] = field(
        default="data/yeast/gold_standard.tsv",
        metadata={
            "help": "TSV containing gold-standard data for benchmarking gene-TF interactions."
        },
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
    use_ddp: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use Distributed Data Parallel to optimize run time for large datasets."
        },
    )
    num_gpus: Optional[int] = field(
        default=1,
        metadata={
            "help": "How many GPUs to use for DDP. If set to 1, regular training will occur."
        },
    )
    new_experiment: Optional[bool] = field(
        default=False,
        metadata={
            "help": "For first time using a dataset, tokenize and cache the dataset for faster useage in the future."
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
    # Torch dtype used for training
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Torch_dtype to use when loading and training the model (e.g. 'float16,' 'bfloat16', etc."
        },
    )
    # hugging face
    hf_repo: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging face directory to save the tokenized data to."},
    )
    
    tokenized_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Local directory to save the tokenized data to."
        },
    )
    cache_tokenized_sequences_repo: Optional[str] = field(
        default="cskokgibbs/yeast-pretokenized-NT-cache",
        metadata={
            "help": "HF Hub repo where pre-tokenized gene and TF DNA sequences are stored."
        },
    )
    cache_tokenized_sequences_file_1: Optional[str] = field(
        default="yeast-pre-tokenized-NT-cache-part1.pt",
        metadata={
            "help": "Filename in HF Hub repo where pre-tokenized gene and TF DNA sequences are stored. "
            + "File is a stored dictionary mapping (gene ID, TF ID) to list of dictionaries containing keys 'input_ids' and 'attention_mask'."
        },
    )
    cache_tokenized_sequences_file_2: Optional[str] = field(
        default="yeast-pre-tokenized-NT-cache-part2.pt",
        metadata={
            "help": "Filename in HF Hub repo where pre-tokenized gene and TF DNA sequences are stored. "
            + "File is a stored dictionary mapping (gene ID, TF ID) to list of dictionaries containing keys 'input_ids' and 'attention_mask'."
        },
    )
    round_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "If set to 0, dataset will save and push to hugging face. If greater than 0, will skip saving."
        },
    )