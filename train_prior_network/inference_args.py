from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceArguments:
    """
    Arguments for GLM-Prior inference (distinct from training data).
    Gene/TF sequence files here may cover the full evaluation set,
    whereas script_args.*_dna_sequences cover only the training set.
    """

    gene_dna_sequences: Optional[str] = field(
        default=None,
        metadata={"help": "TSV of gene DNA sequences for inference (may differ from training set)."},
    )
    tf_dna_sequences: Optional[str] = field(
        default=None,
        metadata={"help": "TSV of TF DNA sequences for inference (may differ from training set)."},
    )
    gpu_batch_size: Optional[int] = field(
        default=64,
        metadata={"help": "Number of gene-TF pairs per GPU forward pass during inference."},
    )
    gold_standard_data: Optional[str] = field(
        default=None,
        metadata={"help": "TSV of gold-standard interactions used by evaluate_predictions.py."},
    )
    slurm_profile: Optional[str] = field(
        default="legacy_a100",
        metadata={"help": "SLURM GPU profile for inference jobs. Supported: 'legacy_a100'."},
    )
