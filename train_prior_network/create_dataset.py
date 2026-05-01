import logging
import pandas as pd
import random

from datasets import Dataset
from typing import Tuple

logger = logging.getLogger(__name__)

def create_gene_tf_dataset(
    gene_dna_sequences_fp: str,
    tf_dna_sequences_fp: str,
    gene_tf_prior_data_fp: str,
    classification_threshold: float = 0.5,
    seed: int = 0,
    sanity_check: bool = False,
) -> Dataset:
    gene_seqs = pd.read_csv(gene_dna_sequences_fp, sep="\t")
    gene_id_to_seq = gene_seqs.set_index("gene_id")["sequence"].to_dict()

    tf_seqs = pd.read_csv(tf_dna_sequences_fp, sep="\t")
    tf_id_to_seq = (
        tf_seqs.groupby("DBID")["Consensus"]
        .apply(lambda x: list(set(x)))
        .to_dict()
    )

    prior = pd.read_csv(gene_tf_prior_data_fp, sep="\t")
    if "Unnamed: 0" in prior.columns:
        prior = prior.rename(columns={"Unnamed: 0": "Gene"}).set_index("Gene")
    if "0" in prior.columns:
        prior = prior.rename(columns={"0": "Gene"}).set_index("Gene")

    if sanity_check:
        prior = prior.iloc[:100, :100]

    tf_set = set(prior.columns) & set(tf_id_to_seq)
    gene_set = set(prior.index) & set(gene_id_to_seq)

    # Filter prior to only genes/TFs we have sequences for, then melt to long form
    prior_filtered = prior.loc[list(gene_set), list(tf_set)]
    prior_long = prior_filtered.stack().reset_index()
    prior_long.columns = ["gene", "TF", "value"]
    prior_long["interaction"] = (prior_long["value"] >= classification_threshold).astype(int)

    # Expand each row for TFs that have multiple DNA sequences
    prior_long["gene_DNA"] = prior_long["gene"].map(gene_id_to_seq)
    prior_long["TF_DNA"] = prior_long["TF"].map(tf_id_to_seq)
    prior_long = prior_long.explode("TF_DNA").reset_index(drop=True)

    prior_long = prior_long[["gene", "gene_DNA", "TF", "TF_DNA", "interaction"]]

    logger.info(
        f"Dataset created: {len(prior_long)} rows "
        f"({prior_long['interaction'].sum()} positive, "
        f"{(prior_long['interaction'] == 0).sum()} negative)"
    )

    return Dataset.from_pandas(prior_long, preserve_index=False)
