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
    gene_id_to_seq = {
        row["gene_id"]: row["sequence"] for _, row in gene_seqs.iterrows()
    }
    tf_seqs = pd.read_csv(tf_dna_sequences_fp, sep="\t")
    tf_seqs = (
        tf_seqs.groupby("DBID")
        .agg(lambda x: x.tolist())
        .reset_index()
        .map(lambda x: list(set(x)) if isinstance(x, list) else x)
    )
    tf_id_to_seq = {row["DBID"]: row["Consensus"] for _, row in tf_seqs.iterrows()}
    prior = pd.read_csv(gene_tf_prior_data_fp, sep="\t")
    if "Unnamed: 0" in prior.columns:
        prior = prior.rename(columns={"Unnamed: 0": "Gene"}).set_index("Gene")
    if "0" in prior.columns:
        prior = prior.rename(columns={"0": "Gene"}).set_index("Gene")

    if sanity_check:
        prior = prior.iloc[:100, :100]

    # Figure out which genes and TFs we actually have sequences for
    random.seed(seed)
    tf_set = set([tf for tf in prior.columns if tf in tf_id_to_seq])
    gene_set = set([gene for gene in prior.index if gene in gene_id_to_seq])

    full_gene_tf_interaction_pairs = []
    for gene, row in prior.iterrows():
        # get TFs that a gene interacts with, as well as the ones that it doesn't
        reg_tfs = set(
            [tf for tf, item in row.items() if item >= classification_threshold]
        )
        nonreg_tfs = list(tf_set - reg_tfs)
        logger.info(
            f"Gene {gene}: {len(reg_tfs)} interactions and {len(nonreg_tfs)} non-interactions."
        )
        if gene not in gene_set:
            logger.warning(f"Could not find DNA sequence for gene {gene}.")
            continue
        gene_dna_seq = gene_id_to_seq[gene]
        for reg_tf in reg_tfs:
            if reg_tf not in tf_set:
                logger.warning(f"Could not find DNA sequence for TF {reg_tf}.")
                continue
            reg_tf_dna_seqs = tf_id_to_seq[reg_tf]
            for reg_tf_dna_seq in reg_tf_dna_seqs:
                data_dict = {
                    "gene": gene,
                    "gene_DNA": gene_dna_seq,
                    "TF": reg_tf,
                    "TF_DNA": reg_tf_dna_seq,
                    "interaction": 1 if row[reg_tf] >= classification_threshold else 0,
                }
                full_gene_tf_interaction_pairs.append(data_dict)

        for non_reg_tf in nonreg_tfs:
            if non_reg_tf not in tf_set:
                logger.warning(
                    f"Could not find DNA sequence for TF {non_reg_tf}. Row: {row}"
                )
                continue
            non_reg_tf_dna_seqs = tf_id_to_seq[non_reg_tf]
            for non_reg_tf_dna_seq in non_reg_tf_dna_seqs:
                data_dict = {
                    "gene": gene,
                    "gene_DNA": gene_dna_seq,
                    "TF": non_reg_tf,
                    "TF_DNA": non_reg_tf_dna_seq,
                    "interaction": 0 if row[non_reg_tf] <  classification_threshold else 1,
                }
                full_gene_tf_interaction_pairs.append(data_dict)

    full_ds = Dataset.from_list(full_gene_tf_interaction_pairs)

    logger.info(f"Full dataset format: {full_ds}")

    return full_ds
