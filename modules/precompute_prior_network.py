"""Run batched inference with the prior network.
"""
import argparse
import gc
import logging
import numpy as np
import os
import pandas as pd
import torch

from .prior_network_grn import PriorNetwork
from collections import defaultdict
from datasets import load_dataset
from datasets import disable_caching
from tqdm import tqdm

logging.basicConfig(level="INFO", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path", type=str, default="cskokgibbs/best_model"
    )
    parser.add_argument("--gpu-batch-size", type=int, default=128)
    parser.add_argument(
        "--gene-dna-sequences",
        type=str,
        default="preprocessing/yeast_genome/gene_DNA_sequences_ordered.tsv",
    )
    parser.add_argument(
        "--tf-dna-sequences",
        type=str,
        default="preprocessing/yeast_genome/TF_info_scores_with_DBID.tsv",
    )
    parser.add_argument(
        "--cached-tokenized-sequences",
        type=str,
        default=None,
        help="HF dataset containing tokenized DNA sequences.",
    )
    parser.add_argument(
        "--start", type=int, help="What gene index to start inference with."
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Which gene index to stop at."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    return args


def load_tokenized_cache(dataset_name: str):
    ds = load_dataset(dataset_name, split="train")
    cache = defaultdict(list)

    def map_fn(ex):
        cache[(ex["gene"], ex["TF"])].append(
            {
                "input_ids": torch.LongTensor(ex["input_ids"]),
                "attention_mask": torch.LongTensor(ex["attention_mask"]),
            }
        )

    ds = ds.map(map_fn, desc="creating cache...")
    return cache


def main():
    args = parse_args()
    start = args.start
    gene_data = pd.read_csv(args.gene_dna_sequences, sep="\t")  # gene_id, sequence
    gene_seqs = list(gene_data["sequence"])
    end = args.end if args.end is not None else len(gene_seqs)
    output_fp = f"{args.output_dir}/prior_matrix_genes_{args.start}-{args.end}.jsonl"
    prior_network = PriorNetwork(
        args.model_name_or_path,
        use_gpu=True,
        initialize_model=True,
        max_length=2048,
    )
    tf_data = (
        pd.read_csv(args.tf_dna_sequences, sep="\t")
        .groupby(["DBID"])
        .agg(Consensus=("Consensus", list))
    )  # DBID, Consensus
    tf_ids = list(tf_data.index)
    tf_seqs = list(tf_data["Consensus"])

    logger.info(f"Len gene_seqs: {len(gene_seqs)}")

    gene_seqs = gene_seqs[start:end]
    gene_ids = list(gene_data["gene_id"])[start:end]

    logger.info(f"Num of genes: {len(gene_seqs)}")
    logger.info(f"Num of TFs: {len(tf_seqs)}")

    if args.cached_tokenized_sequences is not None:
        cache = load_tokenized_cache(args.cached_tokenized_sequences)
        prior_sim_matrix = prior_network.forward_dna_sequences_cached(
            gene_ids, tf_ids, cache, batch_size=args.gpu_batch_size, with_grad=False
        )
    else:
        prior_sim_matrix = prior_network.forward_dna_sequences(
            gene_seqs,
            tf_seqs,
            batch_size=args.gpu_batch_size,
            with_grad=False,
        )
    logger.info(f"Output matrix shape: {prior_sim_matrix.shape}")
    logger.info(f"Num gene IDs: {len(gene_ids)}")
    logger.info(f"Num TF IDs: {len(tf_ids)}")
    output = []
    for i, gene_id in enumerate(gene_ids):
        for j, tf_id in enumerate(tf_ids):
            if prior_sim_matrix[i, j].item() is None or np.isnan(
                prior_sim_matrix[i, j].item()
            ):
                logging.warning(f"{prior_sim_matrix[i, j]} converts to null value.")
            output.append(
                {
                    "gene_id": gene_id,
                    "DBID": tf_id,
                    "prediction": prior_sim_matrix[i, j].item(),
                }
            )
    output = pd.DataFrame(output)

    output.to_json(output_fp, orient="records", lines=True)


if __name__ == "__main__":
    main()
