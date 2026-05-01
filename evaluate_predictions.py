"""
Standalone evaluation: merge inference chunks, compute AUPRC vs gold standard.

Finds all prior_matrix_genes_*-*.jsonl(.gz) files in output_dir, concatenates
them, pivots to a gene x TF matrix, and computes AUPRC against a gold standard.

Basic usage (from GLM-Prior/):
    python evaluate_predictions.py \\
        --output-dir /path/to/run \\
        --gold-fp data/yeast/gold_standard.tsv

With completeness check (raises if any chunks are missing):
    python evaluate_predictions.py \\
        --output-dir /path/to/run \\
        --gold-fp data/yeast/gold_standard.tsv \\
        --gene-tsv data/yeast/gene_DNA_sequences_ordered.tsv \\
        --chunk-size 200
"""

import argparse
import glob
import json
import logging
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
import wandb

from inferelator.postprocessing.model_metrics import CombinedMetric

logger = logging.getLogger(__name__)


def _find_chunk_files(output_dir: str) -> List[Tuple[int, int, str]]:
    """
    Find all prior_matrix_genes_{start}-{end}.jsonl(.gz) files in output_dir.
    Returns list of (start, end, filepath) sorted by (start, end).
    """
    pat = re.compile(r"prior_matrix_genes_(\d+)-(\d+)\.jsonl(\.gz)?$")
    candidates = (
        glob.glob(os.path.join(output_dir, "prior_matrix_genes_*.jsonl"))
        + glob.glob(os.path.join(output_dir, "prior_matrix_genes_*.jsonl.gz"))
    )
    indexed = []
    for fp in candidates:
        m = pat.search(os.path.basename(fp))
        if m:
            indexed.append((int(m.group(1)), int(m.group(2)), fp))
    indexed.sort(key=lambda x: (x[0], x[1]))
    return indexed


def _load_and_pivot_chunks(indexed_files: List[Tuple[int, int, str]]) -> pd.DataFrame:
    """
    Load each JSONL(.gz) chunk, concatenate, and pivot to a wide matrix
    [gene_id x DBID] with values = max(prediction).
    """
    dfs = []
    for s, e, fp in indexed_files:
        try:
            df = pd.read_json(fp, orient="records", lines=True)
        except Exception as ex:
            logger.warning("Skipping %s: %s", fp, ex)
            continue
        missing = {"gene_id", "DBID", "prediction"} - set(df.columns)
        if missing:
            logger.warning("Skipping %s: missing columns %s", fp, missing)
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            "No valid prior_matrix_genes_*-*.jsonl files found after filtering."
        )

    combined = pd.concat(dfs, ignore_index=True)
    mat = combined.pivot_table(
        index="gene_id",
        columns="DBID",
        values="prediction",
        aggfunc="max",
        fill_value=0.0,
    ).astype(float)
    return mat


def _verify_chunk_completeness(
    indexed_files: List[Tuple[int, int, str]],
    output_dir: str,
    gene_tsv: str,
    chunk_size: int,
) -> None:
    """Raise if any expected inference chunks are missing."""
    total = len(pd.read_csv(gene_tsv, sep="\t"))
    expected = {
        (start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    }
    found = {(s, e) for s, e, _ in indexed_files}
    missing = sorted(expected - found)
    if missing:
        task_ids = sorted({s // chunk_size for s, _ in missing})
        parts, run_start, run_end = [], task_ids[0], task_ids[0]
        for tid in task_ids[1:]:
            if tid == run_end + 1:
                run_end = tid
            else:
                parts.append(str(run_start) if run_start == run_end else f"{run_start}-{run_end}")
                run_start = run_end = tid
        parts.append(str(run_start) if run_start == run_end else f"{run_start}-{run_end}")
        raise RuntimeError(
            f"{len(missing)} inference chunk(s) missing from {output_dir!r}. "
            f"Missing ranges: {missing}. "
            f"Re-submit with --array={','.join(parts)} "
            f"(expected {len(expected)} chunks total, found {len(found)})."
        )
    logger.info(
        "Completeness check passed: all %d expected chunks present.", len(expected)
    )


def run_evaluation(
    output_dir: str,
    gold_fp: str,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    gene_tsv: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> dict:
    """
    Merges inference chunks, saves the combined predictions matrix,
    computes AUPRC vs gold, and writes auprc_vs_gold.json.

    If gene_tsv and chunk_size are provided, verifies that every expected
    chunk is present before proceeding.  Optionally logs to an existing wandb run.
    """
    if not os.path.exists(gold_fp):
        raise FileNotFoundError(f"Gold standard not found: {gold_fp}")

    os.makedirs(output_dir, exist_ok=True)

    indexed_files = _find_chunk_files(output_dir)
    if not indexed_files:
        raise FileNotFoundError(
            f"No prior_matrix_genes_*-*.jsonl(.gz) files found in {output_dir}. "
            "Run inference first."
        )
    logger.info("Found %d chunk files — merging...", len(indexed_files))

    if gene_tsv is not None and chunk_size is not None:
        _verify_chunk_completeness(indexed_files, output_dir, gene_tsv, chunk_size)

    inferred_mat = _load_and_pivot_chunks(indexed_files)
    logger.info("Combined matrix shape (genes x TFs): %s", inferred_mat.shape)

    combined_fp = os.path.join(output_dir, "prior_network_predictions.tsv")
    inferred_mat.to_csv(combined_fp, sep="\t")
    logger.info("Wrote combined predictions matrix -> %s", combined_fp)

    gold = pd.read_csv(gold_fp, sep="\t", index_col=0)

    rows = list(set(inferred_mat.index).intersection(set(gold.index)))
    cols = list(set(inferred_mat.columns).intersection(set(gold.columns)))
    if not rows or not cols:
        raise RuntimeError(
            f"No overlap between inferred matrix "
            f"(genes={len(inferred_mat.index)}, TFs={len(inferred_mat.columns)}) "
            f"and gold standard (genes={len(gold.index)}, TFs={len(gold.columns)})."
        )

    inferred_aligned = inferred_mat.loc[rows, cols]
    gold_aligned = gold.loc[rows, cols]
    logger.info(
        "Aligned shapes — inferred: %s, gold: %s",
        inferred_aligned.shape, gold_aligned.shape,
    )

    auprc = float(
        CombinedMetric([inferred_aligned], gold_aligned, "keep_all_gold_standard").aupr
    )

    out_json = os.path.join(output_dir, "auprc_vs_gold.json")
    result = {"auprc": auprc}
    with open(out_json, "w") as f:
        json.dump(result, f)
    logger.info("AUPRC=%.6f -> %s", auprc, out_json)

    if wandb.run is not None:
        wandb.log({"auprc_vs_gold": auprc})
    elif wandb_project and wandb_run_name and wandb_entity:
        try:
            api = wandb.Api()
            runs = list(api.runs(f"{wandb_entity}/{wandb_project}", order="-created_at"))
            matches = [r for r in runs if r.name == wandb_run_name]
            if matches:
                run = wandb.init(
                    entity=wandb_entity,
                    project=wandb_project,
                    id=matches[0].id,
                    resume="must",
                )
                run.log({"auprc_vs_gold": auprc}, commit=True)
                run.finish()
        except Exception as exc:
            logger.warning("Could not log AUPRC to wandb: %s", exc)

    return result


def main():
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate GLM-Prior inference output.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory containing prior_matrix_genes_*.jsonl files.")
    parser.add_argument("--gold-fp", required=True,
                        help="Path to gold-standard TSV (genes x TFs).")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--gene-tsv", default=None,
        help="Path to gene DNA sequences TSV. If provided with --chunk-size, "
             "verifies all inference chunks are present before evaluating.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Inference chunk size used when running inference. Required with --gene-tsv.",
    )
    args = parser.parse_args()

    result = run_evaluation(
        output_dir=args.output_dir,
        gold_fp=args.gold_fp,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        gene_tsv=args.gene_tsv,
        chunk_size=args.chunk_size,
    )
    print(f"AUPRC: {result['auprc']:.6f}")


if __name__ == "__main__":
    main()
