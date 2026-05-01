"""Dynamic SLURM array submission for GLM-Prior inference."""
import glob
import logging
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import pandas as pd


def write_inference_sbatch(
    path: str,
    output_dir: str,
    glm_prior_dir: str,
    gene_tsv: str,
    tf_tsv: str,
    model_name_or_path: str,
    gpu_batch_size: int,
    conda_env: str,
    slurm_profile: str = "legacy_a100",
) -> None:
    output_dir = os.path.abspath(output_dir)
    glm_prior_dir = os.path.abspath(glm_prior_dir)
    gene_tsv = os.path.abspath(gene_tsv)
    tf_tsv = os.path.abspath(tf_tsv)
    if os.path.exists(model_name_or_path):
        model_name_or_path = os.path.abspath(model_name_or_path)

    slurm_outputs = os.path.join(output_dir, "slurm_outputs")

    if slurm_profile == "legacy_a100":
        script = f"""#!/bin/bash
#SBATCH --job-name=glm-prior-inference
#SBATCH --open-mode=append
#SBATCH --output={slurm_outputs}/%x_%A_%a.out
#SBATCH --error={slurm_outputs}/%x_%A_%a.err
#SBATCH --export=NONE
#SBATCH --time=47:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=300GB
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@nyu.edu

set -euo pipefail

source ~/miniforge3/bin/activate {conda_env}
module load gcc

cd {glm_prior_dir}

: "${{TOTAL:?Set TOTAL via --export TOTAL=...}}"
: "${{CHUNK:?Set CHUNK via --export CHUNK=...}}"

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
if (( END > TOTAL )); then END=$TOTAL; fi
if (( START >= TOTAL )); then
  echo "Array task $SLURM_ARRAY_TASK_ID has START >= TOTAL; exiting."
  exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID: start=$START end=$END (TOTAL=$TOTAL, CHUNK=$CHUNK)"

OUT_FILE="{output_dir}/prior_matrix_genes_${{START}}-${{END}}.jsonl"
if [[ -f "$OUT_FILE" ]]; then
  echo "Chunk already exists at $OUT_FILE — skipping."
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m modules.precompute_prior_network \\
  --model-name-or-path="{model_name_or_path}" \\
  --gene-dna-sequences="{gene_tsv}" \\
  --tf-dna-sequences="{tf_tsv}" \\
  --gpu-batch-size={gpu_batch_size} \\
  --output-dir="{output_dir}" \\
  --start="$START" \\
  --end="$END"
"""
    else:
        raise ValueError(
            f"Unknown slurm_profile: {slurm_profile!r}. "
            "Supported values: 'legacy_a100'."
        )

    os.makedirs(slurm_outputs, exist_ok=True)
    with open(path, "w") as f:
        f.write(script)


def count_gene_rows(gene_tsv: str) -> int:
    if not gene_tsv or not os.path.exists(gene_tsv):
        raise FileNotFoundError(f"gene_dna_sequences not found: {gene_tsv}")
    df = pd.read_csv(gene_tsv, sep="\t")
    required = {"gene_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{gene_tsv} is missing required columns: {missing}")
    return int(len(df))


def _expected_chunk_ranges(total: int, chunk: int) -> List[Tuple[int, int]]:
    return [(start, min(start + chunk, total)) for start in range(0, total, chunk)]


def _existing_chunk_ranges(output_dir: str) -> set:
    pat = re.compile(r"prior_matrix_genes_(\d+)-(\d+)\.jsonl(\.gz)?$")
    found = set()
    for fp in (
        glob.glob(os.path.join(output_dir, "prior_matrix_genes_*.jsonl"))
        + glob.glob(os.path.join(output_dir, "prior_matrix_genes_*.jsonl.gz"))
    ):
        m = pat.search(os.path.basename(fp))
        if m:
            found.add((int(m.group(1)), int(m.group(2))))
    return found


def _build_array_spec(task_ids: List[int]) -> str:
    ids = sorted(set(task_ids))
    if not ids:
        raise ValueError("No task IDs to submit.")
    parts = []
    run_start = run_end = ids[0]
    for tid in ids[1:]:
        if tid == run_end + 1:
            run_end = tid
        else:
            parts.append(
                str(run_start) if run_start == run_end else f"{run_start}-{run_end}"
            )
            run_start = run_end = tid
    parts.append(str(run_start) if run_start == run_end else f"{run_start}-{run_end}")
    return ",".join(parts)


def submit_inference_array(
    output_dir: str,
    glm_prior_dir: str,
    gene_tsv: str,
    tf_tsv: str,
    model_name_or_path: str,
    conda_env: str,
    chunk: int = 1000,
    gpu_batch_size: int = 64,
    slurm_profile: str = "legacy_a100",
    sbatch_filename: str = "glm_prior_inference_array.sbatch",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    log = logger or logging.getLogger(__name__)

    output_dir = os.path.abspath(output_dir)
    gene_tsv = os.path.abspath(gene_tsv)
    tf_tsv = os.path.abspath(tf_tsv)

    os.makedirs(output_dir, exist_ok=True)

    total_rows = count_gene_rows(gene_tsv)
    expected = _expected_chunk_ranges(total_rows, chunk)
    existing = _existing_chunk_ranges(output_dir)

    missing_ranges = [(s, e) for s, e in expected if (s, e) not in existing]
    if not missing_ranges:
        log.info("All inference chunks already exist — nothing to submit.")
        return {"total_rows": total_rows, "num_tasks": 0, "chunk": chunk}

    task_ids = [s // chunk for s, _ in missing_ranges]
    array_spec = _build_array_spec(task_ids)
    num_total_tasks = math.ceil(total_rows / chunk)

    log.info(
        f"Submitting {len(missing_ranges)}/{num_total_tasks} missing inference chunks "
        f"(TOTAL={total_rows}, CHUNK={chunk}, profile={slurm_profile}): --array={array_spec}"
    )

    sbatch_path = os.path.join(output_dir, sbatch_filename)
    write_inference_sbatch(
        path=sbatch_path,
        output_dir=output_dir,
        glm_prior_dir=glm_prior_dir,
        gene_tsv=gene_tsv,
        tf_tsv=tf_tsv,
        model_name_or_path=model_name_or_path,
        gpu_batch_size=gpu_batch_size,
        conda_env=conda_env,
        slurm_profile=slurm_profile,
    )

    submit_env = os.environ.copy()
    removed = [k for k in list(submit_env) if k.startswith(("SLURM_", "SBATCH_", "SRUN_"))]
    for k in removed:
        submit_env.pop(k, None)
    if removed:
        log.info(f"Removed inherited scheduler vars: {sorted(removed)}")

    cmd = [
        "sbatch",
        f"--array={array_spec}",
        f"--export=TOTAL={total_rows},CHUNK={chunk}",
        sbatch_path,
    ]
    log.info(f"Submitting: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=submit_env)

    return {"total_rows": total_rows, "num_tasks": len(missing_ranges), "chunk": chunk}
