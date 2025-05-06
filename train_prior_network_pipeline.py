import argparse
import hydra
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import random
import scanpy as sc
import sys
import wandb

from omegaconf import DictConfig, OmegaConf
from pprint import pformat
from process_utils import (
    create_single_gpu_slurm_script,
    create_multiple_gpu_slurm_script,
    create_single_cpu_slurm_script,
    get_output_or_exit,
    convert_nested_dict_to_option_strs,
    run_all_commands_and_wait_until_all_completed,
)
from inferelator.postprocessing.model_metrics import CombinedMetric
from tqdm import tqdm
from typing import Any, Dict, List, Optional

logger = logging.getLogger()
logger.setLevel("INFO")

def run_tokenize(
    cfg: DictConfig,
    output_dir: str,
    model_name_or_path: str,
    current_threshold: float,
    max_length: int,
    gene_tf_prior_path: str,
    round_num: int,
) -> str:
    tokenize_fp = os.path.join(output_dir, "dataset_info.json")
    if not cfg.overwrite and os.path.exists(tokenize_fp):
        logger.info(f"{tokenize_fp} already exists. Skipping training...")
        return output_dir
    py_command = f"python -m train_prior_network.tokenize_sequences training_args.output_dir={output_dir} "
    py_command += f"script_args.model_name_or_path={model_name_or_path} "
    py_command += f"script_args.classification_threshold={current_threshold} "
    py_command += f"script_args.max_length={max_length} "
    py_command += f"script_args.gene_tf_prior_data={gene_tf_prior_path} "
    py_command += f"script_args.round_num={round_num} "

    # Use the existing create_single_cpu_slurm_script function
    slurm_f = create_single_cpu_slurm_script(
        py_command,
        conda_env_name="/ext3/py3.10",
        singularity_overlay_fp=cfg.prior_network_singularity_overlay,
        singularity_img_fp=cfg.prior_network_singularity_img,
        slurm_output_dir=cfg.slurm_output_dir,
        email=cfg.email,
        mem="500GB"
    )

    sbatch_cmd = f"sbatch --parsable --wait --job-name=tokenize {slurm_f.name}"
    get_output_or_exit(sbatch_cmd, blocking=True)
    os.remove(slurm_f.name)
    return output_dir

def train_prior_network(
    cfg: DictConfig,
    output_dir: str,
    model_name_or_path: str,
    run_name: str,
    project_name: str,
    gene_tf_prior_path: str,
    current_threshold: float,
    round_num: int,
    tokenized_data_dir: str,
    new_experiment: bool,
    use_ddp: bool,
    num_gpus: int
) -> str:
    model_fp = os.path.join(output_dir, "model.safetensors")
    if not cfg.overwrite and os.path.exists(model_fp):
        logger.info(f"{model_fp} already exists. Skipping training...")
        return output_dir
    if use_ddp:
        logger.info("Running Train Prior Network with DDP")
        cuda_visible_devices = ",".join(str(i) for i in range(num_gpus))
        py_command = f"NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES={cuda_visible_devices} torchrun --nproc_per_node {num_gpus} "
        py_command += f"-m train_prior_network.finetune_nt training_args.output_dir={output_dir} "
    else:
        logger.info("Running Train Prior Network without DDP")
        py_command = f"python -m train_prior_network.finetune_nt training_args.output_dir={output_dir} "
    py_command += f"script_args.wandb_run_name={run_name} script_args.wandb_project={project_name} "
    py_command += f"script_args.model_name_or_path={model_name_or_path} "
    py_command += f"script_args.sanity_check={cfg.sanity_check} "
    py_command += f"script_args.gene_tf_prior_data={gene_tf_prior_path} "
    py_command += f"script_args.classification_threshold={current_threshold} "
    py_command += f"script_args.round_num={round_num} "
    py_command += f"script_args.tokenized_data_dir={tokenized_data_dir} "
    py_command += f"script_args.new_experiment={new_experiment} "

    # get a fresh seed every round so that we can train on a fresh data shuffle
    seed = random.getrandbits(cfg.num_seed_bits)
    py_command += f"training_args.seed={seed} "
    opts = convert_nested_dict_to_option_strs(cfg["prior_network_training"])
    opts_str = " ".join(opts)
    py_command += opts_str

    if use_ddp:
        slurm_f = create_multiple_gpu_slurm_script(
            py_command,
            conda_env_name="/ext3/py3.10",
            singularity_overlay_fp=cfg.prior_network_singularity_overlay,
            singularity_img_fp=cfg.prior_network_singularity_img,
            slurm_output_dir=cfg.slurm_output_dir,
            email=cfg.email,
            gpu_types=["a100", "h100"],
            mem="1500GB",
            num_gpus=num_gpus
        )
    else:
        slurm_f = create_single_gpu_slurm_script(
            py_command,
            conda_env_name="/ext3/py3.10",
            singularity_overlay_fp=cfg.prior_network_singularity_overlay,
            singularity_img_fp=cfg.prior_network_singularity_img,
            slurm_output_dir=cfg.slurm_output_dir,
            email=cfg.email,
            gpu_types=["a100", "h100"],
            mem="500GB",
        )
    sbatch_cmd = (
        f"sbatch --parsable --wait --job-name=train_prior_network {slurm_f.name}"
    )
    get_output_or_exit(sbatch_cmd, blocking=True)
    os.remove(slurm_f.name)
    return output_dir


def prior_network_inference(
    cfg: DictConfig, output_dir: str, model_name_or_path: str, num_genes: int
) -> str:
    combined_output_fp = f"{output_dir}/prior_network_predictions.tsv"
    if not cfg.overwrite and os.path.exists(combined_output_fp):
        logging.info(f"{combined_output_fp} already exists. Skipping...")
        return combined_output_fp

    # Get the number of genes, and split the inference up into multiple jobs
    gene_batch_size = int(math.ceil(num_genes / cfg.prior_network_num_inference_jobs))

    py_base_command = f"python -m modules.precompute_prior_network "
    py_base_command += f"--model-name-or-path={model_name_or_path} "
    py_base_command += f"--output-dir={output_dir} --gpu-batch-size=64 "
    py_base_command += (
        f"--gene-dna-sequences={cfg.prior_network_training.script_args.gene_dna_sequences} "
        f"--tf-dna-sequences={cfg.prior_network_training.script_args.tf_dna_sequences} "
    )
    slurm_cmds = []
    output_fps = []
    for i in np.arange(0, num_genes, gene_batch_size):
        start = i
        end = i + gene_batch_size
        output_fp = f"{output_dir}/prior_matrix_genes_{start}-{end}.jsonl"
        output_fps.append(output_fp)
        if os.path.exists(output_fp) and not cfg.overwrite:
            logging.info(f"{output_fp} already exists. Skipping...")
            continue
        py_command = f"{py_base_command} --start={start} --end={end} "

        slurm_f = create_single_gpu_slurm_script(
            py_command,
            conda_env_name="/ext3/pmf-grn",
            singularity_overlay_fp=cfg.prior_network_inference_singularity_overlay,
            singularity_img_fp=cfg.prior_network_inference_singularity_img,
            slurm_output_dir=cfg.slurm_output_dir,
            email=cfg.email,
            gpu_types=["a100", "rtx8000"],
            mem="300GB",
        )
        sbatch_cmd = f"sbatch --parsable --wait --job-name=pn-inference {slurm_f.name}"
        slurm_cmds.append(sbatch_cmd)
    run_all_commands_and_wait_until_all_completed(slurm_cmds, ignore_failures=False)
    # Now combine the outputs
    all_outputs = pd.concat(
        [pd.read_json(fp, orient="records", lines=True) for fp in output_fps]
    )
    all_outputs = all_outputs.pivot_table(
        index="gene_id", columns="DBID", values="prediction", fill_value=0
    )

    all_outputs.to_csv(combined_output_fp, sep="\t")
    logging.info(f"Predicted interactions saved to {combined_output_fp}.")
    return combined_output_fp


def get_latest_wandb_run_id(entity: str, project: str, run_name: str) -> Optional[str]:
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", order="-created_at"))

    # Filter runs by run_name
    runs = [run for run in runs if run.name == run_name]

    # Return the ID of the most recent run
    if runs:
        return runs[0].id
    else:
        return None


def compute_prior_network_auprc_vs_gold(
    cfg: DictConfig,
    output_dir: str,
    inferred_grn_fp: str,
    gold_fp: str,
    wandb_project: str,
    wandb_run_name: str,
    wandb_entity: str,
):
    output_fp = os.path.join(output_dir, "auprc_vs_gold.json")
    if os.path.exists(output_fp) and not cfg.overwrite:
        logger.info(f"{output_fp} already exists. Skipping...")
        return
    inferred = (
        pd.read_csv(inferred_grn_fp, sep="\t")
        .rename(columns={"0": "gene_id"})
        .set_index("gene_id")
    )
    gold = sc.read_csv(gold_fp, delimiter="\t", first_column_names=True).to_df()
    rows = list(set(inferred.index).intersection(set(gold.index)))
    cols = list(set(inferred.columns).intersection(set(gold.columns)))
    inferred = inferred[cols].loc[rows]
    gold = gold[cols].loc[rows]
    auprc = CombinedMetric([inferred], gold, "keep_all_gold_standard").aupr
    with open(output_fp, "w") as f:
        f.write(json.dumps({"auprc": float(auprc)}))
    wandb_id = get_latest_wandb_run_id(wandb_entity, wandb_project, wandb_run_name)
    if wandb_id is None:
        logging.warning(
            f"Could not find wandb ID for project {wandb_project} and run name {wandb_run_name} for entity {wandb_entity}."
        )
        return
    run = wandb.init(
        entity=wandb_entity, project=wandb_project, id=wandb_id, resume="must"
    )
    run.log({"auprc_vs_gold": auprc}, commit=True)
    run.finish()


def orchestrate(cfg: DictConfig):
    prior_network_model = cfg.prior_network_initial_model
    run_name_prefix = cfg.working_dir.split("/")[-1]
    num_genes = len(
        list(
            pd.read_csv(
                cfg.prior_network_training.script_args.gene_dna_sequences, sep="\t"
            )["gene_id"]
        )
    )
    random.seed(cfg.seed)

    curr_gene_tf_prior_path = cfg.prior_network_training.script_args.gene_tf_prior_data
    max_length = cfg.prior_network_training.script_args.max_length
    curr_threshold = 0.5

    tokenized_data_dir = cfg.prior_network_training.script_args.tokenized_data_dir
    new_experiment = cfg.prior_network_training.script_args.new_experiment
    logger.info(f"New experiment: {new_experiment}")
    use_ddp = cfg.prior_network_training.script_args.use_ddp
    num_gpus = cfg.prior_network_training.script_args.num_gpus
    logger.info(f"Use DDP: {use_ddp}")

    for i in tqdm(range(cfg.num_training_rounds), desc="Rounds"):
        logger.info(f"Round number: {i}")
        # train prior network
        prior_network_dir = os.path.join(cfg.working_dir, f"prior_network_round{i}")
        prior_network_training_run_name = f"{run_name_prefix}_prior_network_round{i}"
        os.makedirs(prior_network_dir, exist_ok=True)
        
        if (i == 0) and (new_experiment == True):
            logger.info("Running pre-tokenize")
            tokenized_data_dir = os.path.join(prior_network_dir, "tokenized_data")
            logger.info(f"Tokenized data for round {i} will be saved to: {tokenized_data_dir}")
            tokenize_folder = run_tokenize(
                cfg,
                tokenized_data_dir,
                prior_network_model,
                curr_threshold,
                max_length,
                curr_gene_tf_prior_path,
                i,
            )
        new_experiment = False
        prior_network_model = train_prior_network(
            cfg,
            prior_network_dir,
            prior_network_model,
            prior_network_training_run_name,
            run_name_prefix,
            curr_gene_tf_prior_path,
            curr_threshold,
            i,
            tokenized_data_dir,
            new_experiment,
            use_ddp,
            num_gpus
        )
        curr_threshold = float(
            json.loads(
                list(
                    open(f"{prior_network_model}/f1s_and_thresholds.jsonl").readlines()
                )[-1]
            )["best_threshold"]
        )
        logging.info(f"New threshold: {curr_threshold}")
        # get predictions from prior network
        prior_network_predictions = prior_network_inference(
            cfg, prior_network_dir, prior_network_model, num_genes
        )
        # compute AUPRC against gold
        compute_prior_network_auprc_vs_gold(
            cfg,
            prior_network_dir,
            prior_network_predictions,
            cfg.gold_fp,
            run_name_prefix,
            prior_network_training_run_name,
            cfg.wandb_entity,
        )


@hydra.main(version_base=None, config_path="config", config_name="train_prior_network_pipeline")
def main(cfg: DictConfig):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(cfg.log_level.upper())
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.slurm_output_dir is None:
        cfg.slurm_output_dir = os.path.join(os.getcwd(), "slurm_output")
        os.makedirs(cfg.slurm_output_dir, exist_ok=True)
    orchestrate(cfg)


if __name__ == "__main__":
    main()
