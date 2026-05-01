import hydra
import logging
import os
import sys  # used by logging handlers

from omegaconf import DictConfig, OmegaConf
from process_utils import (
    create_single_gpu_slurm_script,
    create_multiple_gpu_slurm_script,
    create_single_cpu_slurm_script,
    get_output_or_exit,
    convert_nested_dict_to_option_strs,
)
from typing import Optional

logger = logging.getLogger()
logger.setLevel("INFO")


def run_tokenize(
    cfg: DictConfig,
    glm_prior_dir: str,
    root_dir: str,
    model_name_or_path: str,
    classification_threshold: float,
    max_length: int,
    gene_tf_prior_path: str,
) -> None:
    tokenize_fp = os.path.join(root_dir, "tokenized_data", "dataset_info.json")
    if not cfg.script_args.overwrite and os.path.exists(tokenize_fp):
        logger.info(f"{tokenize_fp} already exists. Skipping tokenization.")
        return
    py_command = (
        f"cd {glm_prior_dir} && "
        f"python -m train_prior_network.tokenize_sequences "
        f"training_args.output_dir={root_dir} "
        f"script_args.model_name_or_path={model_name_or_path} "
        f"script_args.classification_threshold={classification_threshold} "
        f"script_args.max_length={max_length} "
        f"script_args.gene_tf_prior_data={gene_tf_prior_path} "
    )
    slurm_f = create_single_cpu_slurm_script(
        py_command,
        conda_env_name=cfg.script_args.conda_env,
        slurm_output_dir=cfg.script_args.slurm_output_dir,
        email=cfg.script_args.email,
        mem="500GB",
    )
    sbatch_cmd = f"sbatch --parsable --wait --job-name=glm-prior-tokenize {slurm_f.name}"
    get_output_or_exit(sbatch_cmd, blocking=True)
    os.remove(slurm_f.name)


def train_prior_network(
    cfg: DictConfig,
    glm_prior_dir: str,
    output_dir: str,
    model_name_or_path: str,
    run_name: str,
    project_name: str,
    gene_tf_prior_path: str,
    num_gpus: int,
) -> str:
    model_fp = os.path.join(output_dir, "model.safetensors")
    if not cfg.script_args.overwrite and os.path.exists(model_fp):
        logger.info(f"{model_fp} already exists. Skipping training.")
        return output_dir

    use_ddp = num_gpus > 1
    cuda_visible_devices = ",".join(str(i) for i in range(num_gpus))
    if use_ddp:
        logger.info(f"Running training with DDP ({num_gpus} GPUs)")
        base = (
            f"NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
            f"torchrun --nproc_per_node {num_gpus} "
            f"-m train_prior_network.finetune_nt "
        )
    else:
        logger.info("Running training on single GPU")
        base = "python -m train_prior_network.finetune_nt "

    from omegaconf import OmegaConf
    opts = convert_nested_dict_to_option_strs({
        "training_args": OmegaConf.to_container(cfg["training_args"]),
        "script_args": OmegaConf.to_container(cfg["script_args"]),
    })
    opts_str = " ".join(opts)
    py_command = (
        f"cd {glm_prior_dir} && {base}"
        f"{opts_str} "
        f"script_args.wandb_run_name={run_name} "
        f"script_args.wandb_project={project_name} "
        f"script_args.model_name_or_path={model_name_or_path} "
        f"script_args.gene_tf_prior_data={gene_tf_prior_path} "
    )

    if use_ddp:
        slurm_f = create_multiple_gpu_slurm_script(
            py_command,
            conda_env_name=cfg.script_args.conda_env,
            slurm_output_dir=cfg.script_args.slurm_output_dir,
            email=cfg.script_args.email,
            gpu_types=["a100", "h100"],
            mem="1500GB",
            num_gpus=num_gpus,
        )
    else:
        slurm_f = create_single_gpu_slurm_script(
            py_command,
            conda_env_name=cfg.script_args.conda_env,
            slurm_output_dir=cfg.script_args.slurm_output_dir,
            email=cfg.script_args.email,
            gpu_types=["a100", "h100"],
            mem="500GB",
        )
    sbatch_cmd = f"sbatch --parsable --wait --job-name=glm-prior-train {slurm_f.name}"
    get_output_or_exit(sbatch_cmd, blocking=True)
    os.remove(slurm_f.name)
    return output_dir


def orchestrate(cfg: DictConfig):
    if cfg.training_args.output_dir is None:
        raise ValueError(
            "output_dir must be set when running the pipeline. "
            "Pass it on the command line: training_args.output_dir=/path/to/experiment"
        )
    glm_prior_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = cfg.training_args.output_dir
    run_name_prefix = root_dir.split("/")[-1]
    max_length = cfg.script_args.max_length
    gene_tf_prior_path = cfg.script_args.gene_tf_prior_data
    num_gpus = cfg.script_args.num_gpus
    os.makedirs(root_dir, exist_ok=True)
    run_name = f"{run_name_prefix}_prior_network"

    model_name_or_path = cfg.script_args.model_name_or_path

    # Step 1: Tokenize — saves to {root_dir}/tokenized_data
    run_tokenize(
        cfg,
        glm_prior_dir,
        root_dir,
        model_name_or_path,
        0.5,
        max_length,
        gene_tf_prior_path,
    )

    # Step 2: Train — loads tokenized_data from {root_dir}/tokenized_data, saves model to root_dir
    trained_model_dir = train_prior_network(
        cfg,
        glm_prior_dir,
        root_dir,
        model_name_or_path,
        run_name,
        run_name_prefix,
        gene_tf_prior_path,
        num_gpus,
    )

    # Step 3: Inference — submit any missing chunks then exit (re-run after inference completes)
    from inference_utils import submit_inference_array

    result = submit_inference_array(
        output_dir=model_dir,
        glm_prior_dir=glm_prior_dir,
        gene_tsv=cfg.inference_args.gene_dna_sequences,
        tf_tsv=cfg.inference_args.tf_dna_sequences,
        model_name_or_path=trained_model_dir,
        conda_env=cfg.script_args.conda_env,
        chunk=cfg.script_args.inference_chunk_size,
        gpu_batch_size=cfg.inference_args.gpu_batch_size,
        slurm_profile=cfg.inference_args.slurm_profile,
        logger=logger,
    )
    if result["num_tasks"] > 0:
        logger.info(
            f"Submitted {result['num_tasks']} inference job(s). "
            f"Once complete, run evaluation with:\n"
            f"  python evaluate_predictions.py "
            f"--output-dir {model_dir} "
            f"--gold-fp {cfg.inference_args.gold_standard_data}"
        )
    else:
        logger.info(
            f"All inference chunks already present. Run evaluation with:\n"
            f"  python evaluate_predictions.py "
            f"--output-dir {model_dir} "
            f"--gold-fp {cfg.inference_args.gold_standard_data}"
        )


@hydra.main(version_base=None, config_path="config", config_name="train_prior_network_pipeline")
def main(cfg: DictConfig):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(cfg.script_args.log_level.upper())
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.script_args.slurm_output_dir is None:
        cfg.script_args.slurm_output_dir = os.path.join(os.getcwd(), "slurm_output")
        os.makedirs(cfg.script_args.slurm_output_dir, exist_ok=True)
    orchestrate(cfg)


if __name__ == "__main__":
    main()
