"""Submit GLM-Prior inference SLURM array jobs from config.

Usage:
    python submit_inference.py [overrides...]

Reads inference_args and script_args from config/train_prior_network_pipeline.yaml.
Submits missing chunk jobs to SLURM via sbatch array.
"""
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from inference_utils import submit_inference_array

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config",
    config_name="train_prior_network_pipeline",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(cfg.script_args.log_level.upper())
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    output_dir = cfg.training_args.output_dir
    if output_dir is None:
        raise ValueError(
            "training_args.output_dir must be set. "
            "Pass it on the command line: training_args.output_dir=/path/to/experiment"
        )

    glm_prior_dir = os.path.dirname(os.path.abspath(__file__))

    result = submit_inference_array(
        output_dir=output_dir,
        glm_prior_dir=glm_prior_dir,
        gene_tsv=cfg.inference_args.gene_dna_sequences,
        tf_tsv=cfg.inference_args.tf_dna_sequences,
        model_name_or_path=output_dir,
        conda_env=cfg.script_args.conda_env,
        chunk=cfg.script_args.inference_chunk_size,
        gpu_batch_size=cfg.inference_args.gpu_batch_size,
        slurm_profile=cfg.inference_args.slurm_profile,
        logger=logger,
    )

    if result["num_tasks"] > 0:
        logger.info(
            f"Submitted {result['num_tasks']} inference job(s) "
            f"({result['total_rows']} genes in chunks of {result['chunk']}).\n"
            f"Once complete, run evaluation with:\n"
            f"  python evaluate_predictions.py "
            f"--output-dir {output_dir} "
            f"--gold-fp {cfg.inference_args.gold_standard_data}"
        )
    else:
        logger.info(
            f"All inference chunks already present. Run evaluation with:\n"
            f"  python evaluate_predictions.py "
            f"--output-dir {output_dir} "
            f"--gold-fp {cfg.inference_args.gold_standard_data}"
        )


if __name__ == "__main__":
    main()
