import os
import logging
import hydra

from functools import partial
from train_prior_network.finetune_utils import (
    initialize_dataset,
    map_dataset,
    tokenize_dataset,
)
from train_prior_network.script_args import ScriptArguments
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../config", config_name="train_prior_network_pipeline", version_base="1.2"
)
def main(cfg: DictConfig):
    output_dir = cfg.training_args.output_dir
    seed = cfg.training_args.seed
    logger.info(f"Output directory: {output_dir}")

    cfg_dict = OmegaConf.to_container(cfg)
    script_args = ScriptArguments(**cfg_dict["script_args"])

    logger.info(f"Parsed script_args: {script_args}")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer_kwargs = {
        "max_length": script_args.max_length,
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
    }

    full_ds = initialize_dataset(script_args, seed)
    full_ds = map_dataset(full_ds, tokenizer)
    full_ds = tokenize_dataset(full_ds, tokenizer, tokenizer_kwargs)

    tokenized_data_dir = os.path.join(output_dir, "tokenized_data")
    logger.info(f"Saving tokenized dataset to {tokenized_data_dir}")
    os.makedirs(tokenized_data_dir, exist_ok=True)
    full_ds.save_to_disk(tokenized_data_dir)
    logger.info("Tokenization complete.")

if __name__ == "__main__":
    main()
