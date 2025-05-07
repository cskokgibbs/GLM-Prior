import os
import logging
import hydra

from dataclasses import asdict
from train_prior_network.finetune_utils import *
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="../config/prior_network", config_name="finetune_nt", version_base="1.2"
)
def main(cfg: DictConfig):
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(
        f"Output directory : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    cfg_dict = OmegaConf.to_container(cfg)
    training_args = TrainingArguments(**cfg_dict["training_args"])
    script_args = ScriptArguments(**cfg_dict["script_args"])
    cfg_dict = {**cfg_dict, "training_args": asdict(training_args)}

    logger.info(f"Parsed training_args: {training_args}")
    logger.info(f"Parsed script_args: {script_args}")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer_kwargs = {
        "max_length": script_args.max_length,
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
    }

    full_ds = initialize_dataset(script_args, training_args)
    full_ds = map_dataset(full_ds, tokenizer)  
    full_ds = tokenize_dataset(full_ds, tokenizer, tokenizer_kwargs)
    
    logger.info(f"Saving full tokenized dataset for prior to {training_args.output_dir}")
    full_ds.save_to_disk(training_args.output_dir)
    logger.info(f"Pushing tokenized dataset to Hugging Face repo: {script_args.hf_repo}")
    full_ds.push_to_hub(script_args.hf_repo)
    # create a cache
    manage_cache(script_args, training_args.output_dir, script_args.tokenized_data_dir, load=False)

if __name__ == "__main__":
    main()