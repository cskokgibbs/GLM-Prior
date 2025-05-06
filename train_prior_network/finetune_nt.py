import hydra
import json
import logging
import numpy as np
import os
import pandas as pd
import pprint
import random
import sys
import torch
import wandb

from dataclasses import asdict

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from train_prior_network.balanced_classes_trainer import (
    InteractionsTrainerImbalancedClasses,
)
from train_prior_network.create_dataset import create_gene_tf_dataset
from train_prior_network.script_args import ScriptArguments
from train_prior_network.convert_pretokenized_data_to_cache import cache_pretokenized_data
from train_prior_network.finetune_utils import *
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from typing import Any, Dict, List
from functools import partial
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from datasets import Dataset
from scipy.special import softmax

logger = logging.getLogger(__name__)

def push_dataset_to_hub(dataset, repo_id, repo_type="dataset"):
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        logger.info(f"Repository '{repo_id}' exists. Pushing dataset to hugging face.")
    except RepositoryNotFoundError:
        logger.info(f"Repository '{repo_id}' does not exist. Creating the repository.")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    # push the dataset to the repo
    dataset.push_to_hub(repo_id)


def evaluate_prediction_matrix(
    trainer, combined_ds, gene_tf_prior_data, sanity_check, classification_threshold
):
    if sanity_check:
        logger.info(
            f"Running in sanity-check mode for prediction evaluation. Reducing the size of the dataset."
        )
        combined_ds = combined_ds.select(range(min(len(combined_ds), 1000)))

    predictions, label_ids, _ = trainer.predict(combined_ds)
    predicted_labels = (predictions[:, 1] >= classification_threshold).astype(int)

    prior_knowledge_df = pd.read_csv(gene_tf_prior_data, sep="\t", index_col=0)
    prior_knowledge_long = prior_knowledge_df.melt(
        var_name="TF", value_name="interaction", ignore_index=False
    )
    index_name = (
        "index"
        if prior_knowledge_long.index.name is None
        else prior_knowledge_long.index.name
    )
    prior_knowledge_long.reset_index(inplace=True)
    prior_knowledge_long.rename(columns={index_name: "gene"}, inplace=True)
    prior_knowledge_long["interaction"] = (
        prior_knowledge_long["interaction"] >= classification_threshold
    ).astype(int)

    combined_df = combined_ds.to_pandas()
    combined_df["predicted_interaction"] = predicted_labels

    # evaluate predictions against the prior-knowledge
    evaluation_df = pd.merge(
        combined_df, prior_knowledge_long, on=["gene", "TF"], how="left"
    )
    evaluation_df.rename(columns={"interaction_y": "interaction"}, inplace=True)
    evaluation_df["correct"] = (
        evaluation_df["predicted_interaction"] == evaluation_df["interaction"]
    )

    # calculate metrics
    y_true = evaluation_df["interaction"]
    y_pred = evaluation_df["predicted_interaction"]

    logger.info("Checking for NaNs in y_true and y_pred")
    if y_true.isna().any():
        nan_indices = y_true[y_true.isna()].index
        logger.warning(f"NaN detected in y_true at indices: {nan_indices}")
        y_true.fillna(0, inplace=True)
    if y_pred.isna().any():
        nan_indices = y_pred[y_pred.isna()].index
        logger.warning(f"NaN detected in y_pred at indices: {nan_indices}")
        y_pred.fillna(0, inplace=True)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    confusion = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    logger.info(f"Confusion Matrix:\n{confusion}")
    logger.info(f"Classification Report:\n{report}")

    # calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")
    return evaluation_df


def save_interaction_matrix(evaluation_df, output_dir):
    interaction_matrix = evaluation_df.pivot_table(
        index="gene", columns="TF", values="predicted_interaction", fill_value=0
    )
    interaction_matrix = (interaction_matrix > 0).astype(int)
    output_file = os.path.join(output_dir, "predicted_interactions.tsv")
    interaction_matrix.to_csv(output_file, sep="\t")
    logger.info(f"Interaction matrix saved to {output_file}")


def format_strs(batch, tokenizer):
    """
    Format sequences as <gene DNA sequence><cls><TF DNA sequence>.
    Tokenizer will automatically add another <cls> in front.
    """
    inputs = [
        f"{batch['gene_DNA'][i]}{tokenizer.cls_token}{batch['TF_DNA'][i]}"
        for i in range(len(batch["gene_DNA"]))
    ]
    return {"formatted_inputs": inputs}


def tokenize(batch, tokenizer_kwargs, tokenizer) -> Dict[str, Any]:
    tokenized_outputs = tokenizer(batch["formatted_inputs"], **tokenizer_kwargs)
    return {
        "input_ids": tokenized_outputs["input_ids"],
        "attention_mask": tokenized_outputs["attention_mask"],
        "labels": batch["interaction"],
    }


def get_tokenized_from_cache_apply(row, cache) -> List[Dict[str, Any]]:
    output_dicts = []
    tokenized_dicts = cache[(row["gene"], row["TF"])]
    for dna_pair in tokenized_dicts:
        # TODO: remove padding
        output_dicts.append(
            {
                "gene": row["gene"],
                "TF": row["TF"],
                "input_ids": dna_pair["input_ids"],
                "attention_mask": dna_pair["attention_mask"],
                "labels": row["interaction"],
            }
        )
    return output_dicts


def compute_f1_threshold(references, probabilities):
    """Find the best F1 score and the corresponding threshold."""
    fpr, tpr, thresholds = roc_curve(references, probabilities)
    total_positives = sum(references)
    total_negatives = len(references) - total_positives

    best_f1 = 0
    best_threshold = 0

    for idx, threshold in enumerate(thresholds):
        tp = tpr[idx] * total_positives
        fp = fpr[idx] * total_negatives
        fn = total_positives - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_f1, best_threshold

def compute_metrics(
    eval_pred: EvalPrediction, threshold: float = 0.5, output_thresholds_fp: str = None
):
    """Computes precision, recall, and F1 score with threshold optimization"""
    logits = eval_pred.predictions
    references = (eval_pred.label_ids >= threshold).astype(
        int
    )  # references may be probabilities instead of hard labels

    probabilities = softmax(logits, axis=-1)
    positive_class_probs = probabilities[:, 1]

    try:
        roc_auc = roc_auc_score(references, positive_class_probs)
    except ValueError as e:
        # roc_auc_score will return a ValueError if only one class is represented in the true labels
        logging.warning(e)
        roc_auc = None

    best_f1, best_threshold = compute_f1_threshold(references, positive_class_probs)

    f1_and_thres_dict = {
        "best_f1_score": float(best_f1),
        "best_threshold": float(best_threshold),
    }
    wandb.log(f1_and_thres_dict)
    with open(output_thresholds_fp, "a") as f:
        f.write(json.dumps(f1_and_thres_dict) + "\n")

    thresholded_predictions = (positive_class_probs >= best_threshold).astype(int)

    # Get classification report for precision, recall, and f1-score
    report = classification_report(
        references, thresholded_predictions, output_dict=True, zero_division=0
    )
    mcc = matthews_corrcoef(references, thresholded_predictions)

    wandb.log({"roc_auc": roc_auc})

    # Extract metrics for the positive and negative classes
    logger.info(f"Report: {report}")
    metrics = {
        "accuracy": report["accuracy"],
        "positive_class_precision": report["1"]["precision"] if "1" in report else 0,
        "positive_class_recall": report["1"]["recall"] if "1" in report else 0,
        "positive_class_f1": report["1"]["f1-score"] if "1" in report else 0,
        "negative_class_precision": report["0"]["precision"] if "0" in report else 0,
        "negative_class_recall": report["0"]["recall"] if "0" in report else 0,
        "negative_class_f1": report["0"]["f1-score"] if "0" in report else 0,
        "roc_auc": roc_auc,
        "matthews_corrcoef": mcc,
        "best_f1_from_thresholding": best_f1,
    }

    return metrics


@hydra.main(
    config_path="../config/prior_network", config_name="finetune_nt", version_base="1.2"
)
def main(cfg: DictConfig):
    print(f"Working directory : {os.getcwd()}")
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    print("Python executable:", sys.executable)
    print("Python environment variables:", os.environ.get("CONDA_PREFIX"))
    cfg_dict = OmegaConf.to_container(cfg)
    training_args = TrainingArguments(**cfg_dict["training_args"])
    script_args = ScriptArguments(**cfg_dict["script_args"])
    training_args.report_to = ["wandb"]
    training_args.run_name = script_args.wandb_run_name
    # now get *all training args* now that TrainingArguments has post-processed
    cfg_dict = {**cfg_dict, "training_args": asdict(training_args)}
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(training_args.log_level.upper())
    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(f"Script args:\n{script_args}")

    class_labels = ["non-reg interactions", "reg interactions"]
    class_weights = script_args.class_weights

    logger.info("Class weights:")
    for label, weight in zip(class_labels, class_weights):
        logger.info(f"Class: {label} has weight {weight}")

    os.environ["WANDB_LOG_MODEL"] = "false"
    wandb.init(
        project=script_args.wandb_project,
        name=script_args.wandb_run_name,
        entity=script_args.wandb_entity,
        config=cfg_dict,
    )
    wandb.run.log_artifact = lambda *args, **kwargs: None
    random.seed(training_args.seed)

    ####################### tokenization #######################
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer_kwargs = {
        "max_length": script_args.max_length,
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
    }

    if script_args.sanity_check:
        script_args.new_experiment = True
    
    if script_args.new_experiment:
        logger.info(f"New experiment, creating dataset from matrix: {script_args.gene_tf_prior_data}.")
        full_ds = initialize_dataset(script_args, training_args)
        
        # Process dataset: format, tokenize, and save
        full_ds = map_dataset(full_ds, tokenizer)
        full_ds = tokenize_dataset(full_ds, tokenizer, tokenizer_kwargs)

        if not script_args.sanity_check and script_args.round_num == 0:
            logger.info(f"Sanity check is set to: {script_args.sanity_check}")
            logger.info(f"Train prior network round {script_args.round_num}")
            logger.info("Pushing dataset to hugging face.")
            push_to_hugging_face(full_ds, script_args.hf_repo, script_args.tokenized_data_dir)
            logger.info("Creating cache and pushing to hugging face.")
            manage_cache(script_args, training_args.output_dir, script_args.tokenized_data_dir, load=False)
            
    elif script_args.round_num > 0 and not script_args.sanity_check:
        logger.info(f"Round number is: {script_args.round_num}, creating and mapping dataset")
        full_ds = initialize_dataset(script_args, training_args)
        logger.info("Using cache from an existing experiment.")
        cache = manage_cache(script_args, training_args.output_dir, script_args.tokenized_data_dir, load=True)
        if cache is None:
            raise ValueError("Cache not found. Ensure the cache was created in the previous round.")
        else:
            logger.info("Cache loaded successfully!")
        full_ds = remove_duplicates(full_ds)
        logger.info("Mapping tokenized inputs from cache to dataset.")
        full_ds_df = full_ds.to_pandas()
        outputs = []
        for _, row in tqdm(
            full_ds_df.iterrows(), desc="Getting tokenized inputs from cache..."
        ):
            outputs.extend(get_tokenized_from_cache_apply(row, cache))
        outputs = pd.DataFrame(outputs)
        logger.info(f"Dataset size after extending from cache: {outputs.shape}")
        full_ds = Dataset.from_dict(outputs)

    else:
        # Round 0 but dataset exists already: use the cache
        logger.info("Using cache from an existing experiment.")
        cache = manage_cache(script_args, training_args.output_dir, script_args.tokenized_data_dir, load=True)
        if cache is None:
            raise ValueError("Cache not found. Ensure the cache was created in the previous round.")
        else:
            logger.info("Cache loaded successfully!")
        # Load full dataset and remove duplicates
        full_ds = load_from_local_or_from_hugging_face(script_args.hf_repo, script_args.tokenized_data_dir)
        full_ds = remove_duplicates(full_ds)
        # Map tokenized inputs from cache to their respective labels
        logger.info("Mapping tokenized inputs from cache to dataset.")
        full_ds_df = full_ds.to_pandas()
        outputs = []
        for _, row in tqdm(
            full_ds_df.iterrows(), desc="Getting tokenized inputs from cache..."
        ):
            outputs.extend(get_tokenized_from_cache_apply(row, cache))
        outputs = pd.DataFrame(outputs)
        logger.info(f"Dataset size after extending from cache: {outputs.shape}")
        full_ds = Dataset.from_dict(outputs)

    # Split dataset
    full_ds, train_ds, dev_ds = split_dataset(full_ds, script_args.train_prop, training_args.seed)

    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Dev dataset size: {len(dev_ds)}")

    # Downsample if required
    if script_args.downsample_rate < 1.0:
        logger.info(f"Downsampling negative samples with a rate of {script_args.downsample_rate}")
        train_ds = downsample_negative_samples(train_ds, script_args.downsample_rate, "train")
        #dev_ds = downsample_negative_samples(dev_ds, script_args.downsample_rate, "dev")
    else:
        logger.info(f"No downsampling applied. Rate: {script_args.downsample_rate}")


    ####################### training #######################
    logger.info("Starting training!")
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Dev dataset size: {len(dev_ds)}")
    
    if script_args.sanity_check:
        logger.info(
            f"Running in sanity-check mode. Will reduce the sizes of the datasets and train for only one epoch."
        )
        train_ds = train_ds.select(range(min(len(train_ds), 1000)))
        dev_ds = dev_ds.select(range(min(len(dev_ds), 1000)))
        training_args.num_train_epochs = 1
        training_args.eval_strategy = "steps"
        training_args.eval_steps = 5
    elif (
        script_args.max_eval_examples is not None
        and len(dev_ds) > script_args.max_eval_examples
    ):
        dev_ds = dev_ds.select(range(script_args.max_eval_examples))
    
    model_kwargs = {}
    if script_args.torch_dtype is not None:
        assert hasattr(torch, script_args.torch_dtype)
        model_kwargs["torch_dtype"] = getattr(torch, script_args.torch_dtype)
    model_kwargs["trust_remote_code"] = True
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path, num_labels=2, **model_kwargs
        )
    except Exception as e:
        logger.warning(f"Failed to load model with default config: {e}")
        config = AutoConfig.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",  # Base config
            trust_remote_code=True,
        )
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            config=config,
            **model_kwargs,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    model = model.to(device)
    saved_thresholds_fp = os.path.join(
        training_args.output_dir, "f1s_and_thresholds.jsonl"
    )
      
    ####################### predict and evaluate #######################

    evaluation_df = evaluate_prediction_matrix(
        trainer,
        dev_ds,  # updated to eval on dev split
        script_args.gene_tf_prior_data,
        sanity_check=script_args.sanity_check,
        classification_threshold=cfg.script_args.classification_threshold,
    )

    logger.info(
        "Evaluation DataFrame: \n%s\nShape: %s",
        evaluation_df.head(),
        evaluation_df.shape,
    )
    save_interaction_matrix(evaluation_df, training_args.output_dir)


if __name__ == "__main__":
    main()
