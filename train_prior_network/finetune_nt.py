import hydra
import json
import logging
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import wandb
import time

from dataclasses import asdict
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from scipy.special import softmax
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from train_prior_network.balanced_classes_trainer import InteractionsTrainerImbalancedClasses
from train_prior_network.script_args import ScriptArguments
from train_prior_network.finetune_utils import (
    initialize_dataset,
    map_dataset,
    tokenize_dataset,
    split_dataset,
    downsample_negative_samples,
)
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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

    evaluation_df = pd.merge(
        combined_df, prior_knowledge_long, on=["gene", "TF"], how="left"
    )
    evaluation_df.rename(columns={"interaction_y": "interaction"}, inplace=True)
    evaluation_df["correct"] = (
        evaluation_df["predicted_interaction"] == evaluation_df["interaction"]
    )

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
    references = (eval_pred.label_ids >= threshold).astype(int)

    probabilities = softmax(logits, axis=-1)
    positive_class_probs = probabilities[:, 1]

    try:
        roc_auc = roc_auc_score(references, positive_class_probs)
    except ValueError as e:
        logging.warning(e)
        roc_auc = None

    best_f1, best_threshold = compute_f1_threshold(references, positive_class_probs)

    f1_and_thres_dict = {
        "best_f1_score": float(best_f1),
        "best_threshold": float(best_threshold),
    }
    if wandb.run is not None:
        wandb.log(f1_and_thres_dict)
    if output_thresholds_fp is not None:
        with open(output_thresholds_fp, "a") as f:
            f.write(json.dumps(f1_and_thres_dict) + "\n")

    thresholded_predictions = (positive_class_probs >= best_threshold).astype(int)

    report = classification_report(
        references, thresholded_predictions, output_dict=True, zero_division=0
    )
    mcc = matthews_corrcoef(references, thresholded_predictions)

    if wandb.run is not None:
        wandb.log({"roc_auc": roc_auc})

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


def _has_arrow_dataset(path):
    return path is not None and os.path.exists(os.path.join(path, "dataset_info.json"))


@hydra.main(
    config_path="../config", config_name="train_prior_network_pipeline", version_base="1.2"
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
    cfg_dict = {**cfg_dict, "training_args": asdict(training_args)}
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    is_main_process = training_args.local_rank in (-1, 0)

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
    if is_main_process:
        wandb.init(
            project=script_args.wandb_project,
            name=script_args.wandb_run_name,
            entity=script_args.wandb_entity,
            config=cfg_dict,
        )
        wandb.run.log_artifact = lambda *args, **kwargs: None
    random.seed(training_args.seed)

    ####################### tokenization #######################
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer_kwargs = {
        "max_length": script_args.max_length,
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
    }

    tokenized_data_dir = os.path.join(training_args.output_dir, "tokenized_data")

    if script_args.sanity_check:
        logger.info("sanity_check=True — tokenizing inline from prior matrix.")
        full_ds = initialize_dataset(script_args, training_args.seed)
        full_ds = map_dataset(full_ds, tokenizer)
        full_ds = tokenize_dataset(full_ds, tokenizer, tokenizer_kwargs)
    elif _has_arrow_dataset(tokenized_data_dir):
        logger.info("Loading tokenized dataset from disk (memory-mapped Arrow format).")
        full_ds = load_from_disk(tokenized_data_dir)
        if "label" in full_ds.column_names and "labels" not in full_ds.column_names:
            full_ds = full_ds.rename_column("label", "labels")
    else:
        raise FileNotFoundError(
            f"Tokenized dataset not found at {tokenized_data_dir!r}. "
            "Run the tokenize step first (train_prior_network.tokenize_sequences), "
            "or set sanity_check=True to tokenize inline."
        )

    # Split dataset: train_prop (default 0.99) goes to train, remainder to dev
    full_ds, train_ds, dev_ds = split_dataset(full_ds, script_args.train_prop, training_args.seed)

    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Dev dataset size: {len(dev_ds)}")

    if script_args.downsample_rate < 1.0:
        logger.info(f"Downsampling negative samples with a rate of {script_args.downsample_rate}")
        train_ds = downsample_negative_samples(train_ds, script_args.downsample_rate, "train", seed=training_args.seed)
    else:
        logger.info(f"No downsampling applied. Rate: {script_args.downsample_rate}")

    ####################### training #######################
    logger.info("Starting training!")
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Dev dataset size: {len(dev_ds)}")

    if script_args.sanity_check:
        logger.info(
            "Running in sanity-check mode. Reducing dataset sizes and training for one epoch."
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
            "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
            trust_remote_code=True,
        )
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name_or_path,
            config=config,
            use_safetensors=True,
            **model_kwargs,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    model = model.to(device)
    saved_thresholds_fp = os.path.join(
        training_args.output_dir, "f1s_and_thresholds.jsonl"
    )

    trainer = InteractionsTrainerImbalancedClasses(
        model,
        training_args,
        train_ds,
        dev_ds,
        tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred, cfg.script_args.classification_threshold, saved_thresholds_fp
        ),
        script_args=script_args,
    )

    trainer.evaluate()
    trainer.train()

    trainer.save_model(training_args.output_dir)

    ####################### predict and evaluate #######################

    if is_main_process:
        evaluation_df = evaluate_prediction_matrix(
            trainer,
            dev_ds,
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
