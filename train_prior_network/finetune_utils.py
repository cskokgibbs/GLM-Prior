import os
import random
import logging
import pprint
import pandas as pd
import torch

from collections import defaultdict
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from train_prior_network.create_dataset import create_gene_tf_dataset
from train_prior_network.convert_pretokenized_data_to_cache import cache_pretokenized_data
from train_prior_network.finetune_nt import *
from functools import partial
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

logger = logging.getLogger(__name__)

def initialize_dataset(script_args, training_args):
    """Creates the initial gene-TF dataset."""
    logger.info(f"Creating initial gene-TF dataset from {script_args.gene_tf_prior_data}.")
    full_ds = create_gene_tf_dataset(
        script_args.gene_dna_sequences,
        script_args.tf_dna_sequences,
        script_args.gene_tf_prior_data,
        classification_threshold=float(script_args.classification_threshold),
        seed=training_args.seed,
        sanity_check=script_args.sanity_check,
    )
    return full_ds

def map_dataset(full_ds, tokenizer):
    """Maps and formats the dataset by applying formatting functions."""
    logger.info("Formatting dataset.")
    formatted_ds = full_ds.map(
        partial(format_strs, tokenizer=tokenizer),
        batched=True,
        remove_columns=["gene_DNA", "TF_DNA"],
    )
    logger.info("Logging first few examples of formatted dataset.")
    samples = formatted_ds.select(range(3))
    for ex in samples:
        logger.info(pprint.pformat(ex))
    return formatted_ds

def tokenize_dataset(formatted_ds, tokenizer, tokenizer_kwargs):
    """Tokenizes the formatted dataset."""
    logger.info("Tokenizing dataset.")
    tokenized_ds = formatted_ds.map(
        partial(tokenize, tokenizer_kwargs=tokenizer_kwargs, tokenizer=tokenizer),
        batched=True,
    )
    logger.info("Logging first few examples of tokenized dataset.")
    samples = tokenized_ds.select(range(3))
    for ex in samples:
        logger.info(pprint.pformat(ex))
    return tokenized_ds

def push_to_hugging_face(full_ds, hf_repo, tokenized_data_dir):
    """Pushes the dataset to Hugging Face and saves it locally."""
    logger.info(f"Pushing tokenized dataset to Hugging Face: {hf_repo}")
    push_dataset_to_hub(full_ds, hf_repo)
    logger.info(f"Saving full tokenized dataset to {tokenized_data_dir}")
    full_ds.save_to_disk(os.path.join(tokenized_data_dir, "tokenized_data"))

def load_from_local_or_from_hugging_face(hf_repo, tokenized_data_dir):
    """Loads the dataset from local storage if available, otherwise from Hugging Face Hub."""
    local_path = os.path.join(tokenized_data_dir, "tokenized_data")
    if os.path.exists(tokenized_data_dir):
        logger.info(f"Loading dataset from local path: {tokenized_data_dir}")
        full_ds = load_from_disk(tokenized_data_dir)
        logger.info(f"Full dataset loaded locally with length: {len(full_ds)}")
        return full_ds
    else:
        logger.info(f"Local dataset not found. Loading dataset from Hugging Face: {hf_repo}")
        api = HfApi()
        files_in_repo = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
        parquet_files = [f for f in files_in_repo if f.startswith("data/") and f.endswith(".parquet")]

        datasets_list = []
        for file_name in parquet_files:
            dataset_fp = hf_hub_download(repo_id=hf_repo, filename=file_name, repo_type="dataset")
            df = pd.read_parquet(dataset_fp)
            ds = Dataset.from_pandas(df)
            datasets_list.append(ds)
        
        full_ds = concatenate_datasets(datasets_list)
        logger.info(f"Full dataset loaded with length: {len(full_ds)}")
        return full_ds

def manage_cache(script_args, output_dir, tokenized_data_dir, load=False):
    """Manages creation, splitting, and loading of dataset cache, with upload to Hugging Face."""
    api = HfApi()
    cache_file_path_1 = os.path.join(tokenized_data_dir, script_args.cache_tokenized_sequences_file_1)
    cache_file_path_2 = os.path.join(tokenized_data_dir, script_args.cache_tokenized_sequences_file_2) if script_args.cache_tokenized_sequences_file_2 else None

    if script_args.sanity_check:
        logger.info("Sanity check enabled. Skipping cache creation, upload, and loading.")
        return

    if not load:
        logger.info("Creating and pushing cache to Hugging Face.")
        cache = cache_pretokenized_data(script_args.hf_repo, tokenized_data_dir, script_args.cache_tokenized_sequences_file_1)
        
        try:
            api.repo_info(repo_id=script_args.cache_tokenized_sequences_repo, repo_type="dataset")
            logger.info(f"Repository '{script_args.cache_tokenized_sequences_repo}' already exists.")
        except RepositoryNotFoundError:
            api.create_repo(repo_id=script_args.cache_tokenized_sequences_repo, repo_type="dataset", private=False)
            logger.info(f"Created repository '{script_args.cache_tokenized_sequences_repo}'.")

        # Check cache size and split if necessary
        cache_size = os.path.getsize(cache_file_path_1) / (1024 ** 3)  # Size in GB
        if cache_size > 50:
            split_and_upload_cache(script_args, cache, tokenized_data_dir)
        else:
            # Upload the single cache file
            api.upload_file(
                path_or_fileobj=cache_file_path_1,
                path_in_repo=script_args.cache_tokenized_sequences_file_1,
                repo_id=script_args.cache_tokenized_sequences_repo,
                repo_type="dataset",
            )
            logger.info(f"{script_args.cache_tokenized_sequences_file_1} successfully uploaded to Hugging Face: {script_args.cache_tokenized_sequences_repo}")
    else:
        if os.path.exists(cache_file_path_1):
            logger.info(f"Loading cache from local directory: {cache_file_path_1}")
            cache = torch.load(cache_file_path_1)
            
            # Check if the cache was split into multiple files
            if cache_file_path_2 and os.path.exists(cache_file_path_2):
                logger.info(f"Loading 2nd cache file from local directory: {cache_file_path_2}")
                cache_2 = torch.load(cache_file_path_2)
                if isinstance(cache, dict) and isinstance(cache_2, dict):
                    cache.update(cache_2)
                elif isinstance(cache, list) and isinstance(cache_2, list):
                    cache.extend(cache_2)
                else:
                    logger.warning("Cache types do not match for concatenation.")
            return cache
        else:
            logger.info("Local cache not found. Loading cache from Hugging Face.")
            cache_fp_1 = hf_hub_download(repo_id=script_args.cache_tokenized_sequences_repo, filename=script_args.cache_tokenized_sequences_file_1, repo_type="dataset")
            cache = torch.load(cache_fp_1)
            
            # Check for a second part and load if it exists
            if script_args.cache_tokenized_sequences_file_2:
                cache_fp_2 = hf_hub_download(repo_id=script_args.cache_tokenized_sequences_repo, filename=script_args.cache_tokenized_sequences_file_2, repo_type="dataset")
                cache_2 = torch.load(cache_fp_2)
                if isinstance(cache, dict) and isinstance(cache_2, dict):
                    cache.update(cache_2)
                elif isinstance(cache, list) and isinstance(cache_2, list):
                    cache.extend(cache_2)
                else:
                    logger.warning("Cache types do not match for concatenation.")
            
            return cache


def split_and_upload_cache(script_args, cache, tokenized_data_dir):
    """Splits a large cache into two parts, saves each as a compressed file, and uploads to Hugging Face."""
    logger.info("Splitting large cache for Hugging Face upload.")
    
    # Determine midpoint for splitting
    keys = list(cache.keys())
    values = list(cache.values())
    mid_point = len(keys) // 2

    # Create two separate dictionaries for each half
    cache_part1 = defaultdict(list, zip(keys[:mid_point], values[:mid_point]))
    cache_part2 = defaultdict(list, zip(keys[mid_point:], values[mid_point:]))

    # Define filenames
    part1_file = os.path.join(tokenized_data_dir, script_args.cache_tokenized_sequences_file_1 or "cache-part1.pt.gz")
    part2_file = os.path.join(tokenized_data_dir, script_args.cache_tokenized_sequences_file_2 or "cache-part2.pt.gz")

    # Save each part with compression
    try:
        with gzip.open(part1_file, 'wb') as f:
            torch.save(cache_part1, f)
        with gzip.open(part2_file, 'wb') as f:
            torch.save(cache_part2, f)
        logger.info("Cache split and saved locally.")
    except Exception as e:
        logger.error(f"Error saving split cache files: {e}")
        raise

    # Upload each part to Hugging Face
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=part1_file,
            path_in_repo=script_args.cache_tokenized_sequences_file_1 or "cache-part1.pt.gz",
            repo_id=script_args.cache_tokenized_sequences_repo,
            repo_type="dataset",
        )
        logger.info("Uploaded part 1 to Hugging Face.")

        api.upload_file(
            path_or_fileobj=part2_file,
            path_in_repo=script_args.cache_tokenized_sequences_file_2 or "cache-part2.pt.gz",
            repo_id=script_args.cache_tokenized_sequences_repo,
            repo_type="dataset",
        )
        logger.info("Uploaded part 2 to Hugging Face.")
        
    except Exception as e:
        logger.error(f"Error uploading split cache files to Hugging Face: {e}")
        raise
    
def remove_duplicates(full_ds):
    """Removes duplicate gene-TF pairs."""
    logger.info(f"Removing duplicates from dataset. Initial length: {len(full_ds)}")
    full_ds = Dataset.from_pandas(full_ds.to_pandas().drop_duplicates(subset=["gene", "TF"]))
    logger.info(f"Duplicates removed. New length: {len(full_ds)}")
    return full_ds

def split_dataset(full_ds, train_prop, seed):
    """Splits dataset into train and dev sets."""
    logger.info("Splitting dataset into train and dev sets.")
    split = full_ds.train_test_split(train_size=train_prop, seed=seed)
    full_ds = DatasetDict({"train": split["train"], "dev": split["test"]})
    return full_ds, full_ds["train"], full_ds["dev"]

def downsample_negative_samples(dataset, downsample_rate, dataset_type):
    """Downsamples negative class in the given dataset according to the downsample rate."""
    logger.info(f"Downsampling {dataset_type} dataset's negative class.")
    data_dict = dataset.to_dict()
    non_reg_indices = [i for i, label in enumerate(data_dict['labels']) if label == 0]
    pos_indices = [i for i, label in enumerate(data_dict['labels']) if label > 0]
    
    logger.info(f"Before downsampling: {dataset_type} - Non-reg interactions: {len(non_reg_indices)}, Reg interactions: {len(pos_indices)}")

    num_to_sample = int(len(non_reg_indices) * downsample_rate)
    if num_to_sample > 0:
        downsampled_indices = random.sample(non_reg_indices, k=num_to_sample)
    else:
        logger.warning("Downsample rate resulted in zero samples. No downsampling applied.")
    
    logger.info(f"After downsampling: {dataset_type} - Non-reg interactions: {len(downsampled_indices)}, Reg interactions: {len(pos_indices)}")

    combined_indices = pos_indices + downsampled_indices
    downsampled_dict = {key: [data_dict[key][i] for i in combined_indices] for key in data_dict}
    downsampled_dataset = Dataset.from_dict(downsampled_dict)
    
    logger.info(f"Downsampling complete. New {dataset_type} dataset size: {len(downsampled_dataset)}")
    return downsampled_dataset


