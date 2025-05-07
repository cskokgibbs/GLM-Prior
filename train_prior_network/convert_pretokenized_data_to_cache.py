import os
import gzip
import torch
import pandas as pd
import torch

from collections import defaultdict
from datasets import Dataset, concatenate_datasets, load_dataset
from tensor_utils import remove_padding
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from tqdm import tqdm

def cache_pretokenized_data(pretokenized_data: str, output_dir: str, output_filename: str):
    """
    Cache pre-tokenized dataset indexed by (gene, TF) and remove excess padding.
    If the cache doesn't already exist locally, create it from Hugging Face dataset and push it.
    """
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Found existing local cache at {output_path}")
        return torch.load(output_path)

    print(f"No local cache found. Creating cache from Hugging Face dataset: {pretokenized_data}")
    cache = defaultdict(list)

    ds = load_dataset(pretokenized_data, split="train")

    for ex in tqdm(ds, desc="Caching..."):
        d = {
            "input_ids": torch.LongTensor(ex["input_ids"]),
            "attention_mask": torch.LongTensor(ex["attention_mask"]),
        }
        d = remove_padding(d)
        cache[(ex["gene"], ex["TF"])].append(d)

    torch.save(cache, output_path)
    print(f"Saved cache to local path: {output_path}")

    api = HfApi()
    try:
        api.repo_info(repo_id=pretokenized_data, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=pretokenized_data, repo_type="dataset", private=False)

    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_filename,
        repo_id=pretokenized_data,
        repo_type="dataset"
    )
    print(f"Uploaded cache file to Hugging Face: {pretokenized_data}/{output_filename}")

    return cache
