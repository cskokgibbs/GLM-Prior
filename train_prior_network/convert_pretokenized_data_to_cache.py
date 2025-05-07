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
    """Cache pre-tokenized dataset indexed by (gene, TF) and remove excess padding.
    
    Args:
        pretokenized_data (str): The dataset identifier for loading pre-tokenized data.
        output_dir (str): The directory to save the cached files within
        output_filename (str): The filename to save the cached dataset.
    """
    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        print(f"Found existing local cache at {output_path}")
        return torch.load(output_path)

    print(f"No local cache found. Creating cache from Hugging Face dataset: {pretokenized_data}")  
    cache = defaultdict(list)
    api = HfApi()

    try:
        files_in_repo = api.list_repo_files(
            repo_id=pretokenized_data,
            repo_type="dataset"
            )
    except RepositoryNotFoundError:
        api.create_repo(repo_id=pretokenized_data, repo_type="dataset", private=False)
        print(f"Created new Hugging Face dataset repo: {pretokenized_data}")
        files_in_repo = []

    parquet_files = [f for f in files_in_repo if f.startswith("data/") and f.endswith(".parquet")]
    if not parquet_files:
        raise ValueError(f"No .parquet files found in Hugging Face dataset '{pretokenized_data}'.")
    
    datasets_list = []
    for file_name in parquet_files:
        dataset_fp = hf_hub_download(repo_id=pretokenized_data, filename=file_name, repo_type="dataset")
        df = pd.read_parquet(dataset_fp)
        ds = Dataset.from_pandas(df)
        datasets_list.append(ds)
    
    ds = concatenate_datasets(datasets_list)

    for ex in tqdm(ds, desc="Caching..."):
        d = {
            "input_ids": torch.LongTensor(ex["input_ids"]),
            "attention_mask": torch.LongTensor(ex["attention_mask"]),
        }
        d = remove_padding(d)
        cache[(ex["gene"], ex["TF"])].append(d)

    torch.save(cache, output_path)
    print(f"Saved cache locally to {output_path}")

    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_filename,
        repo_id=pretokenized_data,
        repo_type="dataset"
    )
    print(f"Uploaded {output_filename} to Hugging Face: {pretokenized_data}")
    return cache