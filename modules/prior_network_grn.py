import gc
import logging
import numpy as np
import random
import torch

from tensor_utils import collate_tensors

from contextlib import nullcontext
from itertools import product
from tqdm import tqdm
from torch import nn
from torch.distributions import constraints
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Mapping, Optional, Tuple, Union

logger = logging.getLogger()
logger.setLevel("INFO")


class PriorNetwork(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        use_gpu: bool = True,
        initialize_model: bool = True,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.use_gpu = use_gpu
        # If using the embedding encoder rather than just loading pre-computed embeddings,
        # initialize it here and load it onto the GPU.
        if initialize_model:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, num_labels=2, trust_remote_code=True, force_download=True
            )
            if self.use_gpu:
                self.model = self.model.cuda()
        else:
            self.model = None
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Outputs probabilities of positive class."""
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = model_outputs.logits
        probs = nn.functional.softmax(logits, dim=-1)
        return probs[:, 1]

    def format_and_tokenize_strs(
        self,
        gene_seqs: List[str],
        tf_seqs: List[str],
    ) -> Dict[str, torch.Tensor]:
        assert len(gene_seqs) == len(
            tf_seqs
        ), "gene and TF seqs must be the same length."
        input_pairs = [
            (gene_seq, tf_seq) for gene_seq, tf_seq in zip(gene_seqs, tf_seqs)
        ]
        inputs = [
            f"{gene_seq}{self.tokenizer.cls_token}{tf_seq}"
            for gene_seq, tf_seq in input_pairs
        ]
        tokenizer_kwargs = {
            "max_length": self.max_length,
            "return_tensors": "pt",
            "truncation": True,
            "padding": "longest",
        }
        tokenized_outputs = self.tokenizer(inputs, **tokenizer_kwargs)
        if self.use_gpu:
            tokenized_outputs = tokenized_outputs.to("cuda")
        return tokenized_outputs

    def forward_dna_sequences_cached(
        self,
        gene_ids: List[str],
        tf_ids: List[str],
        gene_and_tf_tokenized_cache: Mapping[
            Tuple[str, str], List[Dict[str, torch.Tensor]]
        ],
        batch_size: int = 2,
        with_grad: bool = True,
    ) -> torch.Tensor:
        """Given a cache of tokenized pairs of |gene DNA|<cls>|tf DNA|, run model inference
        for each possible pair of gene and TF.
        """
        tokenized = []
        nested_tf_lst_sizes = (
            []
        )  # keep track of how many DNA sequences we have have per pair of (gene, TF)
        for i, gene_id in enumerate(gene_ids):
            for tf_id in tf_ids:
                tokenized_seqs = gene_and_tf_tokenized_cache[(gene_id, tf_id)]
                if not tokenized_seqs:
                    logging.warning(f"Found no DNA sequences for {(gene_id, tf_id)}.")
                tokenized.extend(tokenized_seqs)
                if i == 0:
                    # During the first pass, count how many DNA sequences we have per TF
                    nested_tf_lst_sizes.append(len(tokenized_seqs))
        tf_end_idxs = [
            sum(nested_tf_lst_sizes[: i + 1]) for i in range(len(nested_tf_lst_sizes))
        ]
        if with_grad:
            context_manager = nullcontext
            if self.model is not None:
                self.model.train()
        else:
            context_manager = torch.no_grad
            if self.model is not None:
                self.model.eval()
        outputs = []
        with context_manager():
            iterable = np.arange(0, len(tokenized), batch_size)
            iterable = tqdm(iterable, desc="Computing similarity matrix")
            logging.info(f"Length of iterable: {len(iterable)}")
            for batch_num, batch_start_idx in enumerate(iterable):
                batch_tokenized = tokenized[
                    batch_start_idx : batch_start_idx + batch_size
                ]
                batch_inputs = collate_tensors(
                    batch_tokenized, self.tokenizer.pad_token_id
                )
                if self.use_gpu:
                    for k, v in batch_inputs.items():
                        batch_inputs[k] = v.cuda()
                batch_outputs = self.forward(
                    batch_inputs["input_ids"], batch_inputs["attention_mask"]
                )
                if torch.any(torch.isnan(batch_outputs)).item():
                    logging.warning(f"Found NaNs in model outputs: {batch_outputs}")
                outputs.append(batch_outputs)
                if batch_num % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        outputs = torch.cat(outputs)
        if torch.any(torch.isnan(outputs)).item():
            logging.warning(f"Found NaNs in cat-ed model outputs: {outputs}")
        tf_seq_boundaries = [0, *tf_end_idxs]
        logger.info(f"tf_seq_boundaries: {tf_seq_boundaries}")
        output_matrix = []
        num_genes = len(gene_ids)
        num_tfs = len(tf_ids)
        num_tf_dna_sequences = tf_seq_boundaries[-1]
        for gene_idx in range(num_genes):
            for tf_idx in range(num_tfs):
                start_idx = gene_idx * num_tf_dna_sequences + tf_seq_boundaries[tf_idx]
                end_idx = (
                    gene_idx * num_tf_dna_sequences + tf_seq_boundaries[tf_idx + 1]
                )
                pair_preds = outputs[start_idx:end_idx]
                mean_prediction = torch.mean(pair_preds)
                if torch.any(torch.isnan(mean_prediction)).item():
                    logging.warning(f"Found NaN mean for inputs: {pair_preds}")
                output_matrix.append(mean_prediction)
        output_matrix = torch.stack(output_matrix).reshape((num_genes, num_tfs))
        return output_matrix

    def forward_dna_sequences(
        self,
        gene_dna_sequences: List[str] = None,
        tf_dna_sequences: List[List[str]] = None,
        batch_size: int = 2,
        with_grad: bool = True,
    ) -> torch.Tensor:
        """Forward pass for every possible pair of gene and TF.

        Args:
        gene_dna_sequences: list of DNA sequences. Shape: (N,)
        tf_dna_sequences: list of list of TF DNA sequences (since each TF may have multiple DNA sequences). Shape: (M,)
        batch_size: Size of batch to compute outputs for in a single forward pass of the model.

        Returns tensor of shape (N,M) containing the predictions for each pair of gene and TF.
        """
        outputs = []
        num_genes = len(gene_dna_sequences)
        num_tfs = len(tf_dna_sequences)
        if with_grad:
            context_manager = nullcontext
            self.model.train()
        else:
            context_manager = torch.no_grad
            self.model.eval()

        # Record the index at which each nested list ends before flattening
        nested_lst_sizes = [len(l) for l in tf_dna_sequences]
        end_idxs = [
            sum(nested_lst_sizes[: i + 1]) for i in range(len(nested_lst_sizes))
        ]
        # now flatten sequences
        tf_dna_sequences = [x for nested_list in tf_dna_sequences for x in nested_list]

        gene_and_tf_idx_pairs = list(
            product(range(len(gene_dna_sequences)), range(len(tf_dna_sequences)))
        )
        with context_manager():
            iterable = np.arange(0, len(gene_and_tf_idx_pairs), batch_size)
            if not with_grad:
                iterable = tqdm(iterable, desc="Computing original similarity matrix")
            for batch_num, batch_start_idx in enumerate(iterable):
                batch_idx_pairs = gene_and_tf_idx_pairs[
                    batch_start_idx : batch_start_idx + batch_size
                ]
                batch_gene_seqs = [gene_dna_sequences[i] for i, _ in batch_idx_pairs]
                batch_tf_seqs = [tf_dna_sequences[j] for _, j in batch_idx_pairs]
                tokenized = self.format_and_tokenize_strs(
                    batch_gene_seqs, batch_tf_seqs
                )
                outputs.append(
                    self.forward(tokenized.input_ids, tokenized.attention_mask)
                )
                if batch_num % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    log_gpu_memory(prefix_message=f"GPU memory at batch {batch_num}")
        outputs = torch.cat(outputs)
        tf_seq_boundaries = [0, *end_idxs]
        output_matrix = []
        for gene_idx in range(num_genes):
            for tf_idx in range(num_tfs):
                start_idx = gene_idx * len(tf_dna_sequences) + tf_seq_boundaries[tf_idx]
                end_idx = (
                    gene_idx * len(tf_dna_sequences) + tf_seq_boundaries[tf_idx + 1]
                )
                pair_preds = outputs[start_idx:end_idx]
                mean_prediction = torch.mean(pair_preds)
                output_matrix.append(mean_prediction)
        output_matrix = torch.stack(output_matrix).reshape((num_genes, num_tfs))
        return output_matrix

    def forward_pairs(
        self,
        gene_tf_pairs: List[Tuple[str, List[str]]],
        batch_size: int = 2,
        with_grad: bool = True,
    ) -> torch.Tensor:
        """Given pairs of (gene DNA sequence, TF DNA sequence), return predictions for each pair.

        Arguments:
            gene_tf_pairs: List of tuples of (gene DNA sequence, DNA sequences corresponding to a TF)
            batch_size: GPU batch size
            with_grad: whether to turn on gradient computations
        Returns: tensor of same length as gene_tf_pairs
        """
        outputs = []
        if with_grad:
            context_manager = nullcontext
            self.model.train()
        else:
            context_manager = torch.no_grad
            self.model.eval()
        gene_dna_sequences = [pair[0] for pair in gene_tf_pairs]
        tf_dna_sequences = [pair[1] for pair in gene_tf_pairs]
        # Record the index at which each nested list ends before flattening
        nested_lst_sizes = [len(l) for l in tf_dna_sequences]
        end_idxs = [
            sum(nested_lst_sizes[: i + 1]) for i in range(len(nested_lst_sizes))
        ]
        end_idxs = [0, *end_idxs]
        # now flatten sequences
        tf_dna_sequences = [x for nested_list in tf_dna_sequences for x in nested_list]
        gene_and_tf_idx_pairs = []
        for i in range(len(gene_dna_sequences)):
            gene_and_tf_idx_pairs.extend(
                [(i, end_idxs[i] + j) for j in range(nested_lst_sizes[i])]
            )
        with context_manager():
            iterable = np.arange(0, len(gene_and_tf_idx_pairs), batch_size)
            if not with_grad:
                iterable = tqdm(iterable, desc="Computing original similarity matrix")
            for batch_start_idx in iterable:
                batch_idx_pairs = gene_and_tf_idx_pairs[
                    batch_start_idx : batch_start_idx + batch_size
                ]
                batch_gene_seqs = [gene_dna_sequences[i] for i, _ in batch_idx_pairs]
                batch_tf_seqs = [tf_dna_sequences[j] for _, j in batch_idx_pairs]
                tokenized = self.format_and_tokenize_strs(
                    batch_gene_seqs, batch_tf_seqs
                )
                outputs.append(
                    self.forward(tokenized.input_ids, tokenized.attention_mask)
                )
                gc.collect()
                torch.cuda.empty_cache()
        outputs = torch.cat(outputs)
        output_tensor = []
        for i in range(1, len(end_idxs)):
            start_idx = end_idxs[i - 1]
            end_idx = end_idxs[i]
            gene_tf_preds = outputs[start_idx:end_idx]
            output_tensor.append(torch.mean(gene_tf_preds))
        output_tensor = torch.stack(output_tensor)
        return output_tensor

def log_gpu_memory(prefix_message=None):
    if prefix_message is not None:
        logger.info(prefix_message)
    logger.info(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    logger.info(
        "torch.cuda.max_memory_allocated: %fGB"
        % (torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024)
    )
    logger.info(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    logger.info(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )