import gc
import logging
import numpy as np
import pyro
import pyro.distributions as dist
import random
import torch

from .pmf import PMF, log_gpu_memory
from tensor_utils import collate_tensors

from contextlib import nullcontext
from itertools import product
from pyro.nn import PyroModule
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
            for batch_start_idx in iterable:
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
            log_interval = 100
            j = 0
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

                if j % log_interval == 0:
                    log_gpu_memory(prefix_message=f"GPU memory at iter {j}")
                j += 1
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


class PriorNetworkPMF(PMF):
    def __init__(
        self,
        num_u: int,
        dim_u: int,
        V_prior_hparams: torch.tensor,
        dataset_size: int,
        prior_mean_log_U: float = 0.0,
        prior_std_log_U: float = 1.0,
        U_max: float = None,
        truncate_U: bool = False,
        guide_max_mean_log_U: float = 100,
        guide_max_std_log_U: float = 10,
        prior_std_logit_A: float = 1.0,
        prior_std_B: float = 1.0,
        use_gpu: bool = False,
        use_mask: bool = False,
        eps: float = 1e-6,
        min_prior_hparam: float = 0.01,
        max_prior_hparam: float = 0.99,
        model_name_or_path: str = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        gene_dna_sequences: List[str] = None,
        gene_names: List[str] = None,
        tf_dna_sequences: List[Union[str, List[str]]] = None,
        tf_ids: List[str] = None,
        gpu_batch_size: int = 16,
        initialize_similarity_matrix: bool = True,
        sim_matrix_mask_rate: float = 0.99,  # percentage of sim. matrix that will be copied over from prev. iter
        seed: int = 0,
        max_length: int = 2048,
    ):
        super().__init__(
            num_u,
            dim_u,
            V_prior_hparams,
            dataset_size,
            prior_mean_log_U,
            prior_std_log_U,
            U_max,
            truncate_U,
            guide_max_mean_log_U,
            guide_max_std_log_U,
            use_gpu,
            use_mask,
            eps,
        )
        self.min_prior_hparam = min_prior_hparam
        self.max_prior_hparam = max_prior_hparam
        self.prior_std_logit_A = prior_std_logit_A
        self.prior_std_B = prior_std_B
        self.gpu_batch_size = gpu_batch_size
        self.gene_dna_sequences = gene_dna_sequences
        self.gene_names = gene_names
        self.tf_dna_sequences = tf_dna_sequences
        self.tf_ids = tf_ids
        assert self.gene_dna_sequences is not None
        assert self.tf_dna_sequences is not None
        self.model_name_or_path = model_name_or_path
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.prior_network = PriorNetwork(
            model_name_or_path,
            use_gpu=self.use_gpu,
            initialize_model=initialize_similarity_matrix,
            max_length=self.max_length,
        )
        self.tokenizer = self.prior_network.tokenizer
        self.initialize_similarity_matrix = initialize_similarity_matrix
        if sim_matrix_mask_rate is not None:
            assert sim_matrix_mask_rate <= 1.0
        self.sim_matrix_mask_rate = sim_matrix_mask_rate
        self.seed = seed
        random.seed(self.seed)

        if self.initialize_similarity_matrix:
            # self.prev_sim_matrix = self.prior_network.forward_dna_sequences(
            #     self.gene_dna_sequences,
            #     self.tf_dna_sequences,
            #     batch_size=self.gpu_batch_size,
            #     with_grad=False,
            # )
            self.prev_sim_matrix = self.V_prior_hparams.clip(
                min=self.min_prior_hparam, max=self.max_prior_hparam
            )

    def compute_prior_network(self) -> torch.Tensor:
        num_genes = len(self.gene_dna_sequences)
        num_tfs = len(self.tf_dna_sequences)
        combined = torch.clone(self.prev_sim_matrix)
        if self.sim_matrix_mask_rate is not None:
            num_genes_to_infer = int(
                np.ceil(np.sqrt((1.0 - self.sim_matrix_mask_rate)) * num_genes)
            )
            num_tfs_to_infer = int(
                np.ceil(np.sqrt((1.0 - self.sim_matrix_mask_rate)) * num_tfs)
            )
            gene_idxs_to_infer = random.sample(range(num_genes), k=num_genes_to_infer)
            tf_idxs_to_infer = random.sample(range(num_tfs), k=num_tfs_to_infer)
            gene_dna_seqs = [self.gene_dna_sequences[i] for i in gene_idxs_to_infer]
            tf_dna_seqs = [self.tf_dna_sequences[i] for i in tf_idxs_to_infer]
            prior_network_inferred = self.prior_network.forward_dna_sequences(
                gene_dna_seqs,
                tf_dna_seqs,
                batch_size=self.gpu_batch_size,
                with_grad=True,
            )
            inferred_idxs = list(product(gene_idxs_to_infer, tf_idxs_to_infer))
            row_idxs = [p[0] for p in inferred_idxs]
            col_idxs = [p[1] for p in inferred_idxs]
            combined[row_idxs, col_idxs] = prior_network_inferred.flatten()
        else:
            gene_idxs_to_infer = random.sample(range(num_genes), k=self.gpu_batch_size)
            tf_idxs_to_infer = random.sample(range(num_tfs), k=self.gpu_batch_size)
            gene_dna_seqs = [self.gene_dna_sequences[i] for i in gene_idxs_to_infer]
            tf_dna_seqs = [self.tf_dna_sequences[i] for i in tf_idxs_to_infer]
            pairs = [(g, t) for g, t in zip(gene_dna_seqs, tf_dna_seqs)]
            prior_network_inferred = self.prior_network.forward_pairs(
                pairs, batch_size=self.gpu_batch_size, with_grad=True
            )
            combined[gene_idxs_to_infer, tf_idxs_to_infer] = prior_network_inferred
        return combined

    def model(
        self,
        i: torch.LongTensor,
        prior_hparams_U_i: torch.Tensor,
        data: torch.Tensor,
        annealing_factor: float,
        mask: torch.Tensor,
    ):
        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("gene_globals", len(self.V_prior_hparams)):

                A = pyro.sample(
                    "A",
                    dist.TransformedDistribution(
                        dist.Normal(
                            torch.logit(
                                self.V_prior_hparams.clip(
                                    min=self.min_prior_hparam, max=self.max_prior_hparam
                                )
                            ),
                            self.prior_std_logit_A,
                        ).to_event(1),
                        dist.transforms.SigmoidTransform(),
                    ),
                )

                B = pyro.sample(
                    "B",
                    dist.Normal(
                        torch.zeros_like(self.V_prior_hparams), self.prior_std_B
                    ).to_event(1),
                )
                V = A * B
            obs_std = pyro.sample(
                "obs_std", dist.LogNormal(torch.tensor([0.0], device=data.device), 1)
            )

        with pyro.plate("locals", self.dataset_size, subsample=data):
            with pyro.poutine.scale(None, annealing_factor):
                U, seq_depth = self.sample_locals_prior(data, prior_hparams_U_i)
                if self.U_max is not None:
                    U = torch.clip(U, max=self.U_max)

            obs_means = torch.matmul(U, torch.transpose(V, 0, 1)) * seq_depth
            if self.use_mask is True:
                obs_means = obs_means * mask
                obs = data * mask
            else:
                obs = data
            pyro.sample(
                "obs", dist.Normal(obs_means, obs_std + 0.001).to_event(1), obs=obs
            )

    def register_pyro_modules(self):
        pyro.module("prior_network", self.prior_network.model)

    def guide(
        self,
        i: torch.LongTensor,
        prior_hparams_U_i: torch.Tensor,
        data: torch.Tensor,
        annealing_factor: float,
        mask: torch.Tensor,
    ):
        self.initialise_locals_guide(data)
        self.register_pyro_modules()

        self.posterior_stds_logit_A = pyro.param(
            "A_stds",
            torch.ones_like(self.V_prior_hparams) * 0.1,
            constraint=constraints.greater_than(0.001),
            event_dim=-1,
        )
        self.posterior_means_B = pyro.param(
            "B_means", torch.zeros_like(self.V_prior_hparams), event_dim=-1
        )
        self.posterior_stds_B = pyro.param(
            "B_stds",
            torch.ones_like(self.V_prior_hparams) * 0.1,
            constraint=constraints.greater_than(0.001),
            event_dim=-1,
        )

        self.posterior_mean_obs_std = pyro.param(
            "obs_std_mean", torch.tensor([0.0], device=data.device)
        )
        self.posterior_std_obs_std = pyro.param(
            "obs_std_std",
            torch.tensor([1.0], device=data.device),
            constraint=constraints.greater_than(0.001),
        )
        # Only re-infer current sim matrix if in training model.
        if self.is_training:
            self.curr_sim_matrix = self.compute_prior_network()
            self.prev_sim_matrix = None
            gc.collect()
            torch.cuda.empty_cache()
            self.prev_sim_matrix = self.curr_sim_matrix.detach()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self.curr_sim_matrix = self.prev_sim_matrix

        with pyro.poutine.scale(None, annealing_factor):
            with pyro.plate("gene_globals", len(self.V_prior_hparams)):
                A_posterior_mean = self.curr_sim_matrix
                pyro.sample(
                    "A",
                    dist.TransformedDistribution(
                        dist.Normal(
                            A_posterior_mean, self.posterior_stds_logit_A
                        ).to_event(1),
                        dist.transforms.SigmoidTransform(),
                    ),
                )
                pyro.sample(
                    "B",
                    dist.Normal(self.posterior_means_B, self.posterior_stds_B).to_event(
                        1
                    ),
                )
            pyro.sample(
                "obs_std",
                dist.LogNormal(self.posterior_mean_obs_std, self.posterior_std_obs_std),
            )
            with pyro.plate("locals", self.dataset_size, subsample=i):
                self.sample_locals_guide(i)
