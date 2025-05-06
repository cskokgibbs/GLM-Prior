import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from datasets import Dataset
from itertools import cycle
from torch.utils.data import BatchSampler, DataLoader
from train_prior_network.script_args import ScriptArguments
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import seed_worker
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class EvenClassBatchSampler(BatchSampler):
    def __init__(
        self,
        batch_size: int,
        drop_last: bool,
        positive_class_indexes: List[str],
        negative_class_indexes: List[str],
        seed: int = 0,
    ) -> None:
        self.batch_size = batch_size
        if self.batch_size % 2 != 0:
            raise ValueError(
                f"EvenClassBatchSampler requires an even batch size. Current batch size: {self.batch_size}"
            )
        self.drop_last = drop_last
        random.seed(seed)
        self.positive_class_indexes = positive_class_indexes
        self.negative_class_indexes = negative_class_indexes
        random.shuffle(self.positive_class_indexes)
        random.shuffle(self.negative_class_indexes)
        self.total_num_examples = 2 * max(
            len(self.positive_class_indexes), len(self.negative_class_indexes)
        )

    def __iter__(self) -> Iterator[List[int]]:
        if len(self.positive_class_indexes) > len(self.negative_class_indexes):
            cycle_iter = cycle(self.negative_class_indexes)
            other_iter = iter(self.positive_class_indexes)
        elif len(self.negative_class_indexes) > len(self.positive_class_indexes):
            cycle_iter = cycle(self.positive_class_indexes)
            other_iter = iter(self.negative_class_indexes)
        else:
            cycle_iter = iter(self.positive_class_indexes)
            other_iter = iter(self.negative_class_indexes)
        half_batch_size = int(self.batch_size / 2)
        if self.drop_last:
            while True:
                try:

                    sub_batches = [
                        [next(data_iter) for _ in range(half_batch_size)]
                        for data_iter in [cycle_iter, other_iter]
                    ]
                    batch = [
                        pp
                        for p in list(zip(sub_batches[0], sub_batches[1]))
                        for pp in p
                    ]
                    yield batch
                except StopIteration:
                    break
        else:
            zipped = list(zip(cycle_iter, other_iter))
            for start_idx in np.arange(0, len(zipped), half_batch_size):
                end_idx = start_idx + half_batch_size
                batch_pairs = zipped[start_idx:end_idx]
                batch = [pp for p in batch_pairs for pp in p]
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_num_examples // self.batch_size
        return math.ceil(self.total_num_examples / self.batch_size)


class InteractionsTrainerImbalancedClasses(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        tokenizer: PreTrainedTokenizerBase = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        script_args: ScriptArguments = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            **kwargs,
        )
        self.script_args = script_args

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # convert labels to soft labels
        neg_class_probs = 1.0 - labels
        labels = torch.stack([neg_class_probs, labels]).T
        labels = labels.type(torch.float32)

        # forward pass
        del inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        weight = torch.tensor(self.script_args.class_weights).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(
            train_dataset, description="training"
        )

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if self.script_args.use_even_class_sampler:
            # TODO: get class indexes
            pos_class_indexes = []
            neg_class_indexes = []
            for i, ex in enumerate(train_dataset):
                if ex["labels"] >= self.script_args.classification_threshold:
                    pos_class_indexes.append(i)
                else:
                    neg_class_indexes.append(i)
            dataloader_params["batch_sampler"] = EvenClassBatchSampler(
                self._train_batch_size,
                self.args.dataloader_drop_last,
                pos_class_indexes,
                neg_class_indexes,
                seed=self.args.seed,
            )
        else:
            dataloader_params["batch_size"] = self._train_batch_size
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))