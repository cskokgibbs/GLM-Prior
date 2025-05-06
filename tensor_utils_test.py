import os
import random
import tempfile
import torch
import unittest

from tensor_utils import collate_tensors, remove_padding
from torch import testing as torch_testing


class TestTensorUtils(unittest.TestCase):
    def test_collate_tensors(self):
        inputs = [
            {
                "input_ids": torch.LongTensor([1, 2, 3, 4, 5, 1, 1, 1]),
                "attention_mask": torch.LongTensor([1, 1, 1, 1, 1, 0, 0, 0]),
            },
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "attention_mask": torch.LongTensor([1, 1, 1]),
            },
            {
                "input_ids": torch.LongTensor([1, 2, 3, 1]),
                "attention_mask": torch.LongTensor([1, 1, 1, 0]),
            },
        ]
        expected_outputs = {
            "input_ids": torch.LongTensor(
                [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 1, 1],
                    [1, 2, 3, 1, 1],
                ]
            ),
            "attention_mask": torch.LongTensor(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                ]
            ),
        }
        outputs = collate_tensors(inputs, pad_token_id=1)
        torch_testing.assert_close(outputs, expected_outputs)

    def test_remove_padding(self):
        inputs = [
            {
                "input_ids": torch.LongTensor([1, 2, 3, 4, 5, 1, 1, 1]),
                "attention_mask": torch.LongTensor([1, 1, 1, 1, 1, 0, 0, 0]),
            },
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "attention_mask": torch.LongTensor([1, 1, 1]),
            },
            {
                "input_ids": torch.LongTensor([1, 2, 3, 1]),
                "attention_mask": torch.LongTensor([1, 1, 1, 0]),
            },
        ]
        expected_outputs = [
            {
                "input_ids": torch.LongTensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.LongTensor([1, 1, 1, 1, 1]),
            },
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "attention_mask": torch.LongTensor([1, 1, 1]),
            },
            {
                "input_ids": torch.LongTensor([1, 2, 3]),
                "attention_mask": torch.LongTensor([1, 1, 1]),
            },
        ]
        outputs = [remove_padding(i) for i in inputs]
        torch_testing.assert_close(outputs, expected_outputs)


if __name__ == "__main__":
    unittest.main()
