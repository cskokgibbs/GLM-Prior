import unittest

from train_prior_network.balanced_classes_trainer import EvenClassBatchSampler


class TestEvenClassBatchSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 4
        self.seed = 0

    def test_balanced_batch_sampling_drop_last(self):
        positive_class_indexes = [0, 2]
        negative_class_indexes = [1, 3, 4, 5, 6]
        drop_last = True
        sampler = EvenClassBatchSampler(
            self.batch_size,
            drop_last,
            positive_class_indexes,
            negative_class_indexes,
            self.seed,
        )
        self.assertEqual(len(sampler), 2)
        batches = list(sampler)
        expected_batches = [[0, 6, 2, 4], [0, 3, 2, 1]]  # 5 is dropped
        self.assertEqual(batches, expected_batches)

    def test_balanced_batch_sampling_do_not_drop_last(self):
        positive_class_indexes = [0, 2]
        negative_class_indexes = [1, 3, 4, 5, 6]
        drop_last = False
        sampler = EvenClassBatchSampler(
            self.batch_size,
            drop_last,
            positive_class_indexes,
            negative_class_indexes,
            self.seed,
        )
        self.assertEqual(len(sampler), 3)
        batches = list(sampler)
        expected_batches = [[0, 6, 2, 4], [0, 3, 2, 1], [0, 5]]
        self.assertEqual(batches, expected_batches)


if __name__ == "__main__":
    unittest.main()
