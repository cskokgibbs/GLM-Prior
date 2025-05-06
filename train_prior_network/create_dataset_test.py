import json
import os
import pandas as pd
import random
import tempfile
import unittest

from contextlib import contextmanager
from datasets import Dataset
from pprint import pprint, pformat
from train_prior_network.create_dataset import create_gene_tf_dataset
from unittest.mock import call, MagicMock


class TestCreateDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_gene_tf_prior = """
	YBR112C	YBR150C	Fake_TF	YBR239C
YAL068C	0	1	0	1
YAL067C	0	0	1	1
"""
        self.mock_gene_dna_seqs = [
            {"gene_id": "YAL068C", "sequence": "AAA"},
            {"gene_id": "YAL067C", "sequence": "TTT"},
            {"gene_id": "YAL066W", "sequence": "GGG"},
        ]
        self.mock_gene_dna_seqs_df = pd.DataFrame(self.mock_gene_dna_seqs)

        self.mock_tf_dna_seqs = [
            {"DBID": "YBR112C", "Consensus": "ATC"},
            {"DBID": "YBR112C", "Consensus": "CCCC"},
            {"DBID": "YBR150C", "Consensus": "GTGT"},
            {"DBID": "YBR239C", "Consensus": "ACACAC"},
        ]
        self.mock_tf_dna_seqs_df = pd.DataFrame(self.mock_tf_dna_seqs)

    @contextmanager
    def create_tmp_files(self, *args, **kwds):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prior_fp = os.path.join(tmp_dir, "prior.tsv")
            with open(prior_fp, "w") as f:
                f.write(self.mock_gene_tf_prior)
            gene_seqs_fp = os.path.join(tmp_dir, "gene_dna_seqs.tsv")
            self.mock_gene_dna_seqs_df.to_csv(gene_seqs_fp, sep="\t")
            tf_seqs_fp = os.path.join(tmp_dir, "tf_dna_seqs.tsv")
            self.mock_tf_dna_seqs_df.to_csv(tf_seqs_fp, sep="\t")
            yield (prior_fp, gene_seqs_fp, tf_seqs_fp)

    def test_create_gene_tf_dataset(self):
        with self.create_tmp_files() as (prior_fp, gene_seqs_fp, tf_seqs_fp):
            output_ds, _ = create_gene_tf_dataset(gene_seqs_fp, tf_seqs_fp, prior_fp, train_prop=1.0, downsample_rate=1.0)
            exp_gene_tf_tuples = [
                ("YAL068C", "AAA", "YBR150C", "GTGT", 1,),  # gene ID, gene DNA seq, TF ID, TF DNA seq, interaction
                ("YAL068C", "AAA", "YBR239C", "ACACAC", 1),
                ("YAL068C", "AAA", "YBR112C", "ATC", 0),
                ("YAL068C","AAA","YBR112C","CCCC",0),
                ("YAL067C", "TTT", "YBR239C", "ACACAC", 1),
                ("YAL067C", "TTT", "YBR112C", "ATC", 0),
                ("YAL067C", "TTT", "YBR112C", "CCCC", 0),
                ("YAL067C", "TTT", "YBR150C", "GTGT", 0),
            ]
            self.assertEqual(
                len(set([json.dumps(d) for d in output_ds])), len(exp_gene_tf_tuples)
            )  # check length and all unique
            for d in output_ds:
                d_tuple = (d["gene"], d["gene_DNA"], d["TF"], d["TF_DNA"], d["interaction"])
                self.assertIn(d_tuple, exp_gene_tf_tuples)

    def test_create_gene_tf_dataset_train_split(self):
        with self.create_tmp_files() as (prior_fp, gene_seqs_fp, tf_seqs_fp):
            train_ds, dev_ds = create_gene_tf_dataset(gene_seqs_fp, tf_seqs_fp, prior_fp, train_prop=0.5, downsample_rate=1.0, seed=0)
            # for this seed: 
            # Train genes: {'YAL068C'}, dev genes: {'YAL067C'}
            # Train TFs: {'YBR150C'}, Dev TFs: {'YBR112C', 'YBR239C'}
            exp_train_tuples = [
                ("YAL068C", "AAA", "YBR150C", "GTGT", 1,),  # gene ID, gene DNA seq, TF ID, TF DNA seq, interaction
            ]
            exp_dev_tuples = [
                ("YAL068C", "AAA", "YBR239C", "ACACAC", 1),
                ("YAL068C", "AAA", "YBR112C", "ATC", 0),
                ("YAL068C","AAA","YBR112C","CCCC",0),
                ("YAL067C", "TTT", "YBR239C", "ACACAC", 1),
                ("YAL067C", "TTT", "YBR112C", "ATC", 0),
                ("YAL067C", "TTT", "YBR112C", "CCCC", 0),
                ("YAL067C", "TTT", "YBR150C", "GTGT", 0),
            ]
            self.assertEqual(
                len(set([json.dumps(d) for d in train_ds])), len(exp_train_tuples)
            )  # check length and all unique
            for d in train_ds:
                d_tuple = (d["gene"], d["gene_DNA"], d["TF"], d["TF_DNA"], d["interaction"])
                self.assertIn(d_tuple, exp_train_tuples)
            self.assertEqual(
                len(set([json.dumps(d) for d in dev_ds])), len(exp_dev_tuples)
            )  # check length and all unique
            for d in dev_ds:
                d_tuple = (d["gene"], d["gene_DNA"], d["TF"], d["TF_DNA"], d["interaction"])
                self.assertIn(d_tuple, exp_dev_tuples)
            
    def test_downsample_rate(self):
        with self.create_tmp_files() as (prior_fp, gene_seqs_fp, tf_seqs_fp):
            train_ds_full, _ = create_gene_tf_dataset(gene_seqs_fp, tf_seqs_fp, prior_fp, train_prop=1.0, downsample_rate=1.0, seed=0)
            train_ds_downsampled, _ = create_gene_tf_dataset(gene_seqs_fp, tf_seqs_fp, prior_fp, train_prop=1.0, downsample_rate=0.5, seed=0)
            
            non_reg_interactions_full = [item for item in train_ds_full if item['interaction'] == 0]
            non_reg_interactions_downsampled = [item for item in train_ds_downsampled if item['interaction'] == 0]

            expected_downsampled_count = len(non_reg_interactions_full) // 2
            self.assertEqual(len(non_reg_interactions_downsampled), expected_downsampled_count)

if __name__ == "__main__":
    unittest.main()
