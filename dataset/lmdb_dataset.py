import abc
import torch
import lmdb
import pytorch_lightning as pl
import copy
import pickle
import pandas as pd
import numpy as np
import json
import os
import re

from torch.utils.data import DataLoader
from tqdm import tqdm
from Bio.Align import PairwiseAligner
import requests


_10TB = 10995116277760


class LMDBDataset(pl.LightningDataModule):
    """
    Abstract class from which other datasets inherit. We use LMDB database for all subclasses.
    """
    def __init__(self,
                 train_lmdb: str = None,
                 valid_lmdb: str = None,
                 test_lmdb: str = None,
                 dataloader_kwargs: dict = None):
        """
        Args:
            train_lmdb: path to train lmdb
            valid_lmdb: path to valid lmdb
            test_lmdb: path to test lmdb
            dataloader_kwargs: kwargs for dataloader
        """
        super().__init__()
        self.train_lmdb = train_lmdb
        self.valid_lmdb = valid_lmdb
        self.test_lmdb = test_lmdb
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}

        self.env = None
        self.operator = None

    def is_initialized(self):
        return self.env is not None
    
    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()
            
        # open lmdb
        self.env = lmdb.open(path, lock=False, map_size=_10TB)
        self.operator = self.env.begin()
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.operator = None
    
    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: str or int):
        value = self.operator.get(str(key).encode())

        if value is not None:
            value = value.decode()

        return value
     
    def _dataloader(self, stage):
        self.dataloader_kwargs["shuffle"] = True if stage == "train" else False
        lmdb_path = getattr(self, f"{stage}_lmdb")
        dataset = copy.copy(self)
        dataset._init_lmdb(lmdb_path)
        setattr(dataset, "stage", stage)
        
        return DataLoader(dataset, collate_fn=dataset.collate_fn, **self.dataloader_kwargs)
    
    def train_dataloader(self):
        dataloader = self._dataloader("train")
        return dataloader

    def test_dataloader(self):
        return self._dataloader("test")
    
    def val_dataloader(self):
        return self._dataloader("valid")
        
    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def collate_fn(self, batch):
        """
        Datasets should implement it as the function will be set when initializing Dataloader

        Returns:
            inputs: dict
            labels: dict
        """
        raise NotImplementedError

    def find_most_similar(self, target_id, target_seq, all_sequences):
        best_match = None
        best_score = -1

        if target_seq is None or len(target_seq) == 0:
            return best_match, best_score

        # aligner = PairwiseAligner()
        # aligner.mode = "local"

        aligner = PairwiseAligner(scoring="blastp")
        aligner.mode = "global"

        perfect_score = aligner.score(target_seq, target_seq)  # Alignment of target with itself

        for uniprot_id, seq in all_sequences.items():
            try:
                if uniprot_id == target_id:  # Skip self-comparison
                    return uniprot_id, 1.0

                # Compute alignment score
                alignments = aligner.score(target_seq, seq)

                if alignments > best_score:
                    best_score = alignments
                    best_match = uniprot_id
            except Exception:
                continue

        return best_match, best_score / perfect_score

    def fetch_uniprot_sequence(self, uniprot_id):
        try:
            if "-" in uniprot_id:
                uniprot_id = uniprot_id.split("-")[1]

            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            response = requests.get(url)

            if response.status_code == 200:
                fasta_data = response.text
                return "".join(fasta_data.split("\n")[1:])
            else:
                print(f"Failed to retrieve sequence for {uniprot_id}")
                return None
        except Exception:
            print(f"Got Exception while retrieving sequence for {uniprot_id}")
            return None

    def log_normalize(self, values):
        return np.log10(values + 1e-8)  # Adding small epsilon to avoid log(0)

