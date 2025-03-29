import pandas as pd
import json
import numpy as np
import pickle
import random

from torch.utils.data import Subset
from transformers import EsmTokenizer
from ..lmdb_dataset import *
from ..data_interface import register_dataset


@register_dataset
class SaprotAnnotationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 bias_feature: bool = False,
                 max_length: int = 1024,
                 mask_struc_ratio: float = None,
                 plddt_threshold: float = None,
                 **kwargs):
        """

        Args:
            tokenizer: EsmTokenizer config path
            
            bias_feature: If True, structure information will be used
            
            max_length: Max length of sequence
            
            mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
            
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            
            **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.bias_feature = bias_feature
        self.max_length = max_length
        self.mask_struc_ratio = mask_struc_ratio
        self.plddt_threshold = plddt_threshold

        self.proteins_with_ligands_ids = []

    def __getitem__(self, index):
        data = json.loads(self._get(index))
        seq = data['seq']

        # Ligands Extraction
        uniprot_id, ligand_list = data['name'], []
        ligand_list = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id][["smiles", "value", "ic50",
                                                                                    "kd", "ki"]].values.tolist()
        ligand_list = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list]

        if ligand_list and uniprot_id not in self.proteins_with_ligands_ids:
            self.proteins_with_ligands_ids.append(uniprot_id)
            print("proteins with ligands:", len(self.proteins_with_ligands_ids))

        # if uniprot_id in self.ligands_dataset:
        #     print("got here")
        #     ligand_list = [(data_point['smi'], data_point['label']['ic50']) for data_point in
        #                    self.ligands_dataset[uniprot_id] if 'label' in data_point and 'ic50' in data_point['label']]
        #
        #     print("ligand_list", ligand_list)
        #     if index not in self.proteins_with_ligands_indexes and ligand_list:
        #         self.proteins_with_ligands_indexes.append(index)
        #         print("proteins with ligands:", len(self.proteins_with_ligands_indexes))
        #
        # # Train only on proteins with ligands
        # elif self.proteins_with_ligands_indexes:
        #     new_index = random.sample(self.proteins_with_ligands_indexes, 1)[0]     # change to iterate pass
        #     return self.__getitem__(new_index)

        # else:
        #     if uniprot_id not in self.proteins_without_ligands:
        #         best_match, best_score = self.find_most_similar(uniprot_id, self.fetch_uniprot_sequence(uniprot_id),
        #                                                         self.protein_sequences)
        #         if best_score >= self.ligand_score_th:
        #             ligand_list = [(data_point['smi'], data_point['label']['ic50']) for data_point in
        #                            self.ligands_dataset[best_match] if
        #                            'label' in data_point and 'ic50' in data_point['label']]
        #             self.ligands_dataset[uniprot_id] = self.ligands_dataset[best_match]  # Cache the result for next iter
        #         else:
        #             self.proteins_without_ligands.append(uniprot_id)

        # Mask structure tokens
        if self.mask_struc_ratio is not None:
            tokens = self.tokenizer.tokenize(seq)
            mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]
            
            # Randomly select tokens to mask
            mask_num = int(len(mask_candi) * self.mask_struc_ratio)
            mask_idx = np.random.choice(mask_candi, mask_num, replace=False)
            for i in mask_idx:
                tokens[i] = tokens[i][:-1] + "#"
            
            seq = "".join(tokens)
        
        # Mask structure tokens with pLDDT < threshold
        if self.plddt_threshold is not None:
            plddt = data["plddt"]
            tokens = self.tokenizer.tokenize(seq)
            seq = ""
            for token, score in zip(tokens, plddt):
                if score < self.plddt_threshold:
                    seq += token[:-1] + "#"
                else:
                    seq += token
                    
        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)
            
        coords = data['coords'][:self.max_length] if self.bias_feature else None
        
        label = data['label']
        if isinstance(label, str):
            label = json.loads(label)
        
        return seq, label, coords, ligand_list

    def collate_fn(self, batch):
        seqs, labels, coords, ligand_list = zip(*batch)

        model_inputs = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": model_inputs}
        # print(self.tokenizer.convert_ids_to_tokens(inputs['inputs']['input_ids'][0]))
        if self.bias_feature:
            inputs['structure_info'] = (coords,)

        labels = {"labels": torch.tensor(labels, dtype=torch.long)}
        
        return inputs, labels, ligand_list
    
    def __len__(self):
        return int(self._get('length'))
