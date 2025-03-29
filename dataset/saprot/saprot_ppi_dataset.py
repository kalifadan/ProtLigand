import torch
import json
import pickle
import random

from ..lmdb_dataset import LMDBDataset
from transformers import EsmConfig, EsmTokenizer
from ..data_interface import register_dataset


@register_dataset
class SaprotPPIDataset(LMDBDataset):
    def __init__(self,
             tokenizer: str,
             max_length: int = 1024,
             plddt_threshold: float = None,
             **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            
            max_length: Max length of sequence
            
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.plddt_threshold = plddt_threshold

        self.proteins_without_ligands = []
        self.proteins_with_ligands_ids = []
        self.proteins_with_ligands_indexes = []

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq_1, seq_2 = entry['seq_1'], entry['seq_2']

        # # Ligands Extraction
        uniprot_id_1, ligand_list_1 = entry['name_1'], []
        # if uniprot_id_1 in self.ligands_dataset:
        #     ligand_list_1 = [data_point['smi'] for data_point in
        #                      self.ligands_dataset[uniprot_id_1] if 'smi' in data_point]
        # ligand_list_1 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_1]["smiles"].values.tolist()
        # ligand_list_1 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_1][["smiles", "value"]].values.tolist()
        # ligand_list_1 = [(item[0], item[1]) for item in ligand_list_1]

        ligand_list_1 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_1][["smiles", "value", "ic50",
                                                                                        "kd", "ki"]].values.tolist()
        ligand_list_1 = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list_1]
        # ligand_list_1 = [(item[0], [item[1]]) for item in ligand_list_1]

        if ligand_list_1 and uniprot_id_1 not in self.proteins_with_ligands_ids:
            self.proteins_with_ligands_ids.append(uniprot_id_1)
            print("proteins with ligands:", len(self.proteins_with_ligands_ids))

        uniprot_id_2, ligand_list_2 = entry['name_2'], []
        # if uniprot_id_2 in self.ligands_dataset:
        #     ligand_list_2 = [data_point['smi'] for data_point in
        #                      self.ligands_dataset[uniprot_id_2] if 'smi' in data_point]
        # ligand_list_2 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_2]["smiles"].values.tolist()
        # ligand_list_2 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_2][["smiles", "value"]].values.tolist()
        # ligand_list_2 = [(item[0], item[1]) for item in ligand_list_2]

        ligand_list_2 = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id_2][["smiles", "value", "ic50",
                                                                                        "kd", "ki"]].values.tolist()
        ligand_list_2 = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list_2]
        # ligand_list_2 = [(item[0], [item[1]]) for item in ligand_list_2]

        if ligand_list_2 and uniprot_id_2 not in self.proteins_with_ligands_ids:
            self.proteins_with_ligands_ids.append(uniprot_id_2)
            print("proteins with ligands:", len(self.proteins_with_ligands_ids))

        # Mask structure tokens with pLDDT < threshold
        if self.plddt_threshold is not None:
            plddt_1, plddt_2 = entry['plddt_1'], entry['plddt_2']
            tokens = self.tokenizer.tokenize(seq_1)
            seq_1 = ""
            for token, score in zip(tokens, plddt_1):
                if score < self.plddt_threshold:
                    seq_1 += token[:-1] + "#"
                else:
                    seq_1 += token

            tokens = self.tokenizer.tokenize(seq_2)
            seq_2 = ""
            for token, score in zip(tokens, plddt_2):
                if score < self.plddt_threshold:
                    seq_2 += token[:-1] + "#"
                else:
                    seq_2 += token

        tokens = self.tokenizer.tokenize(seq_1)[:self.max_length]
        seq_1 = " ".join(tokens)

        tokens = self.tokenizer.tokenize(seq_2)[:self.max_length]
        seq_2 = " ".join(tokens)

        return seq_1, seq_2, int(entry["label"]), ligand_list_1, ligand_list_2

    # def __getitem__(self, index):
    #     entry = json.loads(self._get(index))
    #     seq_1, seq_2 = entry['seq_1'], entry['seq_2']
    #
    #     # Ligands Extraction
    #     uniprot_id_1, ligand_list_1 = entry['name_1'], []
    #     # if uniprot_id_1 in self.ligands_dataset:
    #     #     # ligand_list_1 = [(data_point['smi'], data_point['label']['ic50']) for data_point in
    #     #     #                  self.ligands_dataset[uniprot_id_1] if 'label' in data_point and 'ic50' in data_point['label']]
    #     #     ligand_list_1 = [data_point['smi'] for data_point in
    #     #                      self.ligands_dataset[uniprot_id_1] if 'smi' in data_point]
    #     #     print("ligand_list_1:", len(ligand_list_1))
    #
    #     # else:
    #     #     if uniprot_id_1 not in self.proteins_without_ligands:
    #     #         best_match, best_score = self.find_most_similar(uniprot_id_1, self.fetch_uniprot_sequence(uniprot_id_1),
    #     #                                                         self.protein_sequences)
    #     #         if best_score >= self.ligand_score_th:
    #     #             ligand_list_1 = [(data_point['smi'], data_point['label']['ic50']) for data_point in
    #     #                              self.ligands_dataset[best_match] if
    #     #                              'label' in data_point and 'ic50' in data_point['label']]
    #     #             self.ligands_dataset[uniprot_id_1] = self.ligands_dataset[best_match]  # Cache the result for next iter
    #     #         else:
    #     #             self.proteins_without_ligands.append(uniprot_id_1)
    #     #
    #     # if uniprot_id_1 not in self.proteins_with_ligands_ids and ligand_list_1:
    #     #     print("new protein received")
    #     #     self.proteins_with_ligands_ids.append(uniprot_id_1)
    #     #     self.proteins_with_ligands_indexes.append(index)
    #     #     print("proteins with ligands:", len(self.proteins_with_ligands_indexes))
    #
    #     uniprot_id_2, ligand_list_2 = entry['name_2'], []
    #     # if uniprot_id_2 in self.ligands_dataset:
    #     #     # ligand_list_2 = [(data_point['smi'], data_point['label']['ic50']) for data_point in
    #     #     #                  self.ligands_dataset[uniprot_id_2] if 'label' in data_point and 'ic50' in data_point['label']]
    #     #     ligand_list_2 = [data_point['smi'] for data_point in
    #     #                      self.ligands_dataset[uniprot_id_2] if 'smi' in data_point]
    #     #     print("ligand_list_2:", len(ligand_list_2))
    #
    #     # else:
    #     #     if uniprot_id_2 not in self.proteins_without_ligands:
    #     #         best_match, best_score = self.find_most_similar(uniprot_id_2, self.fetch_uniprot_sequence(uniprot_id_2),
    #     #                                                         self.protein_sequences)
    #     #         if best_score >= self.ligand_score_th:
    #     #             ligand_list_2 = [(data_point['smi'], data_point['label']['ic50']) for data_point in
    #     #                            self.ligands_dataset[best_match] if
    #     #                            'label' in data_point and 'ic50' in data_point['label']]
    #     #             self.ligands_dataset[uniprot_id_2] = self.ligands_dataset[best_match]  # Cache the result for next iter
    #     #         else:
    #     #             self.proteins_without_ligands.append(uniprot_id_2)
    #
    #     # if uniprot_id_2 not in self.proteins_with_ligands_ids and ligand_list_2:
    #     #     self.proteins_with_ligands_ids.append(uniprot_id_2)
    #     #     self.proteins_with_ligands_indexes.append(index)
    #     #     print("proteins with ligands:", len(self.proteins_with_ligands_indexes))
    #     #
    #     # if not ligand_list_1 and not ligand_list_2:
    #     #     if self.proteins_with_ligands_indexes:
    #     #         new_index = random.sample(self.proteins_with_ligands_indexes, 1)[0]  # change to iterate pass
    #     #         print("new_index:", new_index)
    #     #         return self.__getitem__(new_index)
    #
    #     # Mask structure tokens with pLDDT < threshold
    #     if self.plddt_threshold is not None:
    #         plddt_1, plddt_2 = entry['plddt_1'], entry['plddt_2']
    #         tokens = self.tokenizer.tokenize(seq_1)
    #         seq_1 = ""
    #         for token, score in zip(tokens, plddt_1):
    #             if score < self.plddt_threshold:
    #                 seq_1 += token[:-1] + "#"
    #             else:
    #                 seq_1 += token
    #
    #         tokens = self.tokenizer.tokenize(seq_2)
    #         seq_2 = ""
    #         for token, score in zip(tokens, plddt_2):
    #             if score < self.plddt_threshold:
    #                 seq_2 += token[:-1] + "#"
    #             else:
    #                 seq_2 += token
    #
    #     tokens = self.tokenizer.tokenize(seq_1)[:self.max_length]
    #     seq_1 = " ".join(tokens)
    #
    #     tokens = self.tokenizer.tokenize(seq_2)[:self.max_length]
    #     seq_2 = " ".join(tokens)
    #
    #     return seq_1, seq_2, int(entry["label"]), ligand_list_1, ligand_list_2

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs_1, seqs_2, label_ids, ligand_list_1, ligand_list_2 = tuple(zip(*batch))

        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}

        encoder_info_1 = self.tokenizer.batch_encode_plus(seqs_1, return_tensors='pt', padding=True)
        encoder_info_2 = self.tokenizer.batch_encode_plus(seqs_2, return_tensors='pt', padding=True)
        inputs = {"inputs_1": encoder_info_1,
                  "inputs_2": encoder_info_2}

        ligands = {"ligands_1": ligand_list_1,
                   "ligands_2": ligand_list_2}

        return inputs, labels, ligands
