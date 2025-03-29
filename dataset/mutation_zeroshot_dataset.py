import json
import torch

from .data_interface import register_dataset
from .lmdb_dataset import LMDBDataset


@register_dataset
class MutationZeroShotDataset(LMDBDataset):
    """
    Dataset that deals with mutation data for zero-shot prediction
    """

    def __init__(self, **kwargs):
        """

        Args: **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)

        self.proteins_without_ligands = []
        self.proteins_with_ligands_ids = []
        self.proteins_with_ligands_indexes = []

    def __getitem__(self, index):
        data = json.loads(self._get(index))

        # Ligands Extraction
        # print(data)
        # uniprot_seq, ligand_list = data['seq'], []
        # if uniprot_id in self.ligands_dataset:
        #     print("got here:", uniprot_id)
        #     ligand_list = [(data_point['smi'], data_point['label']['ic50']) for data_point in
        #                    self.ligands_dataset[uniprot_id] if 'label' in data_point and 'ic50' in data_point['label']]
        #     print("ligand_list:", len(ligand_list))
        #     if uniprot_id not in self.proteins_with_ligands_ids and ligand_list:
        #         self.proteins_with_ligands_ids.append(uniprot_id)
        #         print("proteins with ligands:", len(self.proteins_with_ligands_ids))

        return data["seq"], data["mut_info"], data["fitness"]

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, mut_info, fitness = zip(*batch)
        
        plddt = self._get("plddt")
        if plddt is not None:
            plddt = json.loads(plddt)
        
        inputs = {"wild_type": self._get("wild_type"),
                  "seqs": seqs,
                  "mut_info": mut_info,
                  "structure_content": self._get("structure_content"),
                  "structure_type": self._get("structure_type"),
                  "plddt": plddt}

        labels = {"labels": torch.Tensor(fitness)}

        return inputs, labels, None
