import torch
import json
import random
import pickle

from ..data_interface import register_dataset
from transformers import EsmTokenizer
from ..lmdb_dataset import *
from ..lmdb_dataset import *
from utils.others import setup_seed


@register_dataset
class SaprotRegressionDataset(LMDBDataset):
	def __init__(self,
	             tokenizer: str,
	             max_length: int = 1024,
	             min_clip: [float, float] = None,
	             mix_max_norm: [float, float] = None,
				 mask_struc_ratio: float = None,
	             plddt_threshold: float = None,
	             **kwargs):
		"""
		
		Args:
			tokenizer: ESM tokenizer

			max_length: Maximum length of the sequence

			min_clip: [given_value, clip_value]
					  Set the fitness value to a fixed value if it is less than a given value
			
			mix_max_norm: [min_norm, max_norm]
						  Normalize the fitness value to [0, 1] by min-max normalization

			mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
			
			plddt_threshold: If not None, mask structure tokens with pLDDT < threshold

			**kwargs:
		"""
		
		super().__init__(**kwargs)
		self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
		self.max_length = max_length
		self.min_clip = min_clip
		self.mix_max_norm = mix_max_norm
		self.mask_struc_ratio = mask_struc_ratio
		self.plddt_threshold = plddt_threshold

		self.proteins_with_ligands_ids = []

	def __getitem__(self, index):
		entry = json.loads(self._get(index))
		seq = entry['seq']

		# Ligands Extraction
		uniprot_id, ligand_list = entry['name'], []
		ligand_list = self.pdbbind_df[self.pdbbind_df['uniprot_id'] == uniprot_id][["smiles", "value", "ic50",
																					"kd", "ki"]].values.tolist()
		ligand_list = [(item[0], [item[1], item[2], item[3], item[4]]) for item in ligand_list]

		if ligand_list and uniprot_id not in self.proteins_with_ligands_ids:
			self.proteins_with_ligands_ids.append(uniprot_id)
			print("proteins with ligands:", len(self.proteins_with_ligands_ids))

		# if uniprot_id in self.ligands_dataset:
		# 	print("got here:", uniprot_id)
		# 	ligand_list = [(data_point['smi'], data_point['label']['ic50']) for data_point in
		# 				   self.ligands_dataset[uniprot_id] if 'label' in data_point and 'ic50' in data_point['label']]
		# 	print("ligand_list:", len(ligand_list))
		#
		# else:
		# 	if uniprot_id not in self.proteins_without_ligands:
		# 		best_match, best_score = self.find_most_similar(uniprot_id, self.fetch_uniprot_sequence(uniprot_id), self.protein_sequences)
		# 		if best_score >= self.ligand_score_th:
		# 			ligand_list = [(data_point['smi'], data_point['label']['ic50']) for data_point in
		# 						   self.ligands_dataset[best_match] if
		# 						   'label' in data_point and 'ic50' in data_point['label']]
		# 			self.ligands_dataset[uniprot_id] = self.ligands_dataset[best_match]	  # cache the result for next iter
		# 		else:
		# 			self.proteins_without_ligands.append(uniprot_id)
		#
		# if index not in self.proteins_with_ligands_indexes and ligand_list:
		# 	print("new protein received")
		# 	self.proteins_with_ligands_indexes.append(index)
		# 	print("proteins with ligands:", len(self.proteins_with_ligands_indexes))
		#
		# if not ligand_list:
		# 	if self.proteins_with_ligands_indexes:
		# 		new_index = random.sample(self.proteins_with_ligands_indexes, 1)[0]  # change to iterate pass
		# 		print("new_index:", new_index)
		# 		return self.__getitem__(new_index)

		# Mask structure tokens
		if self.mask_struc_ratio is not None:
			tokens = self.tokenizer.tokenize(seq)
			mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]

			# Randomly shuffle the mask candidates and set seed to ensure mask is consistent
			setup_seed(20000812)
			random.shuffle(mask_candi)

			# Mask first n structure tokens
			mask_num = int(len(mask_candi) * self.mask_struc_ratio)
			for i in range(mask_num):
				idx = mask_candi[i]
				tokens[idx] = tokens[idx][:-1] + "#"

			seq = "".join(tokens)
		
		# Mask structure tokens with pLDDT < threshold
		if self.plddt_threshold is not None:
			plddt = entry["plddt"]
			tokens = self.tokenizer.tokenize(seq)
			seq = ""
			for token, score in zip(tokens, plddt):
				if score < self.plddt_threshold:
					seq += token[:-1] + "#"
				else:
					seq += token

		tokens = self.tokenizer.tokenize(seq)[:self.max_length]
		seq = " ".join(tokens)
	
		if self.min_clip is not None:
			given_min, clip_value = self.min_clip
			if entry['fitness'] < given_min:
				entry['fitness'] = clip_value
		
		if self.mix_max_norm is not None:
			min_norm, max_norm = self.mix_max_norm
			entry['fitness'] = (entry['fitness'] - min_norm) / (max_norm - min_norm)
				
		label = entry['fitness']
		return seq, label, ligand_list
	
	def __len__(self):
		return int(self._get("length"))
	
	def collate_fn(self, batch):
		seqs, labels, ligand_list = tuple(zip(*batch))
		labels = torch.tensor(labels)
		labels = {"labels": labels}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
		inputs = {"inputs": encoder_info}

		return inputs, labels, ligand_list
