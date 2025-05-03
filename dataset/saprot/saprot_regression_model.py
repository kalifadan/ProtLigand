import torch.distributed as dist
import os
import torchmetrics
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import cross_entropy, cosine_similarity
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        super().__init__(task="regression", **kwargs)
        self.test_result_path = test_result_path

    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def forward(self, inputs, structure_info=None, ligands=None):
        if structure_info:
            # To be implemented
            raise NotImplementedError

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            representations = torch.stack(self.get_hidden_states(inputs, reduction="mean"))
            x = self.model.classifier.dropout(representations)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x).squeeze(dim=-1)
        else:
            # logits = self.model(**inputs).logits.squeeze(dim=-1)

            output = self.model.esm(**inputs)
            hidden = output[0]

            # # Ligand-Protein Transformers
            if [] not in ligands:
                ligands_embeddings, ligands_labels = self.process_ligands(ligands)
                ligands_embeddings = ligands_embeddings.squeeze(0)
            else:
                ligands_embeddings = self.ligand_generator(hidden[:, 0, :])

            ligands_embeddings = self.ligand_proj(ligands_embeddings)  # [batch, ligand_dim] → [batch, hidden_dim]
            ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)  # Expand for attention

            # Apply cross-attention (Protein as Query, Ligands as Key/Value)
            attn_output, _ = self.cross_attention(
                query=hidden,  # Protein embeddings
                key=ligands_embeddings,
                value=ligands_embeddings
            )

            # attn_output = self.norm(attn_output)  # Optional normalization before addition
            hidden = hidden + attn_output  # Residual connection
            logits = self.model.classifier(hidden).squeeze(dim=-1)

        return logits

    def loss_func(self, stage, outputs, labels, inputs=None, ligands=None, info=None):
        fitness = labels['labels'].to(outputs)
        task_loss = torch.nn.functional.mse_loss(outputs, fitness)

        if stage == "test" and self.test_result_path is not None:
            os.makedirs(os.path.dirname(self.test_result_path), exist_ok=True)
            with open(self.test_result_path, 'a') as w:
                uniprot_id, protein_type = info[0]
                w.write(f"{uniprot_id}\t{protein_type}\t{outputs.detach().float().item()}\t{fitness.float().item()}\n")

        loss = task_loss
        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(outputs.detach().float(), fitness.float())

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["loss"] = loss
            log_dict["task_loss"] = task_loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        print(log_dict)

        self.log_info(log_dict)
        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
