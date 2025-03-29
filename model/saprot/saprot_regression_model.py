import torch.distributed as dist
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

        # # Ligand-Protein Transformers
        # if [] not in ligands:
        #     ligands_embeddings, ligands_labels = self.process_ligands(ligands)
        #     ligands_embeddings = ligands_embeddings.squeeze(0)
        #     # ligands_labels = torch.tensor(ligands_labels, dtype=torch.float32).to(self.model.device)
        # else:
        #     ligands_embeddings = self.default_ligand.unsqueeze(0)
        #     # ligands_labels = self.default_ligand_label.unsqueeze(0)
        #
        # # ligands_labels = self.label_adapter(ligands_labels)

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            representations = torch.stack(self.get_hidden_states(inputs, reduction="mean"))
            x = self.model.classifier.dropout(representations)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x).squeeze(dim=-1)
        else:
            logits = self.model(**inputs).logits.squeeze(dim=-1)
            # output = self.model.esm(**inputs)
            # hidden = output[0]
            # ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)
            # # ligands_labels = ligands_labels.unsqueeze(1).expand(-1, hidden.size(1), -1)
            # hidden = self.ligand_protein_transformer(torch.cat([hidden, ligands_embeddings], dim=-1))
            # logits = self.model.classifier(hidden).squeeze(dim=-1)

        return logits

    def loss_func(self, stage, outputs, labels, inputs=None, ligands=None):
        fitness = labels['labels'].to(outputs)
        task_loss = torch.nn.functional.mse_loss(outputs, fitness)

        # proteins = inputs['inputs']
        # generator_loss = self.protein_new_ligand_loss(proteins, ligands)

        loss = task_loss   # + 0.1 * generator_loss + 0.05 * (discriminator_real_loss + discriminator_fake_loss)

        # if generator_loss is not None:
        #     loss = generator_loss

        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.update(outputs.detach().float(), fitness.float())

        if stage == "train":
            # Skip calculating metrics if the batch size is 1
            # if fitness.shape[0] > 1:
            log_dict = self.get_log_dict("train")
            log_dict["loss"] = loss
            log_dict["task_loss"] = task_loss
            # log_dict["generator_loss"] = generator_loss
            # log_dict["discriminator_real_loss"] = discriminator_real_loss
            # log_dict["discriminator_fake_loss"] = discriminator_fake_loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        if self.test_result_path is not None:
            from torchmetrics.utilities.distributed import gather_all_tensors

            preds = self.test_spearman.preds
            preds[-1] = preds[-1].unsqueeze(dim=0) if preds[-1].shape == () else preds[-1]
            preds = torch.cat(gather_all_tensors(torch.cat(preds, dim=0)))

            targets = self.test_spearman.target
            targets[-1] = targets[-1].unsqueeze(dim=0) if targets[-1].shape == () else targets[-1]
            targets = torch.cat(gather_all_tensors(torch.cat(targets, dim=0)))

            if dist.get_rank() == 0:
                with open(self.test_result_path, 'w') as w:
                    w.write("pred\ttarget\n")
                    for pred, target in zip(preds, targets):
                        w.write(f"{pred.item()}\t{target.item()}\n")

        log_dict = self.get_log_dict("test")

        print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
