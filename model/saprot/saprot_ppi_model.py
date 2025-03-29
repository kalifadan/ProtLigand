import numpy as np
import torchmetrics
import torch

from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import cross_entropy, cosine_similarity
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotPPIModel(SaprotBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: other arguments for SaprotBaseModel
        """
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()
        
        hidden_size = self.model.config.hidden_size * 2
        classifier = torch.nn.Sequential(
                        Linear(hidden_size, hidden_size),
                        ReLU(),
                        Linear(hidden_size, 2)
                    )
        
        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs_1, inputs_2, ligands=None):
        if self.freeze_backbone:
            hidden_1 = torch.stack(self.get_hidden_states(inputs_1, reduction="mean"))
            hidden_2 = torch.stack(self.get_hidden_states(inputs_2, reduction="mean"))
        else:
            hidden_1 = self.model.esm(**inputs_1)[0][:, 0, :]
            hidden_2 = self.model.esm(**inputs_2)[0][:, 0, :]

        # Ligand-Protein Transformer
        # if [] not in ligands['ligands_1']:
        #     ligands_embeddings_1, ligands_labels_1 = self.process_ligands(ligands['ligands_1'])
        #     ligands_embeddings_1 = ligands_embeddings_1.squeeze(0)
        #     # ligands_labels_1 = torch.tensor(ligands_labels_1, dtype=torch.float32).to(self.model.device)
        #     # ligands_labels_1 = self.label_adapter(ligands_labels_1)
        # else:
        #     ligands_embeddings_1 = self.default_ligand.unsqueeze(0)
        #     # ligands_labels_1 = self.default_ligand_label.unsqueeze(0)
        #
        # if [] not in ligands['ligands_2']:
        #     ligands_embeddings_2, ligands_labels_2 = self.process_ligands(ligands['ligands_2'])
        #     ligands_embeddings_2 = ligands_embeddings_2.squeeze(0)
        #     # ligands_labels_2 = torch.tensor(ligands_labels_2, dtype=torch.float32).to(self.model.device)
        #     # ligands_labels_2 = self.label_adapter(ligands_labels_2)
        # else:
        #     ligands_embeddings_2 = self.default_ligand.unsqueeze(0)
        #     # ligands_labels_2 = self.default_ligand_label.unsqueeze(0)
        #
        # # hidden_1 = self.ligand_protein_transformer(torch.cat([hidden_1, ligands_embeddings_1, ligands_labels_1], dim=-1))
        # # hidden_2 = self.ligand_protein_transformer(torch.cat([hidden_2, ligands_embeddings_2, ligands_labels_2], dim=-1))
        #
        # hidden_1 = self.ligand_protein_transformer(torch.cat([hidden_1, ligands_embeddings_1], dim=-1))
        # hidden_2 = self.ligand_protein_transformer(torch.cat([hidden_2, ligands_embeddings_2], dim=-1))

        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        return self.model.classifier(hidden_concat)
    
    def loss_func(self, stage, logits, labels, inputs=None, ligands=None):
        label = labels['labels']
        task_loss = cross_entropy(logits, label)

        # protein_1, ligands_1 = inputs['inputs_1'], ligands['ligands_1']
        # generator_loss_1 = self.protein_embeddings_sim_ligand_loss(protein_1, ligands_1)
        #
        # protein_2, ligands_2 = inputs['inputs_2'], ligands['ligands_2']
        # generator_loss_2 = self.protein_embeddings_sim_ligand_loss(protein_2, ligands_2)
        #
        # # Return task_loss if both losses are None
        # if generator_loss_1 is None and generator_loss_2 is None:
        #     print("both proteins are unknown!")
        #     loss = cross_entropy(logits, label)
        # else:
        #     if generator_loss_1 is None:
        #         loss = generator_loss_2     # + task_loss
        #     elif generator_loss_2 is None:
        #         loss = generator_loss_1     # + task_loss
        #     else:
        #         loss = 0.5 * (generator_loss_1 + generator_loss_2)  # + task_loss

        loss = task_loss
        # protein_1, ligands_1 = inputs['inputs_1'], ligands['ligands_1']
        # generator_loss_1, disc_real_loss_1, disc_fake_loss_1 = self.protein_ligand_loss(protein_1, ligands_1)
        # protein_2, ligands_2 = inputs['inputs_2'], ligands['ligands_2']
        # generator_loss_2, disc_real_loss_2, disc_fake_loss_2 = self.protein_ligand_loss(protein_2, ligands_2)
        # loss += 0.1 * (generator_loss_1 + generator_loss_2)
        # loss += 0.05 * (disc_real_loss_1 + disc_fake_loss_1 + disc_real_loss_2 + disc_fake_loss_1)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            # log_dict["original_task_loss"] = task_loss
            # log_dict["generator_loss"] = generator_loss_1 + generator_loss_2
            # log_dict["discriminator_real_loss"] = disc_real_loss_1 + disc_real_loss_2
            # log_dict["discriminator_fake_loss"] = disc_fake_loss_1 + disc_fake_loss_2
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        print(log_dict)
        self.log_info(log_dict)

        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        self.log_info(log_dict)
        self.reset_metrics("valid")

        self.check_save_condition(log_dict["valid_acc"], mode="max")        # TODO: MAYBE TO CHANGE TO VALID LOSS
        # self.check_save_condition(log_dict["valid_loss"], mode="min")        # TODO: MAYBE TO CHANGE TO VALID LOSS
