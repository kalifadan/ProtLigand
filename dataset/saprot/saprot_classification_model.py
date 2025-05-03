import torchmetrics
import torch

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None, ligands=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)

        else:
            # logits = self.model(**inputs).logits

            output = self.model.esm(**inputs)
            hidden = output[0]

            # Ligand-Protein Transformer
            if [] not in ligands:
                ligands_embeddings, ligands_labels = self.process_ligands(ligands)
                ligands_embeddings = ligands_embeddings.squeeze(0)
                # ligands_labels = torch.tensor(ligands_labels, dtype=torch.float32).to(self.model.device)
            else:
                # ligands_embeddings = self.default_ligand.unsqueeze(0)
                # ligands_labels = self.default_ligand_label.unsqueeze(0)
                ligands_embeddings = self.ligand_generator(hidden[:, 0, :])

            ligands_embeddings = self.ligand_proj(ligands_embeddings)  # [batch, ligand_dim] → [batch, hidden_dim]
            ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)  # Expand for attention

            # Apply cross-attention (Protein as Query, Ligands as Key/Value)
            attn_output, _ = self.cross_attention(
                query=hidden,  # Protein embeddings
                key=ligands_embeddings,
                value=ligands_embeddings
            )

            hidden = hidden + attn_output  # Residual connection
            logits = self.model.classifier(hidden)

        return logits

    def loss_func(self, stage, logits, labels, inputs=None, ligands=None, info=None):
        label = labels['labels']
        task_loss = cross_entropy(logits, label)
        loss = task_loss

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            log_dict["task_loss"] = task_loss
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
        self.check_save_condition(log_dict["valid_acc"], mode="max")
