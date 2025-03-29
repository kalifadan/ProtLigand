import torch
import os
import torchmetrics

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from transformers.modeling_outputs import MaskedLMOutput


@register_model
class SaprotLMModel(SaprotBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)

    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}
    
    def forward(self, inputs, coords=None, ligands=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        outputs = self.model(**inputs)

        ligands_embeddings, ligands_labels = self.process_ligands(ligands)
        ligands_embeddings = ligands_embeddings.squeeze(0)
        # ligands_labels = torch.tensor(ligands_labels, dtype=torch.float32).to(self.model.device)

        output = self.model.esm(**inputs)
        hidden = output[0]

        ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)

        hidden = self.ligand_protein_transformer(torch.cat([hidden, ligands_embeddings], dim=-1))
        logits = self.model.lm_head(hidden)

        # Get hidden representations
        # if "output_hidden_states" in inputs and inputs["output_hidden_states"]:
        #     input_ids = inputs["input_ids"]
        #     ends = (input_ids == 2).int()
        #     indices = ends.argmax(dim=-1)
        #     repr_list = []
        #     hidden_states = outputs["hidden_states"][-1]
        #     for i, idx in enumerate(indices):
        #         repr = hidden_states[i][1:idx].mean(dim=0)
        #         repr_list.append(repr)
        #
        #     reprs = torch.stack(repr_list, dim=0)
        #     outputs["hidden_states"] = reprs

        outputs = MaskedLMOutput(
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
        
        return outputs
    
    def loss_func(self, stage, outputs, labels, inputs=None, ligands=None):
        logits = outputs['logits']
        # merge the first and second dimension of logits
        logits = logits.view(-1, logits.size(-1))
        
        # flatten labels
        labels = labels['labels'].flatten().to(logits.device)

        loss = cross_entropy(logits, labels, ignore_index=-1)
        getattr(self, f"{stage}_acc").update(logits.detach(), labels)

        # Ligand loss
        # ligands_embeddings, ligands_labels = self.process_ligands(ligands)
        # ligands_embeddings = ligands_embeddings.squeeze(0)
        # ligands_labels = torch.tensor(ligands_labels, dtype=torch.float32).to(self.model.device)
        # hidden = torch.stack(self.get_hidden_states(inputs["inputs"], reduction="mean"))
        # predictions = self.ligand_protein_transformer(torch.cat([hidden, ligands_embeddings], dim=-1))
        # generator_ba_loss = torch.nn.functional.mse_loss(predictions, ligands_labels)

        task_loss = loss
        # loss += generator_ba_loss

        if stage == 'train':
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            log_dict["train_mlm_loss"] = task_loss
            # log_dict["train_mse_ba_loss"] = generator_ba_loss
            self.log_info(log_dict)
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
        self.check_save_condition(log_dict["valid_loss"], mode="min")
