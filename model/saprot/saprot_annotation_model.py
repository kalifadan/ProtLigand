import torchmetrics
import torch


from torch.nn.functional import binary_cross_entropy_with_logits
from utils.metrics import count_f1_max
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotAnnotationModel(SaprotBaseModel):
    def __init__(self, anno_type: str, **kwargs):
        """
        Args:
            anno_type: one of EC, GO, GO_MF, GO_CC
            **kwargs: other parameters for SaprotBaseModel
        """
        label2num = {"EC": 585, "GO_BP": 1943, "GO_MF": 489, "GO_CC": 320}
        self.num_labels = label2num[anno_type]
        super().__init__(task="classification", **kwargs)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_aupr": torchmetrics.AveragePrecision(pos_label=1, average='micro')}

    def forward(self, inputs, ligands_info=None, coords=None, ligands=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        # Ligand-Protein Transformer
        if [] not in ligands:
            print("got ligands for protein!")
            ligands_embeddings, ligands_labels = self.process_ligands(ligands)
            ligands_embeddings = ligands_embeddings.squeeze(0)
            ligands_labels = torch.tensor(ligands_labels, dtype=torch.float32).to(self.model.device)
        else:
            ligands_embeddings = self.default_ligand.unsqueeze(0)
            ligands_labels = self.default_ligand_label.unsqueeze(0)

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

            # Ligands-Protein Transformer
            output = self.model.esm(**inputs)
            hidden = output[0]
            ligands_embeddings = ligands_embeddings.unsqueeze(1).expand(-1, hidden.size(1), -1)
            ligands_labels = ligands_labels.unsqueeze(1).expand(-1, hidden.size(1), -1)
            hidden = self.ligand_protein_transformer(torch.cat([hidden, ligands_embeddings, ligands_labels], dim=-1))
            logits = self.model.classifier(hidden)

        return logits

    def loss_func(self, stage, logits, labels, inputs=None, ligands=None):
        label = labels['labels'].to(logits)
        task_loss = binary_cross_entropy_with_logits(logits, label.float())
        aupr = getattr(self, f"{stage}_aupr")(logits.sigmoid().detach(), label)

        # proteins = inputs['inputs']
        # generator_loss, discriminator_real_loss, discriminator_fake_loss = self.protein_ligand_loss(proteins, ligands)
        loss = task_loss     # + 0.1 * generator_loss + 0.05 * (discriminator_real_loss + discriminator_fake_loss)

        # proteins = inputs['inputs']
        # generator_loss = self.protein_new_ligand_loss(proteins, ligands)

        # if generator_loss is None:
        #     loss = task_loss
        # else:
        #     loss = generator_loss   # + task_loss

        if stage == "train":
            log_dict = {"train_loss": loss}
            log_dict["task_loss"] = task_loss
            # log_dict["generator_loss"] = generator_loss
            # log_dict["discriminator_real_loss"] = discriminator_real_loss
            # log_dict["discriminator_fake_loss"] = discriminator_fake_loss
            self.log_info(log_dict)
            self.reset_metrics("train")
        
        return loss
    
    def test_epoch_end(self, outputs):
        preds = self.all_gather(torch.cat(self.test_aupr.preds, dim=-1)).view(-1, self.num_labels)
        target = self.all_gather(torch.cat(self.test_aupr.target, dim=-1)).long().view(-1, self.num_labels)
        fmax = count_f1_max(preds, target)
        
        log_dict = {"test_f1_max": fmax,
                    "test_loss": torch.cat(self.all_gather(outputs), dim=-1).mean(),
                    # "test_aupr": self.test_aupr.compute()
                    }
        self.log_info(log_dict)
        print(log_dict)
        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        aupr = self.valid_aupr.compute()

        preds = self.all_gather(torch.cat(self.valid_aupr.preds, dim=-1)).view(-1, self.num_labels)
        target = self.all_gather(torch.cat(self.valid_aupr.target, dim=-1)).long().view(-1, self.num_labels)
        f1_max = count_f1_max(preds, target)
        
        log_dict = {"valid_f1_max": f1_max,
                    "valid_loss": torch.cat(self.all_gather(outputs), dim=-1).mean(),
                    # "valid_aupr": aupr
                    }
        
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_f1_max"], mode="max")