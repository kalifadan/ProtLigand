import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import cross_entropy, cosine_similarity
import abc
import os
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM

import pytorch_lightning as pl
from utils.lr_scheduler import Esm2LRScheduler
from torch import distributed as dist

# seed = 20000812
# random.seed(seed)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x) / (x.shape[-1] ** 0.5), dim=1)  # Scaling factor for stability
        return (attn_weights * x).sum(dim=1)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class AbstractModel(pl.LightningModule):
    def __init__(self,
                 lr_scheduler_kwargs: dict = None,
                 optimizer_kwargs: dict = None,
                 save_path: str = None,
                 from_checkpoint: str = None,
                 load_prev_scheduler: bool = False,
                 save_weights_only: bool = True,):
        """

        Args:
            lr_scheduler: Kwargs for lr_scheduler
            optimizer_kwargs: Kwargs for optimizer_kwargs
            save_path: Save trained model
            from_checkpoint: Load model from checkpoint
            load_prev_scheduler: Whether load previous scheduler from save_path
            load_strict: Whether load model strictly
            save_weights_only: Whether save only weights or also optimizer and lr_scheduler

        """
        super().__init__()
        self.initialize_model()
        
        self.metrics = {}
        for stage in ["train", "valid", "test"]:
            stage_metrics = self.initialize_metrics(stage)
            # Register metrics as attributes
            for metric_name, metric in stage_metrics.items():
                setattr(self, metric_name, metric)
                
            self.metrics[stage] = stage_metrics

        self.lr_scheduler_kwargs = {"init_lr": 0} if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.init_optimizers()

        self.save_path = save_path
        self.save_weights_only = save_weights_only

        self.step = 0
        self.epoch = 0

        # ProtLigand Parameters
        self.ligand_tokenizer = AutoTokenizer.from_pretrained("pchanda/pretrained-smiles-pubchem10m")
        self.ligand_model = AutoModelForMaskedLM.from_pretrained("pchanda/pretrained-smiles-pubchem10m")

        # self.ligand_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        # self.ligand_model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

        # Freeze the ligand model during training
        for param in self.ligand_model.parameters():
            param.requires_grad = False

        protein_hidden_size = self.model.config.hidden_size
        ligand_hidden_size = self.ligand_model.config.hidden_size

        input_size = protein_hidden_size + ligand_hidden_size  # + 4

        self.max_ligands = 1       # 200

        self.default_ligand = nn.Parameter(torch.randn(ligand_hidden_size) * 0.01)

        # Cross-attention module: Protein (Q) attends to Ligand (K, V)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=self.model.config.hidden_size,  # Hidden size of protein embeddings
            num_heads=4,  # Number of attention heads
            batch_first=True
        )

        # Layer Normalization
        # self.norm = torch.nn.LayerNorm(self.model.config.hidden_size)  # Normalize over the last dimension
        self.attention_pooling = AttentionPooling(self.model.config.hidden_size)  # Use attention pooling

        # Binding Affinity Prediction Head
        self.binding_affinity_predictor = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.binding_affinity_predictor.apply(initialize_weights)

        self.ligand_proj = torch.nn.Linear(ligand_hidden_size, protein_hidden_size)

        self.ligand_generator = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=protein_hidden_size, nhead=4, dim_feedforward=512, dropout=0.1,
                                           batch_first=True), num_layers=2),
            nn.Linear(protein_hidden_size, ligand_hidden_size)
        )

        self.load_prev_scheduler = load_prev_scheduler
        if from_checkpoint:
            self.load_checkpoint(from_checkpoint, load_prev_scheduler)

    @abc.abstractmethod
    def initialize_model(self) -> None:
        """
        All model initialization should be done here
        Note that the whole model must be named as "self.model" for model saving and loading
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward propagation
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_metrics(self, stage: str) -> dict:
        """
        Initialize metrics for each stage
        Args:
            stage: "train", "valid" or "test"
        
        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_func(self, stage: str, outputs, labels, inputs=None, ligands=None, info=None) -> torch.Tensor:
        """

        Args:
            stage: "train", "valid" or "test"
            outputs: model outputs for calculating loss
            labels: labels for calculating loss
            inputs: inputs for calculating loss
            ligands: associated ligands for calculating loss

        Returns:
            loss

        """
        raise NotImplementedError

    def process_ligands(self, ligands_info):
        # print("number of given ligands:", max(len(ligands) for ligands in ligands_info))
        batch_size = len(ligands_info)
        max_ligands = min(self.max_ligands, max(len(ligands) for ligands in ligands_info))
        embedding_dim = self.ligand_model.config.hidden_size
        ligand_embeddings = torch.zeros(batch_size, max_ligands, embedding_dim).to(device=self.ligand_model.device)
        all_labels = []

        for i, ligands in enumerate(ligands_info):
            # batch_labels = []
            smiles_list = []

            num_ligands_to_select = min(self.max_ligands, len(ligands))
            ligands = random.sample(ligands, num_ligands_to_select)

            # ligands = ligands[:num_ligands_to_select]

            for smiles, label in ligands:
                smiles_list.append(smiles)

                # TODO: CHECK IT
                all_labels.append(label)

            # all_labels.append(batch_labels)

            if smiles_list:
                ligand_inputs = self.ligand_tokenizer(smiles_list, return_tensors="pt", padding=True,
                                                      truncation=True).to(self.model.device)
                ligands_outputs = self.ligand_model(**ligand_inputs, output_hidden_states=True)
                final_hidden_state = ligands_outputs.hidden_states[-1].to(self.model.device)
                ligand_representations = final_hidden_state.mean(dim=1)
                ligand_embeddings[i, :ligand_representations.size(0)] = ligand_representations

        return ligand_embeddings, all_labels

    @staticmethod
    def load_weights(model, weights):
        model_dict = model.state_dict()

        unused_params = []
        missed_params = list(model_dict.keys())

        for k, v in weights.items():
            if k in model_dict.keys():
                model_dict[k] = v
                missed_params.remove(k)

            else:
                unused_params.append(k)

        if len(missed_params) > 0:
            print(f"\033[31mSome weights of {type(model).__name__} were not "
                  f"initialized from the model checkpoint: {missed_params}\033[0m")

        if len(unused_params) > 0:
            print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

        model.load_state_dict(model_dict)
    
    # Add 1 to step after each optimizer step
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int = 0,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
        )
        self.step += 1

    def on_train_epoch_end(self):
        self.epoch += 1

    def training_step(self, batch, batch_idx):
        inputs, labels, ligands, info = batch
        outputs = self(**inputs, ligands=ligands)
        loss = self.loss_func('train', outputs, labels, inputs, ligands=ligands, info=info)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels, ligands, info = batch
        outputs = self(**inputs, ligands=ligands)
        loss = self.loss_func('valid', outputs, labels, inputs, ligands=ligands, info=info)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels, ligands, info = batch
        outputs = self(**inputs, ligands=ligands)
        loss = self.loss_func('test', outputs, labels, inputs, ligands=ligands, info=info)
        return loss

    def load_checkpoint(self, from_checkpoint, load_prev_scheduler):
        state_dict = torch.load(from_checkpoint, map_location=self.device)
        self.load_weights(self.model, state_dict["model"])

        # Load additional modules dynamically
        # ("ligand_protein_transformer", self.ligand_protein_transformer),
        # ("binding_affinity_predictor", self.binding_affinity_predictor),
        for key, module in [
            ("cross_attention", self.cross_attention),
            ("ligand_proj", self.ligand_proj),
        ]:
            if key in state_dict:
                module.load_state_dict(state_dict[key])

        # Restore nn.Parameter (default_ligand)
        if "default_ligand" in state_dict:
            self.default_ligand = state_dict["default_ligand"]

        # TODO: ADD TO PARAMETERS!
        generator_path = "weights/Pretrain/final_ligand_generator_model.pt"
        generator_state_dict = torch.load(generator_path, map_location=self.device)
        self.ligand_generator.load_state_dict(generator_state_dict["ligand_generator"])

        if load_prev_scheduler:
            try:
                self.step = state_dict["global_step"]
                self.epoch = state_dict["epoch"]
                self.best_value = state_dict["best_value"]
                self.optimizer.load_state_dict(state_dict["optimizer"])
                self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
                print(f"Previous training global step: {self.step}")
                print(f"Previous training epoch: {self.epoch}")
                print(f"Previous best value: {self.best_value}")
                print(f"Previous lr_scheduler: {state_dict['lr_scheduler']}")
            
            except Exception as e:
                print(e)
                raise KeyError("Wrong in loading previous scheduler, please set load_prev_scheduler=False")

    def save_checkpoint(self, save_info: dict = None) -> None:
        """
        Save model to save_path
        Args:
            save_info: Other info to save
        """
        state_dict = {} if save_info is None else save_info
        state_dict["model"] = self.model.state_dict()

        if not self.save_weights_only:
            state_dict["global_step"] = self.step
            state_dict["epoch"] = self.epoch
            state_dict["best_value"] = getattr(self, f"best_value", None)
            state_dict["optimizer"] = self.optimizers().optimizer.state_dict()
            state_dict["lr_scheduler"] = self.lr_schedulers().state_dict()

        torch.save(state_dict, self.save_path)

    def check_save_condition(self, now_value: float, mode: str, save_info: dict = None) -> None:
        """
        Check whether to save model. If save_path is not None and now_value is the best, save model.
        Args:
            now_value: Current metric value
            mode: "min" or "max", meaning whether the lower the better or the higher the better
            save_info: Other info to save
        """

        assert mode in ["min", "max"], "mode should be 'min' or 'max'"

        if self.save_path is not None:
            dir = os.path.dirname(self.save_path)
            os.makedirs(dir, exist_ok=True)
            
            if dist.get_rank() == 0:
                # save the best checkpoint
                best_value = getattr(self, f"best_value", None)
                if best_value:
                    if mode == "min" and now_value < best_value or mode == "max" and now_value > best_value:
                        setattr(self, "best_value", now_value)
                        self.save_checkpoint(save_info)

                else:
                    setattr(self, "best_value", now_value)
                    self.save_checkpoint(save_info)
    
    def reset_metrics(self, stage) -> None:
        """
        Reset metrics for given stage
        Args:
            stage: "train", "valid" or "test"
        """
        for metric in self.metrics[stage].values():
            metric.reset()
    
    def get_log_dict(self, stage: str) -> dict:
        """
        Get log dict for the stage
        Args:
            stage: "train", "valid" or "test"

        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric values

        """
        return {name: metric.compute() for name, metric in self.metrics[stage].items()}
    
    def log_info(self, info: dict) -> None:
        """
        Record metrics during training and testing
        Args:
            info: dict of metrics
        """
        if getattr(self, "logger", None) is not None:
            info["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            info["epoch"] = self.epoch
            self.logger.log_metrics(info, step=self.step)

    def init_optimizers(self):
        # No decay for layer norm and bias
        no_decay = ['LayerNorm.weight', 'bias']
        
        if "weight_decay" in self.optimizer_kwargs:
            weight_decay = self.optimizer_kwargs.pop("weight_decay")
        else:
            weight_decay = 0.01
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                           lr=self.lr_scheduler_kwargs['init_lr'],
                                           **self.optimizer_kwargs)

        self.lr_scheduler = Esm2LRScheduler(self.optimizer, **self.lr_scheduler_kwargs)
    
    def configure_optimizers(self):
        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.lr_scheduler,
                                 "interval": "step",
                                 "frequency": 1}
                }
