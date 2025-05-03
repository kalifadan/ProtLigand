import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import cross_entropy, cosine_similarity
import abc
import os
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding
import torch.nn.functional as F

import pytorch_lightning as pl
from utils.lr_scheduler import Esm2LRScheduler
from torch import distributed as dist


class SMILESTokenizer:
    def __init__(self):
        self.full_vocab = {"[PAD]":0,"[unused1]":1,"[unused2]":2,"[unused3]":3,"[unused4]":4,"[unused5]":5,"[unused6]":6,"[unused7]":7,"[unused8]":8,"[unused9]":9,"[unused10]":10,"[UNK]":11,"[CLS]":12,"[SEP]":13,"[MASK]":14,"c":15,"C":16,"(":17,")":18,"O":19,"1":20,"2":21,"=":22,"N":23,".":24,"n":25,"3":26,"F":27,"Cl":28,">>":29,"~":30,"-":31,"4":32,"[C@H]":33,"S":34,"[C@@H]":35,"[O-]":36,"Br":37,"#":38,"/":39,"[nH]":40,"[N+]":41,"s":42,"5":43,"o":44,"P":45,"[Na+]":46,"[Si]":47,"I":48,"[Na]":49,"[Pd]":50,"[K+]":51,"[K]":52,"[P]":53,"B":54,"[C@]":55,"[C@@]":56,"[Cl-]":57,"6":58,"[OH-]":59,"\\":60,"[N-]":61,"[Li]":62,"[H]":63,"[2H]":64,"[NH4+]":65,"[c-]":66,"[P-]":67,"[Cs+]":68,"[Li+]":69,"[Cs]":70,"[NaH]":71,"[H-]":72,"[O+]":73,"[BH4-]":74,"[Cu]":75,"7":76,"[Mg]":77,"[Fe+2]":78,"[n+]":79,"[Sn]":80,"[BH-]":81,"[Pd+2]":82,"[CH]":83,"[I-]":84,"[Br-]":85,"[C-]":86,"[Zn]":87,"[B-]":88,"[F-]":89,"[Al]":90,"[P+]":91,"[BH3-]":92,"[Fe]":93,"[C]":94,"[AlH4]":95,"[Ni]":96,"[SiH]":97,"8":98,"[Cu+2]":99,"[Mn]":100,"[AlH]":101,"[nH+]":102,"[AlH4-]":103,"[O-2]":104,"[Cr]":105,"[Mg+2]":106,"[NH3+]":107,"[S@]":108,"[Pt]":109,"[Al+3]":110,"[S@@]":111,"[S-]":112,"[Ti]":113,"[Zn+2]":114,"[PH]":115,"[NH2+]":116,"[Ru]":117,"[Ag+]":118,"[S+]":119,"[I+3]":120,"[NH+]":121,"[Ca+2]":122,"[Ag]":123,"9":124,"[Os]":125,"[Se]":126,"[SiH2]":127,"[Ca]":128,"[Ti+4]":129,"[Ac]":130,"[Cu+]":131,"[S]":132,"[Rh]":133,"[Cl+3]":134,"[cH-]":135,"[Zn+]":136,"[O]":137,"[Cl+]":138,"[SH]":139,"[H+]":140,"[Pd+]":141,"[se]":142,"[PH+]":143,"[I]":144,"[Pt+2]":145,"[C+]":146,"[Mg+]":147,"[Hg]":148,"[W]":149,"[SnH]":150,"[SiH3]":151,"[Fe+3]":152,"[NH]":153,"[Mo]":154,"[CH2+]":155,"%10":156,"[CH2-]":157,"[CH2]":158,"[n-]":159,"[Ce+4]":160,"[NH-]":161,"[Co]":162,"[I+]":163,"[PH2]":164,"[Pt+4]":165,"[Ce]":166,"[B]":167,"[Sn+2]":168,"[Ba+2]":169,"%11":170,"[Fe-3]":171,"[18F]":172,"[SH-]":173,"[Pb+2]":174,"[Os-2]":175,"[Zr+4]":176,"[N]":177,"[Ir]":178,"[Bi]":179,"[Ni+2]":180,"[P@]":181,"[Co+2]":182,"[s+]":183,"[As]":184,"[P+3]":185,"[Hg+2]":186,"[Yb+3]":187,"[CH-]":188,"[Zr+2]":189,"[Mn+2]":190,"[CH+]":191,"[In]":192,"[KH]":193,"[Ce+3]":194,"[Zr]":195,"[AlH2-]":196,"[OH2+]":197,"[Ti+3]":198,"[Rh+2]":199,"[Sb]":200,"[S-2]":201,"%12":202,"[P@@]":203,"[Si@H]":204,"[Mn+4]":205,"p":206,"[Ba]":207,"[NH2-]":208,"[Ge]":209,"[Pb+4]":210,"[Cr+3]":211,"[Au]":212,"[LiH]":213,"[Sc+3]":214,"[o+]":215,"[Rh-3]":216,"%13":217,"[Br]":218,"[Sb-]":219,"[S@+]":220,"[I+2]":221,"[Ar]":222,"[V]":223,"[Cu-]":224,"[Al-]":225,"[Te]":226,"[13c]":227,"[13C]":228,"[Cl]":229,"[PH4+]":230,"[SiH4]":231,"[te]":232,"[CH3-]":233,"[S@@+]":234,"[Rh+3]":235,"[SH+]":236,"[Bi+3]":237,"[Br+2]":238,"[La]":239,"[La+3]":240,"[Pt-2]":241,"[N@@]":242,"[PH3+]":243,"[N@]":244,"[Si+4]":245,"[Sr+2]":246,"[Al+]":247,"[Pb]":248,"[SeH]":249,"[Si-]":250,"[V+5]":251,"[Y+3]":252,"[Re]":253,"[Ru+]":254,"[Sm]":255,"*":256,"[3H]":257,"[NH2]":258,"[Ag-]":259,"[13CH3]":260,"[OH+]":261,"[Ru+3]":262,"[OH]":263,"[Gd+3]":264,"[13CH2]":265,"[In+3]":266,"[Si@@]":267,"[Si@]":268,"[Ti+2]":269,"[Sn+]":270,"[Cl+2]":271,"[AlH-]":272,"[Pd-2]":273,"[SnH3]":274,"[B+3]":275,"[Cu-2]":276,"[Nd+3]":277,"[Pb+3]":278,"[13cH]":279,"[Fe-4]":280,"[Ga]":281,"[Sn+4]":282,"[Hg+]":283,"[11CH3]":284,"[Hf]":285,"[Pr]":286,"[Y]":287,"[S+2]":288,"[Cd]":289,"[Cr+6]":290,"[Zr+3]":291,"[Rh+]":292,"[CH3]":293,"[N-3]":294,"[Hf+2]":295,"[Th]":296,"[Sb+3]":297,"%14":298,"[Cr+2]":299,"[Ru+2]":300,"[Hf+4]":301,"[14C]":302,"[Ta]":303,"[Tl+]":304,"[B+]":305,"[Os+4]":306,"[PdH2]":307,"[Pd-]":308,"[Cd+2]":309,"[Co+3]":310,"[S+4]":311,"[Nb+5]":312,"[123I]":313,"[c+]":314,"[Rb+]":315,"[V+2]":316,"[CH3+]":317,"[Ag+2]":318,"[cH+]":319,"[Mn+3]":320,"[Se-]":321,"[As-]":322,"[Eu+3]":323,"[SH2]":324,"[Sm+3]":325,"[IH+]":326,"%15":327,"[OH3+]":328,"[PH3]":329,"[IH2+]":330,"[SH2+]":331,"[Ir+3]":332,"[AlH3]":333,"[Sc]":334,"[Yb]":335,"[15NH2]":336,"[Lu]":337,"[sH+]":338,"[Gd]":339,"[18F-]":340,"[SH3+]":341,"[SnH4]":342,"[TeH]":343,"[Si@@H]":344,"[Ga+3]":345,"[CaH2]":346,"[Tl]":347,"[Ta+5]":348,"[GeH]":349,"[Br+]":350,"[Sr]":351,"[Tl+3]":352,"[Sm+2]":353,"[PH5]":354,"%16":355,"[N@@+]":356,"[Au+3]":357,"[C-4]":358,"[Nd]":359,"[Ti+]":360,"[IH]":361,"[N@+]":362,"[125I]":363,"[Eu]":364,"[Sn+3]":365,"[Nb]":366,"[Er+3]":367,"[123I-]":368,"[14c]":369,"%17":370,"[SnH2]":371,"[YH]":372,"[Sb+5]":373,"[Pr+3]":374,"[Ir+]":375,"[N+3]":376,"[AlH2]":377,"[19F]":378,"%18":379,"[Tb]":380,"[14CH]":381,"[Mo+4]":382,"[Si+]":383,"[BH]":384,"[Be]":385,"[Rb]":386,"[pH]":387,"%19":388,"%20":389,"[Xe]":390,"[Ir-]":391,"[Be+2]":392,"[C+4]":393,"[RuH2]":394,"[15NH]":395,"[U+2]":396,"[Au-]":397,"%21":398,"%22":399,"[Au+]":400,"[15n]":401,"[Al+2]":402,"[Tb+3]":403,"[15N]":404,"[V+3]":405,"[W+6]":406,"[14CH3]":407,"[Cr+4]":408,"[ClH+]":409,"b":410,"[Ti+6]":411,"[Nd+]":412,"[Zr+]":413,"[PH2+]":414,"[Fm]":415,"[N@H+]":416,"[RuH]":417,"[Dy+3]":418,"%23":419,"[Hf+3]":420,"[W+4]":421,"[11C]":422,"[13CH]":423,"[Er]":424,"[124I]":425,"[LaH]":426,"[F]":427,"[siH]":428,"[Ga+]":429,"[Cm]":430,"[GeH3]":431,"[IH-]":432,"[U+6]":433,"[SeH+]":434,"[32P]":435,"[SeH-]":436,"[Pt-]":437,"[Ir+2]":438,"[se+]":439,"[U]":440,"[F+]":441,"[BH2]":442,"[As+]":443,"[Cf]":444,"[ClH2+]":445,"[Ni+]":446,"[TeH3]":447,"[SbH2]":448,"[Ag+3]":449,"%24":450,"[18O]":451,"[PH4]":452,"[Os+2]":453,"[Na-]":454,"[Sb+2]":455,"[V+4]":456,"[Ho+3]":457,"[68Ga]":458,"[PH-]":459,"[Bi+2]":460,"[Ce+2]":461,"[Pd+3]":462,"[99Tc]":463,"[13C@@H]":464,"[Fe+6]":465,"[c]":466,"[GeH2]":467,"[10B]":468,"[Cu+3]":469,"[Mo+2]":470,"[Cr+]":471,"[Pd+4]":472,"[Dy]":473,"[AsH]":474,"[Ba+]":475,"[SeH2]":476,"[In+]":477,"[TeH2]":478,"[BrH+]":479,"[14cH]":480,"[W+]":481,"[13C@H]":482,"[AsH2]":483,"[In+2]":484,"[N+2]":485,"[N@@H+]":486,"[SbH]":487,"[60Co]":488,"[AsH4+]":489,"[AsH3]":490,"[18OH]":491,"[Ru-2]":492,"[Na-2]":493,"[CuH2]":494,"[31P]":495,"[Ti+5]":496,"[35S]":497,"[P@@H]":498,"[ArH]":499,"[Co+]":500,"[Zr-2]":501,"[BH2-]":502,"[131I]":503,"[SH5]":504,"[VH]":505,"[B+2]":506,"[Yb+2]":507,"[14C@H]":508,"[211At]":509,"[NH3+2]":510,"[IrH]":511,"[IrH2]":512,"[Rh-]":513,"[Cr-]":514,"[Sb+]":515,"[Ni+3]":516,"[TaH3]":517,"[Tl+2]":518,"[64Cu]":519,"[Tc]":520,"[Cd+]":521,"[1H]":522,"[15nH]":523,"[AlH2+]":524,"[FH+2]":525,"[BiH3]":526,"[Ru-]":527,"[Mo+6]":528,"[AsH+]":529,"[BaH2]":530,"[BaH]":531,"[Fe+4]":532,"[229Th]":533,"[Th+4]":534,"[As+3]":535,"[NH+3]":536,"[P@H]":537,"[Li-]":538,"[7NaH]":539,"[Bi+]":540,"[PtH+2]":541,"[p-]":542,"[Re+5]":543,"[NiH]":544,"[Ni-]":545,"[Xe+]":546,"[Ca+]":547,"[11c]":548,"[Rh+4]":549,"[AcH]":550,"[HeH]":551,"[Sc+2]":552,"[Mn+]":553,"[UH]":554,"[14CH2]":555,"[SiH4+]":556,"[18OH2]":557,"[Ac-]":558,"[Re+4]":559,"[118Sn]":560,"[153Sm]":561,"[P+2]":562,"[9CH]":563,"[9CH3]":564,"[Y-]":565,"[NiH2]":566,"[Si+2]":567,"[Mn+6]":568,"[ZrH2]":569,"[C-2]":570,"[Bi+5]":571,"[24NaH]":572,"[Fr]":573,"[15CH]":574,"[Se+]":575,"[At]":576,"[P-3]":577,"[124I-]":578,"[CuH2-]":579,"[Nb+4]":580,"[Nb+3]":581,"[MgH]":582,"[Ir+4]":583,"[67Ga+3]":584,"[67Ga]":585,"[13N]":586,"[15OH2]":587,"[2NH]":588,"[Ho]":589,"[Cn]":590}
        self.allowed_tokens = ['(', ')', '-', '/', '1', '2', '3', '4', '=', 'B', 'C', 'F', 'N', 'O', 'S', '[CLS]',
                               '[SEP]', "[PAD]", 'c', 'n', 's', '[MASK]', '[UNK]']

        self.token_to_id = {tok: self.full_vocab[tok] for tok in self.allowed_tokens if tok in self.full_vocab}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.full_vocab_inv = {v: k for k, v in self.full_vocab.items()}
        self.id_to_allowed_id = {v: idx for idx, v in enumerate(self.token_to_id.values())}
        self.allowed_tokens_sorted = list(self.token_to_id.keys())  # for decoding remapped

        self.pad_token_id = self.id_to_allowed_id[self.token_to_id["[PAD]"]]
        self.cls_token_id = self.id_to_allowed_id[self.token_to_id["[CLS]"]]
        self.sep_token_id = self.id_to_allowed_id[self.token_to_id["[SEP]"]]
        self.mask_token_id = self.id_to_allowed_id[self.token_to_id["[MASK]"]]
        self.vocab_size = len(self.id_to_token)

        self.full_to_allowed_id = {}
        allowed_ids = list(self.token_to_id.values())
        for allowed_idx, full_id in enumerate(allowed_ids):
            self.full_to_allowed_id[full_id] = allowed_idx
        self.allowed_token_ids = list(self.token_to_id.values())

    def tokenize(self, smiles):
        tokens = []
        i = 0
        while i < len(smiles):
            if i + 4 <= len(smiles) and smiles[i:i + 4] in self.token_to_id:
                tokens.append(smiles[i:i + 4])
                i += 4
            elif i + 3 <= len(smiles) and smiles[i:i + 3] in self.token_to_id:
                tokens.append(smiles[i:i + 3])
                i += 3
            elif i + 2 <= len(smiles) and smiles[i:i + 2] in self.token_to_id:
                tokens.append(smiles[i:i + 2])
                i += 2
            elif smiles[i] in self.token_to_id:
                tokens.append(smiles[i])
                i += 1
            else:
                tokens.append("[MASK]")
                i += 1
        return tokens

    def encode(self, smiles):
        tokens = self.tokenize(smiles)
        full_ids = [self.token_to_id.get(t, self.mask_token_id) for t in tokens]
        allowed_ids = [self.full_to_allowed_id.get(fid, self.full_to_allowed_id[self.mask_token_id]) for fid in
                       full_ids]
        token_ids = [self.cls_token_id] + allowed_ids + [self.sep_token_id]
        return token_ids

    def encode_batch(self, smiles_list):
        all_token_ids = [self.encode(smiles) for smiles in smiles_list]
        max_len = max(len(ids) for ids in all_token_ids)

        input_ids = []
        attention_mask = []
        for ids in all_token_ids:
            pad_len = max_len - len(ids)
            padded = ids + [self.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            input_ids.append(padded)
            attention_mask.append(mask)

        data = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        return BatchEncoding(data)

    def decode_remapped(self, remapped_ids, skip_special_tokens=True):
        tokens = []
        for t in remapped_ids:
            t = t.item() if isinstance(t, torch.Tensor) else t
            if not (0 <= t < len(self.allowed_tokens_sorted)):
                # If out of bounds, treat as unknown token
                token = "[UNK]"
            else:
                token = self.allowed_tokens_sorted[t]
            if skip_special_tokens and token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]:
                continue
            tokens.append(token)
        return ''.join(tokens)

    def __call__(self, smiles_list, return_tensors="pt", padding=True, truncation=True):
        return self.encode_batch(smiles_list)


class LigandDecoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, tokenizer, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim) * 0.02)

        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=embedding_dim, nhead=4, dim_feedforward=512, batch_first=True
        # )
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=8, dim_feedforward=512, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, memory, tgt_seq):
        batch_size, tgt_len = tgt_seq.shape
        tgt_emb = self.embedding(tgt_seq) + self.positional_encoding[:, :tgt_len, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_seq.device)
        decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.output_layer(decoded)
        return logits

    def generate(self, memory, max_len=100, top_k=3):
        batch_size = memory.size(0)
        device = memory.device

        generated = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = self.embedding(generated)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(device)
            decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.output_layer(decoded)[:, -1, :]  # (batch_size, vocab_size)

            # Top-k sampling
            next_token = []
            for logit in logits:  # logit is (vocab_size,)
                topk_probs, topk_indices = torch.topk(F.softmax(logit, dim=-1), k=top_k)
                sampled_idx = torch.multinomial(topk_probs, num_samples=1)
                next_token.append(topk_indices[sampled_idx])
            next_token = torch.cat(next_token, dim=0).unsqueeze(1)  # (batch_size, 1)

            generated = torch.cat((generated, next_token), dim=1)

            if (next_token.squeeze(-1) == self.tokenizer.sep_token_id).all():
                break

        return generated


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
                 save_weights_only: bool = True,
                 load_generator: str = None):
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
        self.load_generator = load_generator

        self.step = 0
        self.epoch = 0

        # ProtLigand Parameters
        # self.ligand_tokenizer = AutoTokenizer.from_pretrained("pchanda/pretrained-smiles-pubchem10m")
        # self.ligand_model = AutoModelForMaskedLM.from_pretrained("pchanda/pretrained-smiles-pubchem10m")

        # self.ligand_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.ligand_model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

        self.chemberta_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.ligand_tokenizer = SMILESTokenizer()

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

        # self.ligand_generator = nn.Sequential(
        #     nn.TransformerEncoder(
        #         nn.TransformerEncoderLayer(d_model=protein_hidden_size, nhead=4, dim_feedforward=512, dropout=0.1,
        #                                    batch_first=True), num_layers=2),
        #     nn.Linear(protein_hidden_size, ligand_hidden_size)
        # )
        self.ligand_generator = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=protein_hidden_size, nhead=8, dim_feedforward=512, dropout=0.1,
                                           batch_first=True), num_layers=6),
            nn.Linear(protein_hidden_size, ligand_hidden_size)
        )

        self.ligand_decoder = LigandDecoder(
            embedding_dim=ligand_hidden_size,
            vocab_size=self.ligand_tokenizer.vocab_size,
            tokenizer=self.ligand_tokenizer
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
            smiles_list = []
            num_ligands_to_select = min(self.max_ligands, len(ligands))
            ligands = random.sample(ligands, num_ligands_to_select)

            for smiles, label in ligands:
                smiles_list.append(smiles)
                all_labels.append(label)

            if smiles_list:
                ligand_inputs = self.ligand_tokenizer(smiles_list, return_tensors="pt", padding=True,
                                                      truncation=True).to(self.model.device)
                ligands_outputs = self.ligand_model(**ligand_inputs, output_hidden_states=True)
                final_hidden_state = ligands_outputs.hidden_states[-1].to(self.model.device)
                ligand_representations = final_hidden_state.mean(dim=1)
                ligand_embeddings[i, :ligand_representations.size(0)] = ligand_representations

        return ligand_embeddings, all_labels

    def process_ligands_with_smiles(self, ligands_info):
        batch_size = len(ligands_info)
        max_ligands = min(self.max_ligands, max(len(ligands) for ligands in ligands_info))
        embedding_dim = self.ligand_model.config.hidden_size
        ligand_embeddings = torch.zeros(batch_size, max_ligands, embedding_dim).to(device=self.ligand_model.device)
        all_labels = []

        for i, ligands in enumerate(ligands_info):
            smiles_list = []
            num_ligands_to_select = min(self.max_ligands, len(ligands))
            ligands = random.sample(ligands, num_ligands_to_select)

            for smiles, label in ligands:
                smiles_list.append(smiles)

            if smiles_list:
                ligand_inputs = self.chemberta_tokenizer(smiles_list, return_tensors="pt", padding=True,
                                                      truncation=True).to(self.model.device)
                ligands_outputs = self.ligand_model(**ligand_inputs, output_hidden_states=True)
                final_hidden_state = ligands_outputs.hidden_states[-1].to(self.model.device)
                ligand_representations = final_hidden_state.mean(dim=1)
                ligand_embeddings[i, :ligand_representations.size(0)] = ligand_representations

                all_labels.append(ligand_inputs)

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
        for key, module in [
            ("cross_attention", self.cross_attention),
            ("ligand_proj", self.ligand_proj),
            ("ligand_generator", self.ligand_generator),
            ("ligand_decoder", self.ligand_decoder)
        ]:
            if key in state_dict:
                module.load_state_dict(state_dict[key])

        # Restore nn.Parameter (default_ligand)
        if "default_ligand" in state_dict:
            self.default_ligand = state_dict["default_ligand"]

        if self.load_generator:
            print("loading ligand generator...")
            generator_state_dict = torch.load(self.load_generator, map_location=self.device)
            self.ligand_generator.load_state_dict(generator_state_dict["ligand_generator"])
            self.ligand_decoder.load_state_dict(generator_state_dict["ligand_decoder"])

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
