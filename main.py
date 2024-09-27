"""
   Modified from https://github.com/chengkai-liu/Mamba4Rec
"""

# %cd /gdrive/My Drive/TTT4Rec

import sys
import logging
from logging import getLogger

import torch
from torch import nn

from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.dataset import SequentialDataset
from recbole.model.loss import BPRLoss

from ttt import TTTModel, TTTConfig


class CustomSequentialDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def leave_one_out(self, group_by, leave_one_num=1, leave_one_mode='valid_and_test'):
        """
        rewrite function "leave_one_out" to deal with specific train/valid/test split ratios
        """
        split_ratios = self.config['split_ratio']
        self.logger.info(f'Split ratio for train/valid/test: {split_ratios}')

        train_data, valid_data, test_data = self.split_by_ratio(ratios=split_ratios, group_by=group_by)

        return train_data, valid_data, test_data

class TTT4Rec(SequentialRecommender):
    def __init__(self, ttt_config, rec_config, dataset):
        super(TTT4Rec, self).__init__(rec_config, dataset)

        self.hidden_size = rec_config["hidden_size"]
        self.loss_type = rec_config["loss_type"]
        self.dropout_prob = rec_config["dropout_prob"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.ttt_layers = TTTModel(ttt_config)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        batch_size, seq_length, hidden_size = item_emb.shape
        attention_mask = (torch.arange(seq_length).expand(batch_size, seq_length)).cuda() < item_seq_len.unsqueeze(1)
        attention_mask = attention_mask.long()

        item_emb = self.ttt_layers(inputs_embeds=item_emb,
                                   attention_mask = attention_mask).last_hidden_state

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores

ttt_config = TTTConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=50,
        rope_theta=1000.0,
        ttt_layer_type="mlp",
        ttt_base_lr=1.0,
        mini_batch_size=8,
        use_gate=False,
        pre_conv=False,
        share_qk=False)

rec_config = Config(model=TTT4Rec, config_file_list=['config.yaml'])
init_seed(rec_config['seed'], rec_config['reproducibility'])

# logger initialization
init_logger(rec_config)
logger = getLogger()
logger.setLevel(logging.INFO)
logger.info(sys.argv)
logger.info(rec_config)

# dataset filtering
# dataset = create_dataset(rec_config)
dataset = CustomSequentialDataset(rec_config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(rec_config, dataset)

# model loading and initialization
init_seed(rec_config["seed"] + rec_config["local_rank"], rec_config["reproducibility"])
model = TTT4Rec(ttt_config, rec_config, train_data.dataset).to(rec_config['device'])
logger.info(model)

transform = construct_transform(rec_config)
flops = get_flops(model, dataset, rec_config["device"], logger, transform)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

# trainer loading and initialization
trainer = Trainer(rec_config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, show_progress=rec_config["show_progress"]
)

# model evaluation
test_result = trainer.evaluate(
    test_data, show_progress=rec_config["show_progress"]
)

environment_tb = get_environment(rec_config)
logger.info(
    "The running environment of this training is as follows:\n"
    + environment_tb.draw()
)

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
logger.info(set_color("test result", "yellow") + f": {test_result}")