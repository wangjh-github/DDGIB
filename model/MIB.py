import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import GCNConv
from model.component.transformer_block import TransformerBlock

from trainer.config import args

from model.component.position_emb import PositionEmb

from utils.modules import Encoder

class MIB(nn.Module):
    """
    Get the temporal-invariant representation
    """

    def __init__(self, cov_num, in_feature_list, block_num, hidden, attn_heads, feed_forward_hidden,
                 dropout_trans,
                 p, min_len, max_len):
        """
        :param cov_num: number of GCN-layers
        :param in_feature_list: the first layer dim
        :param hidden: the hidden size of transblock-attention
        :param p: the probability of sampling
        :param min_len: the max length of sampling
        :param max_len: the min length of sampling
        """
        super().__init__()
        # cov_num GCNconv
        in_feature_list.append(hidden)
        self.gcn_layers = nn.ModuleList([GCNConv(in_feature_list[i], in_feature_list[i + 1]) for i in range(cov_num)])
        self.t_feats = nn.Parameter(torch.ones(args.node_num, in_feature_list[0]), requires_grad=True)
        self.h_0 = nn.Parameter(torch.ones(hidden), requires_grad=True)
        self.p = p
        self.min_len = min_len
        self.max_len = max_len
        self.sampler = Sampler()

        self.choose_org_begin = 0
        self.choose_org_end = 0

        self.choose_other_begin = 0
        self.choose_other_end = 0

        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, feed_forward_hidden, dropout_trans)
             for _ in range(block_num)])
        self.position_ecd = PositionEmb(hidden, 50)
        
        self.encoder = Encoder(self.gcn_layers, self.t_feats, self.transformer_layers, self.position_ecd, self.h_0)

        self._init_layers()

    def forward(self, graphs, type=0):
        # encoder
        x_org_sampler, x_org_out, self.choose_org_begin = self.encoder(graphs, type)  # Independent(Normal(loc: torch.Size([1809, 14, 32]), scale: torch.Size([1809, 14, 16])), 1)
        x_org = x_org_sampler.rsample()
        x_org = x_org[:, 0, :].squeeze()
        x_other_sampler, x_other_out, self.choose_other_begin = self.encoder(graphs, type)
        x_other = x_other_sampler.rsample()
        x_other = x_other[:, 0, :].squeeze()

        x_org_loc = x_org_sampler.base_dist.loc[:, 0, :].squeeze()
        x_org_scale = x_org_sampler.base_dist.scale[:, 0, :].squeeze()
        x_other_loc = x_other_sampler.base_dist.loc[:, 0, :].squeeze()
        x_other_scale = x_other_sampler.base_dist.scale[:, 0, :].squeeze()

        kl_org_other = self.log_prob(x_org, x_org_loc, x_org_scale) - self.log_prob(x_org, x_other_loc, x_other_scale)
        kl_other_org = self.log_prob(x_other, x_other_loc, x_other_scale) - self.log_prob(x_other, x_org_loc, x_org_scale)  # N x T
        return x_org, x_other, kl_org_other, kl_other_org

    def get_sp_pos(self):
        return self.choose_org_begin, self.choose_other_begin

    def _init_layers(self):
        for layer in self.gcn_layers:
            nn.init.xavier_uniform_(layer.weight)

    def log_prob(self, value, loc, scale):
        # compute the variance
        var = (scale ** 2)
        log_scale = torch.log(scale)
        return -((value - loc) ** 2) / (2 * var) - log_scale - torch.log(torch.tensor(math.sqrt(2 * math.pi)))

