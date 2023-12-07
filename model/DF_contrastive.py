import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, Normal
from torch_geometric.nn import GCNConv
from model.component.transformer_block import TransformerBlock
from model.component.position_emb import PositionEmb
from trainer.config import args


class DFContrastive(nn.Module):
    """
    Get the dynamic-fluctuate representation
    """

    def __init__(self, cov_num, in_feature_list, dropout_gcn, block_num, hidden, attn_heads, feed_forward_hidden,
                 dropout_trans):
        """
        :param cov_num: number of GCN-layers
        :param in_feature_list: the first layer dim
        :param dropout_gcn: the dropout rate - GCN-layer
        :param block_num: number of Trans-layers
        :param hidden: the hidden size of transblock-attention
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: the hidden size of transblock-FFN
        :param dropout_gcn: the dropout rate - trans-layer
        """
        super(DFContrastive, self).__init__()
        # cov_num GCNconv
        in_feature_list.append(hidden)
        self.gcn_layers = nn.ModuleList(
            [GCNConv(in_feature_list[i], in_feature_list[i + 1]) for i in range(cov_num)])
        self.dropout = nn.Dropout(p=dropout_gcn)
        # block_num transformer block
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, feed_forward_hidden, dropout_trans)
             for _ in range(block_num)])
        # trainable feature
        self.t_feats = nn.Parameter(torch.ones(args.node_num, in_feature_list[0]), requires_grad=True)
        self.position_ecd = PositionEmb(hidden, 30)
        self._init_layers()
        self.net = MLP()

    """
    x: batch_size x time_stamp x dim
    idx: lst(src, tar) - len:time_stamp
    """

    def forward(self, graphs):

        # GCN
        struct_out = []
        for graph in graphs:
            x = graph.x
            if args.use_trainable_feature:
                x = x * self.t_feats
            for gcn in self.gcn_layers:
                x = gcn.forward(x, graph.edge_index)
                x = F.relu(x)
                x = self.dropout(x)
            struct_out.append(x[:, None, :])  # N x 1 x dim - len(T)

        x = torch.cat(struct_out, dim=1)  # N x T x dim
        x = x + self.position_ecd(x)
        for transformer in self.transformer_layers:
            x = transformer.forward(x)
            x = self.net(x)
        mu, rho = x[:, :, : 32], x[:, :, 32:]
        rho = self.clip_by_tensor(rho, -1.1, 1.1)
        rho = torch.exp(rho / 2)  # Make sigma always positive
        dyn_club = Independent(Normal(loc=mu, scale=rho), 1)
        dyn_club_1 = dyn_club.rsample()
        return mu, dyn_club_1, rho

    def _init_layers(self):
        for layer in self.gcn_layers:
            nn.init.xavier_uniform_(layer.weight)

    def clip_by_tensor(self, t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = t.float()

        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        x = self.net(x)
        return x