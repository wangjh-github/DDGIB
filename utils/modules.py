import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from trainer.config import args
from utils.sample import Graph_sampler


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, gcn_layers, t_feats, transformer_layers, position_ecd, h_0):
        super().__init__()
        # cov_num GCNconv
        self.gcn_layers = gcn_layers
        self.t_feats = t_feats
        # block_num transformer block
        self.transformer_layers = transformer_layers
        self.position_ecd = position_ecd
        self.h_0 = h_0
        self.net = MLP()
        self.sampler = Graph_sampler()

    def forward(self, graphs, type):
        # GCN
        struct_out1 = []
        if type == 0:
            graphs, choose_begin, choose_end = self.sampler.forward(graphs=graphs, p=0.75, min_len=3, max_len=10)
        else:
            choose_begin = 0
            choose_end = -1
        for graph in graphs:
            x = graph.x
            if args.use_trainable_feature:
                x = x * self.t_feats
            for gcn in self.gcn_layers:
                x = gcn.forward(x, graph.edge_index)
                x = F.relu(x, inplace=False)
            struct_out1.append(x[:, None, :])  # N x 1 x dim - len(T)

        struct_out2 = torch.cat(struct_out1, dim=1)
        h_0 = self.h_0[None, None, :].repeat(struct_out2.shape[0], 1, 1).to(struct_out2.device)
        struct_out_t = torch.cat((h_0, struct_out2), dim=1)
        p_struct_out2 = self.position_ecd(struct_out_t)
        struct_out_t = struct_out_t + p_struct_out2
        for transformer in self.transformer_layers:
            struct_out_t = transformer.forward(struct_out_t)
            struct_out_t = self.net(struct_out_t)
        mu, sigma = struct_out_t[:, :, : 32], struct_out_t[:, :, 32:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        return Independent(Normal(loc=mu, scale=sigma), 1), struct_out_t, choose_begin, choose_end   # Return a factorized Normal distribution


# Auxiliary network for mutual information estimation
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