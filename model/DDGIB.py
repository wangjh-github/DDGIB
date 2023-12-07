import math

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from model.DF_contrastive import DFContrastive
from model.MIB import MIB
from utils.schedulers import ExponentialScheduler
from utils.Blinear import Blinear
from trainer.config import args


class DDGIB(nn.Module):
    def __init__(self,
                 s_cov_num, s_in_feature_list, s_dropout_gcn,
                 s_block_num, s_hidden, s_attn_heads, s_feed_forward_hidden, s_dropout_trans,
                 p, min_len, max_len,
                 d_cov_num, d_in_feature_list, d_dropout_gcn,
                 d_block_num, d_hidden, d_attn_heads, d_feed_forward_hidden, d_dropout_trans
                 ):
        super(DDGIB, self).__init__()
        self.dynamic_model = DFContrastive(d_cov_num, d_in_feature_list, d_dropout_gcn,
                                           d_block_num, d_hidden, d_attn_heads, d_feed_forward_hidden,
                                           d_dropout_trans)
        self.mib = MIB(s_cov_num, s_in_feature_list, s_block_num, s_hidden, s_attn_heads, s_feed_forward_hidden,
                       s_dropout_trans,
                       p, min_len, max_len)
        self.beta_scheduler = ExponentialScheduler(start_value=1e-3, end_value=1,
                                                   n_iterations=100000, start_iteration=50000)
        self.iterations = 0
        self.f_k = Blinear(n_h=32)

    def forward(self, x, type=0):
        sta_org, sta_other, sta_kl_o_t, sta_kl_t_o = self.mib.forward(x, type)
        dyn, dyn_club, dyn_rho = self.dynamic_model.forward(x)
        return sta_org, sta_other, sta_kl_o_t, sta_kl_t_o, dyn_rho, dyn_club, dyn

    def get_loss(self, feed_dict, sta_org, sta_other, sta_kl_o_t, sta_kl_t_o, dyn_club, dyn_rho,
                 dyn, pos_dis=None, neg_dis=None, global_weight=1, type=0):
        graph, data = feed_dict.values()
        # get emb
        sta_org = self._normalize(sta_org)
        sta_other = self._normalize(sta_other)
        dyn = self._normalize(dyn)
        dyn_club = self._normalize(dyn_club)
        self.graph_loss = 0
        beta = self.beta_scheduler(self.iterations)
        sp_begin, sp_end = self.mib.get_sp_pos()
        global_emb = [torch.cat((sta_org, dyn[:, t, :].squeeze()), dim=1) for t in range(len(data[0]["isIn"]))]
        for i in range(0, len(data) - 1):
            node = data[i]["node"]
            # mib loss
            skl = (sta_kl_o_t[node] + sta_kl_t_o[node]).mean() / 2.

            # Mutual information estimation
            sta_weight = self._count_weight(
                data[i]["isIn"][sp_begin:sp_end].count(True) / len(data[i]["isIn"][sp_begin:sp_end])
            )
            sta_pos = (sta_org[node] * sta_other[node]).sum().exp()
            index = torch.tensor(np.concatenate([data[i]["pos"][-1], data[i]["neg"][-1]]))
            index = index.type(torch.int32).to(sta_pos.device)
            sta_neg = (sta_org[node] * torch.index_select(sta_org, dim=0, index=index)).sum(dim=1).exp().sum()
            mi_gradient = torch.log(sta_weight * (sta_pos / (sta_pos + sta_neg)))
            static_loss = - mi_gradient + beta * skl
            global_loss = 0
            for t in range(0, len(data[i]["isIn"])):
                if data[i]["isIn"][t] is False or (len(data[i]["pos"][t]) == 0) or (len(data[i]["neg"][t]) == 0):
                    continue
                pos = np.random.choice(data[i]["pos"][t], size=1)
                dyn_pos = (global_emb[t][node] * global_emb[t][pos]).sum().exp()
                dyn_neg = (global_emb[t][node] *
                           torch.index_select(global_emb[t], dim=0,
                                              index=torch.tensor(data[i]["neg"][t]).type(torch.int32).to(dyn.device))
                           ).sum(dim=1).exp().sum()
                global_loss += torch.log(dyn_pos / (dyn_pos + dyn_neg))
            self.graph_loss += static_loss - global_weight * global_loss
        self.graph_loss = self.graph_loss / len(data)

        # dynamic loss
        idx = np.random.permutation(dyn_club.shape[1])
        indx = np.random.permutation(dyn_club.shape[1])
        idx_p = np.random.permutation(dyn_club.shape[0])
        dyn_q_club = dyn_club[:, idx, :]
        dyn_q_club = dyn_q_club[idx_p, :, :]
        dyn_p_club = dyn_club[:, indx, :]
        dyn_positive = -(dyn - dyn_p_club) ** 2 / dyn_rho
        dyn_positive = torch.mean(dyn_positive)
        dyn_negative = -(dyn_q_club - dyn) ** 2 / dyn_rho
        dyn_negative = torch.mean(dyn_negative)
        mi_est = dyn_positive - dyn_negative
        h_1 = dyn
        c = torch.mean(h_1, 0)
        sigm = nn.Sigmoid()
        c = sigm(c)
        idx1 = np.random.permutation(h_1.shape[0])
        h_2 = h_1[idx1, :, :]
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_1)
        sc_1 = torch.squeeze(self.f_k(h_1, c_x.contiguous()), 2)
        sc_2 = torch.squeeze(self.f_k(h_2, c_x.contiguous()), 2)
        logits = torch.cat((sc_1, sc_2), 1)
        lbl_1 = torch.ones(args.node, args.dim)   # torch.ones(1809, 13) torch.ones(7576, 23)   torch.ones(143, 16)
        lbl_2 = torch.zeros(args.node, args.dim)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        b_xent = nn.BCEWithLogitsLoss()
        loss = b_xent(logits.cpu(), lbl)
        # print("3", loss)
        dynamic_loss = args.beta * mi_est + (1 - args.beta) * loss
        self.graph_loss = args.alpha * self.graph_loss + (1 - args.alpha) * dynamic_loss

        # discriminator loss
        if pos_dis is None or neg_dis is None:
            dis_loss = 0
        else:
            loss = nn.BCELoss()
            pos_tar = torch.ones(pos_dis.size()).to(dyn.device)
            neg_tar = torch.zeros(neg_dis.size()).to(dyn.device)
            dis_loss = -0.5 * (loss(pos_dis, pos_tar) + loss(neg_dis, neg_tar))
        return self.graph_loss + 0.3 * dis_loss

    def _count_weight(self, x):
        return x if x > 0 else 0.01

    def _normalize(self, *xs):
        data = [None if x is None else F.normalize(x, dim=-1) for x in xs]
        return data[0]
