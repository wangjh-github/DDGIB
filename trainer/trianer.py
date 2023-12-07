import torch
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import time
import copy
import os

from trainer.config import args
from trainer.my_dataset import MyDataset

from utils.logistic_cls import evaluate_classifier, train, test
from trainer.MINE import cacul_mine
from utils.utilize import get_evaluation_data, get_sample

from model.DDGIB import DDGIB
from model.discriminator import Discriminator
import torch.optim as optim

path = "./log/model_checkpoint/"


class Trainer:
    def __init__(self, graphs, nodes, adjs, args, labels, i):

        self.graphs = graphs
        self.adjs = adjs
        self.nodes = nodes
        # self.features = feature
        self.args = args
        self.labels = labels

        self._create_model()
        self._create_dataset()
        self._create_dataloader()

        self._init_path()

        self.auc_score = 0
        self.ap_score = 0

        self.val_auc_score = 0
        self.val_ap_score = 0

        self.acc = 0
        self.f1 = 0
        self.auc = 0
        self.i = i


    def _create_dataset(self):
        self.dataset = MyDataset(self.graphs, self.nodes, self.adjs, len(self.graphs),
                                 self.args.pos_num, self.args.neg_num, self.args.node_num)

    def _create_model(self):
        self.model = DDGIB(self.args.static_cov, self.args.static_feat_list,
                           self.args.static_gcn_dropout,
                           self.args.static_trans_num, self.args.static_hidden, self.args.heads,
                           self.args.static_forward_hidden, self.args.static_trans_dropout,
                           self.args.choose_p, self.args.sample_min_len, self.args.sample_max_len,
                           self.args.dynamic_cov, self.args.dynamic_feat_list,
                           self.args.dynamic_gcn_dropout,
                           self.args.dynamic_trans_num, self.args.dynamic_hidden, self.args.heads,
                           self.args.dynamic_forward_hidden, self.args.dynamic_trans_dropout
                           ).to(self.args.device)
        self.discriminator = Discriminator(self.args.dis_in, self.args.dis_hid).to(self.args.device)
        self.model = torch.nn.DataParallel(self.model)
        self.discriminator = torch.nn.DataParallel(self.discriminator)
        if torch.cuda.is_available() and self.args.use_gpu:
            self.model.cuda()
            self.discriminator.cuda()
        print("create model!!!")

    def _create_dataloader(self):
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.args.batch_size,
                                      # batch_size=100,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=self.dataset.collate_fn)

    def _init_path(self):
        if not os.path.exists(path):
            os.makedirs(path)

    def run(self):
        opt = optim.Adam([
            {'params': self.model.module.mib.parameters()},
            {'params': self.model.module.dynamic_model.parameters()},
            {'params': self.model.module.f_k.parameters()}
        ], lr=self.args.lr, weight_decay=self.args.weight_decay)
        opt_p = optim.Adam(self.model.module.Club.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        opt_dis = optim.Adam(self.discriminator.module.parameters(), lr=self.args.lr,
                             weight_decay=self.args.weight_decay)
        dis_loss = torch.nn.BCELoss()
        t_total0 = time.time()
        best_epoch_val = 0
        patient = 0
        min_loss = 10000000

        # Load evaluation data for link prediction.
        train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
        test_edges_pos, test_edges_neg = get_evaluation_data(self.graphs)
        print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
            len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
            len(test_edges_pos), len(test_edges_neg)))

        for epoch in range(0, self.args.min_epoch - 1):
            # train

            t0 = time.time()
            epoch_loss = []
            epoch_loss_dis = []

            for idx, feed_dict in enumerate(self.data_loader):
                feed_dict = self._to_device(feed_dict)
                self.model.train()
                opt.zero_grad()
                opt_p.zero_grad()

                sta_org, sta_other, sta_kl_o_t, sta_kl_t_o, dyn_rho, dyn_club, dyn = self.model.module(
                    feed_dict["graphs"])
                self.discriminator.eval()
                pos, neg = get_sample(sta_org.to(sta_org.device), dyn, len(self.adjs), self.args.dis_sample_num)
                pos, neg = self.discriminator.module(pos, neg)
                loss = self.model.module.get_loss(feed_dict, sta_org, sta_other, sta_kl_o_t, sta_kl_t_o,
                                                  dyn_club, dyn_rho, dyn, pos, neg,
                                                  global_weight=self.args.global_weight, type=1)
                opt_p.zero_grad()

                loss.backward()
                opt.step()
                epoch_loss.append(loss.item())

                temp_dis_epoch = []
                self.model.eval()
                self.discriminator.train()
                # # Train the generator once and train the discriminator several times
                for j in range(self.args.dis_epoch):
                    opt_dis.zero_grad()

                    pos, neg = get_sample(sta_org, dyn, len(self.adjs), self.args.dis_sample_num)
                    pos = pos.detach()
                    neg = neg.detach()
                    pos, neg = self.discriminator.module(pos, neg)
                    pos_tar = torch.ones(pos.size()).to(pos.device)
                    neg_tar = torch.zeros(neg.size()).to(pos.device)
                    # loss
                    loss_neg = 0.5 * (dis_loss(pos, pos_tar) + dis_loss(neg, neg_tar))
                    loss_neg.backward()

                    opt_dis.step()
                    temp_dis_epoch.append(loss_neg.item())

                epoch_loss_dis.append(np.mean(temp_dis_epoch))

            average_epoch_loss = np.mean(epoch_loss)
            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                patient = 0
                save_path = './data/'
                path = os.path.join(save_path, "{}_best_network_{}_{:.3f}_{:.4f}.pth".format(args.dataset, self.i, args.alpha, args.beta))
                torch.save(self.model.state_dict(), path)
            else:
                patient += 1
                if patient > self.args.patience:
                    print('early stopping')
                    break

            gpu_mem_alloc = torch.cuda.max_memory_allocated(
                self.args.device) / 1000000 if torch.cuda.is_available() else 0

            if epoch == 1 or epoch % self.args.log_interval == 0:
                with torch.no_grad():
                    print("==" * 20)
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss,
                                                                                        time.time() - t0,
                                                                                        gpu_mem_alloc))
        load_path = './data/'
        state_dict = torch.load(load_path + "{}_best_network_{}_{:.3f}_{:.4f}.pth".format(args.dataset, self.i, args.alpha, args.beta))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for idx, feed_dict in enumerate(self.data_loader):
            feed_dict = self._to_device(feed_dict)
            break
        sta, _, _, _, _, _, dyn = self.model.module(feed_dict["graphs"], type=0)
        emb = torch.cat((sta, dyn[:, -1, :].squeeze()), dim=1).detach().cpu()
        idxs = 500
        min_acc = 0
        p = 0
        train_features, test_features, train_labels, test_labels = train_test_split(
            emb, self.labels, test_size=0.1)
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_features, train_labels, test_size=2 / 9)
        train_labels = torch.LongTensor(np.argmax(train_labels, axis=1))
        test_labels = torch.LongTensor(np.argmax(test_labels, axis=1))
        val_labels = torch.LongTensor(np.argmax(val_labels, axis=1))
        for idx in range(idxs):
            _, acc_val = train(train_features, val_features, train_labels, val_labels, idx)
            if idx > 100:
                if acc_val > min_acc:
                    min_acc = acc_val
                    p = 0
                else:
                    p += 1
                    if p > 10:
                        print('train early stopping')
                        break
            if idx % 10 == 0:
                a, b, c, d = test(test_features, test_labels)
                print('acc: {:.4f}'.format(a.item()),
                      'loss: {:.4f}'.format(b.item()),
                      'f1: {:.4f}'.format(c.item()),
                      'auc: {:.4f}'.format(d.item()))
        self.acc, loss_test, self.f1, self.auc = test(test_features, test_labels)
        print("Total time: {}".format(time.time() - t_total0))
        print("test_loss= {:.4f}".format(loss_test.item()),
              "best_accuracy= {:.4f}".format(self.acc.item()),
              "f1_score= {:.4f}".format(self.f1.item()),
              "auc= {:.4f}".format(self.auc.item()),
              )

        return self.acc, self.f1, self.auc

    def _to_device(self, batch):
        feed_dict = copy.deepcopy(batch)
        graphs, data = feed_dict.values()
        # to device
        feed_dict["graphs"] = [g.to(self.args.device) for g in graphs]
        feed_dict["data"] = data

        return feed_dict
