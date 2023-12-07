from trainer.config import args
from utils.utilize import load_graphs, generate_feats
from trainer.trianer import Trainer
import torch
import numpy as np
import sys
import time
import logging

if __name__ == '__main__':
    # Load data
    graphs, nodes, adjs, labels = load_graphs(args.dataset)
    # Load node features
    if args.pre_defined_feature is None:
        if args.use_trainable_feature:
            feats = torch.ones([args.node_num, 1]).to(args.device)
        else:
            feats = generate_feats(graphs, args.device)
    else:
        # Todo: load predefined features
        pass

    # result
    auc_list = []
    f1_list = []
    acc_list = []

    aucd_list = []
    f1d_list = []
    accd_list = []

    # training times
    times = 10

    max_auc = 0
    max_ap = 0

    for i in range(times):
        trainer = Trainer(graphs, nodes, adjs, args, labels, i)
        acc, f1, auc = trainer.run()
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
        max_auc = auc if auc >= max_auc else max_auc
        max_f1 = f1 if f1 >= max_ap else max_ap
        max_acc = acc if acc >= max_ap else max_ap
    f = open("data.txt", "a+")
    print("**" * 10)
    print("Best f1:{}, ACC:{}, AUC:{}".format(max_f1, max_acc, max_auc), file=f, flush=True)
    print("Best f1:{}, ACC:{}, AUC:{}".format(max_f1, max_acc, max_auc), file=sys.stdout)
    print("**" * 10)
    print("mean f1:{}, mean ACC:{}, mean AUC:{}".format(np.mean(f1_list), np.mean(acc_list), np.mean(auc_list)), file=f, flush=True)
    print("mean f1:{}, mean ACC:{}, mean AUC:{}".format(np.mean(f1_list), np.mean(acc_list), np.mean(auc_list)), file=sys.stdout)
    print("**" * 10)
    print("avg f1:{}, avg ACC:{}, avg AUC:{}".format(np.var(f1_list), np.var(acc_list), np.var(auc_list)), file=f, flush=True)
    print("avg f1:{}, avg ACC:{}, avg AUC:{}".format(np.var(f1_list), np.var(acc_list), np.var(auc_list)), file=sys.stdout)
    print("**" * 10)
    print("std f1:{}, std ACC:{}, std AUC:{}".format(np.std(f1_list), np.std(acc_list), np.std(auc_list)), file=f, flush=True)
    print(", std f1:{}, std ACC:{}, std AUC:{}".format(np.std(f1_list), np.std(acc_list), np.std(auc_list)), file=sys.stdout)
