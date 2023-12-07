import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch as tg
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class MyDataset(Dataset):
    def __init__(self, graphs, nodes, adjs, time_step, pos_num, neg_num, node_num):
        super(MyDataset, self).__init__()
        self.graphs = nodes
        self.pyg_graphs = graphs
        self.adjs = adjs
        self.time_steps = time_step
        self.train_nodes = list(self.graphs[self.time_steps - 1].nodes())

        # positive samples and negative samples
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.node_num = node_num
        self.__createitems__()

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    # normalization of node features
    def _normalize_feature(self, features):
        features = np.array(features.todense())
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    # normalization of adjacent matrix
    def _normalize_graph_gcn(self, adj):
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized

    def __createitems__(self):
        self.data_items = {}
        for node in self.train_nodes:
            feed_dict = {}
            in_list = []
            neighbor_list = []
            no_neighbor_list = []
            for i in range(0, self.time_steps):
                g = self.graphs[i]
                nodes = g.nodes()
                if node in nodes:
                    in_list.append(True)
                    neighbors = [n for n in g.neighbors(node)]
                    no_neighbors = list(set(nodes) - set(neighbors) - set([node]))
                    neighbor_list.append(neighbors)
                    if len(no_neighbors) > self.neg_num * 10:
                        no_neighbor_list.append(np.random.choice(no_neighbors, size=self.neg_num * 10, replace=False))
                    else:
                        no_neighbor_list.append(no_neighbors)
                else:
                    in_list.append(False)
                    neighbor_list.append([])
                    no_neighbor_list.append([])
            feed_dict["isIn"] = in_list
            feed_dict["neighbor"] = neighbor_list
            feed_dict["no_neighbor"] = no_neighbor_list
            feed_dict["pos"] = self.pos_num
            feed_dict["neg"] = self.neg_num
            feed_dict["length"] = self.time_steps
            feed_dict["node"] = node
            feed_dict["graphs"] = self.pyg_graphs
            self.data_items[node] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {"graphs": samples[0]["graphs"], "data": []}
        for sample in samples:
            pos_list = []
            neg_list = []
            for i in range(0, sample["length"]):
                if sample["isIn"][i] is False:
                    pos_list.append([])
                    neg_list.append([])
                    continue
                # find pos
                if len(sample["neighbor"][i]) > 0:
                    pos_list.append(np.random.choice(sample["neighbor"][i], size=sample["pos"],
                                                     replace=(len(sample["neighbor"][i]) < sample["pos"])))
                else:
                    pos_list.append([])

                # find neg
                if len(sample["no_neighbor"][i]) > 0:
                    neg_list.append(np.random.choice(sample["no_neighbor"][i], size=sample["neg"],
                                                     replace=(len(sample["no_neighbor"][i]) < sample["neg"])))
                else:
                    neg_list.append([])
            data_dict = {"node": sample["node"], "isIn": sample["isIn"], "pos": pos_list, "neg": neg_list}
            batch_dict["data"].append(data_dict)
        return batch_dict
