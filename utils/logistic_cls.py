import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, auc
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from utils.utilize import *
from trainer.config import args


class MLP(nn.Module):
    def __init__(self, embedding_dimension=64):
        super(MLP, self).__init__()
        # Vanilla MLP
        self.linear_relu = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU()
        )
        self.linear_relu_1 = nn.Sequential(
            nn.Linear(embedding_dimension, args.layer),
            nn.PReLU()
        )
        self.linear_relu_2 = nn.Sequential(
            nn.Linear(embedding_dimension, args.layer),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.linear_relu(x)
        x = self.linear_relu_1(x)
        x = self.linear_relu_2(x)
        return x


loss = nn.CrossEntropyLoss()  # softmax
model = MLP(embedding_dimension=64)


def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, embedding):
    # Downstream logistic regression classifier to evaluate link prediction
    train_pos_feats = np.array(get_link_feats(train_pos, embedding))
    train_neg_feats = np.array(get_link_feats(train_neg, embedding))
    val_pos_feats = np.array(get_link_feats(val_pos, embedding))
    val_neg_feats = np.array(get_link_feats(val_neg, embedding))
    test_pos_feats = np.array(get_link_feats(test_pos, embedding))
    test_neg_feats = np.array(get_link_feats(test_neg, embedding))

    # label
    train_pos_labels = np.array([1] * len(train_pos_feats))
    train_neg_labels = np.array([-1] * len(train_neg_feats))
    val_pos_labels = np.array([1] * len(val_pos_feats))
    val_neg_labels = np.array([-1] * len(val_neg_feats))

    test_pos_labels = np.array([1] * len(test_pos_feats))
    test_neg_labels = np.array([-1] * len(test_neg_feats))

    # data
    train_data = np.vstack((train_pos_feats, train_neg_feats))
    train_labels = np.append(train_pos_labels, train_neg_labels)

    val_data = np.vstack((val_pos_feats, val_neg_feats))
    val_labels = np.append(val_pos_labels, val_neg_labels)

    test_data = np.vstack((test_pos_feats, test_neg_feats))
    test_labels = np.append(test_pos_labels, test_neg_labels)

    # Logistic Model
    logistic = linear_model.LogisticRegression(max_iter=10000)
    logistic.fit(train_data, train_labels)

    test_predict = logistic.predict_proba(test_data)[:, 1]
    val_predict = logistic.predict_proba(val_data)[:, 1]

    # score
    test_roc_score = roc_auc_score(test_labels, test_predict)
    val_roc_score = roc_auc_score(val_labels, val_predict)

    test_ap_score = average_precision_score(test_labels, test_predict)
    val_ap_score = average_precision_score(val_labels, val_predict)

    return test_roc_score, val_roc_score, test_ap_score, val_ap_score


def train(train_features, val_features, train_labels, val_labels, idx):
    model.train()
    embedding = model(train_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_train = loss(embedding, train_labels)
    acc_train = accuracy(embedding, train_labels)
    output = embedding.max(1)[1].to('cpu').detach().numpy()
    output = torch.LongTensor(output)
    sm = nn.Softmax(dim=1)
    f1 = f1_score(train_labels, output, average='weighted')
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    # val
    model.eval()
    embedding = model(val_features)
    embedding = sm(embedding)
    loss_val = loss(embedding, val_labels)
    acc_val = accuracy(embedding, val_labels)
    print('Epoch: {:04d}'.format(idx + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'f1: {:.4f}'.format(f1.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          )
    return loss_train, acc_val


def test(test_features, test_labels):
    model.eval()
    sm = nn.Softmax(dim=1)  # softmax
    emb = model(test_features)
    loss_test = loss(emb, test_labels)
    embedding = sm(emb)
    acc_test = accuracy(embedding, test_labels)
    output = embedding.max(1)[1].to('cpu').detach().numpy()
    f1 = f1_score(test_labels, output, average='weighted')
    auc_test = roc_auc_score(test_labels.detach().numpy(), embedding.detach().numpy(),
                             average='weighted', multi_class='ovr')
    return acc_test, loss_test, f1, auc_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
