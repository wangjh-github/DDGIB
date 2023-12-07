import argparse
import torch
import os

parser = argparse.ArgumentParser(description='DyTed')

# 1.dataset
parser.add_argument('--dataset', type=str, default='viturial', help='dataset')
parser.add_argument('--pre_defined_feature', default=True, help='pre-defined node feature')
parser.add_argument('--use_trainable_feature', default=True, help='pre-defined node feature')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--node_num', type=int, default=1809, help='dim of input feature')


# 2.model
parser.add_argument('--model', type=str, default='DyTed', help='model name')
parser.add_argument('--heads', type=int, default=2, help='attention heads.')

parser.add_argument('--static_cov', type=int, default=2, help='layers of static gcn cov.')
parser.add_argument('--static_feat_list', type=list, default=[128, 256], help='in feature of each layer - static.')
parser.add_argument('--static_gcn_dropout', type=float, default=0.2, help='dropout of gcn - static.')
parser.add_argument('--static_trans_num', type=int, default=3, help='layers of transformer block - static.')
parser.add_argument('--static_hidden', type=int, default=64, help='hidden layer of attn - static.')
parser.add_argument('--static_forward_hidden', type=int, default=128, help='hidden layer of ffn - static.')
parser.add_argument('--static_trans_dropout', type=float, default=0.2, help='dropout of trans - static.')

parser.add_argument('--dynamic_cov', type=int, default=2, help='layers of dynamic gcn cov.')
parser.add_argument('--dynamic_feat_list', type=list, default=[128, 256], help='in feature of each layer - dynamic.')
parser.add_argument('--dynamic_gcn_dropout', type=float, default=0.2, help='dropout of gcn - dynamic.')
parser.add_argument('--dynamic_trans_num', type=int, default=3, help='layers of transformer block - dynamic.')
parser.add_argument('--dynamic_hidden', type=int, default=64, help='hidden layer of attn - dynamic.')
parser.add_argument('--dynamic_forward_hidden', type=int, default=128, help='hidden layer of ffn - dynamic.')
parser.add_argument('--dynamic_trans_dropout', type=float, default=0.2, help='dropout of trans. - dynamic')
parser.add_argument('--alpha', type=float, default=0.999, help='parameter of graph_loss')
parser.add_argument('--beta', type=float, default=0.0001, help='parameter of dynamic_loss')

parser.add_argument('--choose_p', type=float, default=0.75, help='propagate by probability p')
parser.add_argument('--sample_max_len', type=int, default=10, help='max length of sampling.')
parser.add_argument('--sample_min_len', type=int, default=3, help='min length of sampling.')

parser.add_argument('--dis_in', type=int, default=128, help='discriminator in feat.')
parser.add_argument('--dis_hid', type=int, default=64, help='discriminator hidden feat.')

# 3.experiment
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (# nodes)')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('--device', type=str, default='cuda', help='training device')
parser.add_argument('--device_id', type=str, default='2', help='device id for gpu')
parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
parser.add_argument('--node', type=int, default=4000, help='sample num')
parser.add_argument('--time', type=int, default=30, help='timesnops')
parser.add_argument('--dim', type=int, default=30, help='dim')
parser.add_argument('--layer', type=int, default=4, help='class num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')
parser.add_argument('--max_epoch', type=int, default=300, help='max epoch')
parser.add_argument('--pos_num', type=int, default=3, help='positive item')
parser.add_argument('--neg_num', type=int, default=10, help='negative item')
parser.add_argument('--shf_weight', type=float, default=1.0, help='shuffle weight')
parser.add_argument('--global_weight', type=float, default=1.0, help='global weight')
parser.add_argument('--log_interval', type=int, default=1, help='log interval')

parser.add_argument('--dis_start', type=int, default=-1, help='start training discriminator')
parser.add_argument('--dis_epoch', type=int, default=5, help='training discriminator each epoch')
parser.add_argument('--dis_sample_num', type=int, default=500, help='number of positive samples')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

if args.dataset == "Enron":
    args.use_trainable_feature = False
    args.node_num = 143
    args.heads = 4
    args.static_feat_list = [143, 256]
    args.dynamic_feat_list = [143, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 13
    args.sample_min_len = 3
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset == "yelp":
    args.use_trainable_feature = True
    args.heads = 4
    args.static_feat_list = [128, 256]
    args.dynamic_feat_list = [128, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 11
    args.sample_min_len = 3
    args.node_num = 6569
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset == "uci":
    args.use_trainable_feature = True
    args.heads = 4
    args.static_feat_list = [128, 256]
    args.dynamic_feat_list = [128, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 11
    args.sample_min_len = 3
    args.node_num = 1809
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset in ["HepTh"]:
    args.use_trainable_feature = True
    args.heads = 4
    args.static_feat_list = [128, 256]
    args.dynamic_feat_list = [128, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 20
    args.sample_min_len = 6
    args.node_num = 7576
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset in ["Brain"]:
    args.use_trainable_feature = False
    args.heads = 4
    args.static_feat_list = [20, 256]
    args.dynamic_feat_list = [20, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 20
    args.sample_min_len = 6
    args.node_num = 5000
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset in ["reddit"]:
    args.use_trainable_feature = False
    args.heads = 4
    args.static_feat_list = [20, 256]
    args.dynamic_feat_list = [20, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 20
    args.sample_min_len = 6
    args.node_num = 8291
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset in ["DBLP3"]:
    args.use_trainable_feature = False
    args.heads = 4
    args.static_feat_list = [100, 256]
    args.dynamic_feat_list = [100, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 20
    args.sample_min_len = 6
    args.node_num = 4257
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset in ["DBLP5"]:
    args.use_trainable_feature = False
    args.heads = 4
    args.static_feat_list = [100, 256]
    args.dynamic_feat_list = [100, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 20
    args.sample_min_len = 6
    args.node_num = 6606
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

if args.dataset in ["viturial"]:
    args.use_trainable_feature = False
    args.heads = 4
    args.static_feat_list = [32, 256]
    args.dynamic_feat_list = [32, 256]
    args.static_trans_num = 3
    args.dynamic_trans_num = 3
    args.static_hidden = 64
    args.dynamic_hidden = 64
    args.static_forward_hidden = 64
    args.dynamic_forward_hidden = 64
    args.sample_max_len = 20
    args.sample_min_len = 6
    args.node_num = 4000
    args.dis_in = 64
    args.dis_hid = 32
    args.dis_start = 50

# set the running device
if torch.cuda.is_available() and args.use_gpu:
    print('using gpu:{} to train the model'.format(args.device_id))
    args.device_id = list(range(torch.cuda.device_count()))      # torch.cuda.device_count())
    args.device = torch.device("cuda:{}".format(0))

else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

if args.use_trainable_feature:
    print('using trainable feature')




