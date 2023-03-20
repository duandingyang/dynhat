import argparse
import torch
import os

parser = argparse.ArgumentParser(description='Dynhat')
# 1.dataset
parser.add_argument('--dataset', type=str, default='enron184', help='datasets')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=32, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=32, help='dim of output embedding')

# 2.experiments
parser.add_argument('--split_count', type=str, default=12, help='时间切片个数')
parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train.')
parser.add_argument('--testlength', type=int, default=1, help='length for test, default:3')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=1024, help='random seed')
parser.add_argument('--patience', type=int, default=20, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
parser.add_argument('--debug_content', type=str, default='', help='debug_mode content')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--debug_mode', type=int, default=0, help='debug_mode, 0: normal running; 1: debugging mode')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')

# 3.models
parser.add_argument('--model', type=str, default='Dynhat', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', help='Hyperbolic model')
parser.add_argument('--EPS', type=float, default=1e-15, help='eps')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--trainable_feat', type=int, default=0,
                    help='using trainable feat or one-hot feat, default: none-trainable feat')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (1 - keep probability).')
parser.add_argument('--heads', type=int, default=1, help='structural attention heads.')
parser.add_argument('--temporal_attention_layer_heads', type=int, default=1, help='temporal_attention_layer heads')
parser.add_argument('--curvature', type=float, default=1.0, help='curvature value')

args = parser.parse_args()


args.result_txt = '../data/output/results/{}_{}_{}_result.txt'.format(args.dataset, args.model, args.split_count)

if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

