import os
import sys

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.hgcn.layers.hyplayers import HGATConv, TemporalAttentionLayer
from script.hgcn.manifolds import PoincareBall


class Dynhat(nn.Module):
    def __init__(self, args, time_length):
        super(Dynhat, self).__init__()
        self.manifold = PoincareBall()

        self.c = 1
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)

        self.linear = nn.Linear(args.nfeat, args.nout)
        self.hidden_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        
        self.layer1 = HGATConv(self.manifold, args.nout, args.nhid, self.c, self.c,
                                heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=True)
        self.layer2 = HGATConv(self.manifold, args.nhid * args.heads, args.nout, self.c, self.c,
                                heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=False)
        
        self.ddy_attention_layer = TemporalAttentionLayer(
            input_dim=args.nout, 
            n_heads=args.temporal_attention_layer_heads, 
            num_time_steps=time_length,  
            attn_drop=0,  # dropout
            residual=False  
            )
        self.nhid = args.nhid
        self.nout = args.nout
        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.feat)
        glorot(self.linear.weight)
    

    def initHyperX(self, x, c=1.0):
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x


    def forward(self, edge_index, x=None, weight=None):
        x = self.initHyperX(self.linear(x), self.c)
        x = self.manifold.proj(x, self.c)
        x = self.layer1(x, edge_index)
        x = self.manifold.proj(x, self.c)
        x = self.layer2(x, edge_index)
        x = self.toTangentX(x, self.c) 
        return x
