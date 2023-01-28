import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from script.hgcn.layers.hyplayers import HGCNConv, HGATConv, TemporalAttentionLayer
from script.hgcn.manifolds import PoincareBall
from script.models.BaseModel import BaseModel


class Dynhat(BaseModel):
    def __init__(self, args, time_length):
        super(Dynhat, self).__init__(args)
        self.manifold_name = args.manifold
        self.manifold = PoincareBall()

        self.c = Parameter(torch.ones(3, 1).to(args.device) * args.curvature, requires_grad=not 1)
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nout)
        self.hidden_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        if args.aggregation == 'deg':
            self.layer1 = HGCNConv(self.manifold, 2 * args.nout, 2 * args.nhid, self.c[0], self.c[1],
                                   dropout=args.dropout)
            self.layer2 = HGCNConv(self.manifold, 2 * args.nhid, args.nout, self.c[1], self.c[2], dropout=args.dropout)
        if args.aggregation == 'att':
            self.layer1 = HGATConv(self.manifold, 2 * args.nout, 2 * args.nhid, self.c[0], self.c[1],
                                   heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=True)
            self.layer2 = HGATConv(self.manifold, 2 * args.nhid * args.heads, args.nout, self.c[1], self.c[2],
                                   heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=False)
        self.gru = nn.GRUCell(args.nout, args.nout)
        self.ddy_attention_layer = TemporalAttentionLayer(
            input_dim=args.nout, 
            n_heads=args.ddy_attention_layer_heads, 
            num_time_steps=time_length,  
            attn_drop=0,  # dropout
            residual=False  
            )
        self.nhid = args.nhid
        self.nout = args.nout
        self.cat = True
        self.Q = Parameter(torch.ones((args.nout, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.reset_parameters()
        self.init_hiddens()

    def reset_parameters(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.initHyperX(self.hidden_initial)]
        return self.hiddens
    

    def initHyperX(self, x, c=1.0):
        if self.manifold_name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
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
        if x is None:  # using trainable feat matrix
            x = self.initHyperX(self.linear(self.feat), self.c[0])
        else:
            x = self.initHyperX(self.linear(x), self.c[0])
        if self.cat:
            x = torch.cat([x, self.hiddens[-1]], dim=1)

        # layer 1
        x = self.manifold.proj(x, self.c[0])
        x = self.layer1(x, edge_index)

        # layer 2
        x = self.manifold.proj(x, self.c[1])
        x = self.layer2(x, edge_index)

        x = self.toTangentX(x, self.c[2])  # to tangent space
        
        return x
