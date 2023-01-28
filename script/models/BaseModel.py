import torch
import torch.nn as nn
from torch.nn import Parameter
from script.config import args
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class BaseModel(nn.Module):
    def __init__(self, args=None):
        super(BaseModel, self).__init__()
        if args.use_gru:
            self.gru = nn.GRUCell(args.nhid, args.nhid)
        else:
            self.gru = lambda x, h: x

        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nhid)
        self.hidden_initial = torch.ones(args.num_nodes, args.nhid).to(args.device)

        self.model_type = args.model[:3]  # GRU or Dyn
        self.Q = Parameter(torch.ones((args.nhid, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.nhid = args.nhid
        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    

    def forward(self, edge_index, x=None, weight=None):
        if x is None:
            x = self.linear(self.feat)
        else:
            x = self.linear(x)
        if self.model_type == 'Dyn':
            x = self.continuous_encode(edge_index, x, weight)
        if self.model_type == 'GRU':
            x = self.gru_encode(edge_index, x, weight)
        return x
