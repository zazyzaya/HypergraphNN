import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

# As of PyG 2.6 you can't call propagate directly from
# MessagePassing. Hopefully they fix this soon.
class MP(MessagePassing):
    def forward(self, x,ei, size=None):
        return self.propagate(ei, size=size, x=x)

class HyperGNNConv(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, ef_dim=0, edge_aggr='mean', node_aggr='mean'):
        super().__init__()

        self.edge_aggr = edge_aggr
        if hidden_dim:
            self.edge_net = nn.Sequential(
                nn.Linear(in_dim + ef_dim, hidden_dim),
                nn.ReLU()
            )
            self.node_net_feats = hidden_dim
            self.use_edge_net = True

        else:
            self.node_net_feats = in_dim + ef_dim
            self.use_edge_net = False

        self.node_net = nn.Linear(in_dim + self.node_net_feats, out_dim)
        self.node_aggr = MP(node_aggr)

    def forward(self, x, h_edge_index, h_edge_feats=None):
        '''
        Given input
            x:              Matrix of node features
            h_edge_index:   [ Hyper edges, Node ids ]

        Aggregate features of nodes within the same hyper edge
        Aggregate representations of hyperedges for each node
        '''
        src = x[h_edge_index[1]]

        h_edge_aggr_feats = scatter(src, h_edge_index[0], reduce=self.edge_aggr, dim=0)

        # Hyperedge features are the feats of all nodes within them
        # and any features they have individually
        if h_edge_feats is not None:
            h_edge_aggr_feats = torch.cat([h_edge_aggr_feats, h_edge_feats], dim=1)

        if self.use_edge_net:
            h_edge_aggr_feats = self.edge_net(h_edge_aggr_feats)

        # Node features are the aggregate of all of their edge features
        node_feats = self.node_aggr(
            h_edge_aggr_feats,
            h_edge_index
        )[:x.size(0)]

        out = torch.cat([x, node_feats], dim=1)
        out = self.node_net(out)
        norm = out.norm(dim=1, keepdim=True)

        return out / norm