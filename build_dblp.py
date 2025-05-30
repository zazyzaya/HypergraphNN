import torch
from torch_geometric.data import Data
from torch_geometric.datasets import DBLP

from globals import PYG_HOME

def to_hypergraph():
    data = DBLP(f'{PYG_HOME}/dblp/')[0]

    h_edges = data[('author', 'to', 'paper')]['edge_index'][[1,0]]
    h_edge_x = data['paper']['x']
    node_x = data['author']['x']
    node_y = data['author']['y']

    return Data(
        x=node_x,
        y=node_y,
        h_edge_index=h_edges,
        h_edge_attr=h_edge_x
    )