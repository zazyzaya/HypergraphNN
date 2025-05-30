from typing import List

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

def node_sets_to_hypergraph(sets: List[set]) -> torch.Tensor:
    '''
    Takes as input sets of hyperedges
    Returns a Data object where "edge_index" represents the hyper edges
    s.t. the top row is the node ids and the bottom row is the edge id

    E.g. {0,1,2}, {0,2}, {2,3,4} -> [[0,0,0, 1,1, 2,2,2], [0,1,2,0,2,2,3,4]]

    This will make the scatter-gather aggrs in the HGNN simpler
    '''

    dim = sum([len(s) for s in sets])
    ei = torch.empty((2,dim), dtype=torch.long)

    ei[1] = torch.as_tensor(sum([list(s) for s in sets], []))

    lens = torch.as_tensor([len(s) for s in sets])
    ei[0] = torch.arange(lens.size(0)).repeat_interleave(lens)

    return ei

def hypergraph_to_cliques(h_edge_index):
    '''
    Given input of hypergraphs formatted as above, converts into
    a standard graph, where hyperedges become cliques

    E.g. [[0,0,0], [0,1,2]] -> [[0,0,0, 1,1,1, 2,2,2], [0,1,2, 0,1,2, 0,1,2]]

    Not very efficient. TODO make this not O(n^3)
    '''
    x = torch.eye(h_edge_index[1].max() + 1)
    coocurrence = scatter(x[h_edge_index[1]], h_edge_index[0], dim=0)

    edges = set()
    for row in coocurrence:
        uq = row.nonzero()[:,0]
        for i in range(uq.size(0)):
            for j in range(uq.size(0)):
                if i != j:
                    edges.add((uq[i].item(), uq[j].item()))

    src,dst = zip(*edges)
    return torch.tensor([src,dst])

def split_data(g):
    perm = torch.randperm(g.h_edge_index.size(1))
    tr_end = int(perm.size(0) * 0.8)
    va_end = int(perm.size(0) * 0.1) + tr_end

    tr = Data(
        x = g.x,
        y = g.y,
        h_edge_index = g.h_edge_index[:, perm[:tr_end]],
        h_edge_attr = g.h_edge_attr
    )
    va = Data(
        x = g.x,
        y = g.y,
        h_edge_index = g.h_edge_index[:, perm[tr_end:va_end]],
        h_edge_attr = g.h_edge_attr
    )
    te = Data(
        x = g.x,
        y = g.y,
        h_edge_index = g.h_edge_index[:, perm[va_end:]],
        h_edge_attr = g.h_edge_attr
    )

    return tr,va,te

def get_test_edges(sub_hg, target_hg):
    '''
    Given a list of hyperedges and nodes
    Return all edges that node would have if hgs were represented as
    traditional graph cliques
    '''
    full_hg = torch.cat([sub_hg, target_hg], dim=1)
    x = torch.eye(full_hg[1].max() + 1)
    coocurrence = scatter(x[full_hg[1]], full_hg[0], dim=0)

    edges = set()
    for i in range(sub_hg.size(1)):
        hyper_edge,nid = sub_hg[:,i]
        neighbors = coocurrence[hyper_edge].nonzero().flatten()
        for n in neighbors:
            edges.add(
                (nid.item(), n.item())
            )

    src,dst = zip(*edges)
    return torch.tensor([src,dst])