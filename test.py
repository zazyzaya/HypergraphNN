import torch
from torch import nn
from torch_geometric.nn.models import GraphSAGE
from torch.optim import Adam
import pandas as pd
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score

from build_dblp import to_hypergraph
from utils import split_data, hypergraph_to_cliques, get_test_edges
from models.hyper_gnn import HyperGNNConv

EPOCHS = 500
LR = 0.0001

class HyperGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, ef_dim=0, aggrs=('mean', 'mean')):
        super().__init__()

        self.conv1 = HyperGNNConv(in_dim, hidden_dim, ef_dim=ef_dim, edge_aggr=aggrs[0], node_aggr=aggrs[1])
        self.conv2 = HyperGNNConv(hidden_dim, hidden_dim, edge_aggr=aggrs[0], node_aggr=aggrs[1])
        #self.conv3 = HyperGNNConv(hidden_dim, out_dim, edge_aggr=aggrs[0], node_aggr=aggrs[1])

    def forward(self, x, ei, ef=None):
        z = torch.relu(self.conv1(x,ei,ef))
        z = torch.relu(self.conv2(z,ei,ef))
        #z = torch.relu(self.conv3(z,ei,ef))
        return z


'''
(128,32)x3
    v_auc    0.871260
    v_ap     0.876991
    t_auc    0.862588
    t_ap     0.869302
    dtype: float64
    v_auc    0.004652
    v_ap     0.004303
    t_auc    0.004278
    t_ap     0.003963

(128,32)x2
    v_auc    0.929662
    v_ap     0.936852
    t_auc    0.926276
    t_ap     0.933543
    dtype: float64
    v_auc    0.001281
    v_ap     0.000994
    t_auc    0.001126
    t_ap     0.001049

(128,32)x2 with norm and cat
    v_auc    0.931238
    v_ap     0.940469
    t_auc    0.933183
    t_ap     0.942633
    dtype: float64
    v_auc    0.002772
    v_ap     0.002705
    t_auc    0.003122
    t_ap     0.002459

(512,128)x2
    v_auc    0.939383
    v_ap     0.946020
    t_auc    0.931807
    t_ap     0.939606
    dtype: float64
    v_auc    0.000638
    v_ap     0.000637
    t_auc    0.001548
    t_ap     0.001275

(512,128)x2 with norm and cat
    v_auc    0.935818
    v_ap     0.936403
    t_auc    0.936930
    t_ap     0.938894
    dtype: float64
    v_auc    0.004201
    v_ap     0.004778
    t_auc    0.002726
    t_ap     0.002997

(512,128)x2 with norm and cat lr=0.0001
    v_auc    0.971700
    v_ap     0.977121
    t_auc    0.973581
    t_ap     0.978281
    dtype: float64
    v_auc    0.000723
    v_ap     0.000606
    t_auc    0.000870
    t_ap     0.000802

SAGE
(128,32)x2
    v_auc    0.939172
    v_ap     0.947379
    t_auc    0.933941
    t_ap     0.943168
    dtype: float64
    v_auc    0.001030
    v_ap     0.000896
    t_auc    0.001871
    t_ap     0.001417

(512,128)x2
    v_auc    0.954407
    v_ap     0.959783
    t_auc    0.945632
    t_ap     0.953246
    dtype: float64
    v_auc    0.000610
    v_ap     0.000716
    t_auc    0.001332
    t_ap     0.001110

(512,128)x2 lr=0.0001
    v_auc    0.963047
    v_ap     0.970630
    t_auc    0.960025
    t_ap     0.968130
    dtype: float64
    v_auc    0.001080
    v_ap     0.000845
    t_auc    0.001027
    t_ap     0.000856
'''

data = to_hypergraph()

def test():
    tr,va,te = split_data(data)

    #model = HyperGNN(tr.x.size(1), 512, 128)#, 32)
    model = GraphSAGE(tr.x.size(1), 512, 2, 128)

    opt = Adam(model.parameters(), lr=LR)
    tr_edges = get_test_edges(tr.h_edge_index, tr.h_edge_index)
    va_edges = get_test_edges(va.h_edge_index, tr.h_edge_index)
    te_edges = get_test_edges(te.h_edge_index, tr.h_edge_index)

    def get_metrics(preds):
        '''
        Assumes preds are always first half TPs, second half TNs
        '''
        labels = torch.zeros(preds.size(0))
        labels[:preds.size(0)//2] = 1

        auc = auc_score(labels, preds)
        ap = ap_score(labels, preds)

        return auc,ap

    def neg_embs(z, cnt):
        return (
            z[torch.randint(0, z.size(0), (cnt,))] *
            z[torch.randint(0, z.size(0), (cnt,))]
        ).sum(dim=1)

    criterion = nn.BCEWithLogitsLoss()
    best = (0,0,0,0)
    for e in range(EPOCHS):
        opt.zero_grad()
        z = model(tr.x, tr_edges) #tr.h_edge_index)

        pos = (z[tr_edges[0]] * z[tr_edges[1]]).sum(dim=1)
        neg = neg_embs(z, pos.size(0))

        labels = torch.zeros(pos.size(0)*2)
        labels[:pos.size(0)] = 1

        preds = torch.cat([pos,neg])
        loss = criterion.forward(preds, labels)
        loss.backward()
        opt.step()

        auc,ap = get_metrics(preds.detach())
        print(f'[{e}] Loss: {loss.item():0.4f}, AUC: {auc:0.4f}, AP: {ap:0.4f}')

        with torch.no_grad():
            z = model(tr.x, tr_edges) #tr.h_edge_index)

            va_preds = torch.cat([
                (z[va_edges[0]] * z[va_edges[1]]).sum(dim=1),
                neg_embs(z, va_edges.size(1))
            ])
            v_auc, v_ap = get_metrics(va_preds)
            print(f'\tVal AUC: {v_auc:0.4f}, AP: {v_ap:0.4f}')

            te_preds = torch.cat([
                (z[te_edges[0]] * z[te_edges[1]]).sum(dim=1),
                neg_embs(z, te_edges.size(1))
            ])
            t_auc, t_ap = get_metrics(te_preds)
            print(f'\tTe  AUC: {t_auc:0.4f}, AP: {t_ap:0.4f}')

            if v_auc > best[0]:
                best = (v_auc, v_ap, t_auc, t_ap)

    v_auc, v_ap, t_auc, t_ap = best
    print("\nBest: ")
    print(f'\tVal AUC: {v_auc:0.4f}, AP: {v_ap:0.4f}')
    print(f'\tTe  AUC: {t_auc:0.4f}, AP: {t_ap:0.4f}')

    return dict(zip(['v_auc', 'v_ap', 't_auc', 't_ap'],best))

scores = [
    test() for _ in range(10)
]
df = pd.DataFrame(scores)
print(df.mean())
print(df.sem())