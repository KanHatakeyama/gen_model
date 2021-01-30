"""
GNN library
codes adopted from
https://github.com/iwatobipen/playground/blob/master/GCN_chemo.ipynb
https://discuss.dgl.ai/t/cant-run-gcn-chemo-py-an-example-on-github/1280/2

"""

import networkx as nx
import os
from rdkit import Chem
import dgl
from dgl import DGLGraph
import dgl.function as fn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

# constants
MID_DIM = 32
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl',
             'Br', 'Na', 'Ca', 'I', 'B', 'K', 'H',  'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
MAX_ATOMNUM = 60
BOND_FDIM = 5
MAX_NB = 10
PAPER = os.getenv('PAPER', False)

# utility funcs


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return (onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
            + [atom.GetIsAromatic()])


def bond_features(bond):
    bt = bond.GetBondType()
    return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]))


def mol2dgl_single(mols):
    cand_graphs = []
    n_nodes = 0
    n_edges = 0
    bond_x = []

    for mol in mols:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        g = DGLGraph()
        nodeF = []
        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            nodeF.append(atom_features(atom))
        g.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)

            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)
        g.add_edges(bond_src, bond_dst)
        g.ndata['h'] = torch.Tensor(nodeF)
        cand_graphs.append(g)
    return cand_graphs


def collate(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def reduce(nodes):
    # summazation by avarage is different part
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


msg = fn.copy_src(src="h", out="m")

# fix bug
# https://discuss.dgl.ai/t/cant-run-gcn-chemo-py-an-example-on-github/1280/2


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        h = self.linear(g.ndata['h'])
        if self.activation is not None:
            h = self.activation(h)
        return h
