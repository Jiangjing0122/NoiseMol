import torch
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict
import pdb


def load_data(dataset, device):

    #pdb.set_trace()
    data_file = f"../original_datasets_bpe/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    # 
    node_types = set()
    node_types.add("<mask>")
    node_types.add("<pad>")
    node_types.add("<unk>")
    #
    label_types = set()
    tr_len = 0
    for line in file:
        tr_len += 1
        smiles = line.split("\t")[1]
        s = []
        s = smiles.split('|')
        #
        mol = smiles.replace('|','')
        if mol is None:
            continue
            #pdb.set_trace()
        for atom in mol:
           s.append(atom)
        node_types |= set(s)
        label = line.strip().split('\t')[2]
        label_types.add(label)
    file.close()
    #pdb.set_trace()
    te_len = 0
    data_file = f"../original_datasets_bpe/{dataset}/{dataset}_test"
    file = open(data_file, "r")
    for line in file:
        te_len += 1
        smiles = line.split("\t")[1]
        s = []
        s = smiles.split('|')
        mol = smiles.replace('|','')
        if mol is None:
            continue
        for atom in mol:
           s.append(atom)
        node_types |= set(s)
        label = line.strip().split('\t')[2]
        label_types.add(label)
    file.close()

    print(tr_len)
    print(te_len)
    
    node2index = {n: i for i, n in enumerate(node_types)}
    label2index = {l: i for i, l in enumerate(label_types)}

    print(node2index)
    print(label2index)
    max_len = 256
    #pdb.set_trace()
    data_file = f"../original_datasets_bpe/{dataset}/{dataset}_train"
    file = open(data_file, "r")
    train_adjlists = []
    train_features = []
    train_sequence = []
    train_bpe = []
    train_labels = torch.zeros(tr_len)
    for line in file:
        if len(line.strip().split('\t')) < 3:
            continue
        smiles = line.split("\t")[1]
        #label = line.split("\t")[2][:-1]
        label = line.strip().split('\t')[2]
        #
        mol_str = smiles.replace('|','')
        if len(mol_str) > max_len:
            continue
        mol = AllChem.MolFromSmiles(smiles.replace('|',''))
        if mol is None:
            continue
        
        mol_ = smiles.split('|')
        #pdb.set_trace()
        feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
        l = 0
        smiles_seq = []
        smiles_bpe = []
        for atom in mol_str:
            #feature[l, node2index[atom]] = 1
            smiles_seq.append(node2index[atom])
            l += 1
        #pdb.set_trace()
        for atom in mol_:
            smiles_bpe.append(node2index[atom])
        #pdb.set_trace()
        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            typ = bond.GetBondType()
            adj_list[i].append(j)
            adj_list[j].append(i)
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_list[i].append(j)
                adj_list[j].append(i)
        train_labels[len(train_adjlists)]= int(label2index[label])
        train_adjlists.append(adj_list)
        train_features.append(torch.FloatTensor(feature).to(device))
        train_sequence.append(torch.tensor(smiles_seq))
        train_bpe.append(torch.tensor(smiles_bpe))
    file.close()

    data_file = f"../original_datasets_bpe/{dataset}/{dataset}_test"
    file = open(data_file, "r")
    test_adjlists = []
    test_features = []
    test_sequence = []
    test_bpe = []
    test_labels = np.zeros(te_len)
    for line in file:
        smiles = line.split("\t")[1]
        label = line.strip().split('\t')[2]
        mol  = AllChem.MolFromSmiles(smiles.replace('|',''))
        mol_str  = smiles.replace('|','')
        if mol is None:
            continue
        if len(mol_str) > max_len:
            continue
        mol_ = smiles.split('|')
        feature = torch.zeros(len(mol.GetAtoms()), len(node_types))
        l = 0
        smiles_seq = []
        for atom in mol_str:
            #feature[l, node2index[atom]] = 1
            smiles_seq.append(node2index[atom])
            l += 1
        smiles_bpe = []
        for atom in mol_:
            smiles_bpe.append(node2index[atom])

        adj_list = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            typ = bond.GetBondType()
            adj_list[i].append(j)
            adj_list[j].append(i)
            if typ == Chem.rdchem.BondType.DOUBLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
            elif typ == Chem.rdchem.BondType.TRIPLE:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_list[i].append(j)
                adj_list[j].append(i)
        test_labels[len(test_adjlists)] = int(label2index[label])
        test_adjlists.append(adj_list)
        test_features.append(torch.FloatTensor(feature).to(device))
        test_sequence.append(torch.tensor(smiles_seq))
        test_bpe.append(torch.tensor(smiles_bpe))

    file.close()
    
    train_data = {}
    train_data['adj_lists'] = train_adjlists
    train_data['features'] = train_features
    train_data['sequence'] = train_sequence
    train_data['bpe'] = train_bpe
    train_data['dict'] = node2index

    test_data = {}
    test_data['adj_lists'] = test_adjlists
    test_data['features'] = test_features
    test_data['sequence'] = test_sequence
    test_data['bpe'] = test_bpe
    
    return train_data, train_labels, test_data, test_labels