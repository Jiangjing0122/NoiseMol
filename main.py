import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict
from rdkit.Chem import AllChem
from rdkit import Chem

from models import *
torch.backends.cudnn.enabled=False

localtime = time.asctime(time.localtime(time.time()))
localtime = '-'.join(localtime.split())
import pdb
from load_data import load_data


def write_record(args, message):

    fw = open('outputs/data_{}_seq_{}_bpeE_{}_fu_{}_recons_{}_{}.txt'.format(\
         args.dataset, args.sequence, args.bpe, args.fusion, args.recons, localtime), 'a')
    fw.write('{}\n'.format(message))
    fw.close()


def get_args():
    parser = argparse.ArgumentParser(description='pytorch version of GraSeq')

    ''' Graph Settings '''
    parser.add_argument('--agg_func', type=str, default='MEAN')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--b_sz', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', type=bool, default=True, help='use CUDA')
    #parser.add_argument('--gcn', action='store_true')
    parser.add_argument('--unsup_loss', type=str, default='margin')
    parser.add_argument('--max_vali_f1', type=float, default=0)
    parser.add_argument('--name', type=str, default='debug')

    ''' Sequence Settings '''
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_size_graph', type=int, default=64)

    ''' Options '''
    #parser.add_argument('--graph', type=bool, default=True)
    parser.add_argument('--sequence', type=bool, default=True)
    parser.add_argument('--bpe', type=bool, default=False)
    parser.add_argument('--recons', type=bool, default=True)
    parser.add_argument('--fusion', type=bool, default=False)

    parser.add_argument('--attn', type=bool, default=False)

    parser.add_argument('--dataset', type=str)
    #parser.add_argument('--load_long', type=bool, default=False)

    args = parser.parse_args()
    return args


from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc


def evlauation(pred, label):

    macro_precision = precision_score(label, pred)
    macro_recall = recall_score(label, pred)
    macro_f1 = f1_score(label, pred, average='macro')
    micro_f1 = f1_score(label, pred, average='micro')

    return macro_precision, macro_recall, macro_f1, micro_f1


def evlauation_auc(pred, label):

    fpr, tpr, _  = roc_curve(label, pred[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)

    fpr2, tpr2, _  = roc_curve(label, pred[:, 0], pos_label=0)
    auc_score2 = auc(fpr2, tpr2)
    # print('auc_score for label == 0: {}, auc_score for label == 1:
    #                             {}'.format(auc_score, auc_score2))
    return auc_score, fpr, tpr


def main(options, d_name):

    ''' set parameters '''
    args = get_args()
    print(args)
    ''' option settings '''
    args.sequence = options[0]
    args.bpe = options[1]
    args.fusion = options[2]
    args.recons = options[3]
    args.load_long = options[4]

    args.dataset = d_name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))

    write_record(args, '{}'.format(str(args)))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print('current running on the device: {} with loading type: {}'.format(\
                                               args.device, args.load_long))
    write_record(args, 'current running on the device: {} with loading type: {}'.format(\
                                                args.device, args.load_long))

    print('data: {} seq: {} bpe: {} fusion: {} recons: {} attn: {} '.format(\
         args.dataset, args.sequence, args.bpe, args.fusion, args.recons, args.attn))
    write_record(args, 'data: {} seq: {} bpe: {} fusion: {} recons: {} attn: {}'.format(\
          args.dataset, args.sequence, args.bpe, args.fusion, args.recons, args.attn))
    sys.stdout.flush()
    if args.unsup_loss == 'margin':
        num_neg = 4
    elif args.unsup_loss == 'normal':
        num_neg = 100

    args.train_data, args.train_labels, args.test_data, args.test_labels \
                                    = load_data(args.dataset, args.device)
    args.input_size_graph = args.train_data['features'][0].size(1)
    #pdb.set_trace()
    model = Model(args)
    model.to(args.device)

    multiclass_metrics = []

    batch_size = 2
    bad_cases = 0
    for epoch in range(1, args.epochs):
        start_time = time.time()
        print("Training, epoch %d ..." % epoch)
        sys.stdout.flush()

        train_graphs = np.arange(len(args.train_data['adj_lists']))

        np.random.shuffle(train_graphs)
        epoch_len = len(train_graphs)
        losses = 0.0
        
        samples = []

        for idx, s in enumerate(train_graphs):
            samples.append(s)
            if ((idx+1)% batch_size == 0 or idx == epoch_len-1):
                loss = model.train(samples, epoch, int(idx/batch_size))
                losses += loss.item()
                samples.clear()
            if idx % 1000 == 0:
                print ('done %d/%d ... ' % (idx, epoch_len))
        
        test_graphs = np.arange(len(args.test_data['adj_lists']))
        np.random.shuffle(test_graphs)
        #
        print("Testing, epoch %d ..." % epoch)
        sys.stdout.flush()
        outs = torch.zeros(args.test_labels.shape[0], 2)
        for graph_index in tqdm(test_graphs, desc=f'Epoch {epoch}', ascii=True, leave=False):
            out = model.test(graph_index)
            outs[graph_index, :] = out

        test_pred = F.softmax(outs, dim=1)
        test_pred_label = torch.max(test_pred, 1)[1]

        test_pred = test_pred.cpu().detach().numpy()
        test_pred_label = test_pred_label.cpu().detach().numpy()

        pre, rec, maf1, mif1 = evlauation(test_pred_label, args.test_labels)
        auc_score, fpr, tpr = evlauation_auc(test_pred, args.test_labels)

        multiclass_metrics.append([auc_score, pre, rec, maf1, mif1])
        best_auc = sorted(multiclass_metrics, key=lambda x: x[0], reverse=True)[0][0]

        
        print('epoch: {} loss: {:.4f} prec.:{:.4f} rec.:{:.4f} ma-f1:{:.4f} acc:{:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(\
                        epoch, losses, pre, rec, maf1, mif1, auc_score, best_auc))
        write_record(args, 'epoch: {} loss: {:.4f} prec.:{:.4f} rec.:{:.4f} ma-f1:{:.4f} acc:{:.4f} auc:{:.4f} ## best_auc:{:.4f}'.format(\
                               epoch, losses, pre, rec, maf1, mif1, auc_score, best_auc))
        print('Done, epoch %d, took %.1f minutes ...'%(epoch, \
                        float(time.time() - start_time) / 60.))
        print('-------------' * 6)
        #
        #pdb.set_trace()
        if auc_score >= best_auc:
            bad_cases = 0
        else:
            bad_cases += 1
        if bad_cases > 15:
            print ('Early stop at epoch %d ...'%(epoch))
            break
        sys.stdout.flush()

if __name__ == '__main__':
    
    
    # args.sequence = options[0]
    # args.bpe = options[1]
    #args.fusion = options[2]
    # args.recons = option[3]
    # args.load_long = option[4]

    d_name = "bace"
 
    option_list =  [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False],\
                    [False, True, False, False, False], [False, True, False, False, False], [False, True, False, False, False]]

    for op in option_list:
        main(op, d_name)

