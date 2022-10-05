import copy
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data

import pickle
import random
import numpy as np


import time
from tqdm import tqdm

import encoders
import pdb


class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()

        self.device = args.device
        self.layers = args.num_layers
        self.input_size_graph = args.input_size_graph
        self.output_size_graph = args.output_size_graph
        self.train_data = args.train_data
        self.test_data = args.test_data
        self.train_labels = args.train_labels
        self.test_labels = args.test_labels
        self.latent_size = args.latent_size
        self.hidden_size = args.hidden_size
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.warmup = args.warmup

        self.sequence = args.sequence
        self.bpe = args.bpe
        self.recons = args.recons
        self.use_attn = args.attn
        self.use_fusion = args.fusion
        self.drop = 0.1
        self.encoder = encoders.Encoder(self.input_dim, self.hidden_size,\
                                          self.latent_size, self.device,
                                          dropout=self.drop)
        self.AtomEmbedding = nn.Embedding(self.input_size_graph,
                                          self.hidden_size).to(self.device)
        self.AtomEmbedding.weight.requires_grad = True

        self.output_layer = encoders.Classifier(self.latent_size, self.device)

        self.label_criterion = nn.CrossEntropyLoss()

        self.optimizer  = optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=1e-8,
                                     amsgrad=True)
        self.dropout = nn.Dropout(self.drop)

        for name, para in self.named_parameters():
            if para.requires_grad:
                print(name, para.data.shape)
    
    def samples_padding(self, samples, dict_=None, noisy=False,
                       noisy_type='mask', noisy_ratio=0.05):
        # noisy_type=swap：对互换, mask：掩盖, delete：删除
        max_len = 0
        mask_id = torch.tensor(dict_['<mask>'])
        # 
        for s in samples:
            if len(s) > max_len:
                max_len = len(s)
        # 
        x      = np.zeros((len(samples), max_len)).astype('int64')
        x_mask = np.zeros((len(samples), max_len)).astype('float32')
        #pdb.set_trace()
        if noisy:
            for idx in range(len(samples)):
                if noisy_type == 'swap':
                    ss = samples[idx]
                    size_ = len(ss)
                    for ex in range(3):
                        #pdb.set_trace()
                        rdm1 = random.randint(0, size_ - 1)
                        rdm2 = random.randint(0, size_ - 1)
                        tmp = ss[rdm1].item()
                        ss[rdm1] = ss[rdm2]
                        ss[rdm2] = torch.tensor(tmp)
                    buf = ss
                    x[idx, :len(buf)] = buf
                    x_mask[idx, :len(buf)] = 1
                else:
                    ss = samples[idx]
                    buf = []
                    for j in range(len(ss)):
                        rdm = random.randint(0, 99)/100
                        if rdm <= noisy_ratio:
                            if noisy_type == 'mask':
                                buf.append(mask_id)
                            elif noisy_type == 'delete':
                                # nothing
                                sa = 123
                            else:
                                buf.append(mask_id)
                        else:
                            buf.append(ss[j])
                    
                    x[idx, :len(buf)] = buf
                    x_mask[idx, :len(buf)] = 1
        else:
            for idx in range(len(samples)):
               x[idx, :len(samples[idx])] = samples[idx]
               x_mask[idx, :len(samples[idx])] = 1

        return torch.tensor(x).to(self.device), torch.tensor(x_mask).to(self.device)
    # 
    def train(self, graph_index, epoch, batch=1):
        
        enc_states = None
        outputs    = None
        dict_ = self.train_data['dict']
        #noisy = True if epoch % 2 == 0 else False
        noisy = True if batch % 2 == 0 else False
        
            #------------------------------------------------------
        if self.sequence and not self.use_fusion:
            #
            samples_seq = [self.train_data['sequence'][idx] for idx in graph_index]
            x_s, x_s_mask = self.samples_padding(samples_seq, dict_=dict_,
                                                 noisy=noisy)
            x_s_emb = self.dropout(self.AtomEmbedding(x_s))
            enc_states_seq = self.encoder(x_s_emb, x_mask=x_s_mask)
            enc_states = enc_states_seq
            #
            outputs = torch.tanh(torch.mean(enc_states, dim=1))
        elif self.bpe and not self.use_fusion:
            samples_bpe = [self.train_data['bpe'][idx] for idx in graph_index]
            x_b, x_b_mask = self.samples_padding(samples_bpe, dict_=dict_,
                                                 noisy=noisy)
            x_b_emb = self.dropout(self.AtomEmbedding(x_b))
            enc_states_bpe = self.encoder(x_b_emb, x_mask=x_b_mask)
            enc_states = enc_states_bpe
            #
            outputs = torch.tanh(torch.mean(enc_states, dim=1))

        preds  = self.output_layer(outputs)
        # 
        #
        labels = torch.LongTensor([self.train_labels[idx] \
                 for idx in graph_index]).to(self.device)

        self.optimizer.zero_grad()
        # 
        loss = self.label_criterion(preds, labels)
        #
        loss.backward()
        self.optimizer.step()

        return loss

    # 
    def test(self, graph_index):

        enc_states = None
        outputs    = None

        if self.sequence and not self.use_fusion:
            samples_seq = self.test_data['sequence'][graph_index].to(self.device)
            x_s_emb = self.AtomEmbedding(samples_seq)
            x_s_emb = x_s_emb[None, :, :] 
            enc_states_seq = self.encoder(x_s_emb)
            enc_states = enc_states_seq
            outputs = torch.tanh(torch.mean(enc_states, dim=1))
        elif self.bpe and not self.use_fusion:
            samples_bpe = self.test_data['bpe'][graph_index].to(self.device)
            x_b_emb = self.AtomEmbedding(samples_bpe)
            x_b_emb = x_b_emb[None, :, :] 
            enc_states_bpe = self.encoder(x_b_emb)
            enc_states = enc_states_bpe
            outputs = torch.tanh(torch.mean(enc_states, dim=1))
            
        pred = self.output_layer(outputs)

        return pred