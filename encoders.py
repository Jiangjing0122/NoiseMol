import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
torch.backends.cudnn.enabled=False
import math
import pdb

from transformer import TransformerEncoder
from gru_layer import GRU

# 
def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight,gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Classifier(nn.Module):

    def __init__(self, latent_size, device):
        super(Classifier,self).__init__()

        self.latent_size = latent_size

        self.classifier = nn.Sequential(nn.Linear(self.latent_size, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, 2),
                                        )

        self.apply(weights_init)
        self.to(device)

    def forward(self, x):

        out = self.classifier(x)
        return out


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_size, latent_size, device,
                        dropout=0.1):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout = nn.Dropout(dropout)
        '''
        # 
        self.gru_ = GRU(d_inputs=self.input_dim, \
                        d_outputs=self.input_dim,
                        bi_gru=True)
        self.gru = nn.GRU(self.input_dim, int(self.hidden_size/2), \
                              batch_first=True, bidirectional=True,
                              num_layers=1)        
        '''
        # transformer, gru, gru_trans, trans_gru
        #
        self.encoder_type = 'trans'

        if self.encoder_type == 'gru':
            #
            self.gru = nn.GRU(self.input_dim, int(self.hidden_size/2), \
                                  batch_first=True, bidirectional=True,
                                  num_layers=2)
        elif self.encoder_type == 'gru_':
            #
            self.gru_ = GRU(d_inputs=self.input_dim, \
                            d_outputs=self.input_dim,
                            bi_gru=True)
        elif self.encoder_type == 'trans':

            self.transformer = TransformerEncoder(num_layers=4,
                                                  d_model=input_dim,
                                                  dropout=dropout,
                                                  layer_type='self')
        else:
            self.gru = nn.GRU(self.input_dim, int(self.hidden_size/2), \
                             batch_first=True, bidirectional=True, num_layers=1)
            self.transformer = TransformerEncoder(num_layers=6,
                                                  d_model=input_dim,
                                                  dropout=dropout,
                                                  layer_type='self')

        self.apply(weights_init)

        self.to(device)

    def forward(self, x, x_mask=None):
        
        # x.shape = [batch_size, seq_length]
        # x_mask.shape = [batch_size, seq_length]

        if self.encoder_type == 'trans':
            # Transformer Encoder
            x_t = x.transpose(1,0)
            outputs_ = self.transformer(x_t, mask=x_mask)
            outputs_ = outputs_.transpose(1,0)
            if x_mask is not None:
                outputs_ = outputs_*x_mask[:,:,None]
        
        elif self.encoder_type == 'gru':
            # GRU Encoder
            outputs_, _ = self.gru(x)
            outputs_ = self.dropout(outputs_)
            #if x_mask is not None:
            #    outputs_ = outputs_*x_mask[:,:,None]

        encoder_outputs = outputs_

        return encoder_outputs
