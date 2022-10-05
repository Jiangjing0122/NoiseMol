# 

import math
import torch
import torch.nn as nn
import pdb


class GRU(nn.Module):
    """
    Args:
    """

    def __init__(self, d_model=64, \
                       d_inputs=64,
                       d_outputs=64,
                       num_layers=1,
                       bi_gru=True):
        super(GRU, self).__init__()
        self.d_model = d_model
        self.d_inputs = d_inputs
        self.d_outputs = d_outputs
        # 
        self.Wi = nn.Linear(self.d_inputs, self.d_model)
        self.Wo = nn.Linear(self.d_model, self.d_outputs)
        # 
        self.Wu = nn.Linear(self.d_model, self.d_model)
        self.Wr = nn.Linear(self.d_model, self.d_model)
        self.Wx = nn.Linear(self.d_model, self.d_model)
        self.Ux = nn.Linear(self.d_model, self.d_model)
        self.Rx = nn.Linear(self.d_model, self.d_model)
        # 
        self.gate_r = nn.Sigmoid()
        self.gate_u = nn.Sigmoid()
        self.tanh   = nn.Tanh()
        # 
        self.num_layers = num_layers
        self.bi_gru = bi_gru
        
    def forward(self, inputs, mask=None):
        """ See :obj:`EncoderBase.forward()`"""
        # inputs.shape = [batch_size, sentence_length, dim_model]
        #pdb.set_trace()
        assert inputs.dim() is 3 
        
        inputs = self.Wi(inputs)
        # 
        def _gru_encoder(inputs, mask=None):
            batch, length, _ = inputs.size()
            # 
            x_x = self.Wx(inputs)
            x_r = self.Ux(inputs)
            x_u = self.Rx(inputs)
            # 
            h_  = torch.zeros(batch, self.d_model).to('cuda')
            # 
            buf = []
            # 
            for idx in range(length):
                # reset and update gates
                r = self.gate_r(self.Wr(h_) + x_r[:,idx])
                u = self.gate_u(self.Wu(h_) + x_u[:,idx])
                # 
                cand_h = self.tanh(r * h_ + x_x[:,idx])
                # 
                h = u * h_ + (1. - u) * cand_h
                if mask is not None:
                    #
                    h = mask[:,idx][:,None] * h + \
                        (1. - mask[:,idx])[:, None] * h_
                #
                buf.append(h)
                # 
                h_ = h
            # outputs.shape=[batch_size, sentence_length, dim_model]
            outputs = torch.stack(buf, dim=1)
            return outputs
        #
        outputs = _gru_encoder(inputs, mask)
        if self.bi_gru:
            #
            inputs_ = torch.flip(inputs, dims=[1])
            if mask is not None:
                mask = torch.flip(mask, dims=[1])
            out_b = _gru_encoder(inputs_, mask)
            # 
            out_b = torch.flip(out_b, dims=[1])
            #
            outputs = outputs + out_b

        return self.Wo(outputs)
