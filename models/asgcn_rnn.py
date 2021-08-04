# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
#from senticnet.senticnet import SenticNet
from layers.Highway import Highway


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)#[batch,seq,2*hidden] * [2hidden,2hidden]
        #print(hidden.shape)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        if text.shape[1] == adj.shape[1]:
        #print(adj.shape)
            output = torch.matmul(adj, hidden) / denom
            if self.bias is not None:
                return output + self.bias
            else:
                return output
        else:
            adj = adj.repeat(1, 1, 2)
            output = torch.matmul(adj, hidden)/ denom
            if self.bias is not None:
                return output + self.bias
            else:
                return output



class ASGCN_lstm(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_lstm, self).__init__()
        #self.weight_h = nn.Parameter(torch.FloatTensor(batch_size, size_1, size_2))
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.graph_lstm1 = DynamicLSTM(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.graph_lstm2 = DynamicLSTM(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.graph_lstm3 = DynamicLSTM(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc4 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.highway = Highway(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.weight_h = nn.Parameter(torch.FloatTensor(opt.batch_size,2*opt.text_len,opt.text_len))
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        #print(aspect_double_idx)
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    # def highway_output(self, x, text_out):
    #     batch_size = x.shape[0]
    #     seq_len = x.shape[1]
    #     weight_h = nn.Parameter(torch.FloatTensor(batch_size, 2*seq_len, seq_len))
    #     weight_h = weight_h.to(self.opt.device)
    #     highway_in = torch.cat((x, text_out), dim=1)
    #     highway_out = self.highway(highway_in).transpose(1,2)
    #     highway_out = torch.matmul(highway_out, weight_h).transpose(1,2)
    #     return highway_out



    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        #weight_h = nn.Parameter(torch.FloatTensor(x.shape[0]))
        text_len = torch.sum(text_indices != 0, dim=-1)
        #text_len = text_len.cpu().numpy()
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        #print(text.shape)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        lstm_out1, (_, _ ) = self.graph_lstm1(x, text_len)
        x = F.relu(self.gc2(self.position_weight(lstm_out1, aspect_double_idx, text_len, aspect_len), adj))
        lstm_out2, (_, _) = self.graph_lstm2(x, text_len)
        x = F.relu(self.gc3(self.position_weight(lstm_out2, aspect_double_idx, text_len, aspect_len), adj))
        lstm_out3, (_, _) = self.graph_lstm3(x, text_len)
        x = F.relu(self.gc4(self.position_weight(lstm_out3, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output