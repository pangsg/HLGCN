# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
#from senticnet.senticnet import SenticNet
from layers.Highway import Highway
from layers.attention import MatAtt
from layers.attention import Attention


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

class asp_GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(asp_GraphConvolution, self).__init__()
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

class Gate_Module(nn.Module):
    def __init__(self, context_hid_dim, aspect_hid_dim, bias=True):
        # 参数说明：
        # context_hid_dim：输入的上下文隐藏层维度大小
        # aspect_hid_dim：输入的属性词隐藏层维度大小
        # bias：是否需要偏置
        super(Gate_Module, self).__init__()

        self.context_hid_dim = context_hid_dim
        self.aspect_hid_dim = aspect_hid_dim
        self.bias = bias
        self.Wh = nn.Parameter(torch.FloatTensor(self.context_hid_dim, self.context_hid_dim + self.aspect_hid_dim))
        self.Wm = nn.Parameter(torch.FloatTensor(1, self.context_hid_dim))
        if self.bias:
            self.bh = nn.Parameter(torch.FloatTensor(self.context_hid_dim))

        nn.init.uniform_(self.Wh, -0.1, 0.1)
        nn.init.uniform_(self.Wm, -0.1, 0.1)

    def forward(self, context_hid_embed, aspect_hid_embed):
        # 参数说明：
        # context_hid_embed：输入上下文隐含层嵌入，维度为[batch_size, context_len, context_hid_dim]
        # aspect_hid_embed：输入属性词隐含层嵌入，维度为[batch_size, aspect_len, aspect_hid_dim]
        batch_size = context_hid_embed.size(0)
        context_len = context_hid_embed.size(1)
        aspect_len = aspect_hid_embed.size(1)

        # 平均池化后再扩张, 即原文公式va⊙
        va_eN = torch.sum(aspect_hid_embed, dim=1, keepdim=True)
        # print(va_eN.shape)
        va_eN = va_eN.repeat(1, context_len, 1)
        # 沿第三维度进行拼接
        # 原文公式（1）, 对应于原文的维度要转置
        H = torch.cat((context_hid_embed, va_eN), 2).transpose(1, 2)
        # 原文公式（2）
        # 在加上偏置的时候为了方便运算就对torch.matmul(self.Wh, H)结果转置了
        # 加完偏置后再转置回来
        if self.bias:
            M1 = torch.tanh(torch.matmul(self.Wh, H).transpose(1, 2) + self.bh).transpose(1, 2)
        else:
            M1 = torch.tanh(torch.matmul(self.Wh, H))
        # 在最后的matmul的时候上一步得到的M1要在第1,2维转置
        # 只有计算完转置了才能满足实际的维度，论文中的H和M都是与实际的维度在1,2维有转置的关系
        M = torch.matmul(self.Wm, M1).transpose(1, 2)
        # 原文公式（3)，对应于每个词语的权重，具体使用视情况，这里就没有与context_hid_embed点乘
        G = torch.sigmoid(M)
        return G

class Attention1(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention1, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class new_gcn(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(new_gcn, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)

        self.asp_gc11 = asp_GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.asp_gc12 = asp_GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(6*opt.hidden_dim, opt.polarities_dim)
        self.highway = Highway(2*opt.hidden_dim, 2*opt.hidden_dim, 'sigmoid')
        self.highway_sigmoid = Highway(2*opt.hidden_dim, 2*opt.hidden_dim, 'sigmoid')
        self.highway1 = Highway(2 * opt.hidden_dim, 2 * opt.hidden_dim, 'sigmoid')
        self.highway1_sigmoid = Highway(2 * opt.hidden_dim, 2 * opt.hidden_dim, 'sigmoid')
        self.highway2 = Highway(2 * opt.hidden_dim, 2 * opt.hidden_dim, 'sigmoid')
        self.highway2_sigmoid = Highway(2 * opt.hidden_dim, 2 * opt.hidden_dim, 'sigmoid')
        self.highway3 = Highway(2 * opt.hidden_dim, 2 * opt.hidden_dim, 'sigmoid')
        self.highway3_sigmoid = Highway(2 * opt.hidden_dim, 2 * opt.hidden_dim, 'sigmoid')
        self.text_embed_dropout = nn.Dropout(opt.dropout)
        self.adptive_att = Attention1(6 * opt.hidden_dim, 2*opt.hidden_dim)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
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

    def unmask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(1)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, context_indices, aspect_indices, left_indices, adj, asp_adj, asp_adj2, asp_adj3= inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        #context_len = torch.sum(context_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text,text_len)
        #text_out = self.position_weight(text_out, aspect_double_idx, text_len, aspect_len)
        text = F.relu(self.gc1(text_out,adj))#1
        # highway_in = torch.cat((text, text_out),dim=1)#[batch, 2seq, 2hidden]
        # highway_out = self.highway(highway_in)#[batch,2seq,2hidden]
        # text= F.relu(self.gc2(highway_out,adj))#2
        text = F.relu(self.gc2(text, adj))  # 2
       #  highway_in = torch.cat((text, text_out),dim=1)#[batch, 2seq, 2hidden]
       #  highway_out = self.highway(highway_in)#[batch,2seq,2hidden]
       #  text = F.relu(self.gc3(highway_out,adj))#[batch,2seq,2hidden]#3
        text = F.relu(self.gc3(text, adj))  # [batch,2seq,2hidden]#3

        #1-hop gcn
        asp = self.asp_gc11(text_out, asp_adj)
        # highway_in = torch.cat((asp,text_out),dim=1)
        # highway_out = self.highway1_sigmoid(highway_in)
        # asp = self.asp_gc12(highway_out,asp_adj)#[bach,len,2*hidden]
        asp = self.asp_gc12(asp, asp_adj)
        asp = self.position_weight(asp, aspect_double_idx, text_len, aspect_len)
        #2-hop gcn
        asp2 = self.asp_gc11(text_out,asp_adj2)
        # highway_in = torch.cat((asp2, text_out), dim=1)
        # highway_out = self.highway1_sigmoid(highway_in)
        # asp2 = self.asp_gc12(highway_out, asp_adj2)  # [bach,len,2*hidden]
        asp2 = self.asp_gc12(asp2, asp_adj2)
        asp2 = self.position_weight(asp2, aspect_double_idx, text_len, aspect_len)
        #3-hop gcn
        asp3 = self.asp_gc11(text_out,asp_adj3)
        # highway_in = torch.cat((asp3, text_out), dim=1)
        # highway_out = self.highway1_sigmoid(highway_in)
        # asp3 = self.asp_gc12(highway_out, asp_adj3)  # [bach,len,2*hidden]
        asp3 = self.asp_gc12(asp3, asp_adj3)
        asp3 = self.position_weight(asp3, aspect_double_idx, text_len, aspect_len)
        #adaptive  layer

        asp = torch.cat((asp,asp2),dim=-1)
        asp = torch.cat((asp,asp3),dim=-1)#[batch,len,3*hidden]
        seq_len = asp.shape[1]

        asp_final, att = self.adptive_att(asp)#[batch,3*hidden]
        asp_final = asp_final.unsqueeze(dim=1)
        asp_final = asp_final.repeat(1,seq_len,1)




        aspect = self.mask(asp_final, aspect_double_idx)#[batch,len,hidden]
        context = self.unmask(text,aspect_double_idx)
        text_len = torch.tensor(text_len, dtype=torch.float).to(self.opt.device)
        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        a_mean = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        c_mean = torch.div(torch.sum(context, dim=1), text_len.view(text_len.size(0), 1))
        s_mean = torch.div(torch.sum(text_out, dim=1), text_len.view(text_len.size(0), 1))
        # resnet
        c_mean = c_mean.repeat(1, 3)
        s_mean = s_mean.repeat(1, 3)
        final_rep  = s_mean * (1 - torch.sigmoid(a_mean)) + c_mean * torch.sigmoid(a_mean)
        #final_rep = a_mean
        #print(final_rep.shape)
        #final_rep = torch.cat((c_mean,a_mean),dim=-1)
        output = self.fc(final_rep)
        return output