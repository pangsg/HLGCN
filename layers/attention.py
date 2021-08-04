import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class MatAtt(nn.Module):

    def __init__(self, input_dim, output_dim, opt, batch_size=32):
        super(MatAtt, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.opt = opt
        #self.batch_size = batch_size
        #self.opt =opt
        self.W_ct = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.W_tc = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.w_c1 = nn.Linear(input_dim,output_dim, bias=True)
        self.w_c2 = nn.Linear(input_dim, 1, bias=True)
        self.w_a1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w_a2 = nn.Linear(input_dim, 1, bias=True)
        self.w_final = nn.Linear(2*input_dim, 2*output_dim, bias=True)
        self.w_a = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        self.w_c = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        self.W = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        #self.b_c = nn.Parameter(torch.FloatTensor(batch_size,))


        nn.init.uniform_(self.W_ct, -0.1, 0.1)
        nn.init.uniform_(self.W_tc, -0.1, 0.1)
        nn.init.uniform_(self.w_a, -0.1, 0.1)
        nn.init.uniform_(self.w_c, -0.1, 0.1)
        nn.init.uniform_(self.W, -0.1,0.1)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def cdp(self,aspect, context):
        temp = torch.relu(torch.matmul(aspect, self.W))
        aspect = temp * context
        return aspect

    def forward(self, aspect, context):
         #batch_size = text.shape[0]
         batch_size = aspect.shape[0]
         context_len = context.shape[1]
         aspect_len = aspect.shape[1]
         #for i in range(batch_size):
         H_ct = torch.matmul(context, self.W_ct)# [batch_size, aspect_leq, 2*hidden] * [2*hidden, 2*hidden] = [batch_size,context_leq,2*hidden]
         H_ct = torch.matmul(H_ct,aspect.transpose(1,2))#[batch,context,aspect]
         H_tc = torch.matmul(aspect, self.W_tc)
         #属性词矩阵，上下文矩阵
         h_context = torch.matmul(H_ct, aspect) #[batch,context,aspect] * [batch,aspect,2*hidden] = [ba
         H_tc = torch.matmul(H_tc,context.transpose(1,2))#[batch, aspect, context]tch,context_len,2*hidden]
         h_aspect = torch.matmul(H_tc, context)#[batch, aspect_len, 2*hidden]
         #获得上下文和属性词向量 self-attention
         #score_c = torch.matmul(h_context,self.W_1c)#[batch,context,hidden]*[hidden,hidden]->[batch,context,hidden]
         #两层的mlp
         score_c_matrix = torch.relu(self.w_c1(h_context))#[batch,context,hidden]
         score_c_matrix = torch.relu(self.w_c2(score_c_matrix))#[batch,context,1]
         score_a_matrix = torch.relu(self.w_a1(h_aspect))  # [batch,aspect,hidden]
         score_a_matrix = torch.relu(self.w_a2(score_a_matrix))  # [batch,aspect,1]
         #softmax
         score_c = F.softmax(score_c_matrix,dim=1)#[batch,soft,1]
         score_a = F.softmax(score_a_matrix, dim=1)
         context_vec = torch.sum(h_context * score_c, dim=1,keepdim=True)#[batch,1,hidden]
         aspect_vec = torch.sum(h_aspect * score_a, dim=1,keepdim=True)#[batch,1,hidden]
         #print(context_vec.shape)
         # 开始交互注意力
         #context
         #b_c = torch.tensor(torch.zeros([batch_size,context_len,1])).to(self.opt.device)
         b_c = nn.Parameter(torch.FloatTensor(batch_size, context_len)).to(self.opt.device)
         b_c = nn.init.uniform_(b_c,-0.0,0.0)
         #b_c = nn.Parameter(torch.FloatTensor(batch_size,context_len)).to(self.opt.device)
         b_a = nn.Parameter(torch.FloatTensor(batch_size, aspect_len)).to(self.opt.device)
         b_a = nn.init.uniform_(b_a, -0.0, 0.0)
         #b_a = nn.Parameter(torch.FloatTensor(batch_size, aspect_len)).to(self.opt.device)
         context_att = torch.matmul(h_context, self.w_c)#batch len hidden [hidden,hidden]->[batch len hidden]
         context_att = torch.relu((torch.bmm(context_att,aspect_vec.transpose(1,2))).squeeze()+b_c)#[batch,len,hidden]*[batch hidden 1]->[batch,len,1]
         context_att = context_att.unsqueeze(dim=-1)#batch,len,1
         context_score = F.softmax(context_att,dim=1)#[batch,lensoft,1]
         context_rep = torch.sum(context_score * h_context,dim=1)#[batch,lensoft,batch] [batch,len,hidden]
         #aspect
         aspect_att = torch.matmul(h_aspect,self.w_a)
         aspect_att = torch.relu((torch.bmm(aspect_att, context_vec.transpose(1, 2))).squeeze()+b_a)
         aspect_att = aspect_att.unsqueeze(dim=-1)
         aspect_score = F.softmax(aspect_att,dim=1)
         aspect_rep = torch.sum(aspect_score * h_aspect, dim=1)
         #拼接
         #aspect_rep = self.cdp(aspect_rep, context_rep)
         final_rep = torch.cat((aspect_rep,context_rep),dim=1)
         final_rep = torch.relu(self.w_final(final_rep))

         return final_rep


    #def forward(self, aspect, context):
class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0.1):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)#(-1,hidden,len)
            score = torch.bmm(qx, kt)#(-1,len,hidden)*(-1,hidden,len)->(-1,q_len,k_len)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)#(-1,q_len,k_len)*(-1,k_len,hidden)->(-1,len,hidden)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class AM_Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(AM_Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        #print('w shape is ',w.shape)
        beta = torch.softmax(w, dim=1)
        #print('beta shape is',beta.shape)
        return (beta * z).sum(1), beta