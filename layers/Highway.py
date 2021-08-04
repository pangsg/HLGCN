import torch.nn as nn
import torch.nn.functional as F
import torch


class Highway(nn.Module):
    """
    Input shape=(batch_size,dim,dim)
    Output shape=(batch_size,dim,dim)
    """

    def __init__(self, input_size, out_size, function, layer_num=1,):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.function = function
        # self.batch_size = batch_size
        # self.seq_len = seq_len
        self.linear = nn.ModuleList([nn.Linear(input_size, out_size )
                                     for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(input_size, out_size )
                                   for _ in range(self.layer_num)])
       # self.weight = nn.Parameter(torch.FloatTensor(batch_size,2*seq_len, seq_len))

    def forward(self, x):
        for i in range(self.layer_num):
            if self.function == 'relu':
                gate = torch.rrelu(self.gate[i](x))
            else:
                gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.rrelu(self.linear[i](x))
            #weight = nn.Parameter(torch.FloatTensor(x.shape[0], 2*x.shape[1], x.shape[1]))
            x = gate * nonlinear + (1 - gate) * x


        return x
