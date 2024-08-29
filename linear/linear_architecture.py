import numpy as np
import torch
from torch import nn


class LinearNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearNN, self).__init__()
        layers_num = 5
        encode_dims = np.linspace(in_features, int(in_features*0.3), layers_num).astype(int)
        decoder_dims = np.linspace(int(in_features*0.3), out_features, layers_num).astype(int)
        seq = []
        for n in range(layers_num-1):
            seq.append(nn.Linear(encode_dims[n], encode_dims[n+1]))
            seq.append(nn.ReLU())
        for n in range(layers_num-1):
            seq.append(nn.Linear(decoder_dims[n], decoder_dims[n+1]))
            seq.append(nn.ReLU())
        self.encode = nn.Sequential(*seq)

    def forward(self, x):
        out = self.encode(x)
        return out

class LinearEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearEncoder, self).__init__()

        self.att1 = nn.MultiheadAttention(in_features, 1)

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(0.0)

        self.linear_net = nn.Sequential(
            nn.Linear(in_features, int(in_features * 0.3)),
            nn.Dropout(0.0),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_features * 0.3), in_features)
        )

        self.linear_net_final = nn.Sequential(
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        attn_output, attn_output_weights = self.att1(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        x = self.linear_net_final(x)
        return x