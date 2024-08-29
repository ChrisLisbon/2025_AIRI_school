import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttention(nn.Module):
    def __init__(self, channels_in, out_channels):
        # Call the __init__ function of the parent nn.module class
        super(CNNAttention, self).__init__()
        # Define Convolution Layers
        self.conv1 = nn.Conv2d(channels_in, 64, 3, 1, 1, padding_mode='reflect')

        # Define Layer Normalization and Multi-head Attention layers
        self.norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(64, num_heads=1, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))

        # Define additional Convolution Layers
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv4 = nn.ConvTranspose2d(128, 127, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(127)

        self.conv5 = nn.ConvTranspose2d(127, out_channels, 4, 2, 4)


    def use_attention(self, x):
        # Reshape input for multi-head attention
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC

        # Apply Layer Normalization
        x_att = self.norm(x_att)
        # Apply Multi-head Attention
        att_out, att_map = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)

        # Apply self-attention mechanism and add to the input
        x = self.scale * self.use_attention(x)[0] + x

        # Apply batch normalization and ReLU activation
        x = F.relu(x)

        # Additional convolutional layers
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = F.relu(x)
        return x
