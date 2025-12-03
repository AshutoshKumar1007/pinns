import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.Tanh()  # Tanh is smooth, essential for 4th order derivatives

    def forward(self, x):
        res = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + res)


class DRM_Model(nn.Module):
    def __init__(self):
        super(DRM_Model, self).__init__()
        # Increased width to 50 for better capacity on GPU
        self.input_layer = nn.Linear(2, 50)

        # 3 Residual Blocks
        self.resblocks = nn.Sequential(
            ResBlock(50),
            ResBlock(50),
            ResBlock(50)
        )

        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = self.resblocks(out)
        out = self.output_layer(out)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)