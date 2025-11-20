import torch
import numpy as np

class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.L1 = torch.nn.Linear(2,50)
        self.L2 = torch.nn.Linear(50,50)
        self.L3 = torch.nn.Linear(50,50)
        self.L4 = torch.nn.Linear(50,50)
        self.L5 = torch.nn.Linear(50,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y], axis=1)
        x1 = torch.tanh(self.L1(inputs))
        x2 = torch.tanh(self.L2(x1))
        x3 = torch.tanh(self.L3(x2))
        x4 = torch.tanh(self.L4(x3))
        x5 = self.L5(x4)

        return x5



def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
