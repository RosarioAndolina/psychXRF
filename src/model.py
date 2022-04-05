import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import CFReLU


class MPL(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes):
        super().__init__()
        self.out_size = out_size
        self.net_sizes = [in_size] + hidden_sizes
        self.net_struct = []
        for i in range(len(self.net_sizes)-1):
            self.net_struct.append(nn.Linear(self.net_sizes[i], self.net_sizes[i+1]))
            self.net_struct.append(nn.Tanh())
        self.net_struct.append(nn.Linear(self.net_sizes[-1], self.out_size))
        self.network = nn.Sequential(*self.net_struct)
    
    def forward(self, x):
        x = self.network(x)
        x = F.relu(x)
        return x

class MPLSmax(MPL):
    """
    softmax for weight fractions - CFReLU for thicknes
    
    CFReLU: custom FReLU, see activation.py
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MPLSmax, self).__init__(in_size, out_size, hidden_sizes)
        self.split_point = 2
        self.cfrelu = CFReLU(out_size, b = 0.01e-4)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.network(x)
        activated = (
            self.cfrelu(x[:, 0:self.split_point]),
            self.softmax(x[:, self.split_point:])
        )
        return torch.cat(activated, dim = 1)
