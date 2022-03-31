import torch
import torch.nn as nn
import torch.nn.functional as F


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
