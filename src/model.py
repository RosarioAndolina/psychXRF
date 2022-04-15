import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import CFReLU, FReLU, SumNorm, BReLU


class MPL(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes, activation = 'tanh', wfnorm = None, brelu_par = 1.0e-8):
        super().__init__()
        self.out_size = out_size
        self.net_sizes = [in_size] + hidden_sizes
        self.activation = activation
        self.wfnorm = wfnorm
        self.brelu_par = brelu_par
        self.net_struct = []
        for i in range(len(self.net_sizes)-1):
            self.net_struct.append(nn.Linear(self.net_sizes[i], self.net_sizes[i+1]))
            if self.activation == 'tanh':
                self.net_struct.append(nn.Tanh())
            elif self.activation == 'relu':
                self.net_struct.append(nn.ReLU())
            elif self.activation == 'frelu':
                self.net_struct.append(FReLU(self.net_sizes[i+1], b = 0.0))
            elif self.activation == 'brelu':
                self.net_struct.append(BReLU(self.net_sizes[i+1], b = self.brelu_par))
        self.net_struct.append(nn.Linear(self.net_sizes[-1], self.out_size))
        if self.wfnorm == 'softmax':
            self.net_struct.append(nn.Linear(self.out_size, self.out_size))
        self.network = nn.Sequential(*self.net_struct)
    
    def forward(self, x):
        x = self.network(x)
        x = F.relu(x)
        return x

class MFreluSmax(MPL):
    """
    FReLU activated - softmax for weight fractions - CFReLU for thicknes
    
    CFReLU: custom FReLU, see activation.py
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MFreluSmax, self).__init__(in_size, out_size, hidden_sizes, activation = 'frelu', wfnorm = 'softmax')
        self.split_point = 2
        self.cfrelu = CFReLU(out_size, b = 0.01)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.network(x)
        activated = (
            self.cfrelu(x[:, 0:self.split_point]),
            # F.relu(x[:, 0:self.split_point]),
            self.softmax(x[:, self.split_point:])
        )
        return torch.cat(activated, dim = 1)

class MFreluWfNorm(MPL):
    """
    FRelu activated - weight fractions sum-normalized - CFReLU for thicknes
    
    CFReLU: custom FReLU, see activation.py
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MFreluWfNorm, self).__init__(in_size, out_size, hidden_sizes, activation = 'frelu')
        self.split_point = 2
        #activation for thicknes
        self.cfrelu = CFReLU(self.split_point, b = 0.01)
        #activation for weight fractions
        self.sumNorm = SumNorm(self.out_size - self.split_point, dim = 1)
        
    def forward(self, x):
        x = self.network(x)
        x = F.relu(x)
        activated = (
            self.cfrelu(x[:, 0:self.split_point]),
            # F.relu(x[:, 0:self.split_point]),
            self.sumNorm(x[:, self.split_point:])
        )
        return torch.cat(activated, dim = 1)

class MCreluWfNorm(MPL):
    """
    BRelu activated - weight fractions sum-normalized - BReLU for thicknes
    
    BReLU: Biased ReLU, see activation.py
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MCreluWfNorm, self).__init__(in_size, out_size, hidden_sizes, activation = 'brelu')
        self.split_point = 2
        self.brelu = BReLU(self.out_size, b = 0.001)
        #activation for thicknes
        self.I = nn.Identity()
        #activation for weight fractions
        self.sumNorm = SumNorm(self.out_size - self.split_point, dim = 1)
        
    def forward(self, x):
        x = self.network(x)
        x =  self.brelu(x)
        activated = (
            self.I(x[:, 0:self.split_point]),
            # F.relu(x[:, 0:self.split_point]),
            self.sumNorm(x[:, self.split_point:])
        )
        return torch.cat(activated, dim = 1)

class MSplitOut01(MPL):
    """
    multiple outputs - two for thickness one for weight fractions
    
    BReLU activated - weight fraction sum-normalized - BReLU for thicknes
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MSplitOut01, self).__init__(in_size, out_size, hidden_sizes, activation = 'brelu')
        self.split_point = 2
        self.brelu = BReLU(self.out_size, b = 0.001)
        #activation for thickness
        self.I = nn.Identity()
        #activation for weight fractions
        self.sumNorm = SumNorm(self.out_size - self.split_point, dim = 1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.brelu(x)
        activated = (
            self.I(x[:, 0]),
            self.I(x[:, 1]),
            self.sumNorm(x[:, self.split_point:]))
        return activated

class MSplitOut02(MPL):
    """
    multiple outputs - two for thickness one for weight fractions
    
    FReLU activated - weight fraction sum-normalized - BReLU for thicknes
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MSplitOut02, self).__init__(in_size, out_size, hidden_sizes, activation = 'frelu')
        self.split_point = 2
        #activation for thickness
        self.brelu = BReLU(self.out_size, b = 1.0e-3)
        #activation for weight fractions
        self.sumNorm = SumNorm(self.out_size - self.split_point, dim = 1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.brelu(x)
        activated = (
            x[:, 0],
            x[:, 1],
            self.sumNorm(x[:, self.split_point:]))
        return activated

class MSplitOut03(MPL):
    """
    multiple outputs - two for thickness one for weight fractions
    
    FReLU activated - weight fraction sum-normalized - CFReLU for thicknes
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MSplitOut03, self).__init__(in_size, out_size, hidden_sizes, activation = 'frelu')
        self.split_point = 2
        #activation for thickness
        self.cfrelu = CFReLU(self.out_size, b = 1.0e-3)
        #activation for weight fractions
        self.sumNorm = SumNorm(self.out_size - self.split_point, dim = 1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.cfrelu(x)
        activated = (
            x[:, 0],
            x[:, 1],
            self.sumNorm(x[:, self.split_point:]))
        return activated

class MSplitOut04(MPL):
    """
    multiple outputs - two for thickness one for weight fractions
    
    Tanh activated - weight fraction sum-normalized - CFReLU for thicknes
    """
    def __init__(self, in_size, out_size, hidden_sizes):
        super(MSplitOut04, self).__init__(in_size, out_size, hidden_sizes, activation = 'tanh')
        self.split_point = 2
        #activation for thickness
        self.cfrelu = CFReLU(self.out_size, b = 1.0e-3)
        #activation for weight fractions
        self.sumNorm = SumNorm(self.out_size - self.split_point, dim = 1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.cfrelu(x)
        activated = (
            x[:, 0],
            x[:, 1],
            self.sumNorm(x[:, self.split_point:]))
        return activated
