import torch
import torch.nn as nn

#Root Means Squared Log Error
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

#Root Means Squared Error
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))

#Determination Coefficient
class R2Score(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, actual):
        rss = ((actual - pred)**2).sum()
        ym = actual.mean()
        tss = ((actual - ym)**2).sum()
        return 1 - rss/tss

#Adjusted Determination Coefficient
def AR2Score(n, k, r2):
    return 1 - (((1- r2)*(n-1))/(n - k -1))
