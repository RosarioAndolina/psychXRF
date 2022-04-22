import torch.utils.data as data
from numpy.random import rand

class CustomDataset(data.Dataset):
    def __init__(self, inputs, targets, input_transform = None, target_transform = None):
        super(CustomDataset, self).__init__()
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.inputs = inputs
        self.targets = targets
        if self.input_transform:
            self.inputs = self.input_transform(self.inputs)
        if self.target_transform:
            self.targets = self.target_transform(self.targets)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x, y
    
    def __len__(self):
        return len(self.inputs)

class DatasetSF(CustomDataset):
    def __init__(self, inputs, targets, input_transform = None, target_transform = None, sfbounds = None):
        super(DatasetSF, self).__init__(inputs, targets, input_transform, target_transform)
        self.sfbounds = sfbounds
    
    def __getitem__(self, idx):
        scale_factor = self.sfbounds[0] + rand() * (self.sfbounds[1] - self.sfbounds[0])
        x = self.inputs[idx].copy() 
        y = self.targets[idx].copy()
        x = x * scale_factor
        y[2] =  y[2] * scale_factor
        
        return x, y
