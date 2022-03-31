import torch.utilis.data as data

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
