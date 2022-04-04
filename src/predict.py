import argparse
import torch
from .data import DataProcessing, DataTransform
from .model import MPL
import h5py

class psychXRF:
    def __init__(self, model_path, transform_file):
        self.model_path = model_path
        self.h5_inputs_data = None
        self.transform_file = transform_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(self.model_path).to(self.device)
    
    def load_inputs_from_file(self, h5_inputs_data):
        self.h5_inputs_data = h5_inputs_data
        with h5py.File(self.h5_inputs_data, 'r') as fin:
            self.inputs = fin['inputs'][:]
    
    def sample_inputs_from_tiff(tiff_dir, labels):
        pass
    
    def predict(self):
        if hasattr(self, 'inputs'):
            dtrans = DataTransform(self.inputs).load_transform_info(self.transform_file).input_transform()
        else:
            raise ValueError('No inputs yet')
        inputs = torch.Tensor(dtrans.inputs).to(self.device)
        self.predicted = self.model(inputs).cpu().detach().numpy()
        return self.predicted
        
        
