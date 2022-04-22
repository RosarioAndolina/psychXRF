from os.path import join, exists, basename, dirname
from numpy import array, empty, hstack, random, tile, log, quantile, median, arange, ones
from numpy.random import rand
from XRDXRFutils.data import SyntheticDataXRF
from itertools import combinations
import h5py
from .dataset import CustomDataset
import torch

class DataProcessing:
    def __init__(self, poisson = False):
        self.filepath = None
        self.N = 10
        self.data = None
        self.poisson = poisson
        self.shape = None

    def load_h5_data(self, filepath):
        self.filepath = filepath
        synt = SyntheticDataXRF().load_h5(self.filepath)
        synt.data = synt.data.reshape(synt.data.shape[1], synt.data.shape[2])
        synt.labels = synt.labels.reshape(synt.labels.shape[1], synt.labels.shape[2])
        self.shape = synt.data.shape
        self.data = synt
        return self

    @staticmethod
    def get_peaks_area(spectra, condition):
        areas = spectra[:, condition].sum(axis = 1)
        return areas
    
    def get_inputs_from_ROI(self, ROI = None):
        if not ROI:
            ROI = {"Mn" : [[5.67, 6.1]],
                   "Fe" : [[6.2, 6.6]],
                   "Pb" : [[10.3, 10.8]],
                   "Hg" : [[9.7, 10.24]],
                   "Ca" : [[3.5, 3.9]]}
        if self.poisson:
            self.data.data = array([random.poisson(self.data.data) for _ in range(self.N)]).reshape(self.shape[0] * self.N, self.shape[1])
            self.data.reflayer_thickness = tile(self.data.reflayer_thickness, self.N)
            self.data.sublayer_thickness = tile(self.data.sublayer_thickness, self.N)
            self.data.weight_fractions = tile(self.data.weight_fractions, (self.N, 1))
            self.shape = self.data.data.shape
        
        condition = lambda roi : (self.data.energy > roi[0]) & (self.data.energy < roi[1])
        data_dict = {}
        for symbol, roi in ROI.items():
            data_dict[symbol] = array([self.get_peaks_area(self.data.data, condition(r)) for r in roi]).sum(axis = 0)
            _min = data_dict[symbol][data_dict[symbol] > 0].min()
            data_dict[symbol][data_dict[symbol] == 0.0] = _min * 1.0e-2
        datalen = data_dict['Pb'].shape[0]
        combo = list(combinations(self.data.metadata['reflayer_elements'],2))
        features_len = len(self.data.metadata['reflayer_elements']) + len(combo)
        inputs = empty((datalen, features_len))
        for i, k in enumerate(self.data.metadata['reflayer_elements'],0):
            inputs[:,i] = data_dict[k]/data_dict['Pb']
        for i, c in enumerate(combo, len(self.data.metadata['reflayer_elements'])):
            inputs[:,i] = data_dict[c[0]]/data_dict[c[1]]
        return inputs

    def get_inputs_from_labels(self, ratios = False):
        symbols = [l.split('-')[0] for l in self.data.metadata['labels']]
        _min = self.data.labels[self.data.labels > 0].min()
        self.data.labels[self.data.labels == 0.0] = _min
        if ratios:
            combo = list(combinations(symbols[1:],2))
            features_len = len(symbols) - 1 + len(combo)
            inputs = empty((self.data.labels.shape[0], features_len))
            for i, l in enumerate(self.data.labels[:,1:].T,0):
                inputs[:,i] = l/self.data.labels[:,0]
            for i, c in enumerate(combo, self.data.labels.shape[1] - 1):
                inputs[:, i] = self.data.labels[:, symbols.index(c[0]) - 1] / self.data.labels[:, symbols.index(c[1]) - 1]
            return inputs
        else:
            return self.data.labels

    def get_targets(self):
        # convert to micron
        targets = hstack((self.data.reflayer_thickness.reshape(self.shape[0],1)*1.0e4,
                          self.data.sublayer_thickness.reshape(self.shape[0],1)*1.0e4,
                          self.data.weight_fractions))
        return targets

class DataTransform:
    def __init__(self, inputs, targets = None, transform_file = None):
        self.inputs = inputs
        self.targets = targets
        self.transform_file = transform_file
        self.target_transform = None
        
    def load_transform_info(self, transform_file):
        self.metadata = {}
        with h5py.File(transform_file, 'r') as fin:
            for k, v in fin.attrs.items():
                self.metadata[k] = v
            if self.metadata['norm'] == 'gauss':
                self.Q1 = fin['Q1'][:]
                self.median = fin['median'][:]
                self.Q3 = fin['Q3'][:]
        self.metadata['status'] = 'loaded'
        if self.metadata['norm'] == 'zero_to_one':
            self.Max = self.inputs.max()
            self.Min = self.inputs.min()
        return self
    
    def save_transform_info(self):
        if not self.transform_file:
            raise ValueError('Transform file needed')
        if hasattr(self, 'metadata'):
            with h5py.File(self.transform_file, 'w') as fout:
                for k,v in self.metadata.items():
                    fout.attrs[k] = v
                if self.metadata['norm'] == 'gauss':
                    dataset = fout.create_dataset('Q1', data = self.Q1)
                    dataset = fout.create_dataset('median', data = self.median)
                    dataset = fout.create_dataset('Q3', data = self.Q3)
            return self
        else:
            raise ValueError("Nothing to save")
    
    def input_transform(self, Log = True, norm = 'gauss'):
        if not hasattr(self, 'metadata'):
            self.metadata = {}
            self.metadata['Log'] = Log
            self.metadata['norm'] = norm
            self.metadata['status'] = 'new'
            if Log:
                self.inputs = log(self.inputs)
            self.Q1 = quantile(self.inputs, 0.25, axis = 0)
            self.median = median(self.inputs, axis = 0) 
            self.Q3 = quantile(self.inputs, 0.75, axis = 0)
            self.save_transform_info()
        
        if self.metadata['status'] == 'loaded':
            if self.metadata['Log']:
                self.inputs = log(self.inputs)
        if self.metadata['norm'] == 'gauss':
            self.inputs = (self.inputs - self.median)/(self.Q3 - self.Q1)
        elif self.metadata['norm'] == 'zero_to_one':
            self.Min = self.inputs.min()
            self.Max = self.inputs.max()
            self.inputs = (self.inputs - self.Min)/(self.Max - self.Min)
        return self
    
    def _split_train_test(self):
        idx = arange(0,self.inputs.shape[0])
        random.shuffle(idx)
        self.idx = idx
        self.train_size = int(self.inputs.shape[0]*0.8)
        #self.test_size = self.inputs.shape[0] - self.train_size
        return self
    
    def get_training_set(self):
        if not hasattr(self, 'idx'):
            self._split_train_test()
        train_inputs = self.inputs[self.idx[:self.train_size],:]
        train_inputs = torch.Tensor(train_inputs)
        train_targets = self.targets[self.idx[:self.train_size],:]
        train_targets = torch.Tensor(train_targets)
        return CustomDataset(train_inputs, train_targets, target_transform = self.target_transform)
        
    def get_testing_set(self):
        if not hasattr(self, 'idx'):
            self._split_train_test()
        test_inputs = self.inputs[self.idx[self.train_size:],:]
        test_inputs = torch.Tensor(test_inputs)
        test_targets = self.targets[self.idx[self.train_size:],:]
        test_targets = torch.Tensor(test_targets)
        return CustomDataset(test_inputs, test_targets, target_transform = self.target_transform)

class DataProcessingSF(DataProcessing):
    def __init__(self, sfbounds = [0.1, 5]):
        super(DataProcessingSF, self).__init__()
        self.sfbounds = sfbounds
    
    def load_h5_data(self, filepath):
        self = super(DataProcessingSF,self).load_h5_data(filepath)
        if self.sfbounds == None:
            self.scale_factor = ones((self.shape[0],1))
        else:
            self.scale_factor = self.sfbounds[0] + rand(self.shape[0],1) * (self.sfbounds[1] - self.sfbounds[0])
        return self
    
    def new_scale_factor(self):
        if self.sfbounds == None:
            self.scale_factor = ones((self.shape[0],1))
        else:
            self.scale_factor = self.sfbounds[0] + rand(self.shape[0],1) * (self.sfbounds[1] - self.sfbounds[0])
        return self
    
    def get_inputs_from_labels(self, ratios = False):
        inputs = super(DataProcessingSF, self).get_inputs_from_labels(ratios = ratios)
        return inputs * self.scale_factor
    
    def get_targets(self):
        # convert to micron
        targets = hstack((self.data.reflayer_thickness.reshape(self.shape[0],1)*1.0e4,
                          self.data.sublayer_thickness.reshape(self.shape[0],1)*1.0e4,
                          self.scale_factor,
                          self.data.weight_fractions))
        return targets
