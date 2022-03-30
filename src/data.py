from os.path import join, exists, basename, dirname
from numpy import array, empty, hstack, random, tile
from XRDXRFutils.data import SyntheticDataXRF
from itertools import combinations

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
            self.data.reflayer_thicknes = tile(self.data.reflayer_thicknes, self.N)
            self.data.sublayer_thicknes = tile(self.data.sublayer_thicknes, self.N)
            self.data.weight_fractions = tile(self.data.weight_fractions, (self.N, 1))
            self.shape = self.data.data.shape
        
        condition = lambda roi : (self.data.energy > roi[0]) & (self.data.energy < roi[1])
        data_dict = {}
        for symbol, roi in ROI.items():
            data_dict[symbol] = array([self.get_peaks_area(self.data.data, condition(r)) for r in roi]).sum(axis = 0)
            _min = data_dict[symbol][data_dict[symbol] > 0].min()
            data_dict[symbol][data_dict[symbol] == 0] = _min * 1.0e-2
        datalen = data_dict['Pb'].shape[0]
        combo = list(combinations(self.data.metadata['reflayer_elements'],2))
        features_len = len(self.data.metadata['reflayer_elements']) + len(combo)
        inputs = empty((datalen, features_len))
        for i, k in enumerate(self.data.metadata['reflayer_elements'],0):
            inputs[:,i] = data_dict[k]/data_dict['Pb']
        for i, c in enumerate(combo, len(self.data.metadata['reflayer_elements'])):
            inputs[:,i] = data_dict[c[0]]/data_dict[c[1]]
        return inputs

    def get_inputs_from_labels(self):
        symbols = [l.split('-')[0] for l in self.data.metadata['labels']]
        _min = self.data.labels[self.data.labels > 0].min()
        self.data.labels[self.data.labels == 0.0] = _min * 1.0e-2
        combo = list(combinations(symbols[1:],2))
        features_len = len(symbols) - 1 + len(combo)
        inputs = empty((self.data.labels.shape[0], features_len))
        for i, l in enumerate(self.data.labels[:,1:].T,0):
            inputs[:,i] = l/self.data.labels[:,0]
        for i, c in enumerate(combo, self.data.labels.shape[1] - 1):
            inputs[:, i] = self.data.labels[:, symbols.index(c[0]) - 1] / self.data.labels[:, symbols.index(c[1]) - 1]
        return inputs

    def get_targets(self):
        # convert to micron
        targets = hstack((self.data.reflayer_thicknes.reshape(self.shape[0],1)*1.0e4,
                          self.data.sublayer_thicknes.reshape(self.shape[0],1)*1.0e4,
                          self.data.weight_fractions))
        return targets

def input_transform(method = 'labels'):
    if method == 'labels':
        pass
        
