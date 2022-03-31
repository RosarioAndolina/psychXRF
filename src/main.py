#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import MPL
from .data import DataProcessing, DataTransform
from os import getenv

def main():
    # Training settings
    parser = argparse.ArgumentParser('psychxrf')
    parser.add_argument('--h5datafile', type = str, required = True, help = 'HDF5 data file')
    parser.add_argument('--batch-size', type = int, default = 64, help = 'training batch size [64]')
    parser.add_argument('--test-batch-size', type = int, default = 32, help = 'testing batch size [32]')
    parser.add_argument('--num-epoch', type = int, default = 800, help = 'number of epoch to train for [800]')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate [0.01]')
    parser.add_argument('--hidden-sizes', nargs = '+', default = [128], help = 'sequence of hidden layer sizes')
    parser.add_argument('--trans-file', type = str, default = f'/home/{getenv("USER")}/test_transform_file.h5', help = 'HDF5 file were inputs & targets transformation parameters are stored')

    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("###### LOADING DATASETS ######")
    dproc = DataProcessing().load_h5_data(opt.h5datafile)
    dtrans = DataTransform(dproc.get_inputs_from_labels(), dproc.get_targets(), transform_file = opt.trans_file)
    dtrans.input_transform()
    train_set = dtrans.get_training_set()
    test_set = dtrans.get_testing_set()
    training_dataloader = DataLoader(train_set, opt.batch_size, shuffle = True)
    testing_dataloader = DataLoader(test_set, opt.test_bacth_size, shuffle = True)
    # to be continued
    


if __name__ == '__main__':
    main()

