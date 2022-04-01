#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import MPL
from .data import DataProcessing, DataTransform
from .metrics import R2Score, AR2Score, RMSELoss
from os import getenv


# Training settings
parser = argparse.ArgumentParser('psychxrf')
parser.add_argument('--h5datafile', type = str, required = True, help = 'HDF5 data file')
parser.add_argument('--batch-size', type = int, default = 64, help = 'training batch size [64]')
parser.add_argument('--test-batch-size', type = int, default = 32, help = 'testing batch size [32]')
parser.add_argument('--num-epoch', type = int, default = 800, help = 'number of epoch to train for [800]')
parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate [0.01]')
parser.add_argument('--hidden-sizes', nargs = '+', default = [128], help = 'sequence of hidden layer sizes')
parser.add_argument('--optimizer', type = str, default = 'sgd', help = 'Optimizer. One of "sgd" "adam" [sgd]')
parser.add_argument('--trans-file', type = str, default = f'/home/{getenv("USER")}/test_transform_file.h5', help = 'HDF5 file were inputs & targets transformation parameters are stored')

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("###### LOADING DATASETS ######")
dproc = DataProcessing().load_h5_data(opt.h5datafile)
dtrans = DataTransform(dproc.get_inputs_from_labels(), dproc.get_targets(), transform_file = opt.trans_file)
dtrans.input_transform()
train_set = dtrans.get_training_set()
test_set = dtrans.get_testing_set()
training_dataloader = DataLoader(train_set, opt.batch_size, pin_memory = True, shuffle = True)
testing_dataloader = DataLoader(test_set, opt.test_batch_size, pin_memory = True, shuffle = True)

print("###### Building Model ######")
model = MPL(in_size = dtrans.inputs.shape[1], out_size = dtrans.targets.shape[1], hidden_sizes = opt.hidden_sizes).to(device)
criterion = RMSELoss() #nn.MSELoss()
r2score = R2Score()

if opt.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr = opt.lr)
elif opt.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
else:
    raise ValueError('Unused optimizer')
print(model)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_dataloader,1):
        input , target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #print(f"Epoch [{epoch} ({iteration}/{len(training_dataloader)})]: Loss: {loss.item():.4f}")
    if (epoch % 50 == 0):
        print(f"##### Epoch {epoch} Completed: avg. Loss: {(epoch_loss/len(training_dataloader)):.4f}")
    return epoch_loss/len(training_dataloader)

def test(epoch):
    mean_loss = 0
    mean_metric = 0
    with torch.no_grad():
        for batch in testing_dataloader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            loss = criterion(prediction, target)
            metric = r2score(prediction, target)
            mean_loss += loss.item()
            mean_metric += metric.item()
    mean_loss = mean_loss/len(testing_dataloader)
    mean_metric = mean_metric/len(testing_dataloader)
    #adjusted r2score
    ar2score = AR2Score(input.shape[0], input.shape[1], mean_metric)
    if (epoch % 50 == 0):
        print(f"##### Test Loss: {mean_loss:.4f}")
        print(f"##### Test R2: {mean_metric:.4f}")
        print(f"##### Test Adjusted R2: {ar2score:.4f}")
    return mean_loss
    
            

def checkpoint():
    pass

def main():
    train_loss = []
    test_loss = []
    for epoch in range(1, opt.num_epoch + 1):
        train_loss.append(train(epoch))
        test_loss.append(test(epoch))
        # to do: train and test loss plot
    print("##### DONE #####")

if __name__ == '__main__':
    main()

