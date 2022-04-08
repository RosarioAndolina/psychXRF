#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import psychXRF.model as Model
#from .model import MFreluSmax, MFreluWfNorm, MCreluWfNorm, MSplitOut01, MSplitOut02, MSplitOut03, MSplitOut04
from .data import DataProcessing, DataTransform
from .metrics import R2Score, AR2Score, RMSELoss
from os import getenv, makedirs
from os.path import join, exists, basename
from time import localtime
import h5py
from numpy import asarray, arange, zeros, median, sqrt
from sys import exit
import matplotlib.pyplot as plt


# Training settings
parser = argparse.ArgumentParser('psychxrf', formatter_class = RawTextHelpFormatter)
parser.add_argument('--h5data', type = str, required = True, help = 'HDF5 data file to train from')
parser.add_argument('--batch-size', type = int, default = 64, help = 'training batch size [64]')
parser.add_argument('--test-batch-size', type = int, default = 32, help = 'testing batch size [32]')
parser.add_argument('--num-epoch', type = int, default = 800, help = 'number of epoch to train for [800]')
parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate [0.01]')
parser.add_argument('--hidden-sizes', nargs = '+', default = [128], help = 'sequence of hidden layer sizes')
parser.add_argument('--optimizer', type = str, default = 'SDG', help = 'Optimizer')
parser.add_argument('--criterion', type = str, default = 'MSELoss', help = 'Loss function')
parser.add_argument('--plot', action = 'store_true', help = 'animated plot with loss results')
parser.add_argument('--root-dir', type = str, default = f'{join(getenv("HOME"),".psychXRF")}', help = f'root directory to store on [{join(getenv("HOME"),".psychXRF")}]')
parser.add_argument('--trans-file', type = str, default = '', help = 'HDF5 file were inputs & targets transformation parameters\nare stored [ROOT_DIR/transforms/<H5DATA name>_trans.h5]')
parser.add_argument('--model-name', type = str, default = '', help = 'model name used to save checkpoints [model_<timestamp>]')
parser.add_argument('--model', type = str, default = '', help = 'model to train [MSplitOut04]')
parser.add_argument('--checkpoint', type = str, default = '', help = 'load CHEKPOINT and train (to be implemented)')

opt = parser.parse_args()
metadata = {}

lt = localtime()
timestamp = f'{lt.tm_mday}-{lt.tm_mon}-{lt.tm_year}_{lt.tm_hour}-{lt.tm_min}-{lt.tm_sec}'
model_name = opt.model_name if opt.model_name else f'model_{timestamp}'
model_dir = join(opt.root_dir, 'models', model_name)
makedirs(model_dir, exist_ok = True)
metadata['timestamp'] = timestamp
metadata['model_name'] = model_name
metadata['model_dir'] = model_dir

transform_dir = join(opt.root_dir, 'transforms')
makedirs(transform_dir, exist_ok = True)
transform_file = opt.trans_file if opt.trans_file else join(transform_dir, f'{basename(opt.h5data).replace(".h5","")}_trans.h5')
metadata['transform_file'] = transform_file
metadata['leaning_rate'] = opt.lr
metadata['train_data'] = opt.h5data
metadata['batch_size'] = opt.batch_size
metadata['test_batch_size'] = opt.test_batch_size
metadata['hidden_sizes'] = opt.hidden_sizes
metadata['optimizer'] = opt.optimizer

config_dir = join(opt.root_dir, 'config')
# makedirs(config_dir, exist_ok = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(targets, split_point = 2):
    n = lambda x: x/x.max()
    for i in range(split_point):
        targets[:,i] = n(targets[:,i])
    targets[:, split_point:] = targets[:, split_point:]/100.
    return targets
    

print("###### LOADING DATASETS ######")
dproc = DataProcessing().load_h5_data(opt.h5data)
dtrans = DataTransform(dproc.get_inputs_from_labels(), dproc.get_targets(), transform_file = transform_file)
dtrans.input_transform()
dtrans.target_transform = normalize
metadata['reflayer_thicknes_max'] = dtrans.targets[:, 0].max()
metadata['sublayer_thicknes_max'] = dtrans.targets[:, 1].max()

train_set = dtrans.get_training_set()
test_set = dtrans.get_testing_set()
training_dataloader = DataLoader(train_set, opt.batch_size, pin_memory = True, shuffle = True)
testing_dataloader = DataLoader(test_set, opt.test_batch_size, pin_memory = True, shuffle = True)

print("###### Building Model ######")
if opt.model:
    _model = getattr(Model, opt.model)
    model = _model(in_size = dtrans.inputs.shape[1], out_size = dtrans.targets.shape[1], hidden_sizes = [int(x) for x in opt.hidden_sizes]).to(device)
else:
    model = Model.MSplitOut04(in_size = dtrans.inputs.shape[1], out_size = dtrans.targets.shape[1], hidden_sizes = [int(x) for x in opt.hidden_sizes]).to(device)
if hasattr(nn, opt.criterion):
    criterion = getattr(nn, opt.criterion)
elif hasattr(metrics, opt.criterion):
    criterion = getattr(metrics, opt.criterion)
else:
    raise ValueError(f"Criterion {op.criterion} not found")
rl_criterion = criterion() #nn.MSELoss()    
sl_criterion = criterion() #nn.MSELoss()
wf_criterion = criterion() #nn.MSELoss()
metadata['criterion'] = [rl_criterion._get_name(), sl_criterion._get_name(), wf_criterion._get_name()]
if hasattr(optim, opt.optimizer):
    _optim = getattr(optim, opt.optimizer)
else:
    raise ValueError(f"Optimizer {opt.optimizer} not found in torch.optim")
optimizer = _optim(model.parameters(), lr = opt.lr)
metadata['optimizer'] = opt.optimizer
r2score = R2Score()
print(model)

def train(epoch):
    epoch_rl_loss = 0
    epoch_sl_loss = 0
    epoch_wf_loss = 0
    epoch_loss = 0
    for iteration, batch in enumerate(training_dataloader,1):
        input , target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        rl_out, sl_out, wf_out = model(input)
        rl_loss = rl_criterion(rl_out, target[:, 0])
        sl_loss = sl_criterion(sl_out, target[:, 1])
        wf_loss = wf_criterion(wf_out, target[:, 2:])
        epoch_rl_loss += rl_loss.item()
        epoch_sl_loss += sl_loss.item()
        epoch_wf_loss += wf_loss.item()
        loss = (rl_loss*20 + sl_loss*10 + wf_loss*3)/(20+10+3)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #print(f"Epoch [{epoch} ({iteration}/{len(training_dataloader)})]: Loss: {loss.item():.4f}")
    if (epoch % 50 == 0):
        epoch_loss = epoch_loss/len(training_dataloader)
        epoch_rl_loss = epoch_rl_loss/len(training_dataloader)
        epoch_sl_loss = epoch_sl_loss/len(training_dataloader)
        epoch_wf_loss = epoch_wf_loss/len(training_dataloader)
        print(f"##### Epoch {epoch} Completed: avg. Loss: {epoch_loss:.4f}")
        print(f"      avg. reflayer Loss: {epoch_rl_loss:.4f}")
        print(f"      avg. sublayer Loss: {epoch_sl_loss:.4f}")
        print(f"      avg. weigh fractions Loss: {epoch_wf_loss:.4f}\n")
    return epoch_rl_loss, epoch_sl_loss, epoch_wf_loss, epoch_loss

def test(epoch):
    mean_loss = 0
    mean_rl_loss = 0
    mean_sl_loss = 0
    mean_wf_loss = 0
    mean_metric = 0
    with torch.no_grad():
        for batch in testing_dataloader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            rl_loss = rl_criterion(prediction[0], target[:,0])
            sl_loss = sl_criterion(prediction[1], target[:,1])
            wf_loss = wf_criterion(prediction[2], target[:, 2:])
            loss = (rl_loss + sl_loss + wf_loss)/3.
            rl_metric = r2score(prediction[0], target[:,0])
            sl_metric = r2score(prediction[1], target[:,1])
            wf_metric = r2score(prediction[2], target[:,2:])
            metric = (rl_metric + sl_metric + wf_metric)/3.
            mean_rl_loss += rl_loss.item()
            mean_sl_loss += sl_loss.item()
            mean_wf_loss += wf_loss.item()
            mean_loss += loss.item()
            mean_metric += metric.item()
    mean_loss = mean_loss/len(testing_dataloader)
    mean_rl_loss = mean_rl_loss/len(testing_dataloader)
    mean_sl_loss = mean_sl_loss/len(testing_dataloader)
    mean_wf_loss = mean_wf_loss/len(testing_dataloader)
    mean_metric = mean_metric/len(testing_dataloader)
    #adjusted r2score
    ar2score = AR2Score(input.shape[0], input.shape[1], mean_metric)
    if (epoch % 50 == 0):
        print(f"##### Test Loss: {mean_loss:.4f}")
        print(f"      Test reflayer Loss: {mean_rl_loss:.4f}")
        print(f"      Test sublayer Loss: {mean_sl_loss:.4f}")
        print(f"      Test weith fraction Loss: {mean_wf_loss:.4f}")
        print(f"##### Test R2: {mean_metric:.4f}")
        print(f"##### Test Adjusted R2: {ar2score:.4f}")
    return mean_rl_loss, mean_sl_loss, mean_wf_loss, mean_loss
    
            

def checkpoint(epoch):
    model_file = join(model_dir, f'{model_name}_epoch{epoch}.pt')
    if (epoch % 10 == 0):
        torch.save(model, model_file)
        print(f'Checkpoint saved to {model_file}')

def plot_results(train_loss, test_loss):
    fig, ax = plt.subplots()
    x = arange(len(train_loss))
    ax.plot(x, train_loss, label = 'Train Loss')
    ax.plot(x, test_loss, label = 'Test Loss')
    ax.legend()
    plt.show()
    

def main():
    train_loss = zeros((4,opt.num_epoch))
    test_loss = zeros((4, opt.num_epoch))
    trainL = train(1)
    testL = test(1)
    max_loss = max(testL)
    train_loss[:,0] = asarray(trainL)
    test_loss[:,0] = asarray(testL)
    if opt.plot:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, opt.num_epoch)
        ax.set_ylim(0, max_loss)
        ax.set_title(f'Model {model_name}\ntesting set loss')
        x = arange(opt.num_epoch)
        test_line_rl, = ax.plot(x, test_loss[0], label = 'reference layer')
        test_line_sl, = ax.plot(x, test_loss[1], label = 'sublayer')
        test_line_wf, = ax.plot(x, test_loss[2], label = 'weight fractions')
        test_line_total, = ax.plot(x, test_loss[3], label = 'mean')
        # test_line, = ax.plot(x, test_loss, label = 'Test Loss')
        ax.legend()
        print(f"\n\nTraining model {model_name}")
    for epoch in range(2, opt.num_epoch + 1):
        train_loss[:, epoch-1] = train(epoch)
        test_loss[:, epoch-1] = test(epoch)
        checkpoint(epoch)
        # plot results
        if opt.plot:
            # train_line.set_ydata(train_loss)
            test_line_rl.set_ydata(test_loss[0])
            test_line_sl.set_ydata(test_loss[1])
            test_line_wf.set_ydata(test_loss[2])
            test_line_total.set_ydata(test_loss[3])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    # train_loss = asarray(train_loss)
    # test_loss = asarray(test_loss)
    loss_fname = join(model_dir, f'{model_name}_loss.h5')
    print(f'Saving loss file {loss_fname}')
    with h5py.File(loss_fname, 'w') as fout:
        for k, v in metadata.items():
            fout.attrs[k] = v
        dataset = fout.create_dataset('train_loss', data = train_loss)
        dataset = fout.create_dataset('test_loss', data = test_loss)
    print("##### DONE #####")
    if opt.plot:
        plt.ioff()
        plt.show()
    # plot_results(train_loss, test_loss)

if __name__ == '__main__':
    main()

