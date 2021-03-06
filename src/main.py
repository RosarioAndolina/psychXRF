#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import psychXRF.model as Model
#from .model import MFreluSmax, MFreluWfNorm, MCreluWfNorm, MSplitOut01, MSplitOut02, MSplitOut03, MSplitOut04
from .data import DataProcessingSF, DataTransform
from .metrics import R2Score, AR2Score, MAPE
import psychXRF.metrics as Metrics
from os import getenv, makedirs
from os.path import join, exists, basename
from time import localtime
import h5py
from numpy import asarray, arange, zeros, median, sqrt, linspace, sin, where, vstack, int8, ceil
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
metadata['learning_rate'] = opt.lr
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
dproc = DataProcessingSF(sfbounds = [0.1, 2] ).load_h5_data(opt.h5data)
# cnd = where((dproc.data.sublayer_thickness > 20.0e-4) & (dproc.data.sublayer_thickness <= 40.0e-4))
# dproc.data.data = dproc.data.data[cnd]
# dproc.shape = dproc.data.data.shape
# dproc.data.labels = dproc.data.labels[cnd]
# dproc.data.reflayer_thickness = dproc.data.reflayer_thickness[cnd]
# dproc.data.sublayer_thickness = dproc.data.sublayer_thickness[cnd]
# dproc.data.weight_fractions = dproc.data.weight_fractions[cnd]
# dproc.scale_factor = dproc.scale_factor[cnd]

inputs = []
targets = []
for i in range(5):
    inputs.append(dproc.get_inputs_from_labels())
    targets.append(dproc.get_targets())    
    dproc.new_scale_factor()

dtrans = DataTransform(vstack(inputs), vstack(targets), transform_file = transform_file)
print(f"inputs shape: {dtrans.inputs.shape}")
dtrans.input_transform()
dtrans.target_transform = normalize
metadata['reflayer_thickness_max'] = dtrans.targets[:, 0].max()
#metadata['sublayer_thickness_max'] = None
metadata['scale_factor_max'] = dtrans.targets[:,1].max()

train_set = dtrans.get_training_set()
test_set = dtrans.get_testing_set()
training_dataloader = DataLoader(train_set, opt.batch_size, pin_memory = True, shuffle = True)
testing_dataloader = DataLoader(test_set, opt.test_batch_size, pin_memory = True, shuffle = True)

print("###### Building Model ######")
if opt.model:
    _model = getattr(Model, opt.model)
    if 'ResNet' in opt.model:
        model = _model(in_size = dtrans.inputs.shape[1], out_size = dtrans.targets.shape[1]).to(device)
    else:
        model = _model(in_size = dtrans.inputs.shape[1], out_size = dtrans.targets.shape[1], hidden_sizes = [int(x) for x in opt.hidden_sizes]).to(device)
else:
    model = Model.MSplitOut04(in_size = dtrans.inputs.shape[1], out_size = dtrans.targets.shape[1], hidden_sizes = [int(x) for x in opt.hidden_sizes]).to(device)
if hasattr(nn, opt.criterion):
    loss_func = getattr(nn, opt.criterion)
elif hasattr(Metrics, opt.criterion):
    loss_func = getattr(Metrics, opt.criterion)
else:
    raise ValueError(f"Criterion {opt.criterion} not found")
criterion = loss_func() #nn.MSELoss()    
# sl_criterion = criterion() #nn.MSELoss()
# wf_criterion = criterion() #nn.MSELoss()
# metadata['criterion'] = [rl_criterion._get_name(), sl_criterion._get_name(), wf_criterion._get_name()]
metadata['criterion'] = criterion._get_name()
if hasattr(optim, opt.optimizer):
    _optim = getattr(optim, opt.optimizer)
else:
    raise ValueError(f"Optimizer {opt.optimizer} not found in torch.optim")

lambda0 = opt.num_epoch / (opt.num_epoch)
def lambda1(epoch, lambda0=lambda0):
    #return (lambda0 * (sin(0.1 * epoch)) ** 2) ** epoch
    return lambda0 ** epoch

optimizer = _optim(model.parameters(), lr = opt.lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

metadata['optimizer'] = opt.optimizer
r2score = R2Score()
mape = MAPE()
print(model)

def train(epoch):
    epoch_rl_loss = 0
    epoch_wf_loss = 0
    epoch_scale_loss = 0
    epoch_loss = 0
    for iteration, batch in enumerate(training_dataloader,1):
        input , target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        rl_out, scale_out, wf_out = model(input)
        rl_loss = criterion(rl_out, target[:, 0])
        scale_loss = criterion(scale_out, target[:, 1])
        wf_loss = criterion(wf_out, target[:, 2:])
        epoch_rl_loss += rl_loss.item()
        epoch_wf_loss += wf_loss.item()
        epoch_scale_loss += scale_loss.item()
        loss = (rl_loss*10 + scale_loss * 50 + wf_loss*30)/(10+50+30)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #print(f"Epoch [{epoch} ({iteration}/{len(training_dataloader)})]: Loss: {loss.item():.4f}")
    if (epoch % 50 == 0):
        epoch_loss = epoch_loss/len(training_dataloader)
        epoch_rl_loss = epoch_rl_loss/len(training_dataloader)
        epoch_wf_loss = epoch_wf_loss/len(training_dataloader)
        epoch_scale_loss = epoch_scale_loss/len(training_dataloader)
        print(f"##### Epoch {epoch} Completed: avg. Loss: {epoch_loss:.4f}")
        print(f"      avg. reflayer Loss: {epoch_rl_loss:.4f}")
        print(f"      avg. scale factor Loss: {epoch_scale_loss:.4f}")
        print(f"      avg. weigh fractions Loss: {epoch_wf_loss:.4f}\n")
    return epoch_rl_loss, epoch_scale_loss, epoch_wf_loss, epoch_loss

def test(epoch):
    #loss R2 & mape -> LRM
    mean_LRM = torch.zeros(3,3)
    LRM = torch.zeros(3,3)
    LRM_funcs = [criterion, r2score, mape]
    with torch.no_grad():
        for batch in testing_dataloader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            for i in range(3):
                LRM[i,0] = LRM_funcs[i](prediction[0], target[:,0])
                LRM[i,1] = LRM_funcs[i](prediction[1], target[:,1])
                LRM[i,2] = LRM_funcs[i](prediction[2], target[:,2:])
            mean_LRM += LRM
    mean_LRM = mean_LRM/len(testing_dataloader)
    mean_loss = mean_LRM[0].mean()
    mean_metric = mean_LRM[1].mean()
    mean_mape = mean_LRM[2].mean()
    #adjusted r2score
    ar2score = AR2Score(input.shape[0], input.shape[1], mean_metric)
    if (epoch % 10 == 0):
        print(f"EPOCH {epoch}")
        print(f"##### Test Loss: {mean_loss.item():.4f}")
        print(f"      Test reflayer Loss: {mean_LRM[0,0]:.4f}")
        print(f"      Test scale factor Loss: {mean_LRM[0,1]:.4f}")
        print(f"      Test weith fraction Loss: {mean_LRM[0,2]:.4f}")
        print(f"##### Test R2: {mean_metric.item():.4f}")
        print(f"      Test reflayer R2: {mean_LRM[1,0]:.4f}")
        print(f"      Test scale factor R2: {mean_LRM[1,1]:.4f}")
        print(f"      Test weight fractions R2: {mean_LRM[1,2]:.4f}")
        print(f"##### Test Adjusted R2: {ar2score:.4f}")
        print(f"      ACCURACY reflayer: {100 - mean_LRM[2,0]}")
        print(f"      ACCURACY scale factor: {100 - mean_LRM[2,1]}")
        print(f"      ACCURACY weight fractions: {100 - mean_LRM[2,2]}")
        print(f"      mean MAPE: {mean_mape}")
    return torch.cat((mean_LRM[0], mean_loss.reshape(1), 1 - mean_LRM[1]))
    
            

def checkpoint(epoch):
    model_file = join(model_dir, f'{model_name}_epoch{epoch}.pt')
    if (epoch % 10 == 0):
        torch.save(model, model_file)
        print(f'Checkpoint saved to {model_file}')

def plot_results(best_checkpoint, r2 = False):
    if r2:
        metrics_range = range(4, len(best_checkpoint))
    else:
        metrics_range = range(4)
    checkpoint_paths = [
        join(model_dir, f"{model_name}_epoch{best_checkpoint[i]}.pt")
        for i in metrics_range
    ]
    models = [torch.load(x).to('cpu') for x in checkpoint_paths]
    with torch.no_grad():
        inputs = test_set[:][0]
        targets = test_set[:][1]
        predictions = [
            m(inputs)[i] for i, m in enumerate(models[:-1])
        ]
        if r2:
            predictions.append(models[-1](inputs)[-1])
        else:
            predictions.append(models[-1](inputs))
    #x = linspace(1e-3,1,10)
    fig1, ax1 = plt.subplots()
    ax1.scatter(targets[:, 1]*metadata['scale_factor_max'], targets[:,0]*metadata['reflayer_thickness_max'], s = 5, alpha = 0.3)
    ax1.scatter(predictions[1]*metadata['scale_factor_max'], predictions[0]*metadata['reflayer_thickness_max'],  s = 5, alpha = 0.3, label = 'prediction')
    ax1.set_title("reflayer thickness vs scale factor", fontsize = 10)
    ax1.set_xlabel(r"scale factor")
    ax1.set_ylabel(r"rl thickness $\mu$")
    #ax[0].set_xlim(0,1)
    #ax[0].set_ylim(0,1)
    ax1.legend()
    
    #combo = list(combinations[])
    x = len(dproc.data.metadata['reflayer_elements'])
    nrow = int8(sqrt(x))
    ncol = int8(ceil(x/nrow))
    bins = linspace(targets.numpy().min(), targets.numpy().max(), 50)
    fig2, ax2 = plt.subplots(nrow,ncol, figsize = (9,10))
    for i in range(nrow):
        for j in range(ncol):
            try:
                t = targets[:, 2:][:,j::ncol][:,i]
                p = predictions[2][:,j::ncol][:,i]
                ax2[i,j].set_title(f"{dproc.data.metadata['reflayer_elements'][j::ncol][i]}")
                ax2[i,j].hist(t.numpy(), bins = bins, histtype = 'step', label = 'target')
                ax2[i,j].hist(p.numpy(), bins = bins, histtype = 'step', label = 'prediction')
                ax2[i,j].set_yscale('log')
                ax2[i,j].set_xlabel("weight fraction")
            except IndexError:
                pass
            #ax2[i,j].plot(x,x, c = "darkorange", label = "Ideal")
            #ax2[i,j].set_ylabel("prediction")
            #ax2[i,j].set_xlim(0,1)
            #ax2[i,j].set_ylim(1.0e-8,torch.max(t,p).max())
            ax2[i,j].legend()
    fig3, ax3 = plt.subplots(1,2)
    c,b,_ = ax3[0].hist(targets[:,1].numpy()*metadata['scale_factor_max'], bins = 50, histtype = 'step', label = 'target')
    ax3[0].hist(predictions[1].numpy()*metadata['scale_factor_max'], bins = b, histtype = 'step', label = 'prediction')
    ax3[0].set_title("Scale factor")
    ax3[0].set_ylabel("counts")
    ax3[0].legend()
    
    c, b, _ = ax3[1].hist(targets[:,0].numpy()*metadata['reflayer_thickness_max'], bins = 50, histtype = 'step', label = 'target')
    ax3[1].hist(predictions[0].numpy()*metadata['reflayer_thickness_max'], bins = b, histtype = 'step', label = 'prediction')
    ax3[1].set_title("Ref. layer thicknes")
    ax3[1].set_ylabel("counts")
    ax3[1].legend()
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    plt.show()
        

def main():
    train_loss = torch.zeros((4,opt.num_epoch))
    test_loss = torch.zeros((7, opt.num_epoch))
    trainL = train(1)
    testL = test(1)
    max_loss = testL[:4].max()
    train_loss[:,0] = torch.tensor(trainL)
    test_loss[:,0] = testL
    if opt.plot:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, opt.num_epoch)
        ax.set_ylim(0.0005, max_loss)
        ax.set_title(f'Model {model_name}\ntesting set loss')
        x = arange(opt.num_epoch)
        test_line_rl, = ax.plot(x, test_loss[0].numpy(), label = 'reference layer')
        test_line_sf, = ax.plot(x, test_loss[1].numpy(), label = 'scale factor')
        test_line_wf, = ax.plot(x, test_loss[2].numpy(), label = 'weight fractions')
        test_line_total, = ax.plot(x, test_loss[3].numpy(), label = 'mean')
        # test_line, = ax.plot(x, test_loss, label = 'Test Loss')
        ax.legend()
        print(f"\n\nTraining model {model_name}")
    for epoch in range(2, opt.num_epoch + 1):
        try:
            train_loss[:, epoch-1] = torch.tensor(train(epoch))
            scheduler.step()
            test_loss[:, epoch-1] = test(epoch)
            checkpoint(epoch)
            # plot results
            if opt.plot:
                # train_line.set_ydata(train_loss)
                test_line_rl.set_ydata(test_loss[0].numpy())
                test_line_sf.set_ydata(test_loss[1].numpy())
                test_line_wf.set_ydata(test_loss[2].numpy())
                test_line_total.set_ydata(test_loss[3].numpy())
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
        except KeyboardInterrupt:
            break
    # train_loss = asarray(train_loss)
    # test_loss = asarray(test_loss)
    #checkpoint_index = (torch.arange(10, opt.num_epoch + 1, 10)[1:]) - 1
    checkpoint_index = (torch.arange(10, epoch, 10)) - 1
    best_test_loss = (test_loss[:, checkpoint_index].min(dim = 1).indices + 1) *10
    metadata['best_checkpoint'] = best_test_loss.numpy()
    loss_fname = join(model_dir, f'{model_name}_loss.h5')
    print(f'Saving loss file {loss_fname}')
    with h5py.File(loss_fname, 'w') as fout:
        for k, v in metadata.items():
            fout.attrs[k] = v
        dataset = fout.create_dataset('train_loss', data = train_loss)
        dataset = fout.create_dataset('test_loss', data = test_loss)
    print("@@@@@@@@@@@@ BEST CHECKPOINT: ", metadata['best_checkpoint'])
    print("##### DONE #####")
    if opt.plot:
        plt.ioff()
        plt.show()
    plot_results(metadata['best_checkpoint'])

if __name__ == '__main__':
    main()
