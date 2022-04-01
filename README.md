# psychXRF
Predict synthetic multichannel XRF spectra simulation parameters

# Installation
pip install --user .

# Usage
```console
psychxrf [-h] --h5datafile H5DATAFILE [--batch-size BATCH_SIZE] [--test-batch-size TEST_BATCH_SIZE] 
                [--num-epoch NUM_EPOCH] [--lr LR] [--hidden-sizes HIDDEN_SIZES [HIDDEN_SIZES ...]] 
                [--optimizer OPTIMIZER] [--trans-file TRANS_FILE]

options:
  -h, --help            show this help message and exit
  --h5datafile H5DATAFILE
                        HDF5 data file
  --batch-size BATCH_SIZE
                        training batch size [64]
  --test-batch-size TEST_BATCH_SIZE
                        testing batch size [32]
  --num-epoch NUM_EPOCH
                        number of epoch to train for [800]
  --lr LR               learning rate [0.01]
  --hidden-sizes HIDDEN_SIZES [HIDDEN_SIZES ...]
                        sequence of hidden layer sizes
  --optimizer OPTIMIZER
                        Optimizer. One of "sgd" "adam" [sgd]
  --trans-file TRANS_FILE
                        HDF5 file were inputs & targets transformation parameters are stored

```
