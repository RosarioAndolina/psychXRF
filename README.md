# psychXRF
Predict synthetic multichannel XRF spectra simulation parameters

# Installation
pip install --user .

# Usage:
## Train
```console
psychxrf [-h] --h5data H5DATA [--batch-size BATCH_SIZE] [--test-batch-size TEST_BATCH_SIZE] [--num-epoch NUM_EPOCH] 
         [--lr LR] [--hidden-sizes HIDDEN_SIZES [HIDDEN_SIZES ...]] [--optimizer OPTIMIZER] [--plot]
         [--root-dir ROOT_DIR] [--trans-file TRANS_FILE] [--model-name MODEL_NAME]

options:
  -h, --help            show this help message and exit
  --h5data H5DATA       HDF5 data file to train from
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
  --plot                animated plot with loss results
  --root-dir ROOT_DIR   root directory to store on [/home/<user>/.psychXRF]
  --trans-file TRANS_FILE
                        HDF5 file were inputs & targets transformation parameters
                        are stored [ROOT_DIR/transforms/<H5DATA name>_trans.h5]
  --model-name MODEL_NAME
                        model name [model_<timestamp>]
```
