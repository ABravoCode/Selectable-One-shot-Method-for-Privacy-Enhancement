# -*- coding: utf-8 -*-

import torch

def cnn_for_fcn_gradients(input_shape):
    """
    Creates a CNN submodule for Dense layer gradients.
    """
    # Input container
    dim1 = int(input_shape[0])
    dim2 = int(input_shape[1])
    
    cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(1,dim2),stride=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),            
            torch.nn.Linear(100*dim1, 2024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
            )
    
    return cnn
    