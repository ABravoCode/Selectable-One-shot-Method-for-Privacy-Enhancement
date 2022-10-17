# -*- coding: utf-8 -*-

import torch

def fcn_module(inputsize, layer_size=128):
    """
    Creates a FCN submodule. Used in different attack components.
    Args:
    ------
    inputsize: size of the input layer
    """
    fcn = torch.nn.Sequential(
            torch.nn.Linear(inputsize, layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(layer_size, 64)
            )
    
    return fcn
    