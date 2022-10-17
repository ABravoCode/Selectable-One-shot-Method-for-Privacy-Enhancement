# -*- coding: utf-8 -*-

import torch

def encoder(encoder_inputs_size):
    """
    Create encoder model for membership inference attack.
    Individual attack input components are concatenated and passed to encoder.
    """
#    encoder 结构
    encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_inputs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.ReLU()       
            )
    return encoder
    
#    # 这里输入需要判断是encoder_inputs_real还是encoder_inputs，cat的格式等
##    appended = torch.cat(encoder_inputs_real, axis=1)
##    len_appended = appended.shape[1]
#    
#    encoder = torch.nn.Sequential(
#            torch.nn.Linear(len_appended, 256),
#            torch.nn.ReLU()
#            )(appended)
#    encoder = torch.nn.Sequential(
#            torch.nn.Linear(256, 128),
#            torch.nn.ReLU()
#            )(encoder)            
#    encoder = torch.nn.Sequential(
#            torch.nn.Linear(128, 64),
#            torch.nn.ReLU()
#            )(encoder)    
#    encoder = torch.nn.Sequential(
#            torch.nn.Linear(64, 1),
#            torch.nn.ReLU()
#            )(encoder)       
#    return encoder    
    