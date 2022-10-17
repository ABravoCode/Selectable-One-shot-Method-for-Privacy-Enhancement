# -*- coding: utf-8 -*-

import torch
import numpy as np

def CrossEntropyLoss_exampleloss(logits, labels):
    """
    Calculates the softmax cross entropy loss for classification
    predictions.
    """
    # tensor 变为int64的Tensor
    labels = labels.to(torch.int64)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = []
    
    for i in range(len(labels)):
        result = []
        x = logits[i] 
        # 扩充一维
        x = torch.unsqueeze(x,0)
        y = labels[i]
        y = torch.unsqueeze(y,0)
        res = loss_func(x,y)
        result.append(res.data.numpy())
        loss.append(result)
    loss = torch.tensor(np.stack(loss))

    return loss


def CrossEntropyLoss(logits, labels):
    """
    Calculates the softmax cross entropy loss for classification
    predictions.
    """
    # tensor 变为int64的Tensor
    labels = labels.to(torch.int64)
    # 扁平，变为1维的label
    labels = labels.flatten()
#    print(labels)
    loss_func = torch.nn.CrossEntropyLoss()
#    logits = torch.topk(logits, 1)[1].squeeze(1)
#    print(logits)
    loss = loss_func(logits,labels)

    return loss

def mse(true, predicted):
    """
    Computes loss of the attack model on given batch
    Args:
    ----
    """
    loss_func = torch.nn.MSELoss()
    loss = loss_func(true, predicted)
    return loss

