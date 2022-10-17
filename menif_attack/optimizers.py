# -*- coding: utf-8 -*-

# 定义一些优化器
from torch import optim

def optimizer_op(optimizer):
    """
    Returns TensorFlow optimizer for supported optimizers
    """
    op_lower_case = optimizer.lower()
    if op_lower_case == "sgd":
        return optim.SGD
    elif op_lower_case == "adam":
        return optim.Adam
    elif op_lower_case == "adagrad":
        return optim.Adagrad
    elif op_lower_case == "rmsprop":
        return optim.RMSprop
    