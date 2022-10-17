# -*- coding: utf-8 -*-

# 用于使用操作系统接口的包
import os

import math
import torch
from torch import nn
import torch.nn.functional as F


import numpy as np

# 简单的完整性检测，输入层数是否有误
def sanity_check(layers, layers_to_exploit):
    """
    Basic sanity check for layers and gradients to exploit based on model layers
    """
    if layers_to_exploit and len(layers_to_exploit):
        assert np.max(layers_to_exploit) <= len(layers),\
            "layer index greater than the last layer"

# 时间花费
def time_taken(self, start_time, end_time):
    """
    Calculates difference between 2 times
    """
    delta = end_time - start_time
    hours = int(delta / 3600)
    delta -= hours * 3600
    minutes = int(delta / 60)
    delta -= minutes * 60
    seconds = delta
    return hours, minutes, np.int(seconds)            

# 定义attack_utils类，用于成员推测攻击的基本工具函数
class attack_utils():
    """
    Utilities required for conducting membership inference attack
    """

    def __init__(self, directory_name='latest'):
        self.root_dir = os.path.abspath(os.path.join(
                                        os.path.dirname(__file__),
                                        ))
        self.log_dir = os.path.join(self.root_dir, "logs")
#        self.aprefix = os.path.join(self.log_dir,
#                                    directory_name,
#                                    "attack",
#                                    "model_checkpoints")
#        self.dataset_directory = os.path.join(self.root_dir, "datasets")

#        if not os.path.exists(self.aprefix):
#            os.makedirs(self.aprefix)
#        if not os.path.exists(self.dataset_directory):
#            os.makedirs(self.dataset_directory)
            
    def get_gradshape(self, layer):
        """
        Returns the shape of gradient matrices
        Args:
        -----
        model: model to attack 
        """
        row = layer[0].size()[0]
        col = layer[0].size()[1]
        gradshape = (row,col)
        return gradshape          

#    # 求梯度的范数【有修改】
#    def get_gradient_norm(self, gradients):
#            """
#            Returns the norm of the gradients of loss value
#            with respect to the parameters
#            Args:
#            -----
#            gradients: Array of gradients of a batch 
#            """
#            gradient_norms = []
#            for gradient in gradients:
#                #summed_squares = [K.sum(K.square(g)) for g in gradient]
#                summed_squares = [sum(math.pow(g,2)) for g in gradient]                
#                #norm = K.sqrt(sum(summed_squares))
#                norm = math.sqrt(sum(summed_squares))
#                gradient_norms.append(norm)
#            return gradient_norms

#    # 计算预测不确定性（熵）【有修改】
#    def get_entropy(self, model, features, output_classes):
#            """
#            Calculates the prediction uncertainty
#            """
#            entropyarr = []
#            for feature in features:
##                feature = tf.reshape(feature, (1, len(feature.numpy())))
#                feature = torch.reshape(feature, (1, len(feature.numpy())))
#                predictions = model(feature)
##                predictions = tf.nn.softmax(predictions)
#                predictions = nn.Softmax(predictions)
##                mterm = tf.reduce_sum(input_tensor=tf.multiply(predictions,
##                                                               np.log(predictions)))
#                mterm = torch.sum(input_tensor=np.multiply(predictions,
#                                               np.log(predictions)))
#                entropy = (-1/np.log(output_classes)) * mterm
#                entropyarr.append(entropy)
#            return entropyarr
#
    # 【有修改】
    def split(self, x):
        """
        Splits the array into number of elements equal to the
        size of the array. This is required for per example
        computation.
        """
#        split_x = tf.split(x, len(x.numpy()))
        # x 本来是一个batch,转为一个数据样本,e.g,100,1
        split_x = torch.split(x, len(x.numpy()))
        
        return split_x
#
##   【有修改，返回是参数，不是tf.saver】
#    def get_savers(self, attackmodel):
#            """
#            Creates prefixes for storing classification and inference
#            model
#            """
#            # Prefix for storing attack model checkpoints
#            prefix = os.path.join(self.aprefix, "ckpt")
#            # Saver for storing checkpoints
##            attacksaver = Saver(attackmodel.variables)
#            attacksaver = attackmodel.variables
#            return prefix, attacksaver
#
#   【有修改】
    def createOHE(self, num_output_classes):
            """
            creates one hot encoding matrix of all the vectors
            in a given range of 0 to number of output classes.
            """
#            return tf.one_hot(tf.range(0, num_output_classes),
#                              num_output_classes,
#                              dtype=tf.float32)
            return F.one_hot(torch.arange(0, num_output_classes),
                             num_output_classes)
#    【有修改】
    def one_hot_encoding(self, labels, ohencoding):
            """
            Creates a one hot encoding of the labels used for 
            inference model's sub neural network
            Args: 
            ------
            zero_index: `True` implies labels start from 0
            """
#            labels = tf.cast(labels, tf.int64).numpy()
            
            # 转化为int64
#            labels = (torch.tensor(labels).to(torch.int64)).numpy()
            labels = (labels.clone().detach().to(torch.int64)).numpy()
            # 由于使用了list()才能遍历这个map对象，加上使用np.stack增加了一个维度
            result = np.stack(list(map(lambda x: ohencoding[x], labels)))
            
            result = torch.tensor(result,dtype=torch.float32)
#            return tf.stack(list(map(lambda x: ohencoding[x], labels)))           
            return result

#         # 【有修改】
#    def intersection(self, to_remove, remove_from, batch_size):
#            """
#            Finds the intersection between `to_remove` and `remove_from`
#            and removes this intersection from `remove_from` 
#            """
#            # unbatch是tf的一个函数，用于把多个数组元素，转为一个数组维度
#            to_remove = to_remove.unbatch()
#            remove_from = remove_from.unbatch()
#    
#            m1, m2 = dict(), dict()
#            for example in to_remove:
#                hashval = hash(bytes(np.array(example)))
#                m1[hashval] = example
#            for example in remove_from:
#                hashval = hash(bytes(np.array(example)))
#                m2[hashval] = example
#    
#            # Removing the intersection
#            extracted = {key: value for key,
#                         value in m2.items() if key not in m1.keys()}
#            dataset = extracted.values()
#            features, labels = [], []
#            for d in dataset:
#                features.append(d[0])
#                labels.append(d[1])
#            finaldataset = tf.compat.v1.data.Dataset.from_tensor_slices(
#                (features, labels))
#            return finaldataset.batch(batch_size=batch_size)





            