# -*- coding: utf-8 -*-

#### 本代码为数据集划分代码

# 首先将数据集等量划分

import numpy as np
import pandas as pd
import pickle

############## 把数据集汇聚成.npy文件 #############
# 读取数据集

def load_cifar10_batch(cifar10_dataset_folder_path,batch_id):

    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id),mode='rb') as file:
        batch = pickle.load(file, encoding = 'latin1')
        
    # features and labels
    features = batch['data'].reshape((len(batch['data']),3,32,32))
    labels = batch['labels']
    
    return  features, labels  

# 加载所有数据
cifar10_path = './dataset/cifar-10-batches-py'

# 载入数据,去除表头
# 一共有5个batch的训练数据
x_train_dataset, y_train_dataset = load_cifar10_batch(cifar10_path, 1)
for i in range(2,6):
    features,labels = load_cifar10_batch(cifar10_path, i)
    x_train_dataset, y_train_dataset = np.concatenate([x_train_dataset, features]),np.concatenate([y_train_dataset, labels])

# 加载测试数据
with open(cifar10_path + '/test_batch', mode = 'rb') as file:
    batch = pickle.load(file, encoding='latin1')
    x_test_dataset = batch['data'].reshape((len(batch['data']),3,32,32))
    y_test_dataset = batch['labels']

# 将原本的50K的training dataset 10K的test dataset进行合并一个大的数据集
x_dataset = np.vstack((x_train_dataset,x_test_dataset))
y_dataset = np.hstack((y_train_dataset,y_test_dataset))


# ###################
# # 保存为npy文件
np.save("./dataset/cifar10/dataset_cifar10.npy",x_dataset)
np.save("./dataset/cifar10/dataset_cifar10_label.npy",y_dataset)

############################

######### 把数据集按client的数量进行分割
# 载入数据
x_dataset = np.load('./dataset/cifar10/dataset_cifar10.npy')

# 取分类标签，
lables = np.load('./dataset/cifar10/dataset_cifar10_label.npy')

# 均分为训练集和测试集，30K train和30K test
# 即总的训练集和总的测试集
data_size = 30000
x_train_dataset = x_dataset[:data_size]
y_train_dataset = lables[:data_size]

x_test_dataset = x_dataset[data_size:2*data_size]
y_test_dataset = lables[data_size:2*data_size]

# 保存总的训练集和测试集数据
np.save('./dataset/cifar10/x_train_cifar10_all.npy',x_train_dataset)   
np.save('./dataset/cifar10/y_train_cifar10_all.npy',y_train_dataset)   

np.save('./dataset/cifar10/x_test_cifar10_all.npy',x_test_dataset)  
np.save('./dataset/cifar10/y_test_cifar10_all.npy',y_test_dataset) 

#在从30K train和30K test中均分给各个client
num_partition = 10
# 平均每个client的数据集
num_data_size = int(data_size/num_partition)

for i in range(num_partition):
    #划分当前的client的数据集
    num_index = i
    print(num_index)
    partition_start = num_index
    partition_end = num_index + 1
    
    x_train_dataset_save = x_train_dataset[num_data_size*partition_start:num_data_size*partition_end]
    y_train_dataset_save = y_train_dataset[num_data_size*partition_start:num_data_size*partition_end]
        
    x_test_dataset_save = x_test_dataset[num_data_size*partition_start:num_data_size*partition_end]
    y_test_dataset_save = y_test_dataset[num_data_size*partition_start:num_data_size*partition_end]
            
    #保存各个client的数据集    
    np.save('./dataset/cifar10/x_train_cifar10_{}.npy'.format(num_index),x_train_dataset_save)   
    np.save('./dataset/cifar10/y_train_cifar10_{}.npy'.format(num_index),y_train_dataset_save)   
    
    np.save('./dataset/cifar10/x_test_cifar10_{}.npy'.format(num_index),x_test_dataset_save)   
    np.save('./dataset/cifar10/y_test_cifar10_{}.npy'.format(num_index),y_test_dataset_save) 

print('this is cifar10 data partition for {} clients'.format(num_partition))