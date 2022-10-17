# -*- coding: utf-8 -*-

#本代码用于生成并保存最后一个轮次的全局模型和各个client本地模型

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import copy
from torch.optim import lr_scheduler

# 载入总的训练集和测试集
x_train_all = np.load('../dataset/cifar10/x_train_cifar10_all.npy')
y_train_all = np.load('../dataset/cifar10/y_train_cifar10_all.npy')

x_test_all = np.load('../dataset/cifar10/x_test_cifar10_all.npy')
y_test_all = np.load('../dataset/cifar10/y_test_cifar10_all.npy')


##############
# 定义网络
class Net(nn.Module):
# 网络组成子模块的定义
    def __init__(self): 
        super(Net, self).__init__() 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.grad_list = []
        self.layer_list = []        
        self.layer_output_list = []
    
    # 前向过程的定义 网络流程的定义
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) #相当于 reshape 成 16 * 5 * 5 列， 行数自适应
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # 用于测试集时，记录，避免在训练时占用过多计算机内存
    def eval_layer_output(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        self.layer_output_list.append(x)
        x = self.pool(F.relu(self.conv2(x)))
        self.layer_output_list.append(x)
        x = x.view(-1, 16 * 5 * 5)
        self.layer_output_list.append(x)
        x = F.relu(self.fc1(x))
        self.layer_output_list.append(x)
        x = F.relu(self.fc2(x))
        self.layer_output_list.append(x)
        x = self.fc3(x)
        self.layer_output_list.append(x)
        return self.layer_output_list, x
    
    # 根据泄露的层数确定攻击模型实际获取的特征数据
    def get_grad(self, list=[]):        
        return self.grad_list
    
net_g = Net()    
#################
# 联邦学习过程

### 参数设置：
# client 数量
num_client = 10

# 全局训练轮次
g_iteration_count = 10
# 本地训练轮次
l_iteration_count = 10 
 
for g_epoch in range(g_iteration_count):
    # 在每个大的全局轮次中
    # 将当前全局轮次训练好的本地训练放入该数组中
    cur_all_client = []
    
    # 首先每个client下载当前的全局模型
    for client_i in range(num_client):
         print('######each client start to download global net_g ######')
         print('######this is client_{} in g_count_{}######'.format(client_i,g_epoch))    
         cur_net = copy.deepcopy(net_g)
         
         #载入当前client的数据集
         x_train = np.load('../dataset/cifar10/x_train_cifar10_{}.npy'.format(client_i))
         y_train = np.load('../dataset/cifar10/y_train_cifar10_{}.npy'.format(client_i))

         x_test = np.load('../dataset/cifar10/x_test_cifar10_{}.npy'.format(client_i))
         y_test = np.load('../dataset/cifar10/y_test_cifar10_{}.npy'.format(client_i))
         
         # 转成Tensor形式
         x_train = np.array(x_train)
         x_train = torch.FloatTensor(x_train)
        
         y_train = np.array(y_train)
         y_train = torch.LongTensor(y_train).squeeze()
        
         # 进行batch划分
         torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)
         train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                                   batch_size=64, 
                                                  shuffle=False)     
         len_cur_x_train = len(x_train)
         # 开始本地轮次的训练
         loss_func = torch.nn.CrossEntropyLoss()
         model_opt = optim.SGD(params=cur_net.parameters(),lr=0.005)
         # sgd,lr=0.005 10个全局轮次，聚合模型的预测准确率为0.46
         
         scheduler = lr_scheduler.LambdaLR(optimizer=model_opt, lr_lambda=lambda epoch:0.95**epoch)

        # 本地训练
         for epoch in range(l_iteration_count):
             # 基于划分的batch训练
             for step, data in enumerate(train_loader, start=0):
                 x_feature, y_feature = data # 解构出特征和标签
                 output = cur_net(x_feature)     # input x and predict based on x
                 loss = loss_func(output, y_feature)     # must be (1. nn output, 2. target)        

                 model_opt.zero_grad()   # clear gradients for next train
                 loss.backward()         # backpropagation, compute gradients
                 model_opt.step()        # apply gradients
        
                 if step % (int(len_cur_x_train/64)) == (int(len_cur_x_train/64))-1: # 
                     prediction = torch.max(output, 1)[1]#只返回最大值的每个索引,标签
                     pred = prediction.data.numpy()
                     target = y_feature.data.numpy()
                     accuracy = float((pred == target).astype(int).sum()) / float(target.size)
                     print("epoch={},step={} train_result model{} = {}".format(epoch, step, client_i, accuracy))
             scheduler.step()          
         print('save current client_{}'.format(client_i))                
         #测试一下此时的测试准确率         
         # 转成Tensor形式
         x_test = np.array(x_test)
         x_test = torch.FloatTensor(x_test)
        
         y_test = np.array(y_test)
         y_test = torch.LongTensor(y_test) 
         
         out_test = cur_net(x_test)     # input x and predict based on x
         prediction = torch.max(out_test, 1)[1]
         pred_y = prediction.data.numpy()
         target_y = y_test.data.numpy()
         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
         print('test_result_currrent_client_model =', accuracy)
         
          ## 保存模板模型
         torch.save(cur_net, '../model/cifar10/client_{}_g_count_{}.pkl'.format(client_i,g_epoch))  # save entire net
         # torch.save(cur_net.state_dict(),'../model/cifar10/client_{}_g_count_{}_params.pkl'.format(client_i, g_epoch))#网络的参数
         
         # 将训练好的client model保存在数据中，用于后续的第一轮聚合
         cur_all_client.append(cur_net)
    
    # 进行全局模型的聚合
    print('##### start global_model_once_in_g_count_{}'.format(g_epoch))
    
    # 参数平均聚合,更新net_g         
    # 根据模型层数，确定需要聚合的每层参数的个数        
    item_name_arr = []
    # 前面的卷积层
    item_name_arr.append('conv1.weight')
    item_name_arr.append('conv1.bias')
    item_name_arr.append('conv2.weight')
    item_name_arr.append('conv2.bias')
    # 全连接层有三层
    num_model_layer =  3
    for i in range(num_model_layer):
        item_name_arr.append('fc{}.weight'.format(i+1))
    for i in range(num_model_layer):
        item_name_arr.append('fc{}.bias'.format(i+1))
 
    # 赋值net_g参数,这样才能更新修改net_g的参数，因为net_g.state_dict()返回的是一个深copy的副本
    model_dict  = net_g.state_dict()
    
    # 进行参数平均
    for i in range(len(item_name_arr)):
        cur_name = item_name_arr[i]
    #    print('net_g avg before')
    #    test_before = model_dict[cur_name]
    #    print(test_before)
        
        for j in range(len(cur_all_client)):
    #        print(j)
            net_cur_ele = cur_all_client[j]
    #        print(net_cur_ele.state_dict()[cur_name])            
            if j == 0 :
                sum_item = net_cur_ele.state_dict()[cur_name]
            else:
                sum_item = sum_item + net_cur_ele.state_dict()[cur_name]
        # 参数平均
        model_dict[cur_name] = sum_item/(num_client*1.0)

    # 更新聚合的net_g
    net_g.load_state_dict(model_dict)
    print('##### finish global_model_once_in_g_count_{}'.format(g_epoch)) 
    
    ## 保存每次聚合的global model
    torch.save(net_g, '../model/cifar10/global_model_g_count_{}.pkl'.format(g_epoch))  # save entire net
    # torch.save(net_g.state_dict(),'../model/cifar10/global_model_g_count_{}_params.pkl'.format(g_epoch))#网络的参数          
          
    # 测试一下此时聚合模型的测试准确度
    # 转为torch.tensor的格式
    x_test_g = np.array(x_test_all)
    x_test = torch.FloatTensor(x_test_g)
    y_test_g = np.array(y_test_all)
    y_test = torch.LongTensor(y_test_g)   
    
    out_test = net_g(x_test)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y_test.data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    print("################# test_result_net_g_model_in_the_g_count_{} = ".format(g_epoch),accuracy)
#    print(net_g.state_dict())


        
        
    
    