# -*- coding: utf-8 -*-

# 本代码用于将梯度保护后得所有client聚合，生成聚合好的最终总体模型，用于攻击测试

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# 网络结构
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
#        self.layer_output_list.append(x)
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

# 创建初始网络模型
net_g = Net()

# client 数量
num_client = 10

# 之前设定的全局迭代轮次
g_iteration_count = 10

# 字符串标识（用于存储模型名称的后缀）：需要保护的层数layersx，参与聚合前受保护的client数量clientNumx，不同扰动强度epsilonx等
str_name = 'layers{}'.format('345')

# 存储所有client 的数组
cur_all_client = []

##### 根据需求载入本地模型
# 导入所有的client model(或者选择n个保护的本地模型和10-n个未保护的模型进行聚合)
for client_i in range(num_client):    
    # 载入已保护的client
    cur_net = torch.load('../model/cifar10_defend/protected_client_{}_g_count_{}_{}.pkl'.format(client_i, g_iteration_count-1,str_name))
    cur_all_client.append(cur_net)

# 参数平均聚合,更新net_g         
# 根据模型层数，确定需要聚合的每层参数的个数        
item_name_arr = []
# 前面的卷积层
item_name_arr.append('conv1.weight')
item_name_arr.append('conv1.bias')
item_name_arr.append('conv2.weight')
item_name_arr.append('conv2.bias')
num_model_layer = 3
for i in range(num_model_layer):
    item_name_arr.append('fc{}.weight'.format(i+1))
for i in range(num_model_layer):
    item_name_arr.append('fc{}.bias'.format(i+1))
 
# 赋值net_g参数,这样才能更新修改net_g的参数，因为net_g.state_dict()返回的是一个深copy的副本
model_dict  = net_g.state_dict()

# 进行参数平均
for i in range(len(item_name_arr)):
    cur_name = item_name_arr[i]
   
    for j in range(len(cur_all_client)):
        net_cur_ele = cur_all_client[j]        
        if j == 0 :
            sum_item = net_cur_ele.state_dict()[cur_name]
        else:
            sum_item = sum_item + net_cur_ele.state_dict()[cur_name]
    # 参数平均
    model_dict[cur_name] = sum_item/(num_client*1.0)

# 更新聚合的net_g
net_g.load_state_dict(model_dict)
print('##### finish and save protected global_model_once_in_g_count_{}'.format(g_iteration_count-1)) 
## 保存每次聚合的global model
torch.save(net_g, '../model/cifar10_defend/global/protected_global_model_g_count_{}_{}.pkl'.format(g_iteration_count-1,str_name))  # save entire net
# torch.save(net_g.state_dict(),'../model/cifar10_defend/global/protected_global_model_g_count_{}_params_{}.pkl'.format(g_iteration_count-1,str_name))#网络的参数          
    
# 测试一下此时聚合模型的测试准确度
#载入当前global model的数据集
x_test_all = np.load('../dataset/cifar10/x_test_cifar10_{}.npy'.format('all'))
y_test_all = np.load('../dataset/cifar10/y_test_cifar10_{}.npy'.format('all'))
# 转为torch.tensor的格式
x_test_g = np.array(x_test_all)
x_test = torch.FloatTensor(x_test_g)
y_test_g = np.array(y_test_all)
y_test = torch.LongTensor(y_test_g).squeeze()    

out_test = net_g(x_test)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("################# test_result_protected_net_g_model = ",accuracy)      