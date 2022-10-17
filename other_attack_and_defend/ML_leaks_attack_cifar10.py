# -*- coding: utf-8 -*-

# 本代码用于实现首个成员推测攻击（ML-leaks_attack），并测试本文涉及的四个防御方案的性能

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn import metrics
import random
from torch.optim import lr_scheduler

#################

# 载入总的训练集和测试集
x_train_all = np.load('../dataset/cifar10/x_train_cifar10_all.npy')
y_train_all = np.load('../dataset/cifar10/y_train_cifar10_all.npy')

x_test_all = np.load('../dataset/cifar10/x_test_cifar10_all.npy')
y_test_all = np.load('../dataset/cifar10/y_test_cifar10_all.npy')


# 分配数据，总共使用6*data_size条数据，三分之一为D_target, 三分之一为D_shadow（一半为train另一半为out，用于训练attackmodel），三分之一为D_test
# 其中，D_target为member,D_test为non_member，用于测试攻击性能。

data_size = 5000
# 为了快速引入保护方案的target model，这里的范围修改，但范围长度不修改
# D_target
x_target_dataset = x_train_all[:2*data_size]
y_target_dataset = y_train_all[:2*data_size]

# D_shadow
x_shadow_dataset = x_train_all[2*data_size:4*data_size]
y_shadow_dataset = y_train_all[2*data_size:4*data_size]
# 转成Tensor形式
x_shadow_dataset_arr = np.array(x_shadow_dataset)
x_shadow_all = torch.FloatTensor(x_shadow_dataset_arr)

y_shadow_dataset_arr = np.array(y_shadow_dataset)
y_shadow_all = torch.LongTensor(y_shadow_dataset_arr).squeeze()

# D_test
x_test_dataset = x_test_all[:2*data_size]
y_test_dataset = y_test_all[:2*data_size]
# 转成Tensor形式
x_test_dataset_arr = np.array(x_test_dataset)
x_test_dataset = torch.FloatTensor(x_test_dataset_arr)

y_shadow_dataset_arr = np.array(y_test_dataset)
y_test_dataset = torch.LongTensor(y_shadow_dataset_arr).squeeze()


# 划分D_shadow 为train部分和out部分
len_shadow_train = int(len(x_shadow_all)/2)

x_shadow_train  = x_shadow_all[0:len_shadow_train]
y_shadow_train = y_shadow_all[0:len_shadow_train]

x_shadow_out = x_shadow_all[len_shadow_train:]
y_shadow_out = y_shadow_all[len_shadow_train:]

##################
# 训练 Shadow model

# 目标模型结构，shadow model 可以使用，在原文中这个其实也是解除这个约束，但使用中我们先使用
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

net_shadow = Net()

# 用D_shadow_train 训练shadow model
# 进行batch划分
torch_dataset = torch.utils.data.TensorDataset(x_shadow_train, y_shadow_train)
train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=128, 
                                          shuffle=False)

loss_func = torch.nn.CrossEntropyLoss()
la_opt = optim.Adam(params=net_shadow.parameters(),lr=0.006)
# scheduler = lr_scheduler.LambdaLR(optimizer=la_opt, lr_lambda=lambda epoch:0.95**epoch)
# 迭代次数
iteration_count = 35


len_data_len_shadow_train = len(x_shadow_train)

for epoch in range(iteration_count):
    for step, data in enumerate(train_loader, start=0):
        x_a, y_a = data # 解构出特征和标签
        out_a = net_shadow(x_a)     # input x and predict based on x
        loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
#        print(loss_a)
        la_opt.zero_grad()   # clear gradients for next train
        loss_a.backward()         # backpropagation, compute gradients
        la_opt.step()        # apply gradients

        if step % (int(len_data_len_shadow_train/128)) == (int(len_data_len_shadow_train/128)) - 1: 
            prediction_a = torch.max(out_a, 1)[1]#只返回最大值的每个索引,标签
            pred_a = prediction_a.data.numpy()
            target_a = y_a.data.numpy()
            accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
            print("epoch={}, train_result_shadow_model = {}".format(epoch, accuracy_a))

################
# 训练Attack model(一个)           
            
# x_shadow_train 为member,x_shadow_out为non_member            
x_member_train = x_shadow_train
y_member = torch.LongTensor(torch.ones_like(y_shadow_train))

x_nonmember_train = x_shadow_out
y_nonmember = torch.LongTensor(torch.zeros_like(y_shadow_out))

# 先经过shadow model，经过softmax转为概率模式，然后取前三个并且进行排序
x_member_all = F.softmax(net_shadow(x_member_train),dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_member = torch.tensor(x_member)

x_nonmember_all = F.softmax(net_shadow(x_nonmember_train),dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_nonmember = torch.tensor(x_nonmember)

# 整合训练attack model的数据集
x_attack_train = torch.cat((x_member,x_nonmember),0) 
y_attack_train = torch.cat((y_member,y_nonmember),0) 

# 定义二分类器
class Net_two_class(torch.nn.Module):
    def __init__(self, n_feature=3, n_hidden=6, n_output=2):
        super(Net_two_class, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

# 取前3最大概率作为input feature vector
net_attack_model = Net_two_class(n_feature=3,n_hidden=5,n_output=2)

# 进行batch划分
torch_dataset = torch.utils.data.TensorDataset(x_attack_train, y_attack_train)
train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=128, 
                                          shuffle=False)

loss_func = torch.nn.CrossEntropyLoss()
m_attack_opt = optim.SGD(params=net_attack_model.parameters(),lr=0.0035)
scheduler = lr_scheduler.LambdaLR(optimizer=m_attack_opt, lr_lambda=lambda epoch:0.95**epoch)

# 迭代次数
iteration_count = 5

len_data_len_attack_train = len(x_attack_train)

## 需要调整好攻击模型的训练准确率，不能过拟合 ###
for epoch in range(iteration_count):
    for step, data in enumerate(train_loader, start=0):
        x_a, y_a = data # 解构出特征和标签
        out_a = net_attack_model(x_a)     # input x and predict based on x
        loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
#        print(loss_a)
        m_attack_opt.zero_grad()   # clear gradients for next train
        loss_a.backward()         # backpropagation, compute gradients
        m_attack_opt.step()        # apply gradients

        # if step % (int(len_data_len_attack_train/256)) == (int(len_data_len_attack_train/256)) - 1: 
        if step != 0 :
            prediction_a = torch.max(out_a, 1)[1]#只返回最大值的每个索引,标签
            pred_a = prediction_a.data.numpy()
            target_a = y_a.data.numpy()
            accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
            print("epoch={},step={} train_result_attack_model = {}".format(epoch,step, accuracy_a))
    scheduler.step() 

###############################       
### 下面进行模型测试 ####   start#####
################################

################################
# 1、载入 未保护的 target model  ############## 
# 用D_target输入target model，用于被黑盒访问
# 转为Tensor
x_target_train_arr = np.array(x_target_dataset)
x_target_train = torch.FloatTensor(x_target_train_arr)
y_target_dataset_arr = np.array(y_target_dataset)
y_target_train = torch.LongTensor(y_target_dataset_arr).squeeze()

net_target = torch.load('../model/cifar10/global_model_g_count_9.pkl')    

# 非成员数据，y标签为0
y_org_nonmember =  torch.LongTensor(y_test_dataset)
y_nonmember_test = torch.zeros_like(y_org_nonmember)

# 成员数据，y标签为1
y_org_member = y_target_train
y_member_test = torch.ones_like(y_org_member)

# 输入到target model的input feature
x_test_member = x_target_train
x_test_nonmember = x_test_dataset

# 经过target model，取其输出概率前三，大到小
x_member_all = F.softmax(net_target(x_test_member),dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_member_test = torch.tensor(x_member)

# 经过target model，取其输出概率前三，大到小
x_nonmember_all = F.softmax(net_target(x_test_nonmember),dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)
         
# 整合数据集，用于输入到attack model的inputfeature，和对应的真实的成员性标签
x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)


# 测试攻击模型性能
out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about un-protect result:\n',classify_report)

#############################
# 2、载入 DP方案 保护的 target model  ############## 
# 1、+DP
# 定义DP噪声
epsilon = 0.3
mu = 0
sigma = 1/epsilon

# 经过target model，取其输出概率前三，大到小
pred = net_target(x_test_nonmember) 
[row, col] = pred.size()
added_noise = []
for i in range(row):
    cur_row_noise = []
    for j in range(col):
        cur_row_noise.append(random.gauss(mu,sigma))
    added_noise.append(cur_row_noise)
added_noise = torch.tensor(added_noise)
pred = pred + added_noise

x_nonmember_all = F.softmax(pred,dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)

# 经过target model，取其输出概率前三，大到小
pred = net_target(x_test_member)
[row, col] = pred.size()
added_noise = []
for i in range(row):
    cur_row_noise = []
    for j in range(col):
        cur_row_noise.append(random.gauss(mu,sigma))
    added_noise.append(cur_row_noise)
added_noise = torch.tensor(added_noise)
pred = pred + added_noise

x_member_all = F.softmax(pred,dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_member_test = torch.tensor(x_member)

# 整合数据集
x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)

# 测试攻击模型性能
out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about DP-scheme result:\n',classify_report)

#############################
# 3、载入 top3方案 保护的 target model  ############## 
# 经过target model，取其输出概率前三，大到小
pred = net_target(x_test_nonmember)

x_nonmember_all = F.softmax(pred,dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)

# 经过target model，取其输出概率前三，大到小
pred = net_target(x_test_member)

x_member_all = F.softmax(pred,dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_member_test = torch.tensor(x_member)

# 整合数据集
x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)

# 测试攻击模型性能
out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about top3-scheme result:\n',classify_report)

#############################
# 4、载入 top1方案 保护的 target model  ############## 
pred = net_target(x_test_nonmember)

x_nonmember_all = F.softmax(pred,dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    cur_ele = (x_nonmember_all[i].sort(descending=True)).values[0:1].detach().numpy()
    p_arr = np.concatenate((cur_ele,[0])) # 先将p_变成list形式进行拼接，注意输入为一个tuple
    cur_ele = np.append(p_arr,0)
    # sad
    x_nonmember.append(cur_ele)
x_nonmember_test = torch.FloatTensor(x_nonmember)

# 经过target model，取其输出概率前1，大到小
pred = net_target(x_test_member)

x_member_all = F.softmax(pred,dim=1)
x_member = []
for i in range(len(x_member_all)):
    cur_ele = (x_member_all[i].sort(descending=True)).values[0:1].detach().numpy()
    # top1只有1个概率值，补充两个概率为0的元素，变成三个概率值形式
    p_arr = np.concatenate((cur_ele,[0])) # 先将p_变成list形式进行拼接，注意输入为一个tuple
    cur_ele = np.append(p_arr,0)
    
    x_member.append(cur_ele)
x_member_test = torch.FloatTensor(x_member)

# 整合数据集
x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)

# 测试攻击模型性能
#x_test_m_attack = torch.tensor(x_test_m_attack, dtype=torch.float32)
out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about top1-scheme result:\n',classify_report)

#############################
# 5、载入 本文梯度保护方案 保护的 target model  ############## 
# 用D_target输入target model，用于被黑盒访问
# 转为Tensor
x_target_train_arr = np.array(x_target_dataset)
x_target_train = torch.FloatTensor(x_target_train_arr)
y_target_dataset_arr = np.array(y_target_dataset)
y_target_train = torch.LongTensor(y_target_dataset_arr).squeeze()

# 载入梯度保护方案的训练好的模型
net_target = torch.load('../model/cifar10_defend/global/protected_global_model_g_count_9_layers345.pkl')    

# 非成员数据，y标签为0
y_org_nonmember =  torch.LongTensor(y_test_dataset)
y_nonmember_test = torch.zeros_like(y_org_nonmember)

# 成员数据，y标签为1
y_org_member = y_target_train
y_member_test = torch.ones_like(y_org_member)

# 输入到target model的input feature
x_test_member = x_target_train
x_test_nonmember = x_test_dataset

# 经过target model，取其输出概率前三，大到小
x_member_all = F.softmax(net_target(x_test_member),dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_member_test = torch.tensor(x_member)

# 经过target model，取其输出概率前三，大到小
x_nonmember_all = F.softmax(net_target(x_test_nonmember),dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:3].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)
         
# 整合数据集，用于输入到attack model的inputfeature，和对应的真实的成员性标签
x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)

# 测试攻击模型性能
out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about ours-scheme result:\n',classify_report)

########################################### end #######