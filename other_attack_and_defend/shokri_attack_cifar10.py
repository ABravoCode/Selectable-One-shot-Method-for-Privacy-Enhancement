# -*- coding: utf-8 -*-

# 本代码用于实现首个成员推测攻击（shokri_attack），并测试本文涉及的四个防御方案的性能

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

data_size = 2000
# 为了快速引入保护方案的target model，这里的范围修改，但范围长度不修改
# D_target
x_target_dataset = x_train_all[:2*data_size]
y_target_dataset = y_train_all[:2*data_size]

# D_shadow
x_shadow_dataset = x_train_all[4*data_size:6*data_size]
y_shadow_dataset = y_train_all[4*data_size:6*data_size]
# 转成Tensor形式
x_shadow_dataset_arr = np.array(x_shadow_dataset)
x_shadow_dataset = torch.FloatTensor(x_shadow_dataset_arr)

y_shadow_dataset_arr = np.array(y_shadow_dataset)
y_shadow_dataset = torch.LongTensor(y_shadow_dataset_arr).squeeze()

# D_test
x_test_dataset = x_test_all[2*data_size:4*data_size]
y_test_dataset = y_test_all[2*data_size:4*data_size]
# 转成Tensor形式
x_test_dataset_arr = np.array(x_test_dataset)
x_test_dataset = torch.FloatTensor(x_test_dataset_arr)

y_shadow_dataset_arr = np.array(y_test_dataset)
y_test_dataset = torch.LongTensor(y_shadow_dataset_arr).squeeze()

################################################################################
# D_shadow 划分num_dataset个子数据集D_si（随机采样生成，存在overlap），D_test也划分为num_dataset个D_ti.

# D_si的生成,每个data_size长度
d_si_x_set = []
d_si_y_set = []

# D_ti的生成,每个data_size长度
d_ti_x_set = []
d_ti_y_set = []

num_dataset = 9
d_data_size = data_size

for i in range(num_dataset):    
    cur_start = random.randint(0,d_data_size-1)
    cur_end = cur_start + d_data_size
    # D_si
    cur_sub_shadow_x =  x_shadow_dataset[cur_start:cur_end]
    cur_sub_shadow_y =  y_shadow_dataset[cur_start:cur_end]
    d_si_x_set.append(cur_sub_shadow_x)
    d_si_y_set.append(cur_sub_shadow_y)
    
    # D_ti
    cur_sub_test_x =  x_test_dataset[cur_start:cur_end]
    cur_sub_test_y =  y_test_dataset[cur_start:cur_end]
    d_ti_x_set.append(cur_sub_test_x)
    d_ti_y_set.append(cur_sub_test_y)

##################################
# D_si去训练shadow model（一共num_dataset个）
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


net_shadow0 = Net()
net_shadow1 = Net()
net_shadow2 = Net()
net_shadow3 = Net()
net_shadow4 = Net()
net_shadow5 = Net()
net_shadow6 = Net()
net_shadow7 = Net()
net_shadow8 = Net()

net_arr = []
net_arr.append(net_shadow0)
net_arr.append(net_shadow1)
net_arr.append(net_shadow2)
net_arr.append(net_shadow3)
net_arr.append(net_shadow4)
net_arr.append(net_shadow5)
net_arr.append(net_shadow6)
net_arr.append(net_shadow7)
net_arr.append(net_shadow8)

# d_si_shadow 数据集的长度
len_data_len_shadow = len(d_si_x_set[0])

for i in range(num_dataset):
    cur_count_shadow = i
    net_shadow = net_arr[i]
    
    # 进行batch划分
    torch_dataset = torch.utils.data.TensorDataset(d_si_x_set[cur_count_shadow], d_si_y_set[cur_count_shadow])
    train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                               batch_size=128, 
                                              shuffle=False)
    loss_func = torch.nn.CrossEntropyLoss()
    la_opt = optim.Adam(params=net_shadow.parameters(),lr=0.005)
    # 迭代次数
    iteration_count = 20
    
    for epoch in range(iteration_count):
        for step, data in enumerate(train_loader, start=0):
            x_a, y_a = data # 解构出特征和标签
            out_a = net_shadow(x_a)     # input x and predict based on x
            loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
    #        print(loss_a)
            la_opt.zero_grad()   # clear gradients for next train
            loss_a.backward()         # backpropagation, compute gradients
            la_opt.step()        # apply gradients
           
            if step % (int(len_data_len_shadow/128)) == (int(len_data_len_shadow/128)) - 1: #data_size/batch_size 为batch的个数
                prediction_a = torch.max(out_a, 1)[1]#只返回最大值的每个索引,标签
                pred_a = prediction_a.data.numpy()
                target_a = y_a.data.numpy()
                accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
                print("epoch={},step={}, train_result_shadow_model_{} = {}".format(epoch, step, cur_count_shadow, accuracy_a))

            
#############################################################################
# 等数量的D_s1和D_t1，经过shadow model得到输出（看作后续attack model的输入特征）
x_attack_train_set = []
y_attack_train_set = []

for i in range(num_dataset):
    
    cur_count_shadow = i
    net_shadow = net_arr[cur_count_shadow]
    
    # D_si为member,D_ti为nonmember          
    x_member_train = d_si_x_set[cur_count_shadow]
    y_member = torch.LongTensor(torch.ones_like(d_si_y_set[cur_count_shadow]))
    
    x_nonmember_train = d_ti_x_set[cur_count_shadow]
    y_nonmember = torch.LongTensor(torch.zeros_like(d_ti_y_set[cur_count_shadow]))
    
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
    
    # 把数据整合到一个大的集合里，方便后面attack model使用
    x_attack_train_set.append(x_attack_train)
    y_attack_train_set.append(y_attack_train)

###############################################################################
# 使用attack model的输入特征和他们的y标签为成员1，和非成员0(合成一个大的数据集），利用这个数据集训练9个attack model

# 定义二分类器 attack model
class Net_two_class(torch.nn.Module):
    def __init__(self, n_feature=3, n_hidden=6, n_output=2):
        super(Net_two_class, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

# 取前3最大概率作为input feature vector，要9个
net_attack_model0 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model1 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model2 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model3 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model4 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model5 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model6 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model7 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)
net_attack_model8 = Net_two_class(n_feature=3,n_hidden=5,n_output=2)

net_attack_arr = []
net_attack_arr.append(net_attack_model0)
net_attack_arr.append(net_attack_model1)
net_attack_arr.append(net_attack_model2)
net_attack_arr.append(net_attack_model3)
net_attack_arr.append(net_attack_model4)
net_attack_arr.append(net_attack_model5)
net_attack_arr.append(net_attack_model6)
net_attack_arr.append(net_attack_model7)
net_attack_arr.append(net_attack_model8)

# 把多个（input feature，0|1）数据集整合为一个大的数据集
x_attack_train_all = []
y_attack_train_all = []

for i in range(num_dataset):
    x_attack_train_all.extend(x_attack_train_set[i].numpy())
    y_attack_train_all.extend(y_attack_train_set[i].numpy())

# 获取攻击数据集的长度
len_data_len = len(x_attack_train_all)
    
x_attack_train_all = torch.FloatTensor(x_attack_train_all)
y_attack_train_all = torch.LongTensor(y_attack_train_all)


# 训练num_dataset个attack model,可能需要一个一个训练attack model，才比较好控制训练准确率
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
    
    # 进行batch划分
    torch_dataset = torch.utils.data.TensorDataset(x_attack_train_all, y_attack_train_all)
    train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                               batch_size=256, 
                                              shuffle=False)    
    loss_func = torch.nn.CrossEntropyLoss()
    # m_attack_opt = optim.Adam(params=net_attack_model.parameters(),lr=0.0005) 
    m_attack_opt = optim.SGD(params=net_attack_model.parameters(),lr=0.0045)
    scheduler = lr_scheduler.LambdaLR(optimizer=m_attack_opt, lr_lambda=lambda epoch:0.95**epoch)
    # 迭代次数
    iteration_count = 10
    
    for epoch in range(iteration_count):
        for step, data in enumerate(train_loader, start=0):
            x_a, y_a = data # 解构出特征和标签
            out_a = net_attack_model(x_a)     # input x and predict based on x
            loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
            # print(loss_a)
            m_attack_opt.zero_grad()   # clear gradients for next train
            loss_a.backward()         # backpropagation, compute gradients
            m_attack_opt.step()        # apply gradients  
           
            # if step % (int(len_data_len/256)) == (int(len_data_len/256)) - 1: # #size(x_attack_train_all)/batch_size 为batch的个数
            if step != 0:
                prediction_a = torch.max(out_a, 1)[1]#只返回最大值的每个索引,标签
                pred_a = prediction_a.data.numpy()
                target_a = y_a.data.numpy()
                accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
                print("epoch={},step={}, train_result_attack_model_{} = {}".format(epoch, step,cur_count_attack, accuracy_a))
        scheduler.step() 
    # sad   
        
        
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

###################################################################################
# 成员推测时，数据先经过target model，得到输出概率特征，再把这个输出概率特征输入到9个attack model里面，根据多票表决决定成员性分类

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
# 经过9个attack model，进行voting
result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    # 把每个attack model预测的结果保存
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

# 在 count_arr_add 中，如果有半数大于5，则投票投为1，否则为0
count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
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
# 经过9个attack model，进行voting
result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    # 把每个attack model预测的结果保存
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

# 在 count_arr_add 中，如果有半数大于5，则投票投为1，否则为0
count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about DP_scheme result:\n',classify_report)

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
# 经过9个attack model，进行voting
result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    # 把每个attack model预测的结果保存
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

# 在 count_arr_add 中，如果有半数大于5，则投票投为1，否则为0
count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about top3_scheme result:\n',classify_report)


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
# 经过9个attack model，进行voting
result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    # 载入新训练的模型
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    # 把每个attack model预测的结果保存
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

# 在 count_arr_add 中，如果有半数大于5，则投票投为1，否则为0
count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about top1_scheme result:\n',classify_report)


#############################
# 5、载入 本文梯度保护方案 保护的 target model  ############## 
# 用D_target输入target model，用于被黑盒访问
# 转为Tensor
x_target_train_arr = np.array(x_target_dataset)
x_target_train = torch.FloatTensor(x_target_train_arr)
y_target_dataset_arr = np.array(y_target_dataset)
y_target_train = torch.LongTensor(y_target_dataset_arr).squeeze()

net_target = torch.load('../model/cifar10_defend/global/protected_global_model_g_count_9_layers345.pkl')    


###################################################################################
# 成员推测时，数据先经过target model，得到输出概率特征，再把这个输出概率特征输入到9个attack model里面，根据多票表决决定成员性分类

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
# 经过9个attack model，进行voting
result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    # 把每个attack model预测的结果保存
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

# 在 count_arr_add 中，如果有半数大于5，则投票投为1，否则为0
count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about ours-protect result:\n',classify_report)

########################################### end #######