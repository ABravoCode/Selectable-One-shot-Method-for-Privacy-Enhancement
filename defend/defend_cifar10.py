# -*- coding: utf-8 -*-

# 本代码用于输入每个client的原始本地模型，进行梯度保护，重新训练本地模型，在后续聚合代码再进行一次全局聚合，生成保护的最终总体模型
# 一次只对一个本地模型进行梯度保护，并保存模型


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy


class_num = 10

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

####################################################################
#### 一个一个训练，重训练10个，然后再聚合生成一个保护后的聚合模型
# client 个数        
num_client = 10
# 全局迭代轮次
g_iteration_count = 10

# client 编号 0,1,2```9
cur_num_client = 9
# 设置扰动强度eps=0.1
eps_setting = 0.1
# 字符串标识（用于存储模型名称的后缀）：需要保护的层数
str_name = 'layers{}'.format('345')

net_la =torch.load('../model/cifar10/client_{}_g_count_{}.pkl'.format(cur_num_client, g_iteration_count-1))

#载入当前client的数据集
x_train = np.load('../dataset/cifar10/x_train_cifar10_{}.npy'.format(cur_num_client))
y_train = np.load('../dataset/cifar10/y_train_cifar10_{}.npy'.format(cur_num_client))

x_test = np.load('../dataset/cifar10/x_test_cifar10_{}.npy'.format(cur_num_client))
y_test = np.load('../dataset/cifar10/y_test_cifar10_{}.npy'.format(cur_num_client))
 
# 转成Tensor形式
# x_train = np.array(x_train)
x_train = torch.FloatTensor(x_train)

# y_train = np.array(y_train)
y_train = torch.LongTensor(y_train)



#训练部分
x_a = x_train
# 变成一维
y_a = y_train.squeeze()

## 要训练的模型
net_w_b = copy.deepcopy(net_la)

# 网络初始化  
# 冻结层,冻结前面两层卷积层
net_w_b.conv1.weight.requires_grad = False
net_w_b.conv2.weight.requires_grad = False
net_w_b.fc1.weight.requires_grad = True
net_w_b.fc2.weight.requires_grad = True
net_w_b.fc3.weight.requires_grad = True

net_w_b.conv1.bias.requires_grad = False
net_w_b.conv2.bias.requires_grad = False
net_w_b.fc1.bias.requires_grad = True
net_w_b.fc2.bias.requires_grad = True
net_w_b.fc3.bias.requires_grad = True

    
# 只让requires_grad = True 的param进行参数更新
# 如果使用SGD，lr过大的话，会导致LOSS增大|不变，不收敛变小,所以这里使用Adam
net_w_b_opt = optim.Adam(filter(lambda p: p.requires_grad, net_w_b.parameters()), lr=0.0003)

# 保护前的测试准确率
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test)

y_test = np.array(y_test)
y_test = torch.LongTensor(y_test).squeeze() 
 
out_test = net_w_b(x_test)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print('test_result_cifar10_model before protection =', accuracy)



## 模型训练

# 保护的层数，0 1 2 4 5 
# 这里只能填最后一层，2.就算保护前面的层，也是通过最后一层的输出，然后计算概率，计算loss,反向传播去做。
num_protect_layer = 4

input_dataset = x_train
input_dataset_y = y_train
len_example = input_dataset.size()[0]

# 这里使用的是cifar10 10分类的数据集
num_classification = class_num
# 定义根据输出类别数num_output_classes，生成并输出one-hot编码的函数
def createOHE(num_output_classes):
        """
        creates one hot encoding matrix of all the vectors
        in a given range of 0 to number of output classes.
        """
        return F.one_hot(torch.arange(0, num_output_classes),
                         num_output_classes)
# 定义根据输出类别数num_output_classes，生成并输出分类硬标签的函数
def createLabel(num_output_classes):
    res = []
    for i in range(num_output_classes):
        res.append([i])
    return res
#生成one-hot编码列表
y_ext_list_prob = createOHE(num_classification)
y_ext_list_prob = y_ext_list_prob.float()

y_ext_list_label = torch.LongTensor(createLabel(num_classification))

# 直接计算梯度方向，根据LOSS函数，预测结果，与实际结果
# 由于使用了CrossEntropyLoss，已经做了logsoftmax，所以y_ext要变为lable形式，而不是概率形式
# 定义计算（输出）梯度方向的函数，example_input样本输入，net模型网络，protect_number_layer要保护的层数，y_ext_label标签对应的极端概率向量
def get_grad_direction(example_input, net, protect_number_layer, y_ext_label):
    # 最初的输入(模型的输入)
    model_input = example_input     
    # 模型的输出(logits)
    model_input = torch.unsqueeze(model_input,0)
    
    model_output = net(model_input)    
    # 使用CrossEntropyLoss函数
    loss_func = torch.nn.CrossEntropyLoss()    
    loss_a = loss_func(model_output, y_ext_label)
    
    cur_arr = []
    
    # 每次计算梯度前，将网络的梯度初始化为零
    net_w_b_opt.zero_grad()
    
    #torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track梯度
    with torch.no_grad():
        loss_a.backward()
        for name, parms in net.named_parameters():
#            print(parms.grad)      
            # 包含着weight和bias
            # 并且去除没有训练的层的梯度情况（没反向传播的层都是None)
            if parms.grad != None:
                cur_arr.append(parms.grad)  

#    print(cur_arr)        
    return copy.deepcopy(cur_arr)   


# 定义从原始预测输出org_y和最大梯度角对应的预测输出graddif_y之间，用二分法迭代获取最佳分割点。eps为扰动强度，seg_point_start查找起点，seg_point_end查找终点
def get_point_via_bisection(org_y, graddif_y, eps, seg_point_start, seg_point_end):
    # 分割点,从中间开始找
    seg_point = (seg_point_start + seg_point_end)/2
#    print(seg_point)
    
#   对应元素加起来
    cur_y = (1-seg_point)*org_y + seg_point*graddif_y
    
    # 当前元素最大值的下标（对应着分类的label)
    org_label = torch.nonzero(torch.eq(org_y, max(org_y))).squeeze(1)
    # 当遇到元素有相同的时候，取第一个
    if len(org_label) > 1:
        org_label = org_label[0]
    
    cur_label = torch.nonzero(torch.eq(cur_y, max(cur_y))).squeeze(1)
    # 当遇到元素有相同的时候，取第一个
    if len(cur_label) > 1:
        cur_label = cur_label[0]
    
    # 查找划分点的范围精度
    dif_stop = 1e-3

    # 分类标签的准确度约束，eps
    if torch.abs(torch.max(org_y) - torch.max(cur_y)) <= eps:
#        print('在eps范围内')
        # y~与y的最大概率差在eps范围内时，继续在里面搜寻
#        print(org_label,cur_label)
        if org_label != cur_label :
#            print('标签不同')
            # 标签不同，说明准确度没有保证，应该让y~接近org_y
            next_seg_start = seg_point_start
            next_seg_end = seg_point        
            seg_point = get_point_via_bisection(org_y, graddif_y, eps, next_seg_start,next_seg_end)
        else:
#            print('标签相同')
            # 当划分起始点与终点到达范围精度,并且标签相同,并且在eps内时，不需要再进一步搜寻，输出结果
            if seg_point_end - seg_point_start < dif_stop:
#                print('final seg_point')
#                print(seg_point)
                return seg_point            
            else:
                # 当标签相同时，应该进一步寻找最合适的seg_point，让y~更接近y*
                next_seg_start = seg_point
                next_seg_end = seg_point_end        
                seg_point = get_point_via_bisection(org_y, graddif_y, eps, next_seg_start,next_seg_end)
        
    else:
        # y~与y在eps之外，应该缩短距离
#        print('不在eps范围内')
        next_seg_start = seg_point_start
        next_seg_end = seg_point        
        seg_point = get_point_via_bisection(org_y, graddif_y, eps, next_seg_start,next_seg_end)
    
    return seg_point


# 在y(prob)和y*(ext_prob)之间计算获取合适的y~,保证分类准确性的约束。org_y_list原始预测输出列表，max_graddif_y_list最大梯度角偏差对应的预测输出列表，eps扰动强度
def get_modified_y_list(org_y_list,max_graddif_y_list,eps):
    len_data = len(org_y_list)
    modified_y_list = []
    
    for i in range(len_data):
        # 计算每个y到y*的划分点
        seg_point = get_point_via_bisection(org_y_list[i],max_graddif_y_list[i],eps,0,1)
        # 计算y~
        new_cur_y = (1-seg_point)*org_y_list[i] + seg_point*max_graddif_y_list[i]
#        print(i)
#        print(new_cur_y,org_y_list[i])
        modified_y_list.append(new_cur_y)
        
    return modified_y_list

## 找到要修改的y~_list,训练网络让输出y（prob）去靠近
# 找到所有最大梯度角偏差的概率形式的y*
y_ext_list = []
for i in range(len_example):
    # 原来真实的y
    example_y_org = input_dataset_y[i]
    example_y_org = torch.unsqueeze(example_y_org,0)
    # 取其W的梯度
    grad_dir_org = get_grad_direction(input_dataset[i],net_w_b, num_protect_layer,example_y_org)[0]
    
    # 比较的变量的初始化
    y_star_max = torch.tensor(-1)
    max_norm_square = torch.tensor(-1)
    
    for j in range(len(y_ext_list_label)):
        # 取其W的梯度
        grad_dir_ext = get_grad_direction(input_dataset[i],net_w_b, num_protect_layer,y_ext_list_label[j])[0]
        cur_dif = torch.sub(torch.div(grad_dir_ext,torch.norm(grad_dir_ext,p=2)), torch.div(grad_dir_org,torch.norm(grad_dir_org,p=2)))
        cur_norm_square = torch.pow(torch.norm(cur_dif,p=2),2)
#            print(cur_norm_square)
        if cur_norm_square > max_norm_square:
            max_norm_square = cur_norm_square
            y_star_max = y_ext_list_label[j]
#        print(max_norm_square)         
#        print(y_star_max)
    
    # 保存梯度角偏差最大的时候，对应的y*的lable
    # y_star_max

    # 转换为极限值的概率表达
    y_ext_list.append(torch.squeeze(y_ext_list_prob[y_star_max]))
#    print(y_ext_list)
    
# 利用插值法，在原始y与最大梯度角偏差的y*,找到使得分类准确度不变的最大距离的Y~
# 原始的y(logits)
net_w_b.layer_output_list = []
layer_output,_ =  net_la.eval_layer_output(x_train)
cur_layer_output = layer_output[num_protect_layer]
# 转换成概率形式的y(prob)
org_y_list_prob = F.softmax(cur_layer_output,dim=1)

modified_y_list = get_modified_y_list(org_y_list_prob, y_ext_list, eps_setting)
print('done modified_y_list') 
# 自定义损失函数
class MyLoss(nn.Module):     
    def __init__(self):
       super(MyLoss, self).__init__()
       
    def forward(self, org_y_prob_list, modified_y_list):
         sum_loss = torch.tensor(0,dtype=float,requires_grad=True)
         for i in range(len(org_y_prob_list)):
            cur_loss = torch.norm(torch.sub(org_y_prob_list[i],modified_y_list[i]),p=2)
            sum_loss = torch.add(sum_loss,cur_loss)
         return sum_loss

lossfunc = MyLoss()
# 迭代次数
iteration_count = 200
# batch 训练    
# list 再转为numpy array
for i in range(len(modified_y_list)):
    modified_y_list[i] = modified_y_list[i].detach().numpy()

modified_y_list = torch.tensor(modified_y_list)


# 进行batch划分
torch_dataset = torch.utils.data.TensorDataset(input_dataset, modified_y_list)
train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=128, 
                                          shuffle=False)



# 训练net_w_b
for epoch in range(iteration_count):  
    # loss不下降，可能恰好初始化到了一个不太容易下降的参数空间上，需要一些别的初始化（但本身网络就是已经训练好的model了）
    # 下面是一个trick,过程中，重新载入train_loader
     if epoch == 1:
         torch_dataset = torch.utils.data.TensorDataset(input_dataset, modified_y_list)
         train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=128, 
                                          shuffle=False)
    
     for step, data in enumerate(train_loader, start=0):
        input_dataset, modified_y_list = data # 解构出特征和标签
        
        output =  net_w_b(input_dataset)
        y_prob_list = F.softmax(output,dim=1)    
    
        ## 将y~(prob))与org_y(prob)去计算loss，来训练网络
        loss_prob = lossfunc(y_prob_list,modified_y_list)
        print(loss_prob)
        
        net_w_b_opt.zero_grad()   # clear gradients for next train
        loss_prob.backward(retain_graph=True)         # backpropagation, compute gradients
        net_w_b_opt.step()  
        
# 解除freeze，并保存该模型
net_w_b.conv1.weight.requires_grad = True
net_w_b.conv2.weight.requires_grad = True
net_w_b.fc1.weight.requires_grad = True
net_w_b.fc2.weight.requires_grad = True
net_w_b.fc3.weight.requires_grad = True

net_w_b.conv1.bias.requires_grad = True
net_w_b.conv2.bias.requires_grad = True
net_w_b.fc1.bias.requires_grad = True
net_w_b.fc2.bias.requires_grad = True
net_w_b.fc3.bias.requires_grad = True


#print(net_w_b.fc5.weight)  
torch.save(net_w_b, '../model/cifar10_defend/protected_client_{}_g_count_{}_{}.pkl'.format(cur_num_client, g_iteration_count-1,str_name)) # save entire net
# torch.save(net_w_b.state_dict(),'../model/cifar10_defend/protected_client_{}_g_count_{}_params_{}.pkl'.format(cur_num_client, g_iteration_count-1,str_name))#网络的参数   

 # 目标模型的测试集准确度
## 转成Tensor形式
#x_test = np.array(x_test)
#x_test = torch.FloatTensor(x_test)
#
#y_test = np.array(y_test)
#y_test = torch.LongTensor(y_test).squeeze() 
 
out_test = net_w_b(x_test)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print('test_result_cifar10_model after protection =', accuracy)



