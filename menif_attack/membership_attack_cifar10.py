# -*- coding: utf-8 -*-

### 本代码是实现梯度成员推测攻击的入口(在相同的数据集下，对保护前后的全局模型进行攻击测试)
import member_inf_cifar10
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import attack_data_cifar10
import numpy as np
import random

## Model to train attack model on. Should be same as the one trained.
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

## 载入总体的训练集和测试集
x_train_all = np.load('../dataset/cifar10/x_train_cifar10_all.npy')
y_train_all = np.load('../dataset/cifar10/y_train_cifar10_all.npy')

x_test_all = np.load('../dataset/cifar10/x_test_cifar10_all.npy')
y_test_all = np.load('../dataset/cifar10/y_test_cifar10_all.npy')

# 生成一个随机整数，直接进行采样
sample_size = 2000
sample_start = random.randint(0,27000)
sample_end = sample_start + sample_size

# 划分回来，并保存用于攻击测试的数据
x_train_attack = x_train_all[sample_start:sample_end]
y_train_attack = y_train_all[sample_start:sample_end]
x_test_attack = x_test_all[sample_start:sample_end]
y_test_attack = y_test_all[sample_start:sample_end]

np.save('../dataset/cifar10/x_train_cifar10_attack.npy',x_train_attack)
np.save('../dataset/cifar10/y_train_cifar10_attack.npy',y_train_attack)
np.save('../dataset/cifar10/x_test_cifar10_attack.npy',x_test_attack)
np.save('../dataset/cifar10/y_test_cifar10_attack.npy',y_test_attack)

# 载入未保护的目标模型cmodelA(最终的聚合模型）
cprefixA = 'global_model_g_count_9.pkl'
cmodelA = torch.load('../model/cifar10/{}'.format(cprefixA))

# 载入用于攻击测试的 训练集数据和测试数据,分别载入特征和标签，后续再使用TensorDataset去做
train_feature_path = '../dataset/cifar10/x_train_cifar10_{}.npy'.format('attack')
train_lable_path = '../dataset/cifar10/y_train_cifar10_{}.npy'.format('attack')
test_feature_path = '../dataset/cifar10/x_test_cifar10_{}.npy'.format('attack')
test_lable_path = '../dataset/cifar10/y_test_cifar10_{}.npy'.format('attack')

# 攻击保护前后的聚合模型的结果存放在这个数组里
save_list = []

# 输入特征数量
input_features = (32,32)

# 创建攻击模型需要处理的数据
datahandleA = attack_data_cifar10.attack_data(train_feature_path,
                                      train_lable_path,
                                      test_feature_path,
                                      test_lable_path,
                                      batch_size=64,
                                      attack_percentage=50,
                                      input_shape=(input_features,))

## 初始化成员推测攻击实例 gradients_to_exploit 泄露的层数与攻击层数相同
attackobj = member_inf_cifar10.initialize(
        target_train_model = cmodelA,
        target_attack_model = cmodelA,
        train_datahandler = datahandleA,
        attack_datahandler = datahandleA,
#        layers_to_exploit=[],
        # 这里的层数从1开始，，1,2,3,4,5,后面三层是全连接
        gradients_to_exploit=[3,4,5],
#        learning_rate=0.0000000003,
#        optimizer='SGD',
        learning_rate=0.001,
        optimizer='SGD',
        epochs=100,
        model_name='cifar10_dataset_3x2y_classification'
        )

## 实例功能模块测试
attackobj.train_attack()

# # 输出最后20个epoch的攻击结果
# print(attackobj.last20_epochs)

# 选择最后20个epoch的攻击结果的最大值为未保护的聚合模型的攻击准确率
best_attack_accuracy = max(attackobj.last20_epochs)

## 记录该实验结果
str_save = str('model {} before protecting: {} '.format(cprefixA, best_attack_accuracy))
save_list.append(str_save)

# 上述结果对应的下标索引加79即为 后续测试保护模型的攻击准确率的最佳epoch
best_attack_epoch = attackobj.last20_epochs.index(best_attack_accuracy) + (100 - 1 - 20)


######### 已保护的目标模型测试 ############
# 载入已保护的目标模型cmodelB(最终的聚合模型）
cprefixB = 'protected_global_model_g_count_9_layers345.pkl'
cmodelB = torch.load('../cifar10/cifar10_defend/global/{}'.format(cprefixB))

## 初始化成员推测攻击实例 gradients_to_exploit 泄露的层数与攻击层数相同
protected_attackobj = member_inf_cifar10.initialize(
        target_train_model = cmodelB,
        target_attack_model = cmodelB,
        # 使用相同的数据块
        train_datahandler = datahandleA,
        attack_datahandler = datahandleA,
#        layers_to_exploit=[],
        # 这里的层数从1开始，，1,2,3,4,5,后面三层是全连接
        gradients_to_exploit=[3,4,5],
        learning_rate=0.001,
        optimizer='SGD',
        epochs=best_attack_epoch,
        model_name='cifar10_dataset_3x10y_classification'
        )
## 实例功能模块测试
protected_attackobj.train_attack()
# 输出保存最后一个结果，为已保护的聚合模型的攻击准确率
protected_attack_accuracy = protected_attackobj.last20_epochs[-1]

## 记录该实验结果
str_save = str('model{} after protecting: {} '.format(cprefixB,protected_attack_accuracy))
save_list.append(str_save)


# 将保存结果的数组转换为字符串并进行保存
info_save = '\n'.join(save_list)
with open('test_info_save.txt', 'w') as f:     # 打开test.txt   如果文件不存在，创建该文件。
    f.write(info_save) #str_save2一定要转为 字符串




