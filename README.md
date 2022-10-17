# 主要目录结构说明

|-- experiments_cifar10（根目录）
    |-- data_Partitioning_cifar10.py (cifar10数据集根据client个数划分)
    |-- README.md（所有实验、流程等说明）
    |-- requirements.txt（代码运行依赖的第三方python包）
    |-- dataset（存储cifar10数据集、划分后的数据集）

​    |   |-- cifar10

​    |-- defend（放置我们梯度保护方案的代码）
​    |   |-- defend_cifar10.py（每个client进行梯度保护的代码）
​    |   |-- defend_cifar10_aggragator.py（将保护后的client进行聚合的代码）
​    |-- FL_model_generation（放置生成clientmodel和全局聚合model的代码）
​    |   |-- FL_model_for_cifar10.py（联邦学习的训练过程的代码）
​    |-- menif_attack（放置基于梯度泄露的成员推测攻击的代码）
​    |   |-- attack_data_cifar10.py
​    |   |-- attack_model_init.py
​    |   |-- attack_utils_cifar10.py
​    |   |-- create_cnn.py
​    |   |-- create_encoder.py
​    |   |-- create_fcn.py
​    |   |-- logger.py
​    |   |-- losses_cifar10.py
​    |   |-- membership_attack_cifar10.py（攻击文件入口）
​    |   |-- membership_attack_cifar10_withDP.py（DP保护方案在该攻击的防御效果的实验代码）
​    |   |-- member_inf_cifar10.py
​    |   |-- optimizers.py
​    |   |-- test_info_save.txt
​    |   |-- logs

​    |-- model(用于放置生成的模型（client模型、全局聚合模型和保护后的模型）)
​    |   |-- cifar10 (放置原始生成的client模型、全局聚合模型，具体模型类型看本文档下面的后缀说明)

​    |   |-- cifar10_defend (放置保护的client模型、全局聚合模型，具体模型类型看本文档下面的后缀说明)

​    |       |-- global（专门放置保护后的全局聚合模型）

​    |-- other_attack_and_defend（放置shokri攻击和ML-LEAKS攻击，并测试未保护的方案、本文梯度保护的方案、DP防御方案、top3和top1防御方案） 
​        |-- ML_leaks_attack_cifar10.py
​        |-- shokri_attack_cifar10.py

# Anaconda创建新环境运行实验代码

创建python环境，conda create --name wuwu python=3.7.10; （wuwu为自定义的用户名）

切换到wuwu环境（用户名环境），conda activate wuwu;

安装spyder（在Anaconda界面）; 

安装torchvison(0.9.1)和torch(1.8.1)，安装scikit-learn(0.24.2)，安装pandas(1.2.4)，安装numpy(1.19.2)。

# 主要执行代码说明

## 步骤0、首先，根据参与方个数，对数据集进行等量划分

### data_Partitioning_cifar10.py

​cifar10数据集，把原本默认划分的50K训练数据和10K的测试数据进行合并，为60K的数据集，然后划分（并保存）一半用于训练集（30K），一半用于测试集（30K）。接着，把训练集和测试集，平均分给10个client，每个client3K的训练数据和3K的测试数据。

保存路径在dataset/cifar10/x1_cifar10_x2，其中x1为训练或测试数据的特征与标签，x2为0-9对应的client编号或all为整个35K训练集和35K测试集，并把划分好的数据存储在./dataset/cifar10文件夹中。（由于是多维数据，不能保存为.csv文件，保存为了.npy文件）

## 步骤1、生成联邦学习的最终总体模型

### ./FL_model_generation/FL_model_for_cifar10.py 

​针对cifar10数据集，生成并保存联邦学习中clientmodel和全局model，并把原始的模型保存在 ./model/cifar10文件夹中。		

​流程：载入数据，定义网络，设定client个数，全局迭代轮次与本地迭代轮次，模型结构的层数，利用两个for循环实现联邦学习的训练过程。过程中有设置输出每一轮全局迭代的clientmodel的预测性能与全局model的预测性能。

## 步骤2、用攻击代码攻击原始最终总体模型

### ./menif_attack/membership_attack_cifar10.py

​针对cifar10数据集和全局模型，进行成员推测攻击，输入最后聚合的模型（不管是保护还是未保护，或者是不同保护参数设置的保护），输出一个攻击准确率。

​流程：该文件是攻击的入口文件，首先定义网络，输入总体的训练集和测试集，在其中分别进行随机采样N条，载入训练集和测试集的特征和标签的路径，创建攻击模型需要处理的数据，初始化成员推测攻击实例，最后进行实例功能模块测试，输出成员推测攻击的攻击准确率。		

​具体步骤说明：

​步骤1：首先，对于原始的攻击目标模型（未保护的聚合模型），运行白盒成员推测攻击时，先载入未保护的聚合模型，载入用于训练和测试白盒攻击模型的数据（例如随机采样2K条训练数据和2K条测试数据），基于上述数据集构建数据块（用于载入攻击模型），接着创建成员推测攻击模型实例（指定攻击层数和攻击模型训练的轮次），然后进行攻击测试。

​步骤2：在攻击测试的过程中，查看最后20个epoch对应的攻击准确率，保存起来，并以这20epoch中攻击成功率最大的结果为最佳攻击成功率，并记录此时epoch为最佳攻击epoch。

​步骤3：接着，攻击已保护的聚合模型。首先载入以保护的聚合模型，在与步骤1相同的数据块下（保证测试和训练的数据集不变），创建攻击模型实例（指定与步骤1相同的攻击层数、学习率、优化器、攻击模型结构），但此时攻击模型训练的轮次设置为步骤2计算得到的最佳攻击epoch，最后进行攻击测试，输出在这个最佳攻击epoch对应的攻击准确率，记录这个结果为已保护的聚合模型下的攻击准确率。

## 步骤3、对联邦学习中的每个参与方进行梯度保护，从而聚合得到一个保护后的最终总体模型

### ./defend/defend_cifar10.py

​这是用于保护单个clientmodel的代码，最后保护梯度，重新训练得到并保存新的梯度保护后的client model，保存在./model/cifar10_defend。

​流程：先定义网络，选择需要保护的client编号和设置扰动强度，载入数据，冻结网络层数（只训练改变需要保护的层数），找到使模型梯度角偏差最大的预测标签并进行one-hot编码，接着在原始预测概率向量和梯度角偏差最大的one-hot编码之间使用二分查找，找到满足约束（扰动强度约束和预测标签不变性）的预测概率向量，基于该预测概率向量重新训练本地模型，输出并保存梯度保护后的client model。

​注意：这里可以根据不同实验需求，修改算法参数，例如扰动强度，保护的层数等，生成不同需求保护的client模型，并用于后续的全局模型聚合。

### ./defend/defend_cifar10_aggragator.py

​用于聚合defend_cifar10.py 生成保护梯度的client，输出并保存得到保护后的global model，保存在./model/cifar10_defend/global。

​流程：按照实验需求载入所有本地模型，做一轮全局聚合，输出并保存该保护后的全局模型。用于后续的攻击测试。			

## 步骤4、用攻击代码攻击保护后的最终总体模型

​与步骤2类似，只是此时的攻击的目标模型设置为步骤3得到的保护后的最终总体模型。

# 其他说明

## 关于实验保护的模型后缀名说明

对于cifar10数据集：

默认下是10分类，10个全局迭代轮次训练，在第10轮次聚合，扰动强度epsilon=0.1，保护10个client后才聚合为一个全局模型。

client_x1_g_count_x1.pkl，是普通未保护下的client模型，其中x1为client编号（0-9），x2为第几轮全局轮次（0-9）。

global_model_g_count_x1.pkl，是普通未保护下的global模型，其中x1为第几轮全局轮次（0-9）。

protected_client_x1_g_count_x2_layersx3.pkl，是保护后的client模型，其中x1为client编号（0-9），x2为第几轮全局轮次（0-9），x3为保护的层次和层数（1-5，可单层或多层）。

类似地，protected_client_x1_g_count_x2_y3,pkl,y3是保护算法修改参数的类型，除了layers后缀，还有其他后缀：epsilonx，x代表扰动强度；classx，x代表输出类别数量；clientNumx，x代表保护client个数;gIterationx，x代表保护第几轮的聚合（特别地，unprotect_gIterationx，x代表原始的第几轮的聚合）。

## ./other_attack_and_defend/shokri_attack_cifar10.py

该代码用于实现首个成员推测攻击（shokri_attack），并测试本文涉及的四个防御方案的性能。

流程：取部分数据集并将其换分为目标数据集（看作member）、影子数据集和测试数据集（看作non_member)；然后将影子数据集划分10个，然后用于训练10个shadow model；测试数据集也划分10个；然后将等数量的影子子集和测试子集输入到shadow model，得到概率输出（看做是attack model的特征输入，并标签化为1（member）和0（non-member）），并用于训练9个attack model；最后根据不同保护方案（target model）的输出概率向量进行去前三高的概率值作为输入特征进行攻击（多个attack model 进行投票表决，确定最终的结果），输出概率结果。

注意：attack模型训练的过程中，可能需要一个一个调整，使其不要过拟合。

类似地，./other_attack_and_defend/ML_leaks_attack_cifar10.py ML-leaks攻击类似shokri_attack，但只用了一个影子数据集和训练一个攻击模型，同样实现了黑盒的成员推测攻击。



# 函数输入输出变量说明

已写在代码里。



# Selectable-One-shot-Method-for-Privacy-Enhancement
