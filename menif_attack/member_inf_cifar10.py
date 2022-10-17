# -*- coding: utf-8 -*-

import datetime

from logger import get_logger
from attack_utils_cifar10 import attack_utils, sanity_check
from create_fcn import fcn_module
from create_cnn import cnn_for_fcn_gradients
from create_encoder import encoder
import torch
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score
import numpy as np
from attack_model_init import attack_model_init
from losses_cifar10 import CrossEntropyLoss,CrossEntropyLoss_exampleloss,mse
import copy
from optimizers import optimizer_op
import os
import json
from torch.optim import lr_scheduler
import random

class initialize(object):
    # 定义默认参数
    def __init__(self,
                 target_train_model,
                 target_attack_model,
                 train_datahandler,
                 attack_datahandler,
                 optimizer="Adam",            
                 layers_to_exploit=None,
                 gradients_to_exploit=None,
                 exploit_loss=True,
                 exploit_label=True,
                 learning_rate=0.001,
                 epochs=100,
                 model_name='sample',
                 if_withDP = [False, 1, 1],
                 ):
        # Set self.loggers (directory according to todays date)
        time_stamp = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        self.attack_utils = attack_utils()
        self.logger = get_logger(self.attack_utils.root_dir, "attack",
                               "meminf", "info", time_stamp)  
        self.target_train_model = target_train_model
        self.target_attack_model = target_attack_model
        self.train_datahandler = train_datahandler
        self.attack_datahandler = attack_datahandler        
        self.optimizer = optimizer_op(optimizer)
        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.exploit_label = exploit_label
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_name = model_name
        self.output_size = int(len(list(target_train_model.parameters())[-1]))
        self.ohencoding = self.attack_utils.createOHE(self.output_size)
        self.last20_epochs = []
        self.if_withDP = if_withDP
        
        # Create input containers for attack & encoder model.
        self.create_input_containers()
        # torch框架下，将网络变为一层层，每层包含weight和bias
        self.layers = self.fix_layers()
        
        # basic sanity checks
        sanity_check(self.layers, layers_to_exploit)
        sanity_check(self.layers, gradients_to_exploit)
        
        # Create individual attack components
        self.create_attack_components(self.layers)
        
        # Initialize the attack model
        self.initialize_attack_model()

        # Log info
        self.log_info()
    
    def log_info(self):
        """
        Logs vital information pertaining to training the attack model.
        Log files will be stored in `/ml_privacy_meter/logs/attack_logs/`.
        """
        self.logger.info("`exploit_loss` set to: {}".format(self.exploit_loss))
        self.logger.info(
            "`exploit_label` set to: {}".format(self.exploit_label))
        self.logger.info("`layers_to_exploit` set to: {}".format(
            self.layers_to_exploit))
        self.logger.info("`gradients_to_exploit` set to: {}".format(
            self.gradients_to_exploit))
        self.logger.info("Number of Epochs: {}".format(self.epochs))
        self.logger.info("Learning Rate: {}".format(self.learning_rate))
        self.logger.info("Optimizer: {}".format(self.optimizer))       
    
    def create_input_containers(self):
        """
        Creates arrays for inputs to the attack and 
        encoder model. 
        (NOTE: Although the encoder is a part of the attack model, 
        two sets of containers are required for connecting 
        the TensorFlow graph).
        再torch中，只能先保存使用到的模块；当有数据输入到模型，才能得到输入输出，再存在xx_real中
        """        
        self.attackinputs = []
        self.encoderinputs = []
        self.encoderinputs_size = 0
        self.attackinputs_size = []

        
    def fix_layers(self):
        """
        把网络层参数分为一层层，一层有weight和bias（原来是直接输出，导致层数为两倍）
        """
        fix_result = []
        wb_params = list(self.target_train_model.parameters())
        for i in range(len(wb_params)):
            if i%2 == 0:
                fix_result.append((wb_params[i],wb_params[i+1]))
#        print(fix_result)    
#        print(len(fix_result))
        return fix_result
    
    # 构造攻击成分
    def create_attack_components(self, layers):
        """
        Creates FCN and CNN modules constituting the attack model.  
        """
        model = self.target_train_model
        
        # for layer outputs
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.create_layer_components(layers)
            
        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_component(self.output_size)
            
        # for loss
        if self.exploit_loss:
            self.create_loss_component()
            
        # for gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.create_gradient_components(model, layers)

        # encoder module,已放在初始化attack_model_init里面
#        self.encoder = encoder(self.encoderinputs_real)            
            
    def create_layer_components(self, layers):
        """
        Creates CNN or FCN components for layers to exploit
        """
        for l in self.layers_to_exploit:
            # For each layer to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[l-1]   
#            print(layer)
            input_shape = len(layer[0])
#            print(input_shape)           
            
            #先只做全连接的网络components，对于图像数据，用CNN的模块
            module = fcn_module(input_shape, 100)
            
            # 把模块的输入节点和输出节点保存
            # 但由于torch没有类似于TensorFlow计算节点的写法，
            #所以先把module保存，当有真正的输入数据时在另外保存使用                
            self.attackinputs.append(module)
            self.encoderinputs_size += 64
            self.encoderinputs.append(module)  
            self.attackinputs_size.append(input_shape)

    def create_label_component(self, output_size):
        """
        Creates component if OHE label is to be exploited
        """
        module = fcn_module(output_size)
        self.attackinputs.append(module)
        self.encoderinputs_size += 64
        self.encoderinputs.append(module)
        self.attackinputs_size.append(output_size)         
            
    def create_loss_component(self):
        """
        Creates component if loss value is to be exploited
        """
        module = fcn_module(1, 100)
        self.attackinputs.append(module)
        self.encoderinputs_size += 64
        self.encoderinputs.append(module) 
        self.attackinputs_size.append(1)    
        
    def create_gradient_components(self, model, layers):
        """
        Creates CNN/FCN component for gradient values of layers of gradients to exploit
        """
        for layerindex in self.gradients_to_exploit:
            # For each gradient to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[layerindex-1]
            shape = self.attack_utils.get_gradshape(layer)   
            module = cnn_for_fcn_gradients(shape)
            self.attackinputs.append(module)      
            self.encoderinputs_size += 256
            self.encoderinputs.append(module)        
            self.attackinputs_size.append(shape[1])  
            
    # 初始化攻击模型的结构
    def initialize_attack_model(self):
        """
        Initializes a `tf.keras.Model` object for attack model.
        The output of the attack is the output of the encoder module.
        """
        # 使用定义一个并联网络
        self.attackmodel = attack_model_init(self.attackinputs,self.encoderinputs_size,self.attackinputs_size)
#        print(self.attackmodel)
        
#        output = self.encoder
#        self.attackmodel = tf.compat.v1.keras.Model(inputs=self.attackinputs,
#                                                    outputs=output)

    def get_layer_outputs(self, model, features):
        """
        Get the intermediate computations (activations) of 
        the hidden layers of the given target model.
        """
        # 获取所有层的输出,用模型自定义的方法获取
        layer_outputs,_ = model.eval_layer_output(features)
        
        # 根据泄露层数确认输出的列表
        for l in self.layers_to_exploit:
            self.inputArray.append(layer_outputs[l-1])
 

    def get_labels(self, labels):
        """
        Retrieves the one-hot encoding of the given labels.
        """
        ohe_labels = self.attack_utils.one_hot_encoding(
            labels, self.ohencoding)
        return ohe_labels

    def get_loss(self, model, features, labels):
        """
        Computes the loss for given model on given features and labels
        """
        logits = model(features)
        # 这里应该输出每个example的loss，而不是一个batch的loss
        loss = CrossEntropyLoss_exampleloss(logits, labels)   
        # tensor(1.2832, grad_fn=<NllLossBackward>)


        return loss

    def compute_gradients(self, model, features, labels):
        """
        Computes gradients given the features and labels using the loss
        """
        # 一个batch输入
#        print(len(features))
#        split_features = self.attack_utils.split(features)   
#        split_labels = self.attack_utils.split(labels)     
        
        gradient_arr = []
        # 把batch分为一个个data去计算
        torch_dataset = torch.utils.data.TensorDataset(features, labels)
        train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1,
                                           shuffle=False)
        
        for (step,(feature,label)) in enumerate(train_loader):
            cur_arr = []
            logits = model(feature)
            loss = CrossEntropyLoss(logits, label)
            
            with torch.no_grad():
                loss.backward()                
                for name, parms in model.named_parameters():
                    # Add gradient wrt crossentropy loss to gradient_arr
                    # 包含着weight和bias
                    cur_arr.append(parms.grad)
            # 这里用深拷贝，由于cur_arr是一个list，直接append会指向同一空间        
            gradient_arr.append(copy.deepcopy(cur_arr))

        return gradient_arr  
        
    def get_gradients(self, model, features, labels):
        """
        Retrieves the gradients for each example.
        """    
        gradient_arr = self.compute_gradients(model, features, labels)  
        batch_gradients = []     
        # gradient_arr 一个batch，每个data example
        for grads in gradient_arr:
            # gradient_arr is a list of size of number of layers having trainable parameters
            gradients_per_example = []
            for g in self.gradients_to_exploit:
                # 这里只取了weight
                g = (g-1)*2                  
#                shape = grads[g].size()                                
                # 变为三维，输入torch定义的CNN的要求:[batch_size, channels, Width, Height]
                toappend = torch.unsqueeze(grads[g],0)
#                reshaped = (1,int(shape[0]), int(shape[1]))
#                toappend = torch.reshape(grads[g], reshaped
                # DP方案需要在攻击模型梯度计算中引入DP噪声
                if(self.if_withDP[0] == True):
                    temp_toappend = torch.squeeze(toappend,0)
                    epsilon = self.if_withDP[2]
                    mu = 0
                    sigma = 1/epsilon                    
                    # 引入DP噪声
                    [row, col] = temp_toappend.size() 
                    added_noise = []
                    for i in range(row):
                        cur_row_noise = []
                        for j in range(col):
                            cur_row_noise.append(random.gauss(mu,sigma))
                        added_noise.append(cur_row_noise)
                    added_noise = torch.tensor(added_noise)
                     #加入噪声
                    temp_toappend +=added_noise
                    toappend = torch.unsqueeze(temp_toappend,0)


#                 每个exmple添加泄露层的权值梯度
                gradients_per_example.append(toappend.numpy())                  
            batch_gradients.append(gradients_per_example)

        # 将batch_gradients的每个examle的同一层梯度抽离出来作为batch
        # 最后可能根据数据格式修改这里的返回值
        exploit_layer_num = len(batch_gradients[0])
        for i in range(exploit_layer_num):
            array = []
            for example in batch_gradients:
                array.append(example[i])
            # 同样使用深拷贝
#            self.inputArray.append(copy.deepcopy(array))
            self.inputArray.append(torch.tensor(np.stack(array)))
                
    def get_gradient_norms(self, model, features, labels):
        """
        Retrieves the gradients for each example
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        # gradient_arr 一个batch，每个data example
        for grads in gradient_arr:
            grad_per_example = []
            for g in range(int(len(grads)/2)):
                g = g*2
                # 计算二范数
                grad_per_example.append(np.linalg.norm(grads[g]))
            batch_gradients.append(grad_per_example)  
        # 按列（同一层）求和再求平均作为整个batch的结果
        result = np.sum(batch_gradients, axis=0) / len(gradient_arr)
        
        return result   
        
        
    def forward_pass(self, model, features, labels):
        """
        Computes and collects necessary inputs for attack model
        """
        # container to extract and collect inputs from target model
        self.inputArray = []

        # Getting the intermediate layer computations
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.get_layer_outputs(model, features)
            
        # Getting the one-hot-encoded labels
        if self.exploit_label:
            ohelabels = self.get_labels(labels)
            self.inputArray.append(ohelabels)
         # Getting the loss value
        if self.exploit_loss:
#            一个example一个loss
            loss = self.get_loss(model, features, labels)
#            loss = torch.reshape(loss, (len(loss), 1))
            self.inputArray.append(loss)
         # Getting the gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.get_gradients(model, features, labels)    
                 
        attack_outputs = self.attackmodel(self.inputArray)
#        exit
        return attack_outputs    
            
    def train_attack(self):
        """
        Trains the attack model
        """
        print('start train_attack module')
        m_features, m_labels, nm_features, nm_labels = self.train_datahandler.load_train()

        model = self.target_train_model
        
        # 目标模型的训练准确度
        pred = model(m_features)
         # 如果使用DP方案，则在预测输出引入DP噪声
        if(self.if_withDP[0] == True):
            # 定义DP噪声
            epsilon = self.if_withDP[1]
            mu = 0
            sigma = 1/epsilon
            
            # 引入DP噪声
            [row, col] = pred.size() 
            added_noise = []
            for i in range(row):
                cur_row_noise = []
                for j in range(col):
                    cur_row_noise.append(random.gauss(mu,sigma))
                added_noise.append(cur_row_noise)
            added_noise = torch.tensor(added_noise)
            # print(added_noise)
             #加入噪声
            pred +=added_noise 
            # print(pred)
        
        acc_train = accuracy_score(m_labels, np.argmax(pred.detach().numpy(), axis=1))
        print('Target model train accuracy = ', acc_train)
        
        # 目标模型的测试准确度
        pred = model(nm_features)
        
         # 如果使用DP方案，则在预测输出引入DP噪声
        if(self.if_withDP[0] == True):
            # 定义DP噪声
            epsilon = self.if_withDP[1]
            mu = 0
            sigma = 1/epsilon
            
            # 引入DP噪声
            [row, col] = pred.size() 
            added_noise = []
            for i in range(row):
                cur_row_noise = []
                for j in range(col):
                    cur_row_noise.append(random.gauss(mu,sigma))
                added_noise.append(cur_row_noise)
            added_noise = torch.tensor(added_noise)
            # print(added_noise)
             #加入噪声
            pred +=added_noise 
            # print(pred)
            
        acc = accuracy_score(nm_labels, np.argmax(pred.detach().numpy(), axis=1))
        print('Target model test accuracy = ', acc)
        
#        # 计算成员数据集与非成员数据集的每层的梯度总差异
#        norm_diff = self.grad_norm_diff()
#        self.logger.info("gradient norm differece(nonmember-member) for each layers = {}".format(norm_diff)) 
#        print("gradient norm differece(nonmember-member) for each layers = {}".format(norm_diff))
        
        #menber,nonmemer for training
        mtrainset_loader, nmtrainset_loader = self.attack_datahandler.load_train_datasetLoader()

        #menber,nonmemer for testing
        mtestset_loader, nmtestset_loader = self.attack_datahandler.load_test_datasetLoader()

        # main training procedure begins
        best_accuracy = 0
        
        attackmodel_opt = self.optimizer(params=self.attackmodel.parameters(),lr=self.learning_rate)
        scheduler = lr_scheduler.LambdaLR(optimizer=attackmodel_opt, lr_lambda=lambda epoch:0.95**epoch)
        
        print(list(self.attackmodel.parameters())[-1])      
        
        
        for e in range(self.epochs): 
            print('this is epoch ',e)
            zipped = zip(mtrainset_loader, nmtrainset_loader)
            for(_, ((mfeatures, mlabels), (nmfeatures, nmlabels))) in enumerate(zipped):
                # Getting outputs of forward pass of attack model
                # batch形式的数据传入                
                moutputs = self.forward_pass(model, mfeatures, mlabels)
                nmoutputs = self.forward_pass(model, nmfeatures, nmlabels)

                # Computing the true values for loss function according
                memtrue = torch.ones(moutputs.shape)
                nonmemtrue = torch.zeros(nmoutputs.shape)
                
                target = torch.cat((memtrue, nonmemtrue), 0)
                probs =  torch.cat((moutputs, nmoutputs), 0)
#                print(target)
#                print(probs)
                
                attackloss = mse(target, probs)
#                print('this is loss:')
                # print(attackloss)
                
                attackmodel_opt.zero_grad()   
                attackloss.backward(retain_graph=True)
                attackmodel_opt.step()
            scheduler.step()
#            print(list(self.attackmodel.parameters()))
                
            # Calculating Attack accuracy            
            attack_accuracy = self.attack_accuracy(mtestset_loader, nmtestset_loader)
            if attack_accuracy > best_accuracy:
                    best_accuracy = attack_accuracy
                    
            print("Epoch {} over :"
                  "Attack test accuracy: {}, Best accuracy : {}"
                  .format(e, attack_accuracy, best_accuracy))
            if (self.epochs - e <=20):
                self.last20_epochs.append(attack_accuracy)
            # 保存最后20个epoch    
            self.logger.info("Epoch {} over,"
                                 "Attack loss: {},"
                                 "Attack accuracy: {}"
                                 .format(e, attackloss, attack_accuracy))
            print(list(self.attackmodel.parameters())[-1]) 
            
#            # 数据记录
#            data = None
#            if os.path.isfile("logs/attack/results") and os.stat("logs/attack/results").st_size > 0:
#                with open('logs/attack/results', 'r+') as json_file:
#                    data = json.load(json_file)
#                    if data:
#                        data = data['result']
#                    else:
#                        data = []
#            if not data:
#                data = []
#            data.append(
#                {self.model_name: {'target_acc': float(acc), 'attack_acc': float(best_accuracy)}})
#            with open('logs/attack/results', 'w+') as json_file:
#                json.dump({'result': data}, json_file)
    
            # logging best attack accuracy
            self.logger.info("Best attack accuracy %.2f%%\n\n",
                             100 * best_accuracy)  
                            
                    
    def attack_accuracy(self, members, nonmembers):
        """
        Computes attack accuracy of the attack model.
        """
        model = self.target_train_model
        
        zipped = zip(members, nonmembers)
        best_accuracy_batch = 0
        for (_,(membatch, nonmembatch)) in enumerate(zipped):
            mfeatures, mlabels = membatch
            nmfeatures, nmlabels = nonmembatch
            
            # Computing the membership probabilities
            mprobs = self.forward_pass(model, mfeatures, mlabels)
            nonmprobs = self.forward_pass(model, nmfeatures, nmlabels)
            probs = torch.cat((mprobs, nonmprobs), 0)

            # true and false matrix
            target_ones = torch.ones(mprobs.shape, dtype=bool)
            target_zeros = torch.zeros(nonmprobs.shape, dtype=bool)
            target = torch.cat((target_ones, target_zeros), 0)
            
            # probs 大于0.5项设置为1，与target的true对比，计算accuracy
            probs_trans = []
            for i in range(len(probs)):     
                probs_trans.append(probs[i] > torch.tensor(0.5))
            probs_trans = torch.tensor(np.stack(probs_trans))
             # accuracy_score(y_true, y_pred)
#            print(target)
#            print(probs_trans)
            acc_result = accuracy_score(target,probs_trans)
            print(acc_result)
#            sad
            if acc_result>best_accuracy_batch:
                best_accuracy_batch = acc_result
                
        return best_accuracy_batch

    def grad_norm_diff(self): 
        '''
        计算成员数据集与非成员数据集的梯度范数
        '''
        #menber,nonmemer for testing
        mtestset_loader, nmtestset_loader = self.attack_datahandler.load_test_datasetLoader()
        model = self.target_train_model
        zipeed = zip(mtestset_loader, nmtestset_loader)
        mresult = []
        nmresult = []
        # member 的梯度范数    
        for (_,((mfeatures, mlabels),(nmfeatures, nmlabels))) in enumerate(zipeed):
            # 一个batch的结果（按每层，每个example的求和平均值）
            mgradientnorm = self.get_gradient_norms(
                model, mfeatures, mlabels)
            nmgradientnorm = self.get_gradient_norms(
                model, nmfeatures, nmlabels)
            mresult.append(mgradientnorm)
            nmresult.append(nmgradientnorm)
        
        # 每个batch 加起来
        memberresult = np.sum(mresult, axis=0)
        nonmemberresult = np.sum(nmresult, axis=0)
        
        # 做差
        diff_nm_m = nonmemberresult - memberresult
#        print(memberresult)
#        print(nonmemberresult)
#        print(diff_nm_m)
        return diff_nm_m


            
                                                                   
        

        
        
        