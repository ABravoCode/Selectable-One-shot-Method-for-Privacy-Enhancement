# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np


class attack_data:
    """
    Attack data class to perform operations on dataset.
    """
    def __init__(self, train_feature_path, train_lable_path, test_feature_path,
                 test_lable_path, batch_size, attack_percentage, normalization=False,
                 input_shape=None):
        self.batch_size = batch_size
        
        # Loading the training (member) dataset
#        self.train_feature = pd.read_csv(train_feature_path)   
        self.train_feature = np.load(train_feature_path)
        self.train_feature = self.generate(self.train_feature)
#        self.train_lable = pd.read_csv(train_lable_path)
        self.train_lable = np.load(train_lable_path)
        self.train_lable = self.generateLong(self.train_lable)
        self.training_size = len(self.train_feature)
                
        # Loading the test (nonmember) dataset
#        self.test_feature = pd.read_csv(test_feature_path)
        self.test_feature = np.load(test_feature_path)
        self.test_feature = self.generate(self.test_feature)
#        self.test_lable = pd.read_csv(test_lable_path)
        self.test_lable = np.load(test_lable_path)
        self.test_lable = self.generateLong(self.test_lable)
        self.testing_size = len(self.test_feature)
        
        
        self.attack_size = int(attack_percentage /
                               float(100) * self.training_size)
        
         # Specifically for image datasets
        self.input_shape = input_shape
        
        self.input_channels = self.input_shape[-1]
        
        self.normalization = normalization        
    
        
    def load_train(self):
        """
        Loads, normalizes and batches training data.
        Returns a tf.data.Dataset object for training
        """
        asize = self.attack_size
        member_feature = self.train_feature[:asize]        
        member_lable = self.train_lable[:asize]
        
        nonmember_feature = self.test_feature[:asize]        
        nonmember_lable = self.test_lable[:asize]

        return member_feature, member_lable, nonmember_feature, nonmember_lable      

    def load_train_datasetLoader(self):
        """
        Loads, normalizes and batches training data.
        Returns a tf.data.Dataset object for training
        """
        asize = self.attack_size
        member_feature = self.train_feature[:asize]        
        member_lable = self.train_lable[:asize]
        
        nonmember_feature = self.test_feature[:asize]        
        nonmember_lable = self.test_lable[:asize]

        # 进行batch划分
        menber_dataset = torch.utils.data.TensorDataset(member_feature, member_lable)
        menber_loader = torch.utils.data.DataLoader(menber_dataset, self.batch_size,
                                                   shuffle=False)        
        # 进行batch划分
        nonmenber_dataset = torch.utils.data.TensorDataset(nonmember_feature, nonmember_lable)
        nonmenber_loader = torch.utils.data.DataLoader(nonmenber_dataset, self.batch_size,
                                                   shuffle=False)   
#        print(list(menber_loader))
        return menber_loader, nonmenber_loader                   
    
    def load_test(self):
        """
        Loads, normalizes and batches training data.
        Returns a tf.data.Dataset object for test attack model
        """
        asize = self.attack_size
        member_feature_test = self.train_feature[asize:]        
        member_lable_test = self.train_lable[asize:]
        
        basize = asize*2
        nonmember_feature_test = self.test_feature[asize:basize]        
        nonmember_lable_test = self.test_lable[asize:basize]

        return member_feature_test, member_lable_test, nonmember_feature_test, nonmember_lable_test
    
    def load_test_datasetLoader(self):
        """
        Loads, normalizes and batches training data.
        Returns a tf.data.Dataset object for training
        """
        asize = self.attack_size
        member_feature = self.train_feature[asize:]        
        member_lable = self.train_lable[asize:]
        
        # 取相等长度的
        basize = asize*2
        nonmember_feature = self.test_feature[asize:basize]        
        nonmember_lable = self.test_lable[asize:basize]

        # 进行batch划分
        menber_dataset = torch.utils.data.TensorDataset(member_feature, member_lable)
        menber_loader = torch.utils.data.DataLoader(menber_dataset, self.batch_size,
                                                   shuffle=False)        
        # 进行batch划分
        nonmenber_dataset = torch.utils.data.TensorDataset(nonmember_feature, nonmember_lable)
        nonmenber_loader = torch.utils.data.DataLoader(nonmenber_dataset, self.batch_size,
                                                   shuffle=False)   
#        print(list(menber_loader))
        return menber_loader, nonmenber_loader   
    
    def generate(self, dataset):
        """
        Parses each record of the dataset and extracts 
        the class (first column of the record) and the 
        features. This assumes 'csv' form of data.
        """     
        features = torch.FloatTensor(dataset)   
#        print(features)
        return features
                                      
    def generateLong(self, dataset):
        """
        Parses each record of the dataset and extracts 
        the class (first column of the record) and the 
        features. This assumes 'csv' form of data.
        """     
        features = torch.LongTensor(dataset)   
#        print(features)
        return features                                 

    
    



