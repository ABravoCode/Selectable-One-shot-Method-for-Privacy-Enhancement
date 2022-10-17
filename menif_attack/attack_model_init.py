# -*- coding: utf-8 -*-

# 攻击模型初始化
import torch
from torch import nn
from create_encoder import encoder

class attack_model_init(nn.Module):
    def __init__(self,attackinputs,encoderinputs_size,attackiputs_size=0):
         super(attack_model_init, self).__init__()
         
         # 把attackinputs列表的模块单独分离出来
         self.len_attackinputs = len(attackinputs)
         for i in range(self.len_attackinputs):
             locals()["block" + str(i)] = attackinputs[i]        
#             print(locals()["block" + str(i)])              
        ### 分别用不同 变量名 定义每个并联模块 ###
         for i in range(self.len_attackinputs):
#             self.Block.('{}'.format(i)) = locals()["block" + str(i)]
             strname = "Block" + str(i)          
             setattr(self,strname,locals()["block" + str(i)])
         self.encoder = encoder(encoderinputs_size) 
         self.attackiputs_size = attackiputs_size                   
             
    def forward(self,inputArray):
        for i in range(len(inputArray)):
            locals()["input" + str(i)] = inputArray[i]            
        
        ### 并联forward ### 把对应数据输入到对应子模块，记录其输出，传递给encoder
        
        # 批量处理的情况
        outputlist = []
        for i in range(len(inputArray)):
            strname = "Block" + str(i)
            module_block = getattr(self,strname)
            result = module_block(locals()["input" + str(i)])
            outputlist.append(result)   
    
        # 扁平化所有子模块的输出,与数据长度到encoder的输入大小一致
        encoder_input = torch.cat((outputlist),1)
        attackoutput = self.encoder(encoder_input)
        return attackoutput  
             

         
         