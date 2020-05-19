import  torch
import random
import  torch.nn as nn
import torch.nn.functional as F

# p_logit: [batch, class_num]
# q_logit: [batch, class_num]
tmps = []
'''
for i in range(20):
    tmp= []
    for j in range(8):
        p_logit = torch.randn(32,16)

        p = F.softmax(p_logit,dim=-1)
        tmp += [p]

    tmps.append(tmp)
print(len(tmps),len(tmp),tmps[0][0].shape)

print(kl_categorical(tmps))
    # return _kl

# print(kl_categorical(p,q))
#py
dd_
# print(_kl2)
'''

t = torch.randn(3, 6,5).cuda()  # 初始化一个tensor，从0到23，形状为（2,3,4）
print("t--->", t)

index1 = torch.randint(low=0,high=5,size=(3,1,1)) # 要选取数据的位置
index1 = index1.repeat(1,1,5)
print("index--->", index1)

data1 = torch.gather(t,dim=1, index=index1.cuda())  # 第一个参数:从第1维挑选， 第二个参数:从该维中挑选的位置
print("data1--->", data1,data1.shape)



# a = torch.randn(,8,30)
#
# b =torch.split(a,1,1)
# print(len(b))
# print(b[0].shape)


